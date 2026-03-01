use std::cmp::Ordering;
use std::collections::HashMap as StdHashMap;
use std::fs;
use std::io::{Read, Write};
use std::path::Path;
use std::time::Instant;

use dary_heap::OctonaryHeap;
use fancy_regex::Regex;
use pyo3::prelude::*;

use ahash::{AHashMap, AHashSet};
use compact_str::CompactString;
use rayon::prelude::*;

/// Default GPT-4 style regex pattern for splitting text into chunks.
/// Each match is categorically homogeneous: pure letters, digit runs,
/// whitespace+letter sequences, punctuation clusters, or whitespace.
/// This categorical separation guarantees that no pair of bytes that
/// straddles two adjacent chunk matches can also appear within a single
/// chunk match — a property the 2-phase trainer relies on.
const GPT4_PATTERN: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+";

/// Splits text into superchunks at hard punctuation boundaries.
/// Adjacent GPT4 chunks within one superchunk can be merged across
/// their boundary in Phase 2; chunks in different superchunks cannot.
const SUPER_CHUNK_PATTERN: &str = r#"[,;:!?()\[\]{}\r\n"]++|\.(?=\s|$)"#;

type Pair = (u32, u32);

/// A Byte Pair Encoding tokenizer.
/// When allow_superchunk=true, uses 2-phase training:
///   Phase 1 — standard BPE over individual GPT4 chunks (within-chunk merges).
///   Phase 2 — BPE over sequences of Phase-1 tokens (cross-chunk merges only).
/// The two merge sequences are interleaved by descending frequency,
/// with Phase 1 winning ties to guarantee prerequisites always land first.
#[pyclass]
pub struct Tokenizer {
    pub merges: StdHashMap<Pair, u32>,
    pub pattern: String,
    compiled_pattern: Regex,
}

fn merge_seq_fingerprint(seq: &[(Pair, u64)]) -> u64 {
    // Deterministic, cheap fingerprint (not cryptographic).
    let mut h: u64 = 14695981039346656037;
    for &(p, f) in seq {
        h ^= p.0 as u64;
        h = h.wrapping_mul(1099511628211);
        h ^= p.1 as u64;
        h = h.wrapping_mul(1099511628211);
        h ^= f;
        h = h.wrapping_mul(1099511628211);
    }
    h
}

fn bytes_to_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}

fn load_phase1_cache(path: &str, vocab_size: u32, pattern: &str) -> Option<Vec<(Pair, u64)>> {
    let p = Path::new(path);
    if !p.exists() { return None; }
    let mut s = String::new();
    fs::File::open(p).ok()?.read_to_string(&mut s).ok()?;
    let mut lines = s.lines();
    let magic = lines.next()?;
    let v2 = if magic == "RUSTBPE_PHASE1_CACHE_V2" {
        true
    } else {
        if magic != "RUSTBPE_PHASE1_CACHE_V1" { return None; }
        false
    };

    let vs_line = lines.next()?;
    let vs = vs_line.strip_prefix("vocab_size=")?.parse::<u32>().ok()?;
    if vs != vocab_size { return None; }

    let pat_line = lines.next()?;
    let pat = pat_line.strip_prefix("pattern=")?;
    if pat != pattern { return None; }

    let mut out: Vec<(Pair, u64)> = Vec::new();
    for line in lines {
        if line.trim().is_empty() { continue; }
        let mut it = line.split_whitespace();
        let a = it.next()?.parse::<u32>().ok()?;
        let b = it.next()?.parse::<u32>().ok()?;
        let f = it.next()?.parse::<u64>().ok()?;
        // V2 may have trailing debug fields; we intentionally ignore them.
        let _ = v2;
        out.push(((a, b), f));
    }
    Some(out)
}

fn save_phase1_cache(path: &str, vocab_size: u32, pattern: &str, phase1_seq: &[(Pair, u64)]) {
    let f = match fs::File::create(path) {
        Ok(x) => x,
        Err(_) => return,
    };
    let mut f = std::io::BufWriter::new(f);

    if writeln!(f, "RUSTBPE_PHASE1_CACHE_V2").is_err() { return; }
    if writeln!(f, "vocab_size={}", vocab_size).is_err() { return; }
    if writeln!(f, "pattern={}", pattern).is_err() { return; }

    let mut token_bytes: Vec<Vec<u8>> = (0..256_u32).map(|i| vec![i as u8]).collect();
    for (k, &((a, b), _)) in phase1_seq.iter().enumerate() {
        let mid = 256 + k as u32;
        if token_bytes.len() <= mid as usize { token_bytes.resize(mid as usize + 1, Vec::new()); }
        let mut mb = token_bytes[a as usize].clone();
        mb.extend(&token_bytes[b as usize]);
        token_bytes[mid as usize] = mb;
    }

    for (k, &((a, b), freq)) in phase1_seq.iter().enumerate() {
        let mid = 256 + k as u32;
        let ahex = bytes_to_hex(&token_bytes[a as usize]);
        let bhex = bytes_to_hex(&token_bytes[b as usize]);
        let mhex = bytes_to_hex(&token_bytes[mid as usize]);
        if writeln!(f, "{} {} {} {} {} {}", a, b, freq, ahex, bhex, mhex).is_err() { return; }
    }
}

fn load_phase2_cache(
    path: &str,
    vocab_size: u32,
    pattern: &str,
    phase1_fp: u64,
) -> Option<Vec<(Pair, u64)>> {
    let p = Path::new(path);
    if !p.exists() { return None; }
    let mut s = String::new();
    fs::File::open(p).ok()?.read_to_string(&mut s).ok()?;
    let mut lines = s.lines();
    let magic = lines.next()?;
    let v2 = if magic == "RUSTBPE_PHASE2_CACHE_V2" {
        true
    } else {
        if magic != "RUSTBPE_PHASE2_CACHE_V1" { return None; }
        false
    };

    let vs_line = lines.next()?;
    let vs = vs_line.strip_prefix("vocab_size=")?.parse::<u32>().ok()?;
    if vs != vocab_size { return None; }

    let pat_line = lines.next()?;
    let pat = pat_line.strip_prefix("pattern=")?;
    if pat != pattern { return None; }

    let fp_line = lines.next()?;
    let fp = fp_line.strip_prefix("phase1_fp=")?.parse::<u64>().ok()?;
    if fp != phase1_fp { return None; }

    let mut out: Vec<(Pair, u64)> = Vec::new();
    for line in lines {
        if line.trim().is_empty() { continue; }
        let mut it = line.split_whitespace();
        let l = it.next()?.parse::<u32>().ok()?;
        let r = it.next()?.parse::<u32>().ok()?;
        let f = it.next()?.parse::<u64>().ok()?;
        // V2 may have trailing debug fields; we intentionally ignore them.
        let _ = v2;
        out.push(((l, r), f));
    }
    Some(out)
}

fn save_phase2_cache(
    path: &str,
    vocab_size: u32,
    pattern: &str,
    phase1_fp: u64,
    phase1_seq: &[(Pair, u64)],
    phase2_seq: &[(Pair, u64)],
) {
    let f = match fs::File::create(path) {
        Ok(x) => x,
        Err(_) => return,
    };
    let mut f = std::io::BufWriter::new(f);

    if writeln!(f, "RUSTBPE_PHASE2_CACHE_V2").is_err() { return; }
    if writeln!(f, "vocab_size={}", vocab_size).is_err() { return; }
    if writeln!(f, "pattern={}", pattern).is_err() { return; }
    if writeln!(f, "phase1_fp={}", phase1_fp).is_err() { return; }

    // Build token bytes for Phase 1 and Phase 2 internal id spaces so we can write
    // human-debuggable merged strings to the cache.
    let phase2_id_offset: u32 = phase1_seq.len() as u32;
    let mut token_bytes: Vec<Vec<u8>> = (0..256_u32).map(|i| vec![i as u8]).collect();

    for (k, &((l, r), _)) in phase1_seq.iter().enumerate() {
        let mid = 256 + k as u32;
        if token_bytes.len() <= mid as usize { token_bytes.resize(mid as usize + 1, Vec::new()); }
        let mut mb = token_bytes[l as usize].clone();
        mb.extend(&token_bytes[r as usize]);
        token_bytes[mid as usize] = mb;
    }

    for (k, &((l, r), _)) in phase2_seq.iter().enumerate() {
        let mid = 256 + phase2_id_offset + k as u32;
        if token_bytes.len() <= mid as usize { token_bytes.resize(mid as usize + 1, Vec::new()); }
        let mut mb = token_bytes[l as usize].clone();
        mb.extend(&token_bytes[r as usize]);
        token_bytes[mid as usize] = mb;
    }

    for (k, &((l, r), freq)) in phase2_seq.iter().enumerate() {
        let mid = 256 + phase2_id_offset + k as u32;
        let lhex = bytes_to_hex(&token_bytes[l as usize]);
        let rhex = bytes_to_hex(&token_bytes[r as usize]);
        let mhex = bytes_to_hex(&token_bytes[mid as usize]);
        if writeln!(f, "{} {} {} {} {} {}", l, r, freq, lhex, rhex, mhex).is_err() { return; }
    }
}

impl Default for Tokenizer {
    fn default() -> Self { Self::new() }
}

// ========================= Internal helpers =========================

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct Span {
    chunk_start: u32,
    chunk_end: u32,
    start: usize,
    end: usize,
}

impl Span {
    #[inline]
    fn within_chunk(chunk: u32, start: usize, end: usize) -> Self {
        Self { chunk_start: chunk, chunk_end: chunk, start, end }
    }

    /// Two spans can merge within-chunk when contiguous inside the same chunk.
    #[inline]
    fn can_within_merge(a: Span, b: Span) -> bool {
        a.chunk_start == a.chunk_end
            && b.chunk_start == b.chunk_end
            && a.chunk_start == b.chunk_start
            && a.end == b.start
    }

    /// A span covers its full chunk if it starts at 0 and ends at chunk_len.
    #[inline]
    fn is_full_chunks(&self, chunk_lens: &[usize]) -> bool {
        let ce = self.chunk_end as usize;
        self.start == 0 && chunk_lens.get(ce).is_some_and(|&len| self.end == len)
    }

    /// Two spans may merge across a chunk boundary when both are full-chunk spans
    /// on adjacent chunks.
    #[inline]
    fn adjacent_full_chunks(a: Span, b: Span, chunk_lens: &[usize]) -> bool {
        a.is_full_chunks(chunk_lens)
            && b.is_full_chunks(chunk_lens)
            && a.chunk_end + 1 == b.chunk_start
    }
}

#[derive(Clone, Debug)]
struct Word {
    ids: Vec<u32>,
    spans: Vec<Span>,
    chunk_lens: Vec<usize>,
}

impl Word {
    /// Build a Word from a single contiguous byte sequence (all tokens in chunk 0).
    #[inline]
    fn new_single_chunk(ids: Vec<u32>) -> Self {
        let n = ids.len();
        let spans = (0..n).map(|i| Span::within_chunk(0, i, i + 1)).collect();
        Self { ids, spans, chunk_lens: vec![n] }
    }

    #[cfg(test)]
    /// Build a Word from a sequence of string chunks. Each chunk becomes a run
    /// of byte-level token IDs in its own chunk slot.
    #[inline]
    fn new_from_chunks(chunks: &[CompactString]) -> Self {
        let mut ids = Vec::new();
        let mut spans = Vec::new();
        let mut chunk_lens = Vec::with_capacity(chunks.len());
        for (ci, chunk) in chunks.iter().enumerate() {
            let bytes = chunk.as_bytes();
            chunk_lens.push(bytes.len());
            for (j, &b) in bytes.iter().enumerate() {
                ids.push(b as u32);
                spans.push(Span::within_chunk(ci as u32, j, j + 1));
            }
        }
        Self { ids, spans, chunk_lens }
    }

    /// Build a Phase 2 Word from already-compressed token IDs.
    /// Each token ID occupies its own chunk slot (chunk_len == 1), so:
    ///   - can_within_merge is never true (different chunk_start per token)
    ///   - adjacent_full_chunks is always true for neighbours (allow_superchunk=true)
    /// This restricts Phase 2 training to cross-chunk merges only.
    #[inline]
    fn new_phase2(token_ids: Vec<u32>) -> Self {
        let n = token_ids.len();
        let spans = (0..n).map(|i| Span::within_chunk(i as u32, 0, 1)).collect();
        Self { ids: token_ids, spans, chunk_lens: vec![1usize; n] }
    }

    #[inline]
    fn pairs_mergeable(&self, allow_superchunk: bool) -> impl Iterator<Item = Pair> + '_ {
        self.ids.windows(2).enumerate().filter_map(move |(i, w)| {
            let sa = self.spans[i];
            let sb = self.spans[i + 1];
            let ok = Span::can_within_merge(sa, sb)
                || (allow_superchunk && Span::adjacent_full_chunks(sa, sb, &self.chunk_lens));
            ok.then_some((w[0], w[1]))
        })
    }

    fn merge_pair(&mut self, pair: Pair, new_id: u32, allow_superchunk: bool) -> Vec<(Pair, i32)> {
        let (a, b) = pair;
        let n = self.ids.len();
        if n < 2 { return Vec::new(); }

        let mut out = Vec::with_capacity(n);
        let mut out_spans = Vec::with_capacity(n);
        let mut deltas: Vec<(Pair, i32)> = Vec::with_capacity(6);
        let mut i = 0;

        while i < n {
            let is_match = i + 1 < n
                && self.ids[i] == a
                && self.ids[i + 1] == b
                && (Span::can_within_merge(self.spans[i], self.spans[i + 1])
                    || (allow_superchunk && Span::adjacent_full_chunks(
                        self.spans[i], self.spans[i + 1], &self.chunk_lens)));

            if is_match {
                let left = out.last().copied();
                let right = if i + 2 < n { Some(self.ids[i + 2]) } else { None };
                if let Some(x) = left {
                    deltas.push(((x, a), -1));
                    deltas.push(((x, new_id), 1));
                }
                deltas.push(((a, b), -1));
                if let Some(y) = right {
                    deltas.push(((b, y), -1));
                    deltas.push(((new_id, y), 1));
                }
                let sa = self.spans[i];
                let sb = self.spans[i + 1];
                let merged_span = if Span::can_within_merge(sa, sb) {
                    Span { chunk_start: sa.chunk_start, chunk_end: sa.chunk_end, start: sa.start, end: sb.end }
                } else {
                    Span { chunk_start: sa.chunk_start, chunk_end: sb.chunk_end, start: 0, end: self.chunk_lens[sb.chunk_end as usize] }
                };
                out.push(new_id);
                out_spans.push(merged_span);
                i += 2;
            } else {
                out.push(self.ids[i]);
                out_spans.push(self.spans[i]);
                i += 1;
            }
        }
        self.ids = out;
        self.spans = out_spans;
        deltas
    }
}

#[derive(Debug, Eq)]
struct MergeJob {
    pair: Pair,
    count: u64,
    pos: Vec<usize>,
}

impl PartialEq for MergeJob {
    fn eq(&self, other: &Self) -> bool { self.count == other.count && self.pair == other.pair }
}
impl PartialOrd for MergeJob {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}
impl Ord for MergeJob {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.count != other.count { self.count.cmp(&other.count) }
        else { other.pair.cmp(&self.pair) }
    }
}

#[cfg(test)]
#[inline]
fn count_pairs_parallel(words: &[Word], counts: &[i32]) -> (AHashMap<Pair, i32>, AHashMap<Pair, Vec<usize>>) {
    count_pairs_parallel_with_options(words, counts, false)
}

#[inline]
fn count_pairs_parallel_with_options(
    words: &[Word],
    counts: &[i32],
    allow_superchunk: bool,
) -> (AHashMap<Pair, i32>, AHashMap<Pair, Vec<usize>>) {
    words.par_iter().enumerate()
        .map(|(i, w)| {
            let mut local_pc: AHashMap<Pair, i32> = AHashMap::new();
            let mut local_wtu: AHashMap<Pair, Vec<usize>> = AHashMap::new();
            if w.ids.len() >= 2 && counts[i] != 0 {
                for pair in w.pairs_mergeable(allow_superchunk) {
                    *local_pc.entry(pair).or_default() += counts[i];
                    local_wtu.entry(pair).or_default().push(i);
                }
            }
            (local_pc, local_wtu)
        })
        .reduce(
            || (AHashMap::new(), AHashMap::new()),
            |(mut apc, mut awtu), (pc, wtu)| {
                for (k, v) in pc { *apc.entry(k).or_default() += v; }
                for (k, s) in wtu { awtu.entry(k).or_default().extend(s); }
                (apc, awtu)
            },
        )
}

fn superchunk_ranges(text: &str, splitter: &Regex) -> Vec<(usize, usize)> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    if bytes.is_empty() { return out; }
    let mut start = 0usize;
    for m in splitter.find_iter(text) {
        let mut end = m.expect("regex match failed").end();
        while end < bytes.len() && bytes[end].is_ascii_whitespace() { end += 1; }
        if start < end { out.push((start, end)); }
        start = end;
    }
    if start < bytes.len() { out.push((start, bytes.len())); }
    out
}

// ========================= Core BPE training =========================

/// Run BPE on words/counts and return the ordered merge sequence.
/// Entry k is (pair, freq_at_merge_time); the corresponding new_id is 256 + k.
/// Does not mutate any external state.
fn train_and_record(
    mut words: Vec<Word>,
    counts: Vec<i32>,
    vocab_size: u32,
    allow_superchunk: bool,
    id_offset: u32,
) -> Vec<(Pair, u64)> {
    assert!(vocab_size >= 256, "vocab_size must be at least 256");
    let num_merges = vocab_size - 256;
    let mut merge_sequence: Vec<(Pair, u64)> = Vec::with_capacity(num_merges as usize);

    log::info!(
        "train_and_record: {} merges requested over {} unique sequences (superchunk={})",
        num_merges, words.len(), allow_superchunk
    );

    let (mut pair_counts, mut where_to_update) =
        count_pairs_parallel_with_options(&words, &counts, allow_superchunk);

    log::info!("train_and_record: {} unique pairs in initial heap", pair_counts.len());

    let mut heap = OctonaryHeap::with_capacity(pair_counts.len());
    for (pair, pos) in where_to_update.drain() {
        let c = *pair_counts.get(&pair).unwrap_or(&0);
        if c > 0 { heap.push(MergeJob { pair, count: c as u64, pos }); }
    }

    let mut merges_done = 0u32;
    let mut last_log_percent = 0u32;

    while merges_done < num_merges {
        let Some(mut top) = heap.pop() else { break };

        let current = *pair_counts.get(&top.pair).unwrap_or(&0);
        if current <= 0 { continue; }
        if top.count != current as u64 {
            top.count = current as u64;
            heap.push(top);
            continue;
        }

        let new_id = 256 + id_offset + merges_done;
        merge_sequence.push((top.pair, top.count));

        let mut local_pos_updates: AHashMap<Pair, AHashSet<usize>> = AHashMap::new();
        for &word_idx in &top.pos {
            let changes = words[word_idx].merge_pair(top.pair, new_id, allow_superchunk);
            for (pair, delta) in changes {
                let delta_total = delta * counts[word_idx];
                if delta_total != 0 {
                    *pair_counts.entry(pair).or_default() += delta_total;
                    if delta > 0 { local_pos_updates.entry(pair).or_default().insert(word_idx); }
                }
            }
        }

        for (pair, pos) in local_pos_updates {
            let cnt = *pair_counts.get(&pair).unwrap_or(&0);
            if cnt > 0 {
                heap.push(MergeJob { pair, count: cnt as u64, pos: pos.into_iter().collect() });
            }
        }

        merges_done += 1;

        let pct = if num_merges > 0 { (merges_done * 100) / num_merges } else { 100 };
        if pct > last_log_percent {
            log::info!(
                "train_and_record: {}% ({}/{}) superchunk={} — {:?} -> {} (freq {})",
                pct, merges_done, num_merges, allow_superchunk, top.pair, new_id, top.count
            );
            last_log_percent = pct;
        }
    }

    log::info!("train_and_record: done — {} merges recorded (superchunk={})", merge_sequence.len(), allow_superchunk);
    merge_sequence
}

// ========================= Phase 2 helpers =========================

/// Encode a single GPT4 chunk to its Phase 1 token ID by applying Phase 1
/// merges greedily (lowest new_id first). Panics if the chunk does not reduce
/// to exactly one token, indicating vocab_size was too small for Phase 1.
#[cfg(test)]
fn encode_single_chunk(chunk: &str, phase1_seq: &[(Pair, u64)]) -> Option<u32> {
    let merges: AHashMap<Pair, u32> = phase1_seq
        .iter()
        .enumerate()
        .map(|(k, &(pair, _))| (pair, 256 + k as u32))
        .collect();

    encode_single_chunk_with_merges(chunk, &merges)
}

fn encode_single_chunk_with_merges(chunk: &str, merges: &AHashMap<Pair, u32>) -> Option<u32> {
    let mut ids: Vec<u32> = chunk.bytes().map(|b| b as u32).collect();
    loop {
        let best = ids.windows(2).enumerate()
            .filter_map(|(i, w)| merges.get(&(w[0], w[1])).map(|&nid| (i, nid)))
            .min_by_key(|&(_, nid)| nid);
        match best {
            None => break,
            Some((idx, new_id)) => { ids[idx] = new_id; ids.remove(idx + 1); }
        }
    }

    (ids.len() == 1).then_some(ids[0])
}

/// Build Phase 2 Word objects from superchunk sequences.
/// Each chunk is encoded to its Phase 1 token ID. The resulting Words use
/// new_phase2 so every adjacent pair satisfies adjacent_full_chunks only —
/// Phase 2 training learns cross-chunk merges exclusively.
fn build_phase2_words(
    seq_counts: AHashMap<Vec<CompactString>, i32>,
    phase1_seq: &[(Pair, u64)],
) -> (Vec<Word>, Vec<i32>) {
    let total = seq_counts.len();
    let phase1_merges: AHashMap<Pair, u32> = phase1_seq
        .iter()
        .enumerate()
        .map(|(k, &(pair, _))| (pair, 256 + k as u32))
        .collect();
    log::info!(
        "build_phase2_words: encoding {} unique superchunk sequences to Phase 1 tokens",
        total
    );

    let mut words = Vec::with_capacity(seq_counts.len());
    let mut counts = Vec::with_capacity(seq_counts.len());
    let mut skipped = 0usize;
    let mut processed = 0usize;
    let mut chunk_cache: AHashMap<CompactString, Option<u32>> = AHashMap::new();
    let start = Instant::now();
    let log_every = 1_000_000usize;
    for (chunk_seq, count) in seq_counts {
        // Convert chunk sequence to Phase 1 token IDs, but salvage usable spans:
        // we keep only contiguous runs of compressible chunks (length >= 2).
        let mut run: Vec<u32> = Vec::new();
        let mut any_bad = false;
        for c in &chunk_seq {
            let enc = if let Some(v) = chunk_cache.get(c) {
                *v
            } else {
                let v = encode_single_chunk_with_merges(c.as_str(), &phase1_merges);
                chunk_cache.insert(c.clone(), v);
                v
            };
            match enc {
                Some(id) => { run.push(id); }
                None => {
                    any_bad = true;
                    if run.len() >= 2 {
                        words.push(Word::new_phase2(std::mem::take(&mut run)));
                        counts.push(count);
                    } else {
                        run.clear();
                    }
                }
            }
        }
        if run.len() >= 2 {
            words.push(Word::new_phase2(run));
            counts.push(count);
        }
        if any_bad { skipped += 1; }

        processed += 1;
        if processed % log_every == 0 || processed == total {
            let pct = if total > 0 { (processed * 100) / total } else { 100 };
            let elapsed = start.elapsed().as_secs_f64();
            let rate = if elapsed > 0.0 { processed as f64 / elapsed } else { 0.0 };
            let remaining = total.saturating_sub(processed) as f64;
            let eta_s = if rate > 0.0 { remaining / rate } else { 0.0 };
            log::info!(
                "build_phase2_words: {}% ({}/{}) — emitted_words={} skipped_seq={} rate={:.0}/s eta={:.0}s",
                pct, processed, total, words.len(), skipped, rate, eta_s
            );
        }
    }

    if skipped > 0 {
        log::info!(
            "build_phase2_words: skipped {} sequences containing chunks not compressed to a single Phase 1 token",
            skipped
        );
    }
    log::info!(
        "build_phase2_words: built {} Phase 2 words in {:.2}s",
        words.len(),
        start.elapsed().as_secs_f64()
    );
    (words, counts)
}

/// Remap a token ID from phase-local numbering to global numbering.
/// Base bytes (< 256) pass through unchanged. IDs >= 256 must have an entry
/// in id_remap; a missing entry indicates a prerequisite merge was not yet applied.
#[inline]
fn remap_id(id: u32, id_remap: &AHashMap<u32, u32>) -> u32 {
    if id < 256 { id } else {
        *id_remap.get(&id).unwrap_or_else(|| panic!(
            "Token ID {} referenced before its producing merge was processed. \
             This indicates a frequency-ordering invariant violation.", id
        ))
    }
}

/// Interleave Phase 1 and Phase 2 merge sequences into a single global merge table.
///
/// At each step the phase with the higher current-head frequency is taken;
/// Phase 1 wins ties, guaranteeing that every within-chunk merge that produces a
/// token referenced by a Phase 2 pair lands before that Phase 2 merge (provable
/// because cross-chunk pair frequency <= min frequency of its constituent tokens).
///
/// Token IDs from each phase's internal numbering (256 + local_index) are remapped
/// to a single monotonically increasing global sequence starting at 256.
fn interleave_merge_sequences(
    vocab_size: u32,
    phase1_seq: Vec<(Pair, u64)>,
    phase2_seq: Vec<(Pair, u64)>,
) -> StdHashMap<Pair, u32> {
    let total = phase1_seq.len() + phase2_seq.len();
    log::info!(
        "interleave_merge_sequences: {} Phase 1 + {} Phase 2 = {} total merges",
        phase1_seq.len(), phase2_seq.len(), total
    );

    let mut id_remap: AHashMap<u32, u32> = AHashMap::with_capacity(total);
    let mut final_merges: StdHashMap<Pair, u32> = StdHashMap::with_capacity(total);
    let mut next_id: u32 = 256;
    let mut ptr1 = 0usize;
    let mut ptr2 = 0usize;

    while (ptr1 < phase1_seq.len() || ptr2 < phase2_seq.len()) && next_id < vocab_size {
        // Phase 1 wins on tie: guarantees cross-chunk prerequisites land first.
        let take_phase1 = ptr2 >= phase2_seq.len()
            || (ptr1 < phase1_seq.len() && phase1_seq[ptr1].1 >= phase2_seq[ptr2].1);

        let phase2_id_offset: u32 = phase1_seq.len() as u32;

        let (pair, internal_new_id) = if take_phase1 {
            let (pair, _) = phase1_seq[ptr1];
            let iid = 256 + ptr1 as u32;
            ptr1 += 1;
            (pair, iid)
        } else {
            let (pair, _) = phase2_seq[ptr2];
            let iid = 256 + phase2_id_offset + ptr2 as u32;
            ptr2 += 1;
            (pair, iid)
        };

        let real_pair = (remap_id(pair.0, &id_remap), remap_id(pair.1, &id_remap));

        if let Some(&existing_id) = final_merges.get(&real_pair) {
            // The same merge pair can appear in both phases (or be encountered twice).
            // Reuse the existing global token ID to avoid creating gaps and missing-token
            // references later when materializing token byte strings.
            id_remap.insert(internal_new_id, existing_id);
        } else {
            let real_new_id = next_id;
            next_id += 1;
            id_remap.insert(internal_new_id, real_new_id);
            final_merges.insert(real_pair, real_new_id);
        }
    }

    log::info!("interleave_merge_sequences: produced {} global merge entries", final_merges.len());
    final_merges
}

// ========================= Python-facing Tokenizer =========================

#[pymethods]
impl Tokenizer {
    #[new]
    pub fn new() -> Self {
        Self {
            merges: StdHashMap::new(),
            pattern: String::new(),
            compiled_pattern: Regex::new("").expect("empty regex should always compile"),
        }
    }

    /// Train from a streaming iterator.
    ///
    /// allow_superchunk=False (default): standard single-phase BPE over GPT4 chunks.
    /// allow_superchunk=True: 2-phase training.
    ///   Phase 1 learns within-chunk merges; Phase 2 learns cross-chunk merges over
    ///   superchunk sequences where each chunk is already a single Phase 1 token.
    ///   Results are interleaved by frequency into one merge table.
    ///
    /// The streaming pass always collects both chunk_counts and seq_counts in a
    /// single iteration when allow_superchunk=True, so the iterator is consumed once.
    #[pyo3(signature = (iterator, vocab_size, pattern=None, allow_superchunk=false, max_superchunk_chunks=4096, tokenizer_dir=None))]
    #[pyo3(text_signature = "(self, iterator, vocab_size, pattern=None, allow_superchunk=False, max_superchunk_chunks=4096, tokenizer_dir=None)")]
    pub fn train_from_iterator(
        &mut self,
        py: pyo3::Python<'_>,
        iterator: &pyo3::Bound<'_, pyo3::PyAny>,
        vocab_size: u32,
        pattern: Option<String>,
        allow_superchunk: bool,
        max_superchunk_chunks: usize,
        tokenizer_dir: Option<String>,
    ) -> PyResult<()> {
        let pattern_str = pattern.unwrap_or_else(|| GPT4_PATTERN.to_string());
        self.pattern = pattern_str.clone();
        self.compiled_pattern = Regex::new(&pattern_str).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Invalid regex pattern: {}", e))
        })?;

        let (phase1_cache_path, phase2_cache_path) = if let Some(ref d) = tokenizer_dir {
            let dir = Path::new(d);
            (
                Some(dir.join("phase1_cache.txt").to_string_lossy().to_string()),
                Some(dir.join("phase2_cache.txt").to_string_lossy().to_string()),
            )
        } else {
            (None, None)
        };

        // Fast path: if both caches exist and are valid, skip ingestion/training.
        if allow_superchunk {
            if let (Some(ref p1), Some(ref p2)) = (&phase1_cache_path, &phase2_cache_path) {
                if let Some(p1_seq) = load_phase1_cache(p1, vocab_size, &pattern_str) {
                    let fp = merge_seq_fingerprint(&p1_seq);
                    if let Some(p2_seq) = load_phase2_cache(p2, vocab_size, &pattern_str, fp) {
                        log::info!(
                            "train_from_iterator: loaded Phase 1 + Phase 2 caches from {}",
                            tokenizer_dir.as_deref().unwrap_or("<unknown>")
                        );
                        self.merges = interleave_merge_sequences(vocab_size, p1_seq, p2_seq);
                        return Ok(());
                    }
                }
            }
        }

        let mut cached_phase1: Option<Vec<(Pair, u64)>> = None;
        if allow_superchunk {
            if let Some(ref p) = phase1_cache_path {
                cached_phase1 = load_phase1_cache(p, vocab_size, &pattern_str);
                if let Some(ref seq) = cached_phase1 {
                    log::info!(
                        "train_from_iterator: loaded Phase 1 cache from {} ({} merges)",
                        p,
                        seq.len()
                    );
                }
            }
        }

        let collect_chunk_counts = allow_superchunk && cached_phase1.is_none();

        let py_iter: pyo3::Py<pyo3::PyAny> = unsafe {
            pyo3::Py::from_owned_ptr_or_err(py, pyo3::ffi::PyObject_GetIter(iterator.as_ptr()))?
        };

        // Phase 1 data: frequency of each unique GPT4 chunk.
        let mut chunk_counts: AHashMap<CompactString, i32> = AHashMap::new();
        // Phase 2 data: frequency of each unique superchunk sequence.
        // Only populated when allow_superchunk=true.
        let mut seq_counts: AHashMap<Vec<CompactString>, i32> = AHashMap::new();

        let superchunk_splitter =
            Regex::new(SUPER_CHUNK_PATTERN).expect("SUPER_CHUNK_PATTERN must compile");

        let buffer_size: usize = 8192;
        let mut buf: Vec<String> = Vec::with_capacity(buffer_size);

        log::info!(
            "train_from_iterator: starting ingestion (vocab_size={}, allow_superchunk={}, buffer_size={})",
            vocab_size, allow_superchunk, buffer_size
        );
        let mut total_texts = 0u64;

        let refill = |buf: &mut Vec<String>| -> PyResult<bool> {
            pyo3::Python::attach(|py| {
                buf.clear();
                let it = py_iter.bind(py);
                loop {
                    if buf.len() >= buffer_size { return Ok(false); }
                    let next_obj = unsafe {
                        pyo3::Bound::from_owned_ptr_or_opt(py, pyo3::ffi::PyIter_Next(it.as_ptr()))
                    };
                    match next_obj {
                        Some(obj) => buf.push(obj.extract()?),
                        None => return if pyo3::PyErr::occurred(py) {
                            Err(pyo3::PyErr::fetch(py))
                        } else { Ok(true) },
                    }
                }
            })
        };

        loop {
            let exhausted = refill(&mut buf)?;
            if buf.is_empty() && exhausted { break; }

            total_texts += buf.len() as u64;
            let pattern = self.compiled_pattern.clone();

            if allow_superchunk {
                let sc_splitter = superchunk_splitter.clone();
                if collect_chunk_counts {
                    // Single parallel pass collecting BOTH chunk_counts and seq_counts.
                    // For each text: split into superchunks, then regex-split each superchunk.
                    // Each GPT4 chunk contributes to chunk_counts; each multi-chunk superchunk
                    // contributes its chunk sequence to seq_counts.
                    let (local_chunks, local_seqs): (
                        AHashMap<CompactString, i32>,
                        AHashMap<Vec<CompactString>, i32>,
                    ) = py.detach(|| {
                        buf.par_iter()
                            .map(|s| {
                                let mut chunk_m: AHashMap<CompactString, i32> = AHashMap::new();
                                let mut seq_m: AHashMap<Vec<CompactString>, i32> = AHashMap::new();
                                for (a, b) in superchunk_ranges(s, &sc_splitter) {
                                    let sc = &s[a..b];
                                    let mut pieces: Vec<CompactString> = Vec::new();
                                    for mat in pattern.find_iter(sc) {
                                        let cs = CompactString::from(mat.expect("regex match failed").as_str());
                                        *chunk_m.entry(cs.clone()).or_default() += 1;
                                        pieces.push(cs);
                                    }
                                    // Only multi-chunk superchunks produce cross-chunk pairs.
                                    if pieces.len() > 1 {
                                        if max_superchunk_chunks == 0 || pieces.len() <= max_superchunk_chunks {
                                            *seq_m.entry(pieces).or_default() += 1;
                                        } else {
                                            for chunk in pieces.chunks(max_superchunk_chunks) {
                                                *seq_m.entry(chunk.to_vec()).or_default() += 1;
                                            }
                                        }
                                    }
                                }
                                (chunk_m, seq_m)
                            })
                            .reduce(
                                || (AHashMap::new(), AHashMap::new()),
                                |(mut ac, mut aseq), (bc, bseq)| {
                                    for (k, v) in bc { *ac.entry(k).or_default() += v; }
                                    for (k, v) in bseq { *aseq.entry(k).or_default() += v; }
                                    (ac, aseq)
                                },
                            )
                    });
                    for (k, v) in local_chunks { *chunk_counts.entry(k).or_default() += v; }
                    for (k, v) in local_seqs { *seq_counts.entry(k).or_default() += v; }
                } else {
                    // Phase 1 cached: collect seq_counts only (avoid chunk-count allocations).
                    let local_seqs: AHashMap<Vec<CompactString>, i32> = py.detach(|| {
                        buf.par_iter()
                            .map(|s| {
                                let mut seq_m: AHashMap<Vec<CompactString>, i32> = AHashMap::new();
                                for (a, b) in superchunk_ranges(s, &sc_splitter) {
                                    let sc = &s[a..b];
                                    let mut pieces: Vec<CompactString> = Vec::new();
                                    for mat in pattern.find_iter(sc) {
                                        let cs = CompactString::from(mat.expect("regex match failed").as_str());
                                        pieces.push(cs);
                                    }
                                    if pieces.len() > 1 {
                                        if max_superchunk_chunks == 0 || pieces.len() <= max_superchunk_chunks {
                                            *seq_m.entry(pieces).or_default() += 1;
                                        } else {
                                            for chunk in pieces.chunks(max_superchunk_chunks) {
                                                *seq_m.entry(chunk.to_vec()).or_default() += 1;
                                            }
                                        }
                                    }
                                }
                                seq_m
                            })
                            .reduce(AHashMap::new, |mut a, b| {
                                for (k, v) in b { *a.entry(k).or_default() += v; }
                                a
                            })
                    });
                    for (k, v) in local_seqs { *seq_counts.entry(k).or_default() += v; }
                }
            } else {
                // Standard mode: collect chunk_counts only.
                let local: AHashMap<CompactString, i32> = py.detach(|| {
                    buf.par_iter()
                        .map(|s| {
                            let mut m: AHashMap<CompactString, i32> = AHashMap::new();
                            for mat in pattern.find_iter(s) {
                                let cs = CompactString::from(mat.expect("regex match failed").as_str());
                                *m.entry(cs).or_default() += 1;
                            }
                            m
                        })
                        .reduce(AHashMap::new, |mut a, b| {
                            for (k, v) in b { *a.entry(k).or_default() += v; }
                            a
                        })
                });
                for (k, v) in local { *chunk_counts.entry(k).or_default() += v; }
            }

            if exhausted { break; }
        }

        log::info!(
            "train_from_iterator: ingestion complete — {} texts, {} unique chunks, {} unique superchunk sequences",
            total_texts, chunk_counts.len(), seq_counts.len()
        );

        if allow_superchunk {
            // ---- Phase 1: within-chunk BPE (or cache) ----
            let phase1_seq = if let Some(seq) = cached_phase1.take() {
                seq
            } else {
                log::info!("train_from_iterator: Phase 1 — within-chunk BPE over {} unique chunks", chunk_counts.len());
                let (p1_words, p1_counts): (Vec<Word>, Vec<i32>) = chunk_counts.iter()
                    .map(|(chunk, &count)| (
                        Word::new_single_chunk(chunk.as_bytes().iter().map(|&b| b as u32).collect()),
                        count,
                    ))
                    .unzip();
                let seq = train_and_record(p1_words, p1_counts, vocab_size, false, 0);
                log::info!("train_from_iterator: Phase 1 complete — {} within-chunk merges", seq.len());
                if let Some(ref p) = phase1_cache_path {
                    save_phase1_cache(p, vocab_size, &pattern_str, &seq);
                    log::info!("train_from_iterator: wrote Phase 1 cache to {}", p);
                }
                seq
            };
            let phase1_fp = merge_seq_fingerprint(&phase1_seq);

            // Free Phase 1 / ingestion memory before Phase 2.
            drop(chunk_counts);
            buf.clear();
            buf.shrink_to_fit();

            // ---- Phase 2: cross-chunk BPE ----
            log::info!("train_from_iterator: Phase 2 — cross-chunk BPE over {} unique sequences", seq_counts.len());
            let phase2_seq = if let Some(ref p2) = phase2_cache_path {
                if let Some(seq) = load_phase2_cache(p2, vocab_size, &pattern_str, phase1_fp) {
                    log::info!("train_from_iterator: loaded Phase 2 cache from {} ({} merges)", p2, seq.len());
                    seq
                } else {
                    let (p2_words, p2_counts) = build_phase2_words(seq_counts, &phase1_seq);
                    let phase2_id_offset: u32 = phase1_seq.len() as u32;
                    let seq = train_and_record(p2_words, p2_counts, vocab_size, true, phase2_id_offset);
                    log::info!("train_from_iterator: Phase 2 complete — {} cross-chunk merges", seq.len());
                    save_phase2_cache(p2, vocab_size, &pattern_str, phase1_fp, &phase1_seq, &seq);
                    log::info!("train_from_iterator: wrote Phase 2 cache to {}", p2);
                    seq
                }
            } else {
                let (p2_words, p2_counts) = build_phase2_words(seq_counts, &phase1_seq);
                let phase2_id_offset: u32 = phase1_seq.len() as u32;
                let seq = train_and_record(p2_words, p2_counts, vocab_size, true, phase2_id_offset);
                log::info!("train_from_iterator: Phase 2 complete — {} cross-chunk merges", seq.len());
                seq
            };

            // ---- Interleave ----
            log::info!("train_from_iterator: interleaving merge sequences");
            self.merges = interleave_merge_sequences(vocab_size, phase1_seq, phase2_seq);
            log::info!(
                "train_from_iterator: done — {} merge rules ({} total tokens)",
                self.merges.len(), 256 + self.merges.len()
            );
        } else {
            log::info!("train_from_iterator: standard single-phase BPE");
            let (words, counts): (Vec<Word>, Vec<i32>) = chunk_counts.iter()
                .map(|(chunk, &count)| (
                    Word::new_single_chunk(chunk.as_bytes().iter().map(|&b| b as u32).collect()),
                    count,
                ))
                .unzip();
            let seq = train_and_record(words, counts, vocab_size, false, 0);
            self.merges = seq.iter().enumerate()
                .map(|(k, &(pair, _))| (pair, 256 + k as u32))
                .collect();
            log::info!("train_from_iterator: done — {} merge rules", self.merges.len());
        }

        Ok(())
    }

    pub fn get_pattern(&self) -> String { self.pattern.clone() }

    #[getter]
    pub fn vocab_size(&self) -> u32 { 256 + self.merges.len() as u32 }

    pub fn get_mergeable_ranks(&self) -> Vec<(Vec<u8>, u32)> {
        let mut token_bytes: Vec<Vec<u8>> = (0..256_u32).map(|i| vec![i as u8]).collect();
        let mut ranks: Vec<(Vec<u8>, u32)> = token_bytes.iter().zip(0u32..).map(|(b, i)| (b.clone(), i)).collect();

        let mut sorted: Vec<_> = self.merges.iter().collect();
        sorted.sort_by_key(|&(_, &tid)| tid);

        for (&(left, right), &mid) in &sorted {
            let mut mb = token_bytes[left as usize].clone();
            mb.extend(&token_bytes[right as usize]);
            if token_bytes.len() <= mid as usize { token_bytes.resize(mid as usize + 1, Vec::new()); }
            token_bytes[mid as usize] = mb.clone();
            ranks.push((mb, mid));
        }
        ranks
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut all_ids = Vec::new();
        for m in self.compiled_pattern.find_iter(text) {
            let chunk = match m {
                Ok(mat) => mat.as_str(),
                Err(e) => { log::warn!("encode: regex error, skipping chunk: {}", e); continue; }
            };
            let mut ids: Vec<u32> = chunk.bytes().map(|b| b as u32).collect();
            while ids.len() >= 2 {
                let best = ids.windows(2).enumerate()
                    .filter_map(|(i, w)| self.merges.get(&(w[0], w[1])).map(|&nid| (i, nid)))
                    .min_by_key(|&(_, nid)| nid);
                match best {
                    None => break,
                    Some((idx, new_id)) => { ids[idx] = new_id; ids.remove(idx + 1); }
                }
            }
            all_ids.extend(ids);
        }
        all_ids
    }

    pub fn decode(&self, ids: Vec<u32>) -> PyResult<String> {
        let mut vocab: Vec<Vec<u8>> = (0..256u32).map(|i| vec![i as u8]).collect();
        let mut sorted: Vec<_> = self.merges.iter().collect();
        sorted.sort_by_key(|&(_, &tid)| tid);
        for (&(left, right), &mid) in &sorted {
            let mut mb = vocab.get(left as usize).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!("Invalid token id {} in merge", left))
            })?.clone();
            mb.extend(vocab.get(right as usize).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!("Invalid token id {} in merge", right))
            })?);
            if vocab.len() <= mid as usize { vocab.resize(mid as usize + 1, Vec::new()); }
            vocab[mid as usize] = mb;
        }
        let mut bytes = Vec::new();
        for &id in &ids {
            bytes.extend(vocab.get(id as usize).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!("Unknown token id: {}", id))
            })?);
        }
        String::from_utf8(bytes).map_err(|e| {
            pyo3::exceptions::PyUnicodeDecodeError::new_err(format!("Decoded bytes not valid UTF-8: {}", e))
        })
    }

    #[pyo3(signature = (texts))]
    #[pyo3(text_signature = "(self, texts)")]
    pub fn batch_encode(&self, py: Python<'_>, texts: Vec<String>) -> PyResult<Vec<Vec<u32>>> {
        Ok(py.detach(|| texts.par_iter().map(|t| self.encode(t)).collect()))
    }
}

#[pymodule]
fn rustbpe(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();
    m.add_class::<Tokenizer>()?;
    Ok(())
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Span helpers ----

    #[test]
    fn test_span_can_within_merge_true() {
        let a = Span::within_chunk(0, 0, 1);
        let b = Span::within_chunk(0, 1, 2);
        assert!(Span::can_within_merge(a, b));
    }

    #[test]
    fn test_span_can_within_merge_gap() {
        // Non-contiguous within same chunk — not mergeable.
        let a = Span::within_chunk(0, 0, 1);
        let b = Span::within_chunk(0, 2, 3);
        assert!(!Span::can_within_merge(a, b));
    }

    #[test]
    fn test_span_can_within_merge_different_chunks() {
        let a = Span::within_chunk(0, 0, 1);
        let b = Span::within_chunk(1, 0, 1);
        assert!(!Span::can_within_merge(a, b));
    }

    #[test]
    fn test_span_adjacent_full_chunks_true() {
        let a = Span { chunk_start: 0, chunk_end: 0, start: 0, end: 2 };
        let b = Span { chunk_start: 1, chunk_end: 1, start: 0, end: 2 };
        assert!(Span::adjacent_full_chunks(a, b, &[2, 2]));
    }

    #[test]
    fn test_span_adjacent_full_chunks_not_full() {
        let a = Span { chunk_start: 0, chunk_end: 0, start: 0, end: 2 };
        let b = Span { chunk_start: 1, chunk_end: 1, start: 0, end: 1 }; // partial
        assert!(!Span::adjacent_full_chunks(a, b, &[2, 2]));
    }

    #[test]
    fn test_span_adjacent_full_chunks_non_adjacent() {
        let a = Span { chunk_start: 0, chunk_end: 0, start: 0, end: 1 };
        let b = Span { chunk_start: 2, chunk_end: 2, start: 0, end: 1 }; // skip chunk 1
        assert!(!Span::adjacent_full_chunks(a, b, &[1, 1, 1]));
    }

    // ---- Word construction ----

    #[test]
    fn test_word_new_single_chunk_pairs() {
        let word = Word::new_single_chunk(vec![1, 2, 3, 4]);
        let pairs: Vec<Pair> = word.pairs_mergeable(false).collect();
        assert_eq!(pairs, vec![(1, 2), (2, 3), (3, 4)]);
    }

    #[test]
    fn test_word_new_single_chunk_empty() {
        assert!(Word::new_single_chunk(vec![]).pairs_mergeable(false).collect::<Vec<_>>().is_empty());
    }

    #[test]
    fn test_word_new_single_chunk_one_element() {
        assert!(Word::new_single_chunk(vec![42]).pairs_mergeable(false).collect::<Vec<_>>().is_empty());
    }

    #[test]
    fn test_word_new_phase2_no_within_pairs() {
        // Phase 2 Words must never yield within-chunk pairs (each token is its own chunk).
        let word = Word::new_phase2(vec![256, 257, 258]);
        assert!(word.pairs_mergeable(false).collect::<Vec<_>>().is_empty(),
            "Phase 2 words must not yield within-chunk pairs");
    }

    #[test]
    fn test_word_new_phase2_cross_chunk_pairs() {
        let word = Word::new_phase2(vec![256, 257, 258]);
        let cross: Vec<Pair> = word.pairs_mergeable(true).collect();
        assert_eq!(cross, vec![(256, 257), (257, 258)]);
    }

    #[test]
    fn test_word_new_phase2_single_element_no_pairs() {
        let word = Word::new_phase2(vec![256]);
        assert!(word.pairs_mergeable(true).collect::<Vec<_>>().is_empty());
    }

    #[test]
    fn test_word_from_chunks_within_only() {
        // "ab" | "cd": within-chunk pairs are (a,b) and (c,d); cross-boundary (b,c) excluded.
        let chunks = vec![CompactString::from("ab"), CompactString::from("cd")];
        let word = Word::new_from_chunks(&chunks);
        let pairs: Vec<Pair> = word.pairs_mergeable(false).collect();
        assert_eq!(pairs, vec![(97, 98), (99, 100)]);
    }

    #[test]
    fn test_word_from_chunks_cross_eligible_when_single_byte() {
        // Single-byte chunks are already full-chunk spans.
        let chunks = vec![CompactString::from("a"), CompactString::from("b")];
        let word = Word::new_from_chunks(&chunks);
        let cross: Vec<Pair> = word.pairs_mergeable(true).collect();
        assert_eq!(cross, vec![(97, 98)]);
    }

    // ---- Word::merge_pair ----

    #[test]
    fn test_merge_pair_basic() {
        let mut word = Word::new_single_chunk(vec![1, 2, 3, 1, 2]);
        word.merge_pair((1, 2), 99, false);
        assert_eq!(word.ids, vec![99, 3, 99]);
    }

    #[test]
    fn test_merge_pair_non_overlapping_even() {
        // [1,2,1,2] -> [99, 99]
        let mut word = Word::new_single_chunk(vec![1, 2, 1, 2]);
        word.merge_pair((1, 2), 99, false);
        assert_eq!(word.ids, vec![99, 99]);
    }

    #[test]
    fn test_merge_pair_non_overlapping_odd_run() {
        // [a,a,a] -> [256, a] (left-to-right, non-overlapping)
        let mut word = Word::new_single_chunk(vec![97, 97, 97]);
        word.merge_pair((97, 97), 256, false);
        assert_eq!(word.ids, vec![256, 97]);
    }

    #[test]
    fn test_merge_pair_non_overlapping_even_run() {
        let mut word = Word::new_single_chunk(vec![97, 97, 97, 97]);
        word.merge_pair((97, 97), 256, false);
        assert_eq!(word.ids, vec![256, 256]);
    }

    #[test]
    fn test_merge_pair_no_match_unchanged() {
        let mut word = Word::new_single_chunk(vec![1, 2, 3]);
        let deltas = word.merge_pair((4, 5), 99, false);
        assert_eq!(word.ids, vec![1, 2, 3]);
        assert!(deltas.iter().all(|(_, d)| *d == 0));
    }

    #[test]
    fn test_merge_pair_delta_correctness() {
        // [1, 2, 3, 1, 2] merge (1,2)->99 => [99, 3, 99]
        // Removed: (1,2)x2, (2,3)x1, (3,1)x1
        // Added:   (99,3)x1, (3,99)x1
        let mut word = Word::new_single_chunk(vec![1, 2, 3, 1, 2]);
        let deltas = word.merge_pair((1, 2), 99, false);
        let mut dm: StdHashMap<Pair, i32> = StdHashMap::new();
        for (p, d) in deltas { *dm.entry(p).or_default() += d; }

        assert_eq!(dm.get(&(1, 2)), Some(&-2));
        assert_eq!(dm.get(&(2, 3)), Some(&-1));
        assert_eq!(dm.get(&(3, 1)), Some(&-1));
        assert_eq!(dm.get(&(99, 3)), Some(&1));
        assert_eq!(dm.get(&(3, 99)), Some(&1));
    }

    #[test]
    fn test_merge_pair_cross_chunk() {
        // Two single-byte chunks; cross-chunk merge.
        let chunks = vec![CompactString::from("a"), CompactString::from("b")];
        let mut word = Word::new_from_chunks(&chunks);
        word.merge_pair((97, 98), 256, true);
        assert_eq!(word.ids, vec![256]);
    }

    #[test]
    fn test_merge_pair_cross_chunk_blocked_without_flag() {
        let chunks = vec![CompactString::from("a"), CompactString::from("b")];
        let mut word = Word::new_from_chunks(&chunks);
        word.merge_pair((97, 98), 256, false); // should not merge cross-chunk
        assert_eq!(word.ids, vec![97, 98]);    // unchanged
    }

    // ---- count_pairs_parallel ----

    #[test]
    fn test_count_pairs_basic() {
        let words = vec![
            Word::new_single_chunk(vec![1, 2, 3]),
            Word::new_single_chunk(vec![1, 2, 4]),
        ];
        let (pc, pos) = count_pairs_parallel(&words, &[1, 2]);
        assert_eq!(*pc.get(&(1, 2)).unwrap(), 3); // 1 + 2
        assert_eq!(*pc.get(&(2, 3)).unwrap(), 1);
        assert_eq!(*pc.get(&(2, 4)).unwrap(), 2);
        let p12 = pos.get(&(1, 2)).unwrap();
        assert!(p12.contains(&0) && p12.contains(&1));
    }

    #[test]
    fn test_count_pairs_zero_count_ignored() {
        let words = vec![Word::new_single_chunk(vec![1, 2])];
        let (pc, _) = count_pairs_parallel(&words, &[0]);
        assert!(pc.is_empty());
    }

    #[test]
    fn test_count_pairs_empty() {
        let (pc, pos) = count_pairs_parallel(&[], &[]);
        assert!(pc.is_empty() && pos.is_empty());
    }

    // ---- superchunk_ranges ----

    #[test]
    fn test_superchunk_ranges_basic() {
        let s = "Hi. Bye! Ok?  Last";
        let splitter = Regex::new(SUPER_CHUNK_PATTERN).unwrap();
        let ranges = superchunk_ranges(s, &splitter);
        let parts: Vec<&str> = ranges.iter().map(|(a, b)| &s[*a..*b]).collect();
        assert_eq!(parts, vec!["Hi. ", "Bye! ", "Ok?  ", "Last"]);
    }

    #[test]
    fn test_superchunk_ranges_empty_string() {
        let splitter = Regex::new(SUPER_CHUNK_PATTERN).unwrap();
        assert!(superchunk_ranges("", &splitter).is_empty());
    }

    #[test]
    fn test_superchunk_ranges_no_split() {
        let s = "hello world";
        let splitter = Regex::new(SUPER_CHUNK_PATTERN).unwrap();
        let ranges = superchunk_ranges(s, &splitter);
        assert_eq!(ranges, vec![(0, s.len())]);
    }

    // ---- train_and_record ----

    #[test]
    fn test_train_and_record_selects_highest_freq_first() {
        let words = vec![
            Word::new_single_chunk(vec![97, 98]),  // "ab" freq 10
            Word::new_single_chunk(vec![99, 100]), // "cd" freq 5
        ];
        let seq = train_and_record(words, vec![10, 5], 257, false, 0);
        assert_eq!(seq.len(), 1);
        assert_eq!(seq[0].0, (97, 98));
        assert_eq!(seq[0].1, 10);
    }

    #[test]
    fn test_train_and_record_chained_merges() {
        // "aaa": (a,a)->256 first, then (256,a)->257
        let words = vec![Word::new_single_chunk(vec![97, 97, 97])];
        let seq = train_and_record(words, vec![10], 258, false, 0);
        assert_eq!(seq.len(), 2);
        assert_eq!(seq[0].0, (97, 97));
        assert_eq!(seq[1].0, (256, 97));
    }

    #[test]
    fn test_train_and_record_no_pairs_returns_empty() {
        let words = vec![Word::new_single_chunk(vec![97])];
        let seq = train_and_record(words, vec![10], 257, false, 0);
        assert!(seq.is_empty());
    }

    #[test]
    fn test_train_and_record_freq_ordering() {
        // (1,2) @ 100 must land before (3,4) @ 50
        let words = vec![
            Word::new_single_chunk(vec![1, 2]),
            Word::new_single_chunk(vec![3, 4]),
        ];
        let seq = train_and_record(words, vec![100, 50], 258, false, 0);
        assert_eq!(seq[0].0, (1, 2));
        assert_eq!(seq[1].0, (3, 4));
    }

    #[test]
    fn test_train_and_record_phase2_cross_chunk_only() {
        // Phase 2 word with two single-byte chunks: only cross-chunk pair (97,98) emerges.
        let words = vec![Word::new_phase2(vec![97, 98])];
        let seq = train_and_record(words, vec![5], 257, true, 0);
        assert_eq!(seq.len(), 1);
        assert_eq!(seq[0].0, (97, 98));
    }

    // ---- encode_single_chunk ----

    #[test]
    fn test_encode_single_chunk_single_byte_no_merges() {
        assert_eq!(encode_single_chunk("a", &[]), Some(97));
    }

    #[test]
    fn test_encode_single_chunk_basic_merge() {
        // (h=104, i=105) -> 256
        let seq = vec![((104u32, 105u32), 10u64)];
        assert_eq!(encode_single_chunk("hi", &seq), Some(256));
    }

    #[test]
    fn test_encode_single_chunk_chained() {
        // (a,b)->256 then (256,c)->257
        let seq = vec![((97u32, 98u32), 10u64), ((256u32, 99u32), 8u64)];
        assert_eq!(encode_single_chunk("abc", &seq), Some(257));
    }

    #[test]
    fn test_encode_single_chunk_picks_earliest_merge_rank() {
        // "abc": (b,c)->256 rank 0, (a,256)->257 rank 1 — should resolve to 257
        let seq = vec![((98u32, 99u32), 10u64), ((97u32, 256u32), 8u64)];
        assert_eq!(encode_single_chunk("abc", &seq), Some(257));
    }

    #[test]
    fn test_encode_single_chunk_returns_none_when_not_fully_compressed() {
        assert_eq!(encode_single_chunk("ab", &[]), None);
    }

    // ---- build_phase2_words ----

    #[test]
    fn test_build_phase2_words_no_within_pairs() {
        let mut seq_counts: AHashMap<Vec<CompactString>, i32> = AHashMap::new();
        seq_counts.insert(vec![CompactString::from("a"), CompactString::from("b")], 5);
        let (words, counts) = build_phase2_words(seq_counts, &[]);
        assert_eq!(counts[0], 5);
        // No within-chunk pairs; one cross-chunk pair.
        assert!(words[0].pairs_mergeable(false).collect::<Vec<_>>().is_empty());
        assert_eq!(words[0].pairs_mergeable(true).collect::<Vec<_>>(), vec![(97, 98)]);
    }

    #[test]
    fn test_build_phase2_words_chunk_lens_all_one() {
        let mut seq_counts: AHashMap<Vec<CompactString>, i32> = AHashMap::new();
        seq_counts.insert(vec![
            CompactString::from("a"),
            CompactString::from("b"),
            CompactString::from("c"),
        ], 1);
        let (words, _) = build_phase2_words(seq_counts, &[]);
        assert_eq!(words[0].chunk_lens, vec![1, 1, 1]);
    }

    #[test]
    fn test_build_phase2_words_uses_phase1_token_ids() {
        // "ab" with (a=97,b=98)->256: Phase 2 word should contain token 256.
        let phase1_seq = vec![((97u32, 98u32), 10u64)];
        let mut seq_counts: AHashMap<Vec<CompactString>, i32> = AHashMap::new();
        seq_counts.insert(vec![CompactString::from("ab"), CompactString::from("c")], 3);
        let (words, _) = build_phase2_words(seq_counts, &phase1_seq);
        assert_eq!(words[0].ids, vec![256, 99]);
    }

    #[test]
    fn test_build_phase2_words_skips_sequence_with_non_compressible_chunk() {
        // With an empty Phase 1 merge table, "ab" will not compress to one token.
        // We should still salvage merges among compressible chunks by splitting into runs.
        let mut seq_counts: AHashMap<Vec<CompactString>, i32> = AHashMap::new();
        seq_counts.insert(vec![
            CompactString::from("a"),
            CompactString::from("b"),
            CompactString::from("ab"),
            CompactString::from("a"),
            CompactString::from("b"),
        ], 3);
        let (words, counts) = build_phase2_words(seq_counts, &[]);
        assert_eq!(words.len(), 2);
        assert_eq!(counts, vec![3, 3]);
        assert_eq!(words[0].ids, vec![97, 98]);
        assert_eq!(words[1].ids, vec![97, 98]);
    }


    #[test]
    fn test_build_phase2_words_three_chunk_sequence() {
        let mut seq_counts: AHashMap<Vec<CompactString>, i32> = AHashMap::new();
        seq_counts.insert(vec![
            CompactString::from("a"),
            CompactString::from("b"),
            CompactString::from("c"),
        ], 2);
        let (words, _) = build_phase2_words(seq_counts, &[]);
        let cross: Vec<Pair> = words[0].pairs_mergeable(true).collect();
        assert_eq!(cross, vec![(97, 98), (98, 99)]);
    }

    // ---- interleave_merge_sequences ----

    #[test]
    fn test_interleave_phase1_only() {
        let p1 = vec![((1u32, 2u32), 100u64), ((3u32, 4u32), 50u64)];
        let merges = interleave_merge_sequences(258, p1, vec![]);
        assert_eq!(merges.get(&(1, 2)), Some(&256));
        assert_eq!(merges.get(&(3, 4)), Some(&257));
    }

    #[test]
    fn test_interleave_phase2_only_base_bytes() {
        let p2 = vec![((97u32, 98u32), 80u64), ((98u32, 99u32), 40u64)];
        let merges = interleave_merge_sequences(258, vec![], p2);
        assert_eq!(merges.get(&(97, 98)), Some(&256));
        assert_eq!(merges.get(&(98, 99)), Some(&257));
    }

    #[test]
    fn test_interleave_ordered_by_frequency() {
        // p1: 100, 30 / p2: 80, 20 -> expected order: p1[0], p2[0], p1[1], p2[1]
        let p1 = vec![((1u32, 2u32), 100u64), ((3u32, 4u32), 30u64)];
        let p2 = vec![((10u32, 11u32), 80u64), ((12u32, 13u32), 20u64)];
        let merges = interleave_merge_sequences(260, p1, p2);
        assert_eq!(merges.get(&(1, 2)),   Some(&256));
        assert_eq!(merges.get(&(10, 11)), Some(&257));
        assert_eq!(merges.get(&(3, 4)),   Some(&258));
        assert_eq!(merges.get(&(12, 13)), Some(&259));
    }

    #[test]
    fn test_interleave_tie_breaking_favors_phase1() {
        // Equal frequency: Phase 1 must get the lower global ID.
        let p1 = vec![((1u32, 2u32), 50u64)];
        let p2 = vec![((10u32, 11u32), 50u64)];
        let merges = interleave_merge_sequences(258, p1, p2);
        assert_eq!(merges.get(&(1, 2)),   Some(&256)); // Phase 1 wins tie
        assert_eq!(merges.get(&(10, 11)), Some(&257));
    }

    #[test]
    fn test_interleave_id_remapping_phase1_chain() {
        // p1: (a,b)->p1-local-256, (p1-local-256,c)->p1-local-257
        // Both remap to global 256 and 257 respectively.
        let p1 = vec![((97u32, 98u32), 100u64), ((256u32, 99u32), 50u64)];
        let merges = interleave_merge_sequences(258, p1, vec![]);
        assert_eq!(merges.get(&(97, 98)), Some(&256));
        assert_eq!(merges.get(&(256, 99)), Some(&257));
    }

    #[test]
    fn test_interleave_id_remapping_cross_phase() {
        // p1: (a,b)->p1-local-256 @ 100
        // p2: (p2-local-256, c)->p2-local-256 @ 50  (p2-local-256 == p1-local-256 for "ab")
        // Expected: global 256=(a,b), global 257=(256,c)
        let p1 = vec![((97u32, 98u32), 100u64)];
        let p2 = vec![((256u32, 99u32), 50u64)];
        let merges = interleave_merge_sequences(258, p1, p2);
        assert_eq!(merges.get(&(97, 98)), Some(&256));
        assert_eq!(merges.get(&(256, 99)), Some(&257));
    }

    #[test]
    fn test_interleave_prerequisite_guarantee() {
        // p1 produces token 256 at freq 110; p2 uses it at freq 100.
        // 110 >= 100 -> Phase 1 wins, so 256=(a,b) always lands before 257=(256,c).
        let p1 = vec![((97u32, 98u32), 110u64)];
        let p2 = vec![((256u32, 99u32), 100u64)];
        let merges = interleave_merge_sequences(258, p1, p2);
        assert_eq!(merges.get(&(97, 98)), Some(&256));
        assert_eq!(merges.get(&(256, 99)), Some(&257));
    }

    #[test]
    fn test_interleave_empty_both() {
        assert!(interleave_merge_sequences(256, vec![], vec![]).is_empty());
    }

    // ---- Full pipeline: train_and_record + build_phase2_words + interleave ----

    #[test]
    fn test_two_phase_pipeline_cross_chunk_only() {
        // Single-byte chunks: no Phase 1 merges. Phase 2 merges cross-chunk pair.
        let p1_seq: Vec<(Pair, u64)> = vec![];
        let mut seq_counts: AHashMap<Vec<CompactString>, i32> = AHashMap::new();
        seq_counts.insert(vec![CompactString::from("a"), CompactString::from("b")], 10);
        let (p2_words, p2_counts) = build_phase2_words(seq_counts, &p1_seq);
        let p2_seq = train_and_record(p2_words, p2_counts, 257, true, p1_seq.len() as u32);
        let merges = interleave_merge_sequences(257, p1_seq, p2_seq);
        assert_eq!(merges.get(&(97, 98)), Some(&256));
    }

    #[test]
    fn test_two_phase_pipeline_within_before_cross() {
        // "ab" chunk (freq 100) must merge within-chunk before participating
        // in the cross-chunk merge (freq 80) as a single token.
        let p1_words = vec![Word::new_single_chunk(vec![97, 98])];
        let p1_seq = train_and_record(p1_words, vec![100], 257, false, 0);
        assert_eq!(p1_seq[0].0, (97, 98));

        let mut seq_counts: AHashMap<Vec<CompactString>, i32> = AHashMap::new();
        seq_counts.insert(vec![CompactString::from("ab"), CompactString::from("c")], 80);
        let (p2_words, p2_counts) = build_phase2_words(seq_counts, &p1_seq);
        let p2_seq = train_and_record(p2_words, p2_counts, 257, true, p1_seq.len() as u32);

        let merges = interleave_merge_sequences(258, p1_seq, p2_seq);
        assert_eq!(merges.get(&(97, 98)), Some(&256)); // within-chunk first
        assert_eq!(merges.get(&(256, 99)), Some(&257)); // cross-chunk after
    }

    #[test]
    fn test_two_phase_pipeline_chained_cross_chunk() {
        // Three chunks "a","b","c". Phase 2 merges (a,b)->256 then (256,c)->257.
        let p1_seq: Vec<(Pair, u64)> = vec![];
        let mut seq_counts: AHashMap<Vec<CompactString>, i32> = AHashMap::new();
        seq_counts.insert(vec![
            CompactString::from("a"),
            CompactString::from("b"),
            CompactString::from("c"),
        ], 10);
        let (p2_words, p2_counts) = build_phase2_words(seq_counts, &p1_seq);
        // Need vocab_size=258 so Phase 2 can make 2 merges.
        let p2_seq = train_and_record(p2_words, p2_counts, 258, true, p1_seq.len() as u32);
        assert_eq!(p2_seq.len(), 2);

        let merges = interleave_merge_sequences(258, vec![], p2_seq);
        // First merge of (a,b) must produce 256; second of (256,c) must produce 257.
        assert_eq!(merges.get(&(97, 98)), Some(&256));
        assert_eq!(merges.get(&(256, 99)), Some(&257));
    }

    // ---- Tokenizer public API ----

    #[test]
    fn test_tokenizer_new_state() {
        let tok = Tokenizer::new();
        assert!(tok.merges.is_empty());
        assert!(tok.pattern.is_empty());
        assert_eq!(tok.vocab_size(), 256);
    }

    #[test]
    fn test_tokenizer_default() {
        let tok = Tokenizer::default();
        assert!(tok.merges.is_empty());
    }

    #[test]
    fn test_vocab_size_increments() {
        let mut t = Tokenizer::new();
        t.merges.insert((0, 1), 256);
        t.merges.insert((256, 2), 257);
        assert_eq!(t.vocab_size(), 258);
    }

    #[test]
    fn test_get_mergeable_ranks_no_panic_with_gapped_ids() {
        // get_mergeable_ranks should not assume token ids are contiguous.
        // This can happen if the merge table contains duplicates or has been interleaved.
        let mut t = Tokenizer::new();
        t.merges.insert((0, 1), 256);
        t.merges.insert((0, 1), 999);
        let _ = t.get_mergeable_ranks();
    }

    #[test]
    fn test_encode_no_pattern_returns_empty() {
        assert!(Tokenizer::new().encode("hello").is_empty());
    }

    #[test]
    fn test_encode_no_merges_returns_bytes() {
        let tok = Tokenizer {
            merges: StdHashMap::new(),
            pattern: r"\w+".to_string(),
            compiled_pattern: Regex::new(r"\w+").unwrap(),
        };
        assert_eq!(tok.encode("hi"), vec![104, 105]);
    }

    #[test]
    fn test_encode_applies_merge() {
        let mut merges = StdHashMap::new();
        merges.insert((104, 105), 256);
        let tok = Tokenizer { merges, pattern: r"\w+".to_string(), compiled_pattern: Regex::new(r"\w+").unwrap() };
        assert_eq!(tok.encode("hi"), vec![256]);
        assert_eq!(tok.encode("hip"), vec![256, 112]);
    }

    #[test]
    fn test_encode_chained_merges() {
        let mut merges = StdHashMap::new();
        merges.insert((97, 97), 256);
        merges.insert((256, 97), 257);
        let tok = Tokenizer { merges, pattern: r"\w+".to_string(), compiled_pattern: Regex::new(r"\w+").unwrap() };
        assert_eq!(tok.encode("aaa"), vec![257]);
        assert_eq!(tok.encode("aaaa"), vec![256, 256]);
        assert_eq!(tok.encode("aaaaa"), vec![256, 257]);
    }

    #[test]
    fn test_encode_empty_string() {
        let tok = Tokenizer { merges: StdHashMap::new(), pattern: r"\w+".to_string(), compiled_pattern: Regex::new(r"\w+").unwrap() };
        assert!(tok.encode("").is_empty());
    }

    #[test]
    fn test_encode_no_matches() {
        let tok = Tokenizer { merges: StdHashMap::new(), pattern: r"\w+".to_string(), compiled_pattern: Regex::new(r"\w+").unwrap() };
        assert!(tok.encode("   ").is_empty());
    }

    #[test]
    fn test_decode_raw_bytes() {
        assert_eq!(Tokenizer::new().decode(vec![104, 105]).unwrap(), "hi");
    }

    #[test]
    fn test_decode_empty() {
        assert_eq!(Tokenizer::new().decode(vec![]).unwrap(), "");
    }

    #[test]
    fn test_decode_invalid_token() {
        assert!(Tokenizer::new().decode(vec![300]).is_err());
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let mut merges = StdHashMap::new();
        merges.insert((104, 105), 256);
        let tok = Tokenizer { merges, pattern: r"\w+|\s+".to_string(), compiled_pattern: Regex::new(r"\w+|\s+").unwrap() };
        let text = "hi there";
        assert_eq!(tok.decode(tok.encode(text)).unwrap(), text);
    }

    #[test]
    fn test_encode_decode_roundtrip_chained() {
        let mut merges = StdHashMap::new();
        merges.insert((104, 101), 256); // he
        merges.insert((108, 108), 257); // ll
        merges.insert((256, 257), 258); // hell
        let tok = Tokenizer { merges, pattern: r"\w+|\s+".to_string(), compiled_pattern: Regex::new(r"\w+|\s+").unwrap() };
        let text = "hello world";
        assert_eq!(tok.decode(tok.encode(text)).unwrap(), text);
    }

    #[test]
    fn test_get_mergeable_ranks_base_only() {
        let tok = Tokenizer::new();
        let ranks = tok.get_mergeable_ranks();
        assert_eq!(ranks.len(), 256);
        assert_eq!(ranks[0], (vec![0u8], 0));
        assert_eq!(ranks[255], (vec![255u8], 255));
    }

    #[test]
    fn test_get_mergeable_ranks_with_merge() {
        let mut merges = StdHashMap::new();
        merges.insert((65, 66), 256);
        let tok = Tokenizer { merges, pattern: String::new(), compiled_pattern: Regex::new("").unwrap() };
        let ranks = tok.get_mergeable_ranks();
        assert_eq!(ranks.len(), 257);
        assert_eq!(ranks[256], (vec![65u8, 66u8], 256));
    }

    #[test]
    fn test_get_mergeable_ranks_chained_merges() {
        let mut merges = StdHashMap::new();
        merges.insert((65, 66), 256);  // AB
        merges.insert((256, 67), 257); // ABC
        let tok = Tokenizer { merges, pattern: String::new(), compiled_pattern: Regex::new("").unwrap() };
        let ranks = tok.get_mergeable_ranks();
        assert_eq!(ranks[256], (vec![65u8, 66u8], 256));
        assert_eq!(ranks[257], (vec![65u8, 66u8, 67u8], 257));
    }
}
