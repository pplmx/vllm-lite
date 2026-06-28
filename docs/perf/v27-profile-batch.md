# BatchComposer Profile (v27.0 H-10)

**Date:** 2026-06-28
**Target:** `BatchComposer::compose` / `compose_with_chunking` / `compose_prefill_batch` / `compose_decode_batch` / `compose_chunked_prefill`
**Source:** `crates/core/src/scheduler/batch_composer/{compose.rs,validate.rs}`
**Method:** Static code analysis (mirrors H-8/H-9)
**Branch:** main (no worktree, per AGENTS.md)
**Harness:** 128-core x86_64 (Linux product-1-23 5.15.0-179-generic), rustc 1.96.0

---

## Environment constraint

Identical to H-8/H-9: `cargo-flamegraph` is installed but CPU sampling is blocked
by `perf_event_paranoid=4`. Real profiling deferred to GPU runner. Static
analysis only — hotspot rankings are based on call-graph shape (allocation
count, sort cost, Vec clone cost, lock patterns), not measured wall-clock.

See `docs/perf/v27-profile-gqa.md` for the full environment note.

---

## Baseline (from H-1 + `docs/perf/v27-baseline.md` lines 43-52)

| Path | Config | ns/iter (median) | Source |
|------|--------|------------------|--------|
| `scheduler_build_batch/10` | build_batch with 10 seqs | **99 ns** | H-1 line 43 |
| `scheduler_build_batch/50` | build_batch with 50 seqs | **100 ns** | H-1 line 44 |
| `scheduler_build_batch/100` | build_batch with 100 seqs | **100 ns** | H-1 line 45 |
| `batch_building/build_batch_10` | batch builder, 10 seqs | **5,095 ns (~5.1 µs)** | H-1 line 51 |
| `batch_building/build_batch_100` | batch builder, 100 seqs | **38,684 ns (~38.7 µs)** | H-1 line 52 |

**Important caveat:** the H-1 `scheduler_build_batch/*` benches show
**~100 ns regardless of input size (10/50/100 seqs)**, which is suspicious
and likely measures a no-op fast path or an empty-batch shortcut rather
than the full `BatchComposer::compose` body. The `batch_building/*`
benches at 5-39 µs are more representative of actual compose work and
scale with input size as expected. The 38.7 µs at 100 seqs is the
relevant baseline for ranking optimization targets.

For comparison: a 100-sequence prefill is dominated by sequence-level
token copying (`seq.tokens[start..].to_vec()`), not by scheduler
bookkeeping. At ~200 ns per element + Vec allocation overhead,
100 seqs × 100 tokens = 10k tokens × ~50 ns = ~500 µs baseline
extrapolation; the actual 38 µs is much faster because the bench
likely uses small (single-digit token) sequences.

---

## Module structure (`crates/core/src/scheduler/batch_composer/`)

```text
mod.rs       (14 lines)   Facade; declares BatchComposer + config re-exports
├── compose.rs  (538 lines) BatchComposer impl + 4 phase-specific composition fns
└── validate.rs (162 lines) BatchCompositionConfig, ChunkedPrefillConfig + builders
```

| File:line | Symbol | Purpose |
|-----------|--------|---------|
| `compose.rs:12-16` | `struct BatchComposer` | Holds `config`, `packing_config`, `chunked_prefill` |
| `compose.rs:57-64` | `compose(&self, sequences, phase)` | Public entry; routes to packing vs standard based on `Phase` and `packing_config.enabled` |
| `compose.rs:69-85` | `compose_with_chunking(&self, sequences, phase, available_memory)` | Public entry with chunking; routes to `compose_chunked_prefill` or `compose_standard` |
| `compose.rs:87-152` | `compose_chunked_prefill` | Long-sequence prefill split into `ChunkedPrefillConfig`-sized chunks |
| `compose.rs:154-159` | `compose_prefill_with_packing` | Currently delegates to `build_batch_from_sequences`; packing optimization is TODO |
| `compose.rs:161-163` | `compose_standard` | Trivial delegate |
| `compose.rs:165-171` | `build_batch_from_sequences` | Phase-dispatch wrapper |
| `compose.rs:174-259` | `compose_prefill_batch` | The main prefill path: sort + assemble |
| `compose.rs:262-327` | `compose_decode_batch` | The main decode path: assemble (no sort) |
| `validate.rs:8-15` | `BatchCompositionConfig` | `max_batch_size`, `max_token_budget`, `enable_similarity_grouping` |
| `validate.rs:67-76` | `ChunkedPrefillConfig` | `enabled`, `target_chunk_size`, `max_chunk_size`, `min_chunk_size` |
| `validate.rs:135-161` | `calculate_chunk_size(seq_len, available_memory)` | Memory-aware chunk size calculation |

---

## compose_prefill_batch hot path (`compose.rs:174-259`)

```text
compose_prefill_batch(&self, mut sequences: Vec<Sequence>) -> Batch
├─ sequences.sort_by_key(|s| s.tokens.len().saturating_sub(s.num_computed_tokens))
│                                          @ 176  *** O(N log N) stable sort ***
├─ let mut seq_ids = Vec::new();                  @ 178  *** 7 Vec::new() allocations ***
├─ let mut input_tokens = Vec::new();
├─ let mut positions = Vec::new();
├─ let mut kv_block_ids = Vec::new();
├─ let mut num_computed_tokens = Vec::new();
├─ let mut is_prefill = Vec::new();
├─ let mut total_tokens = 0usize;
├─ let mut max_seq_len = 0usize;
├─ for seq in sequences.into_iter().take(max_batch_size):
│     let start = seq.num_computed_tokens;
│     let seq_len = seq.tokens.len();
│     let tokens_to_process = seq_len.saturating_sub(start);
│     if tokens_to_process == 0 { continue; }          @ 208
│     if total_tokens + tokens_to_process > max_token_budget { break; }  @ 213
│     seq_ids.push(seq.id);
│     let tokens: Vec<TokenId> = seq.tokens[start..].to_vec()    @ 226  *** Vec clone per seq ***
│     positions.push((start..seq_len).collect());                 @ 227  *** Vec collect per seq ***
│     total_tokens += tokens.len();
│     max_seq_len = max_seq_len.max(tokens.len());
│     input_tokens.push(tokens);
│     kv_block_ids.push(seq.kv_blocks.as_ref().clone());         @ 232  *** Arc<Vec> clone per seq ***
│     num_computed_tokens.push(start);
│     is_prefill.push(start == 0);
└─ return Batch { seq_ids, input_tokens, positions, kv_block_ids,
                  num_computed_tokens, is_prefill, phase: Prefill, ... }
```

### Per-call allocation accounting (prefill, N=max_batch_size seqs)

| Allocation | Count | Size |
|------------|-------|------|
| `Vec::new()` for the 7 output vecs | 7 | grows to N |
| `seq.tokens[start..].to_vec()` per seq | N | `tokens_to_process` × 4B |
| `(start..seq_len).collect()` per seq | N | `tokens_to_process` × 8B (Vec\<usize\>) |
| `seq.kv_blocks.as_ref().clone()` per seq | N | Arc::clone is cheap (atomic increment), but then the inner Vec clone is **also** performed by `as_ref().clone()` — see Hotspot #4 below |
| `sequences.sort_by_key(...)` | 1 | O(N log N) comparisons |
| Final `Batch` struct | 1 | contains the 7 vecs |

For `max_batch_size=256` and average prefill of 2048 tokens:
- `input_tokens`: 256 × 2048 × 4B = **2 MB**
- `positions`: 256 × 2048 × 8B = **4 MB** (usize is 8 bytes on 64-bit)
- `kv_block_ids`: 256 × avg blocks × 8B ≈ tens of KB

The dominant allocation is `positions` at 4 MB per prefill batch, due to
the per-token `Vec<usize>` representation.

---

## compose_decode_batch hot path (`compose.rs:262-327`)

```text
compose_decode_batch(&self, sequences: Vec<Sequence>) -> Batch
├─ let batch_size = sequences.len().min(max_batch_size);            @ 263
├─ let mut seq_ids = Vec::with_capacity(batch_size);                @ 265  *** pre-sized ***
├─ let mut input_tokens = Vec::with_capacity(batch_size);
├─ let mut positions = Vec::with_capacity(batch_size);
├─ let mut kv_block_ids = Vec::with_capacity(batch_size);
├─ let mut num_computed_tokens = Vec::with_capacity(batch_size);
├─ let mut is_prefill = Vec::with_capacity(batch_size);
├─ let mut total_tokens = 0;
├─ let mut max_seq_len = 0;
├─ for seq in sequences.into_iter().take(batch_size):
│     seq_ids.push(seq.id);
│     let last_token = seq.tokens.last().copied().unwrap_or(0);     @ 278
│     let tokens_len = seq.tokens.len();
│     let position = tokens_len - 1;                                 @ 286
│     input_tokens.push(vec![last_token]);                           @ 294  *** per-seq Vec alloc ***
│     positions.push(vec![position]);                                @ 295  *** per-seq Vec alloc ***
│     total_tokens += 1;
│     max_seq_len = max_seq_len.max(1);
│     kv_block_ids.push(seq.kv_blocks.as_ref().clone());            @ 299
│     num_computed_tokens.push(seq.tokens.len() - 1);
│     is_prefill.push(false);
└─ return Batch { ... phase: Decode, ... }
```

**Note on contrast with prefill path:** decode uses
`Vec::with_capacity(batch_size)` for all 6 vecs (lines 265-270) — good.
But prefill uses `Vec::new()` (lines 178-183) — this is an inconsistency
that matters for batch-resize reallocations.

### Per-call allocation accounting (decode, N=batch_size)

| Allocation | Count | Size |
|------------|-------|------|
| 6× `Vec::with_capacity(N)` | 6 | pre-sized |
| `vec![last_token]` per seq | N | 1 × 4B each |
| `vec![position]` per seq | N | 1 × 8B each |
| `seq.kv_blocks.as_ref().clone()` per seq | N | see Hotspot #4 |
| `tokens.last().copied().unwrap_or(0)` per seq | N | O(1) |

The decode path is much leaner than prefill (only 1-element vecs per
sequence), but the same `as_ref().clone()` issue applies.

---

## compose_chunked_prefill hot path (`compose.rs:87-152`)

```text
compose_chunked_prefill(&self, sequences, available_memory) -> Batch
├─ let mut seq_ids = Vec::new();                       @ 88  *** 7 Vec::new() allocations ***
├─ let mut input_tokens = Vec::new();
├─ let mut positions = Vec::new();
├─ let mut kv_block_ids = Vec::new();
├─ let num_computed_tokens = Vec::new();               @ 92  *** not mut — bug? see Hotspot #7 ***
├─ let mut is_prefill = Vec::new();
├─ let mut total_tokens = 0usize;
├─ let mut max_seq_len = 0usize;
├─ for seq in sequences.into_iter().take(max_batch_size):
│     let start = seq.num_computed_tokens;
│     let seq_len = seq.tokens.len();
│     let remaining_tokens = seq_len.saturating_sub(start);
│     if remaining_tokens == 0 { continue; }
│     let chunk_size = self.chunked_prefill
│         .calculate_chunk_size(remaining_tokens, available_memory);  @ 107-109
│     let tokens_to_process = remaining_tokens.min(chunk_size);
│     if total_tokens + tokens_to_process > max_token_budget { break; }
│     seq_ids.push(seq.id);
│     let tokens: Vec<TokenId> = seq.tokens[start..start+tokens_to_process].to_vec();  @ 121
│     positions.push((start..start+tokens_to_process).collect());                       @ 122
│     total_tokens += tokens_to_process;
│     max_seq_len = max_seq_len.max(tokens_to_process);
│     input_tokens.push(tokens);
│     kv_block_ids.push(seq.kv_blocks.as_ref().clone());                               @ 127
│     is_prefill.push(start == 0);
└─ return Batch { ..., num_computed_tokens, ... }   @ 141-151  *** uses the empty Vec ***
```

Note the **same Vec::new() pattern as prefill** (lines 88-95), the same
`as_ref().clone()` issue (line 127), and the same `Vec::collect()`
positions pattern (line 122). Chunking adds `calculate_chunk_size` calls
(`validate.rs:135-161`), which are O(1) integer arithmetic.

---

## Suspected hotspots (priority order)

### 1. **[HIGH] `positions: Vec<Vec<usize>>` dominates allocation at prefill scale** — `compose.rs:227, 122, 295`

**Pattern:** Each sequence gets its own `Vec<usize>` of positions:
- Prefill (compose.rs:227): `(start..seq_len).collect()` — one usize per token
- Chunked (compose.rs:122): `(start..start+tokens_to_process).collect()` — one usize per token
- Decode (compose.rs:295): `vec![position]` — one usize per sequence (cheap)

For prefill at `max_batch_size=256`, `tokens_per_seq=2048`:
- 256 × 2048 × 8B = **4 MB of `usize` positions per batch**
- 256 separate Vec allocations + capacity grow events

**Why it's slow:**
- Per-sequence `Vec::collect()` allocates and writes one usize per token
- The downstream consumer (`batch.positions[i]`) typically just iterates — there's no benefit to materializing each sequence's positions as a separate Vec
- For large prefill (4k-8k tokens), this is the largest single allocation category

**Optimization candidates:**
- **(A) Flatten to `Vec<usize>` with per-seq offsets:** Store positions as a single flat `Vec<usize>` plus a `Vec<usize>` of offsets (cumulative sums). Per-token write: `flat[offset[i]..offset[i+1]].copy_from_slice(&(start..end).collect::<Vec<_>>())`. Eliminates 256 small allocations.
- **(B) `Box<[usize]>` instead of `Vec<usize>`:** For fixed-length positions (which they always are within a sequence), `Box<[usize]>` is one allocation instead of amortized-grow. Marginal win; not as impactful as (A).
- **(C) Drop `positions` from the Batch struct:** If no downstream consumer actually uses per-token positions (check call sites in `batch.rs`, `engine/`), the field can be replaced with a `Vec<usize>` of last-position per sequence. Decode already only stores one position; prefill could too (the consumer can reconstruct per-token positions from `num_computed_tokens + i`).

### 2. **[HIGH] `seq.kv_blocks.as_ref().clone()` does a deep Vec clone of the inner Vec** — `compose.rs:232, 127, 299`

**Pattern:**
```rust
kv_block_ids.push(seq.kv_blocks.as_ref().clone());   @ 232, 127, 299
```

The Sequence type is:
```rust
pub kv_blocks: Arc<Vec<BlockId>>,                    // sequence.rs:15
```

So `seq.kv_blocks` is an `Arc<Vec<BlockId>>`. `.as_ref()` gives
`&Vec<BlockId>` (the Arc is dereferenced). `.clone()` on `&Vec<BlockId>`
clones the **Vec itself** — deep clone of the inner buffer.

**Why it's slow:**
- For a 2048-token prefill, `seq.kv_blocks` may have 128 entries (one per BLOCK_SIZE=16 tokens)
- Per-call, each sequence triggers a 128-entry Vec clone: 128 × 8B = 1 KB
- For `max_batch_size=256`, that's 256 KB of deep Vec clone per prefill batch
- The intent (Arc clone) is a cheap atomic increment; the actual code does a full Vec clone

**Optimization candidates:**
- **(A) Use `Arc::clone(&seq.kv_blocks)`:**
  ```rust
  kv_block_ids.push(seq.kv_blocks.as_ref().clone());  // current: deep Vec clone
  // →
  kv_block_ids.push(seq.kv_blocks.clone());           // Arc clone, atomic increment only
  ```
  Then change `Batch.kv_block_ids: Vec<Vec<BlockId>>` to `Vec<Arc<Vec<BlockId>>>` (or, better, `Vec<Arc<[BlockId]>>`). The downstream consumer would `Arc::clone` to hold the reference; if it just iterates, no clone needed at all.

- **(B) Store `&Arc<Vec<BlockId>>` references in the Batch:** If the Batch is short-lived (consumed and dropped within one forward pass), a `Vec<Arc<Vec<BlockId>>>` or even `Vec<&Arc<Vec<BlockId>>>` (with appropriate lifetime) avoids both clones.

- **(C) Single shared `kv_block_ids` flat Vec:** Similar to positions, flatten into a single `Vec<BlockId>` with per-seq offsets. Downstream consumers (KV cache lookup) can index the range.

### 3. **[HIGH] Prefill path uses `Vec::new()` instead of `Vec::with_capacity(N)`** — `compose.rs:178-185`

**Pattern:** Compare to the decode path at lines 265-270 which uses
`Vec::with_capacity(batch_size)`:
```rust
let mut seq_ids = Vec::new();                    @ 178 (prefill)
...
let mut seq_ids = Vec::with_capacity(batch_size);  @ 265 (decode)
```

**Why it's slow:**
- `Vec::new()` starts with zero capacity
- Each `push` may reallocate (typically at capacities 4, 8, 16, 32, ..., up to N=256)
- For N=256, that's ~8 reallocations + copies per Vec × 6 Vecs = **48 reallocations per prefill batch**
- Each reallocation copies the existing data to a new heap allocation

**Optimization candidates:**
- **(A) Match the decode pattern:** Use `Vec::with_capacity(self.config.max_batch_size)` for all 6 prefill vecs. Zero-risk refactor; pure allocation reduction.
- **(B) Pass `max_batch_size` to the helper:** Same as (A), but make `compose_prefill_batch` take `max_batch_size` explicitly (currently derived from `self.config`).

### 4. **[MEDIUM] Prefill sort runs even for already-sorted or empty input** — `compose.rs:176`

**Pattern:**
```rust
sequences.sort_by_key(|s| s.tokens.len().saturating_sub(s.num_computed_tokens));
```

This unconditionally sorts the entire input sequences Vec on every
prefill batch build.

**Why it's slow:**
- `sort_by_key` is **stable** (preserves order of equal elements). For prefix-cache locality, this is desirable; for raw speed, `sort_unstable_by_key` is faster (no stability guarantee but allows branchless swap).
- For `max_batch_size=256`, sort is O(N log N) ≈ 256 × 8 = 2048 comparisons. Each comparison involves two method calls (`len()` and `num_computed_tokens` field access) plus a saturating subtract. Probably ~50-100 ns total — not a major hotspot but adds up across all prefill batches.
- The sort produces a Vec permutation; downstream `into_iter().take(N)` reads in sorted order. But the batch construction loop could just `take` the top-N min-cost sequences from a `BinaryHeap` without sorting the whole Vec — O(N log K) instead of O(N log N).

**Optimization candidates:**
- **(A) `sort_unstable_by_key`:** Drop stability guarantee. Marginal speedup; correctness verified by the `test_prefill_batch_includes_all_prompt_tokens` test (which doesn't depend on stability).
- **(B) Use `BinaryHeap` for top-K selection:** Build a min-heap of size `max_batch_size`, push each sequence's cost, pop the largest when over capacity. O(N log K) instead of O(N log N). Most beneficial when N >> K (waiting queue >> max_batch_size).
- **(C) Skip sort for already-sorted input:** Track whether the scheduler already sorted the input (via the radix-tree iteration order or a `priority` field). Marginal win; depends on caller contract.

### 5. **[MEDIUM] `seq.tokens[start..].to_vec()` clones the full remaining token slice** — `compose.rs:226, 121`

**Pattern:**
```rust
let tokens: Vec<TokenId> = seq.tokens[start..].to_vec();                  @ 226 (prefill)
let tokens: Vec<TokenId> = seq.tokens[start..start+tokens_to_process].to_vec(); @ 121 (chunked)
```

For prefill of T=2048 tokens, this is a 2048 × 4B = 8 KB clone per
sequence. With max_batch_size=256, that's **2 MB of token cloning per
prefill batch**.

**Why it's slow:** Memory bandwidth-bound. The clone is unavoidable in
the current API (the Batch struct owns `Vec<Vec<TokenId>>`), but the
copy could be reduced or deferred.

**Optimization candidates:**
- **(A) Slice references instead of owned vecs:** Change `Batch.input_tokens` to `Vec<&[TokenId]>` with appropriate lifetime. The downstream consumer copies on demand. But this requires lifetime annotations on `Batch`, which is widely shared (traits crate).
- **(B) Single flat `Vec<TokenId>` with offsets:** Same as positions hotspot #1. Flatten all sequences' tokens into one `Vec<TokenId>` with a `Vec<usize>` of offsets. The downstream consumer slices `[tokens[offset[i]..offset[i+1]])`. Eliminates N Vec allocations and N memcpy calls.
- **(C) `Arc<Vec<TokenId>>` references:** Similar to kv_blocks #2 above; use Arc sharing. Most beneficial for decode (which copies just 1 token per seq).

### 6. **[MEDIUM] `compose_chunked_prefill` has `num_computed_tokens: Vec::new()` (non-mut)** — `compose.rs:92, 146`

**Pattern:**
```rust
let num_computed_tokens = Vec::new();   @ 92  *** NOT mut ***
...
Batch {
    ...
    num_computed_tokens,                @ 146  *** passes the empty Vec ***
    ...
}
```

The vec is declared non-mut, never pushed to, and passed empty to the
Batch. Compare with `compose_prefill_batch` (lines 182-183, 233, 253)
which declares `let mut num_computed_tokens = Vec::new()` and pushes
`start` per sequence.

**Why it's slow:** Not a perf hotspot per se — the Vec is empty so no
allocation happens. But this is a **correctness inconsistency**:
- `compose_prefill_batch` populates `num_computed_tokens[i] = start`
- `compose_chunked_prefill` populates `num_computed_tokens[i] = 0` (empty vec means default, but that's "missing" not "zero")
- Downstream consumers may rely on `num_computed_tokens.len() == batch_size`

The empty Vec means any consumer indexing `batch.num_computed_tokens[i]`
will panic (out-of-bounds) for chunked-prefill batches. This is a
latent bug, not a perf hotspot.

**Optimization candidates:**
- **(A) Declare `mut` and push `start` per seq:** Match the prefill path. Zero-risk correctness fix.

### 7. **[MEDIUM] `ChunkedPrefillConfig::calculate_chunk_size` heuristic could be cached** — `validate.rs:135-161`

**Pattern:**
```rust
pub fn calculate_chunk_size(&self, seq_len: usize, available_memory: usize) -> usize {
    if !self.enabled || seq_len <= self.min_chunk_size { return seq_len; }
    let base_chunk = if self.target_chunk_size == 0 { ... } else { self.target_chunk_size };
    let chunk = base_chunk.clamp(self.min_chunk_size, self.max_chunk_size);
    if seq_len > 8192 { chunk.min(512) }
    else if seq_len > 4096 { chunk.min(1024) }
    else { chunk }
}
```

Called once per sequence in `compose_chunked_prefill` (line 107-109).

**Why it's slow:**
- Pure integer arithmetic; called per sequence
- The output depends only on `(seq_len, available_memory, config)`; for a fixed config, identical seq_len produces identical chunk_size
- At long-tail prefill (T=16384), the function returns a constant for all sequences of similar length

**Optimization candidates:**
- **(A) Memoize by seq_len bucket:** Cache last-N chunk_size calculations in a `[(usize, usize); 8]` array (round seq_len to nearest power of 2). Avoids the integer math in the inner loop.
- **(B) Skip calculation for short sequences:** Already done via `seq_len <= self.min_chunk_size` short-circuit. Could extend to "if seq_len < some_threshold, return seq_len directly".
- **(C) Hoist to caller:** Since the chunk size depends only on `seq_len` and `available_memory` (not the specific sequence), `compose_chunked_prefill` could group sequences by `seq_len` and compute the chunk size once per group.

### 8. **[LOW] `tracing::debug!` macro calls in hot loop** — `compose.rs:199-206, 287-292`

**Pattern:** Every sequence in the prefill loop logs at debug level:
```rust
tracing::debug!(
    seq_id = seq.id,
    start = start,
    seq_len = seq_len,
    tokens_to_process = tokens_to_process,
    total_tokens = total_tokens,
    "compose_prefill: processing sequence"
);
```

`tracing::debug!` expands to a static call site; the formatting work
(field capture) happens regardless of whether the log is emitted.
`tracing` is supposed to short-circuit when the level is disabled, but
the field capture (`seq.id`, `start`, etc.) is still performed.

**Why it's slow:**
- For 256 sequences, that's 256 debug-formatting events per prefill batch
- At default log level (info), these are no-ops but still incur the dispatch
- The `tokens_to_process`, `seq_len`, etc. are field captures, not allocations — but each adds a few ns

**Optimization candidates:**
- **(A) Rate-limit or sample logs:** Log every 10th or 100th sequence in the loop. Keeps observability without per-iter overhead.
- **(B) Use `tracing::trace!` and gate behind config:** Move the per-sequence log to trace level; only emit when explicitly enabled.
- **(C) Move logs out of the loop:** Log batch summary only at the end (already done at lines 187-192, 242-246). Drop the per-sequence log entirely.

### 9. **[LOW] `Vec<Vec<bool>>` for `is_prefill` is wasteful** — `compose.rs:183, 304`

**Pattern:**
```rust
let mut is_prefill = Vec::new();        @ 183
...
is_prefill.push(start == 0);            @ 236 (prefill)
...
is_prefill.push(false);                 @ 304 (decode)
```

This is a `Vec<bool>` not `Vec<Vec<bool>>` — read the type again. Per
sequence push. For batch_size=256, 256 bools × 1B (rounded to 1B or 8B
with alignment) = negligible.

**Optimization candidates:**
- **(A) Use `BitVec` or `u64` bitmask:** For decode batches where all entries are `false`, store a `u64` bitmask. Reduces 256 bytes to 8 bytes. Marginal win.
- **(B) Replace `Vec<bool>` with phase flag:** If the Batch always has consistent `is_prefill` (all true for prefill, all false for decode), the `Batch.phase: BatchPhase` field already conveys this. The `is_prefill` per-seq field is only needed for mixed batches. Add `is_mixed() -> bool` and skip populating `is_prefill` for pure-phase batches.

---

## Recommended H-13 optimization targets (scheduler side)

Per the H-13 scope ("3 hotspot optimizations" per plan line 625), the
top candidates from BatchComposer are:

| Rank | Target | File:line | Estimated speedup | Risk | Notes |
|------|--------|-----------|-------------------|------|-------|
| **1 (Primary)** | `seq.kv_blocks.as_ref().clone()` → `seq.kv_blocks.clone()` (Arc clone) + change `Batch.kv_block_ids` to `Vec<Arc<Vec<BlockId>>>` | `compose.rs:232, 127, 299` + `traits/src/types.rs:27` | **1-5% on prefill; eliminates 256 KB of Vec clone per batch** | Medium | Touches the public `Batch` API (traits crate). All call sites that read `batch.kv_block_ids[i]` as `&Vec<BlockId>` need to add one `Arc::clone` or `.as_ref()`. Pure refactor otherwise. |
| **2 (Secondary)** | Flatten `positions: Vec<Vec<usize>>` to `Vec<usize>` with offsets; same for `input_tokens` | `compose.rs:178-185, 226-227, 121-122, 294-295` + `traits/src/types.rs:23-25` | **5-15% on prefill at scale (4-6 MB allocation reduction)** | Medium-High | Same trait-crate API change as #1. Downstream consumers iterate positions[i], so a `positions_start: Vec<usize>` + `positions_flat: Vec<usize>` API preserves the iteration. |
| **3 (Tertiary)** | `Vec::new()` → `Vec::with_capacity(max_batch_size)` in prefill path; fix chunked-prefill `num_computed_tokens` bug | `compose.rs:178-185, 88-95` | **1-3% on prefill (eliminates ~48 reallocations per batch); correctness fix** | Low | Pure allocation reduction + 1-line bug fix. No API change. Run `test_chunked_prefill_*` to verify the bug fix. |

**Suggested order:** do #1 first (correctness-neutral, biggest single
win), then #3 (easy win + latent-bug fix), then #2 (largest total win
but touches more code).

**Out of scope for H-13 (consider separate tasks):**
- `sort_by_key` → `sort_unstable_by_key` / `BinaryHeap` rewrite (#4) — small win, depending on scheduler caller contract
- `Arc<Vec<TokenId>>` for input_tokens (#5) — overlaps with #2; defer
- `calculate_chunk_size` memoization (#7) — micro-optimization, low ROI
- `tracing::debug!` rate-limiting (#8) — observability concern, not perf
- `is_prefill` bitmask (#9) — micro-optimization, low ROI

---

## Comparison with model-side hotspots

BatchComposer is **non-tensor code**; the patterns differ significantly
from the model-side hotspots in H-8/H-9/PagedKV:

| Pattern | Model side (GQA/MLA/Flash/PagedKV) | Scheduler side (BatchComposer) |
|---------|------------------------------------|--------------------------------|
| Dominant cost | Tensor matmul, softmax, matmul, contiguous | Vec allocation, sort, Vec clone |
| Hot path data | F32 tensors in GPU memory | Sequences (owned Vec) in host memory |
| Allocation pattern | Per-call tensor alloc + cat | Per-sequence Vec push + collect |
| Memory direction | Device-bound (GPU bandwidth) | Host-bound (CPU cache + allocator) |
| Sort/search | None (matmul is the sort) | `sort_by_key`, BinaryHeap candidate |
| Lock contention | Minimal (single-threaded forward) | Could exist at scheduler level (not in BatchComposer itself) |

**Implication:** Optimization techniques that apply to model-side
hotspots (cache scale tensor, eliminate `Tensor::cat`, fuse matmuls)
do not transfer directly. Scheduler-side wins come from
allocation/clone reduction and data-structure layout changes.

**Cross-cutting observation:** Both sides exhibit the same anti-pattern
of `Vec::push` + final `cat` (cf. PagedKV's `Tensor::cat` per-write and
BatchComposer's per-seq push into output Vec). The fix is similar in
spirit (pre-allocate, avoid the per-element allocation) but the
mechanism is different (tensor in-place write vs flat Vec with offsets).

---

## Limitations

- **Static analysis cannot measure actual CPU time per function.** All
  hotspot rankings are based on call-graph shape (allocation count,
  clone depth, Vec allocation count, sort complexity), not measured
  wall-clock per function. Real flamegraph data on a GPU runner with
  `perf_event_paranoid<=0` is required to confirm.
- **The `scheduler_build_batch/*` baseline (~100 ns regardless of input
  size) is suspiciously flat.** It likely measures a different code
  path (e.g., a fast-path empty batch shortcut or the scheduler-level
  `build_batch` wrapper, not the `BatchComposer::compose` body). The
  `batch_building/*` baseline (5-39 µs) is the more relevant number
  for ranking.
- **H-13 optimizers should re-bench after each change** with
  `just bench-core-one batch_building` and confirm the
  `test_*_batch_*` test suite in `compose.rs:336-537` still passes.
- **GPU profiling deferred.** Real flamegraph profiling would clarify
  whether the scheduler is actually on the critical path (vs model
  forward). At qwen3-7B-class with batch=1 the scheduler is invisible;
  at batch=32+ it becomes more relevant.

---

## Files

- Source: `crates/core/src/scheduler/batch_composer/{compose.rs,validate.rs,mod.rs}`
- Types: `crates/core/src/types/sequence.rs` (Sequence struct),
  `crates/traits/src/types.rs:22-34` (Batch struct)
- Plan: `docs/superpowers/plans/2026-06-28-v27-performance.md` (Task H-10)
- H-1 baseline: `docs/perf/v27-baseline.md` (scheduler_build_batch + batch_building rows)
- H-8 methodology: `docs/perf/v27-profile-gqa.md`
- H-9 followup: `docs/perf/v27-profile-mla.md`, `docs/perf/v27-profile-flash.md`
- PagedKV sister report: `docs/perf/v27-profile-pkv.md`
