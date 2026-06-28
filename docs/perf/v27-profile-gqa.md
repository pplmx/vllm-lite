# GQA Profile (v27.0 H-8)

**Date:** 2026-06-28
**Target:** `GqaAttention::forward` (`crates/model/src/components/attention/gqa.rs:132`)
**Helper site:** `expand_kv`, `paged_attention`, `tiled_attention`, `GqaFlashAttention`
**Method:** Static code analysis + criterion smoke baseline
**Branch:** main (no worktree, per AGENTS.md)
**Harness:** 128-core x86_64 (Linux product-1-23 5.15.0-179-generic), rustc 1.96.0

---

## Environment constraint

`cargo-flamegraph` was installed (`cargo install flamegraph` succeeded at v0.6.13),
but **CPU sampling is infeasible in this sandbox**:

| Tool | Status | Reason |
|------|--------|--------|
| `cargo-flamegraph` | installed | `~/.cargo/bin/cargo-flamegraph` |
| `perf` (Linux profiler) | **missing** | `command not found: perf` |
| `perf_event_paranoid` | **4** (most restrictive) | `/proc/sys/kernel/perf_event_paranoid` |
| `dtrace` | not installed | N/A |

When invoked, `cargo flamegraph` errors with: `Error: perf is not installed or not
present in $PATH`. Even with `perf` present, `perf_event_paranoid=4` blocks all
CPU sampling (would require root + `sysctl kernel.perf_event_paranoid<=0`).

**Fallback used:** Static code analysis of the GQA forward path, confirmed by the
H-2 criterion baseline. Real flamegraph profiling requires a GPU runner with
sampling permission — deferred to follow-up profiling on GPU hardware.

---

## GQA forward structure (line-by-line, `crates/model/src/components/attention/gqa.rs`)

```text
forward(x: (B, S, H_hidden)) @ gqa.rs:132
├─ q,k,v = q_proj/k_proj/v_proj(x)              # 3 matmuls (143-145)
├─ q,k,v = reshape((B, S, n_heads, head_dim))   # view only (147-149)
├─ q = apply_q_norm(q)                          # 4 reshape/transpose if has_qk_norm (262-274)
├─ k = apply_k_norm(k)                          # same (276-288)
├─ q = q.transpose(1,2)?.contiguous()?          # (B, n_heads, S, D), forced copy (154)
├─ branch on self.config.use_fused:
│   ├─ TRUE:                                    # use_fused path (156-168)
│   │   ├─ k_heads = k.transpose(1,2)?.contiguous()?
│   │   ├─ v_heads = v.transpose(1,2)?.contiguous()?
│   │   ├─ attn = GqaFlashAttention::forward(q, k_heads, v_heads)  # causal=HARDCODED false
│   │   ├─ attn.transpose(1,2)?.reshape(...)
│   │   └─ o = o_proj(attn)
│   └─ FALSE:                                   # "standard" matmul path (170-194)
│       ├─ k = expand_kv(k, n_heads, n_kv_heads)?   # ALLOCATES new tensor (repeat)
│       ├─ v = expand_kv(v, n_heads, n_kv_heads)?   # ALLOCATES new tensor (repeat)
│       ├─ k.transpose(1,2)?                       # view
│       ├─ v.transpose(1,2)?                       # view
│       ├─ k_t = k.transpose(2,3)?.contiguous()?   # forced copy
│       ├─ qk = matmul(q, k_t)
│       ├─ qk = qk * scale_tensor                 # scale_tensor allocated per call
│       ├─ attn_weights = softmax(qk, dim=3).contiguous()?  # forced copy
│       ├─ attn_output = matmul(attn_weights, v.contiguous()?)  # forced copy
│       ├─ attn_output.transpose(1,2)?.reshape(...)
│       └─ o = o_proj(attn_output)
└─ return o
```

**Two public attention pathways:**

| Helper | Causal mask? | Where called |
|--------|--------------|--------------|
| `forward()` (direct) | **NO** in both `use_fused` (causal hardcoded false @ gqa.rs:160) and "standard" (no mask added @ gqa.rs:177-181) | The H-2 bench exercises this |
| `paged_attention_fn` / `tiled_attention_fn` / `flash_attention_fn` / `run_attention_fn` | **YES** (paged_attention adds mask @ util.rs:150; flash sets causal=true @ gqa.rs:236) | External call sites |

The discrepancy between `forward()` (no mask) and the helpers (with mask) is a
**correctness concern** but not a perf hotspot. See "Limitations" below.

---

## Baseline (from H-2 + re-run on 2026-06-28)

Criterion `--bench gqa_forward --bench gqa_forward_smoke/cpu_smoke --sample-size 10`
(Run output: `/tmp/v27_h8_gqa_baseline.txt`)

| Path | seq_len | Dims | ns/iter (median) | Source |
|------|---------|------|------------------|--------|
| `gqa_forward_smoke/cpu_smoke` | 16 | hidden=64, h=2, h_kv=1, d=32 | **39,530 ns (~39.5 µs)** | this run |
| `gqa_forward_smoke/cpu_smoke` | 16 | (H-2 recorded) | 38,445 ns | `docs/perf/v27-baseline.md` |
| `gqa_forward/standard` | 128/512/2048 | hidden=896, h=14, h_kv=2, d=64 | TBD (GPU required) | `docs/perf/v27-baseline.md` |

**CPU smoke is for path-correctness only**, not for measuring realistic
qwen3-7B perf. Numbers are within 3% of H-2 baseline (no regression in master).

---

## Suspected hotspots (priority order)

### 1. **[HIGH] `expand_kv` called twice on K AND V** — `gqa.rs:170-171`, helper at `util.rs:59-90`

**Pattern:**
```rust
let k = self.expand_kv(&k, self.num_heads, self.num_kv_heads)?;  // gqa.rs:170
let v = self.expand_kv(&v, self.num_heads, self.num_kv_heads)?;  // gqa.rs:171
```

The helper (`util.rs:84-89`) materializes a new tensor via `kv.repeat(&[1, 1,
repeat_factor, 1])`. For qwen3-7B (`num_heads=14, num_kv_heads=2`), each call
allocates a tensor **7× the size of the input K/V** (and again for V).

**Why it's slow:**
- Full-tensor materialization of replicated data — wastes memory bandwidth
- For a typical prefill at `seq_len=2048, batch=4, head_dim=64`, each expansion
  is `4 × 2048 × 14 × 64 × 4B = 28 MB`. K+V expansions = **56 MB** of redundant
  data materialization on every forward pass.
- The same pattern exists in `flash_attention_v3.rs:249-258` for the fused path,
  where `expand_kv(k)` and `expand_kv(v)` similarly materialize.

**Optimization candidates:**
- **(A) Lazy broadcast:** Replace `repeat` with a `view + broadcast` matmul
  (`Tensor::matmul` on strided tensors). Same output, no materialization.
- **(B) Fuse QKV projection:** Project QKV into one matmul (already done in
  some LLM stacks) and reshape to `(B, S, n_heads + 2*n_kv_heads, head_dim)`,
  keeping KV in their native `(B, S, n_kv_heads, D)` layout.
- **(C) Cache expanded K/V** when KV-cache is in use (decode path) — the
  expansion is per-token after the first decode.

### 2. **[HIGH] `Tensor::new(&[scale], ...)` re-allocated every forward** — `gqa.rs:180`, `util.rs:155`, `util.rs:203`, `flash_attention_v3.rs:262`

**Pattern:**
```rust
let scale = 1.0 / (self.head_dim as f32).sqrt();
let qk = qk.mul(&Tensor::new(&[scale], q.device())?.broadcast_as(qk.dims())?)?;  // gqa.rs:179-180
```

`scale` is a function of `head_dim` only — it's **fixed at instance construction
time**. Yet it's re-allocated as a 1-element tensor, then broadcast to the
full qk shape, on every forward call. For `tiled_attention`
(`util.rs:202-204`), this happens **per tile**.

**Why it's slow:**
- Two allocations (the scalar tensor + the broadcast view) per call
- The full-sized broadcast tensor dominates: O(B × H × S × S) float copies
- For a tiled forward with 8 tiles at `seq_len=2048, head_dim=64`, this happens
  8× per forward

**Optimization candidates:**
- **(A) Cache `scale` as `OnceLock<Tensor>` or struct field** initialized in
  `new()`. Eliminates per-call allocation.
- **(B) Use `Tensor::affine(scale, 0.0)`** instead of `mul(broadcast(scale))` —
  fuses the scaling into the existing kernel without materializing a broadcast
  tensor. `affine` is already used elsewhere in candle for similar patterns.

### 3. **[HIGH] Excessive `transpose().contiguous()` round-trips** — `gqa.rs:154, 157-158, 162, 173-176, 183, 185`

**Pattern (counted from the file):**

| Line | Operation | Allocates? |
|------|-----------|------------|
| 154 | `q.transpose(1,2)?.contiguous()?` | YES |
| 157 | `k.transpose(1,2)?.contiguous()?` (fused) | YES |
| 158 | `v.transpose(1,2)?.contiguous()?` (fused) | YES |
| 162 | `attn_output.transpose(1,2)?` (fused) | view only |
| 173 | `k.transpose(1,2)?` | view only |
| 174 | `v.transpose(1,2)?` | view only |
| 176 | `k.transpose(2,3)?.contiguous()?` | YES (k_t) |
| 183 | `v.contiguous()?` | YES |
| 185 | `attn_output.transpose(1,2)?` | view only |

**Six forced `.contiguous()` materializations** per forward in the standard
path (4 in the fused path). Each is a full-tensor O(B·S·H·D) copy.

**Why it's slow:**
- Memory bandwidth bound: each copy reads + writes the full tensor
- For seq_len=2048, hidden=896, F32: each contiguous copy is ~7 MB; 6 copies =
  ~42 MB of redundant traffic per forward
- The contiguous calls are forced by downstream ops that don't accept strided
  inputs (e.g., the matmul in the standard path needs K_t contiguous; the
  softmax needs `attn_weights.contiguous()`)

**Optimization candidates:**
- **(A) Keep tensors in `(B, H, S, D)` layout from projection onward** —
  reshape projections directly into `(B, S, n_heads, head_dim).transpose(1,2)`
  and rely on candle's strided matmul. Eliminates 2-3 contiguous calls.
- **(B) Fuse Q/K/V transpose+contiguous into one pass** (custom kernel or
  batched slice). Reduces materializations from 6 → 2.
- **(C) Use a fused attention kernel** (already exists for the GPU path:
  `GqaFlashAttention`, `FlashAttentionV3`). The fused path still has the
  contiguous issue, but softmax+matmul fusion eliminates the attn_weights
  materialization.

### 4. **[MEDIUM] `tiled_attention` Vec::push + Tensor::cat pattern** — `util.rs:182, 208, 211`

**Pattern:**
```rust
let mut output_parts = Vec::new();
for tile_idx in 0..num_tiles {
    ...
    let out = Tensor::matmul(&attn, &v_tile)?;
    output_parts.push(out);                                  // util.rs:208
}
let attn_output = Tensor::cat(&output_parts, 2)?;            // util.rs:211
```

**Why it's slow:**
- `Vec::push` grows the vec (amortized reallocations)
- `Tensor::cat` allocates a new output tensor and copies all parts in
- For a 2048-token sequence with tile_size=16, that's 128 Vec pushes + 1 cat
  that copies 128 tile-sized tensors into a fresh allocation
- Per-tile overhead: `causal_mask_tile` allocates a Vec<f32> per tile
  (`util.rs:119-130`), scale tensor re-allocated per tile (`util.rs:202-203`)

**Optimization candidates:**
- **(A) Pre-allocate output buffer** of full seq_len shape; narrow+write each
  tile directly. Eliminates `cat` and Vec.
- **(B) Cache `causal_mask_tile` outputs** per (start, tile_len) — these are
  deterministic given seq_len/tile_size.
- **(C) Cache scale tensor** per GqaAttention instance (see hotspot #2).

### 5. **[MEDIUM] `causal_mask` allocates 5 tensors per call** — `util.rs:96-106`, also at `flash_attention_v3.rs:292-303`

**Pattern:** Two arange + reshape + broadcast + two scalar broadcasts + where_cond.
For `tiled_attention`, called PER TILE (so `num_tiles` times).

**Why it's slow:** O(seq_len^2) allocation; per-tile invocation in tiled path.

**Optimization candidates:**
- **(A) Cache masks** in `AttentionConfig` or a `MaskCache` struct keyed by
  seq_len, shared via `Arc<Tensor>`.
- **(B) Skip mask when seq_len == 1** (decode path, no causal structure
  needed).
- **(C) For fixed-size decode batches**, pre-compute masks once.

### 6. **[MEDIUM] `apply_q_norm`/`apply_k_norm` redundant transpose pairs** — `gqa.rs:262-288`

**Pattern:**
```rust
fn apply_q_norm(&self, q, batch_size, seq_len) -> Result<Tensor> {
    if let Some(ref q_norm) = self.q_norm {
        let q = q.transpose(1, 2)?;                          // T1
        let reshape_size = batch_size * self.num_heads * seq_len;
        let q = q.reshape((reshape_size, self.head_dim))?;   // R1
        let q = q_norm.forward(&q)?;
        let q = q.reshape((batch_size, self.num_heads, seq_len, self.head_dim))?;  // R2
        let q = q.transpose(1, 2)?;                          // T2
        Ok(q)
    }
}
```

**4 reshape/transpose ops for a single LayerNorm.** Notably,
`apply_q_norm_impl` (gqa.rs:335-351) is a **cleaner variant** without the
transposes but is not called in the hot `forward()` path.

**Why it's slow:**
- T1 breaks contiguity → R1 may force a copy → LN reads non-contiguous → R2
  may force another copy → T2 view.
- Total: up to 3 forced copies per LN call, when only LN is needed

**Optimization candidates:**
- **(A) Use `apply_q_norm_impl` (or a no-transpose variant)** in the hot path —
  apply LN on the `(B, S, H, D)` layout directly; reshape to `(B*S*H, D)` only.
- **(B) Fuse QK-norm with the projection matmul** as a custom kernel (used in
  some optimized stacks like vLLM-v1).

### 7. **[LOW] `.contiguous()` after softmax** — `gqa.rs:181`

`candle_nn::ops::softmax` already returns a contiguous tensor; the explicit
`.contiguous()?` forces a redundant check. Tiny overhead on its own, but
compounds with hotspot #3.

---

## No-issue areas

- **Q/K/V projections (gqa.rs:143-145):** Direct `Linear::forward` calls — minimal overhead. The 3 separate matmuls are an obvious fusion candidate but the current candle `Linear` API doesn't support fused QKV; defer to kernel-level work.
- **Output projection (`o_proj` at gqa.rs:190, 165, 216, 227, 243):** Single matmul, no obvious wins.
- **Module structure:** Clean separation between `forward`, `paged_attention_fn`, `tiled_attention_fn`, `flash_attention_fn`, `run_attention_fn` — the API surface is consistent and well-named.

---

## Recommended H-11 optimization targets

| Rank | Target | File:line | Estimated speedup | Risk | Notes |
|------|--------|-----------|-------------------|------|-------|
| **1 (Primary)** | Cache `scale` tensor; use `affine` instead of `mul(broadcast(scale))` | gqa.rs:179-180, util.rs:155,203, flash_attention_v3.rs:262 | **2-5% on standard path; up to 15% on tiled path** (per-tile) | Low | Pure refactor; correctness trivial to verify (existing test `test_gqa_attention_fused_matches_standard` already covers numerical parity) |
| **2 (Secondary)** | Eliminate one `.contiguous()` after softmax; restructure to keep tensors in `(B, H, S, D)` layout from projection | gqa.rs:154, 157-158, 176, 181, 183 | **5-10% on standard path; ~3% on fused path** | Medium | Requires reshaping projections to `(B, S, H, D).transpose(1,2)` and verifying matmul accepts strided. Use `test_gqa_attention_fused_matches_standard` to validate. |
| **3 (Stretch)** | Lazy `expand_kv` via broadcast view, or fused QKV projection | gqa.rs:170-171, util.rs:59-90, flash_attention_v3.rs:249-258 | **5-15% on prefill (memory-bandwidth-bound); higher on long sequences** | Medium-High | Largest perf win but touches shape semantics; need `test_gqa_attention_expand_kv_correct` and a numerical-parity test against the materialized version |

**Suggested order:** do #1 first (lowest risk, easy to validate), then #2, then
#3. Each step should re-bench with `just bench-model-one gqa_forward` and run
`just nextest` to confirm no regressions.

**Out of scope for H-11 (consider H-12/H-13 or a separate task):**
- `tiled_attention` Vec::push + Tensor::cat rewrite (#4) — touches `util.rs`
  helpers, affects paged/tiled decode paths
- `causal_mask` caching (#5) — needs API change to share masks across instances
- `apply_q_norm` rewrite (#6) — only matters for Qwen3+ models with `has_qk_norm=true`; check whether qwen3.5 hybrid path matters before optimizing

---

## Correctness note (carried forward for H-9/H-10)

`GqaAttention::forward` (the path the H-2 bench exercises) **does NOT apply
causal masking**:

- `use_fused=true` branch: `GqaFlashAttention::new(..., false)` hardcodes causal
  to false (`gqa.rs:160`).
- `use_fused=false` branch: no mask added between QK matmul and softmax
  (`gqa.rs:177-181`).

The helper methods (`paged_attention_fn`, `tiled_attention_fn`,
`flash_attention_fn`, `run_attention_fn`) DO apply causal masks. The H-2 bench
correctness is unaffected (no causal structure in `attn.forward(&x)` with
`AttentionConfig::default()`), but this divergence between the `forward()`
entry point and the helpers is a latent bug. Flag for H-9/H-10 review.

---

## Limitations

- **Static analysis cannot measure actual CPU time per function.** All
  hotspot rankings are based on call-graph shape (allocation count, tensor
  materialization count, loop bodies), not measured wall-clock time per
  function. Real flamegraph data on GPU hardware is required to confirm.
- **CPU smoke numbers (~39 µs at seq_len=16) are dominated by overhead, not
  by the GQA math.** Per-tensor allocations and `candle_core::Tensor` dispatch
  dominate at this scale; on GPU with larger seq_len, the matmul/softmax cost
  would dominate and our allocation hotspots would be relatively smaller.
- **H-11 optimizers should re-bench after each change** with
  `just bench-model-one gqa_forward` and confirm the standard-vs-fused
  numerical-parity test still passes
  (`test_gqa_attention_fused_matches_standard` in gqa.rs:446).
- **GPU profiling deferred.** Re-run this analysis with
  `cargo flamegraph --bench gqa_forward` on a GPU runner with
  `perf_event_paranoid<=0` to get real self-time numbers; expected to confirm
  or refine the rankings above.

---

## Files

- Bench output: `/tmp/v27_h8_gqa_baseline.txt`
- GQA source: `crates/model/src/components/attention/gqa.rs`
- Helper functions: `crates/model/src/components/attention/util.rs`
- Flash v3: `crates/model/src/components/attention/flash_attention_v3.rs`
- Plan: `docs/superpowers/plans/2026-06-28-v27-performance.md` (Task H-8)
- H-2 baseline: `docs/perf/v27-baseline.md` (gqa_forward_smoke rows)
