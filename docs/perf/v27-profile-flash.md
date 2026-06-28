# Flash Attention Profile (v27.0 H-9)

**Date:** 2026-06-28
**Target:** `FlashAttentionKernel::forward` and variants
**Source:** `crates/model/src/kernels/flash_attention/kernel.rs`
**Method:** Static code analysis (same as H-8)
**Branch:** main (no worktree, per AGENTS.md)
**Harness:** 128-core x86_64 (Linux product-1-23 5.15.0-179-generic), rustc 1.96.0

---

## Environment constraint

Identical to H-8/H-9 (MLA): `cargo-flamegraph` installed but
`perf_event_paranoid=4` blocks CPU sampling. Static analysis only.
Real flamegraph deferred to GPU runner. See
`docs/perf/v27-profile-gqa.md` for the full note.

---

## Forward structure

The flash attention subsystem exposes **three forward pathways** via the
`FlashAttention` trait and a dispatcher:

```text
FlashAttentionKernel::forward(q, k, v)              @ kernel.rs:502
├─ if variant == Tiled: forward_tiled(...)          @ kernel.rs:513 → 504
├─ elif use_sliding_window: forward_sliding_window  @ kernel.rs:519
└─ else: self.attention.forward(q, k, v)            @ kernel.rs:509 (dispatch)

FlashAttentionKernel::forward_tiled(q, k, v)        @ kernel.rs:513
└─ select_tile_size(seq_len, config)                 @ config.rs:69
└─ self.attention.forward_tiled(q, k, v, tile_size)  @ kernel.rs:516

Three backend implementations:
├─ ScaledDotProductAttention                        @ kernel.rs:39
│   ├─ forward (l.444)         = qk matmul + scale + softmax + v matmul
│   ├─ forward_tiled (l.359)   = nested-batch × head × tile loops + cat
│   └─ compute_sliding_window  = narrow then standard forward
├─ FlashAttentionV2                                 @ kernel.rs:46
│   ├─ forward (l.75)          = standard if S≤128 else flash_v2
│   ├─ forward_standard (l.86) = matmul + softmax + matmul
│   ├─ forward_flash_v2 (l.94) = nested-batch × head × block online softmax
│   ├─ compute_flash_attention_block (l.124) = per-block online softmax
│   ├─ forward_with_causal_mask (l.185)
│   ├─ forward_flash_v2_with_causal (l.196)
│   ├─ compute_flash_attention_causal (l.225) = block loop + causal_mask
│   └─ create_causal_mask (l.285) = Vec<f32> push loop + from_slice
└─ FlashAttentionKernel dispatcher                  @ kernel.rs:475
    └─ variant → Box<dyn FlashAttention>            @ kernel.rs:483-493
```

**Public API:**

| Entry point | Variant | Notes |
|-------------|---------|-------|
| `FlashAttentionKernel::forward` | Standard / Flash / FlashV2 / Tiled | Dispatched by `AttentionVariant` enum (`config.rs:8`) |
| `FlashAttentionKernel::forward_sliding_window` | any | Used when `use_sliding_window` is set |
| `ScaledDotProductAttention::forward` / `compute_tiled` / `compute_sliding_window` | n/a | Direct entry, no dispatch |

---

## Baseline (from H-4 + re-run on 2026-06-28)

Criterion `--bench flash_attention_smoke/cpu_smoke --sample-size 10`
(Run output: `/tmp/v27_h4_baseline.txt`, recorded in
`docs/perf/v27-baseline.md` line 174)

| Path | Config | ns/iter (median) | Source |
|------|--------|------------------|--------|
| `flash_attention_smoke/cpu_smoke` | b1_h2_s16_d32 | **11,285 ns (~11 µs)** | H-4 baseline |
| `flash_attention/standard` | b1_h14_s512_d64 | TBD (GPU required) | `docs/perf/v27-baseline.md` |
| `flash_attention/standard` | b1_h14_s2048_d64 | TBD | `docs/perf/v27-baseline.md` |
| `flash_attention/standard` | b4_h14_s512_d64 | TBD | `docs/perf/v27-baseline.md` |

**CPU smoke is for path-correctness only.** At seq_len=16, head_dim=32 the
implementation overhead (tensor dispatch, scale broadcast) dominates the
matmul cost. Realistic b1_h14_s2048_d64 forward on CPU would be ~minutes;
the smoke number is a sanity check.

---

## Suspected hotspots (priority order)

### 1. **[HIGH] `compute_flash_attention_block` per-block tensor explosion** — `kernel.rs:124-179`

**Pattern (per block inside the `for block_idx in 0..num_blocks` loop):**

```rust
let qk_block = q.matmul(&k_block.t()?)?;                     // alloc #1: (seq_q, block)
let qk_scaled = qk_block.broadcast_mul(&scale_tensor)?;      // alloc #2
let block_m = qk_scaled.max_keepdim(1)?;                     // alloc #3: reduction
let block_p = qk_scaled.broadcast_sub(&block_m)?.exp()?;     // alloc #4 + #5 (exp)
let block_l = block_p.sum_keepdim(1)?;                       // alloc #6: reduction

let m_diff = block_m.broadcast_sub(&running_m)?;             // alloc #7
let correction = m_diff.exp()?;                              // alloc #8 (exp)
let scaled_output = if let Some(...) { ... scaled.broadcast_div(&block_l)? };  // alloc #9
let block_out = block_p.matmul(&v_block)?;                   // alloc #10
final_output = scaled_output.broadcast_add(&block_out)?;     // alloc #11
```

**~8-11 tensor allocations per block.** For `seq_len_k=2048, block_size=64`:
**32 blocks × 10 allocations = ~320 tensor allocations per head per forward**.
For `B=4, H=14`: **~4,480 tensor allocations** in one forward pass.

**Why it's slow (this is the central algorithmic hotspot):**
- This implementation **materializes the full QK block per iteration**,
  contrary to the original FlashAttention 2 design which keeps QK in
  registers/SRAM and never writes it to HBM
- The "online softmax" recurrence (running_m, running_l) is the correct
  *math* but the per-block tensor materialization defeats the memory
  bandwidth advantage that motivated the algorithm
- Each `exp()` and `broadcast_sub/broadcast_mul/broadcast_add` is a
  full kernel launch with full-tensor memory traffic
- On GPU, the original FlashAttention paper claims **~2-4× speedup over
  standard attention**; this implementation will see much smaller gains
  (probably just the softmax-in-numerically-stable-form benefit, no
  memory savings)

**Optimization candidates:**
- **(A) Move to `candle_nn::ops::softmax_last_dim`** and pre-allocate
  output buffers to reduce per-block allocation count from 11 → ~4.
- **(B) Fuse online softmax into a single kernel** (true FlashAttention 2
  port): requires custom CUDA kernel via the `kernels/cuda.rs`
  integration point. Out of scope for this H-9 report — flags for
  future kernel work.
- **(C) Reduce broadcast ops** — many `broadcast_mul/sub/add/div` calls
  create intermediate tensors. Replace with in-place ops where possible
  (e.g., `block_p.broadcast_sub_assign(...)` if API supports it).

### 2. **[HIGH] `create_causal_mask` builds mask via Vec<f32> push loop** — `kernel.rs:285-308`

**Pattern (called per block in `compute_flash_attention_causal`):**
```rust
fn create_causal_mask(&self, dims: &[usize], start_k: usize, device) -> Result<Tensor> {
    let (seq_len_q, block_size) = (dims[0], dims[1]);
    let mut mask_data = Vec::with_capacity(seq_len_q * block_size);  // Vec alloc
    for q_idx in 0..seq_len_q {
        for k_idx in 0..block_size {
            let global_k_idx = start_k + k_idx;
            if q_idx > global_k_idx {
                mask_data.push(f32::NEG_INFINITY);                     // push #1
            } else {
                mask_data.push(0.0);                                   // push #2
            }
        }
    }
    Tensor::from_slice(&mask_data, (seq_len_q, block_size), device)    // alloc + copy
}
```

For `seq_len_q=2048, block_size=64`: **131,072 f32 pushes per block**, **32
blocks per forward = ~4.2M pushes** to build a mask that's
**deterministic** given `(seq_len_q, block_size, start_k)`.

**Why it's slow:**
- Vec push loop: 131k allocations worth of writes per block (though
  amortized via `with_capacity`)
- `Tensor::from_slice` then allocates a fresh `(seq_len_q, block_size)`
  tensor and copies 131k f32s in
- Total per causal forward: 32 × (Vec::with_capacity + 131k pushes +
  131k f32 copy) = **~8 MB of redundant mask materialization**
- The mask structure is **purely positional** — no data dependency

**Optimization candidates:**
- **(A) Replace with broadcast comparison** (the standard pattern):
  ```rust
  let q_idx = Tensor::arange_step(0, seq_len_q, 1, device)?;
  let k_idx = Tensor::arange_step(start_k, start_k + block_size, 1, device)?;
  let mask = q_idx.unsqueeze(1)?.broadcast_lt(&k_idx.unsqueeze(0)?)?
                       .to_dtype(candle_core::DType::F32)?
                       .mul(&Tensor::new(f32::NEG_INFINITY, device)?)?
                       .broadcast_as((seq_len_q, block_size))?;
  ```
  Same result, no Vec, no per-element push loop. Or use the
  `causal_mask` helper already in `components/attention/util.rs:96-106`
  (the same pattern GQA uses).
- **(B) Cache the mask** per (seq_len, start_k) — deterministic, no reason
  to rebuild it.
- **(C) Apply mask via `where_cond`** on the qk_scaled tensor instead of
  pre-allocating — fuses the mask into the softmax computation.

### 3. **[HIGH] `forward_flash_v2` and `forward_flash_v2_with_causal` triple-nested loops** — `kernel.rs:99-122, 200-223`

**Pattern:**
```rust
let mut outputs = Vec::with_capacity(batch_size);                  // Vec alloc #1
for b in 0..batch_size {
    let q_b = q.narrow(0, b, 1)?.squeeze(0)?;                      // 3 view ops
    let k_b = k.narrow(0, b, 1)?.squeeze(0)?;
    let v_b = v.narrow(0, b, 1)?.squeeze(0)?;

    let mut head_outputs = Vec::with_capacity(num_heads_q);        // Vec alloc per batch
    for h in 0..num_heads_q {
        let q_h = q_b.narrow(0, h, 1)?.squeeze(0)?;                // 3 view ops per head
        let k_h = k_b.narrow(0, h % num_heads_k, 1)?.squeeze(0)?;
        let v_h = v_b.narrow(0, h % num_heads_k, 1)?.squeeze(0)?;

        let out_h = self.compute_flash_attention_block(&q_h, &k_h, &v_h)?;
        head_outputs.push(out_h);                                  // Vec push per head
    }

    let batch_out = Tensor::stack(&head_outputs, 0)?;              // STACK alloc per batch
    outputs.push(batch_out);                                       // Vec push per batch
}

Tensor::stack(&outputs, 0)                                          // final STACK alloc
```

For `B=4, H=14`: **4 batch iterations × 14 head iterations = 56 squeeze/narrow
calls + 4 `Tensor::stack` operations + 56 `compute_flash_attention_block`
calls**.

**Why it's slow:**
- Per (batch, head) call to `compute_flash_attention_block` re-builds the
  scale_tensor (cached locally at l.131 but `running_m` / `final_output`
  are re-allocated per call) — these allocations can be hoisted
- `Tensor::stack` allocates new tensors and copies inputs in
- Per-call narrow+squeeze × 6 ops: 336 view ops per forward (minor
  overhead but adds up)

**Optimization candidates:**
- **(A) Batch across heads:** `flash_attention_v2` does the same
  computation per head independently — fuse into a single matmul over
  `(B, H, S, D)` tensors. Eliminates the per-head squeeze+stack.
- **(B) Batch across batch dim:** similarly fuse across batch. Eliminates
  the per-batch squeeze+stack.
- **(C) Pre-allocate output buffers:** allocate `final_output` /
  `running_m` / `running_l` outside the loop and reuse for each (b, h)
  pair. Saves ~3 allocations per head.

### 4. **[HIGH] `compute_tiled` Vec::push + Tensor::cat** — `kernel.rs:359-417`

**Pattern:**
```rust
let mut all_outputs: Vec<Tensor> = Vec::with_capacity(batch_size);  // Vec alloc
for _b in 0..batch_size {
    let mut head_outputs: Vec<Tensor> = Vec::with_capacity(num_heads);  // Vec alloc per batch
    for h in 0..num_heads {
        let q_bh = q.narrow(1, h, 1)?.squeeze(1)?;                     // 3 view ops
        let k_bh = k.narrow(1, h, 1)?.squeeze(1)?;
        let v_bh = v.narrow(1, h, 1)?.squeeze(1)?;

        let mut tile_outputs: Vec<Tensor> = Vec::new();                // Vec alloc per head
        for start in (0..seq_len).step_by(tile_size) {
            let end = (start + tile_size).min(seq_len);
            let actual_tile_size = end - start;
            let q_tile = q_bh.narrow(1, start, actual_tile_size)?;     // view per tile
            let k_tile = k_bh.narrow(1, 0, end)?;
            let v_tile = v_bh.narrow(1, 0, end)?;

            let qk = q_tile.matmul(&k_tile.t()?)?;                     // alloc per tile
            let qk_scaled = qk.broadcast_mul(&scale_tensor)?;
            let attn = softmax_last_dim(&qk_scaled)?;
            let out_tile = attn.matmul(&v_tile)?;                      // alloc per tile
            tile_outputs.push(out_tile);                               // Vec push per tile
        }
        let head_output = Tensor::cat(&tile_outputs, 0)?;              // CAT alloc per head
        head_outputs.push(head_output);                                // Vec push per head
    }
    let batch_output = Tensor::stack(&head_outputs, 0)?;               // STACK alloc per batch
    all_outputs.push(batch_output);                                    // Vec push per batch
}
let result = Tensor::stack(&all_outputs, 0)?;                          // final STACK
```

For `B=4, H=14, seq_len=2048, tile_size=64`: **4 × 14 × 32 = 1,792 tile
iterations + 56 cat/stack allocations** per forward.

**Why it's slow:**
- **Identical pattern to GQA tiled_attention (H-8 hotspot #4)**: `Vec::push`
  grows + `Tensor::cat` allocates and copies all tiles
- The inner loop runs `seq_len / tile_size` times and creates 4 tensors
  per iteration (qk, qk_scaled, attn, out_tile) = **128 tensor allocations
  per head** for seq_len=2048, tile_size=64
- Per-tile softmax is **non-causal** — sees the full KV, contradicting
  the `Tiled` variant name's implied causal structure

**Optimization candidates:**
- **(A) Pre-allocate output buffer** of shape `(seq_q, head_dim)` for each
  head. Write each tile via narrow+copy. Eliminates the `Tensor::cat`.
- **(B) Fuse softmax into matmul via online softmax** (see hotspot #1).
- **(C) Batch matmul over tiles** — compute multiple tile outputs in a
  single matmul call.

### 5. **[MEDIUM] `Tensor::new(self.scale, q.device())?` per-call re-allocation** — `kernel.rs:88, 131, 232, 376, 446`

**Pattern (5 occurrences):**
```rust
let scale_tensor = Tensor::new(self.scale, q.device())?;          // l.88 (forward_standard)
let scale_tensor = Tensor::new(self.scale, q.device())?;          // l.131 (compute_flash_attention_block)
let scale_tensor = Tensor::new(self.scale, q.device())?;          // l.232 (compute_flash_attention_causal)
let scale_tensor = Tensor::new(self.scale, q.device())?;          // l.376 (compute_tiled)
let scale_tensor = Tensor::new(self.scale, q.device())?;          // l.446 (SDPA::forward)
```

`self.scale` is a `f32` stored as a struct field (e.g.,
`FlashAttentionV2.scale` at l.47, `ScaledDotProductAttention.scale` at
l.40). It's **constant after construction** — yet re-allocated as a
0-D tensor on every forward.

**Note: This is less severe than GQA's per-tile `Tensor::new(&[scale], ...)`
pattern.** Flash's scale_tensor is a **0-D tensor** (no broadcast), so the
allocation is just one scalar tensor (4 bytes payload). The cost is the
allocation + dispatch overhead, not memory bandwidth.

**Why it's still slow (minor):**
- One 0-D Tensor allocation per forward call (5 call sites, but only one
  is hit per variant)
- The 0-D tensor is then used as `broadcast_mul` argument — could use
  in-place affine or scalar mul

**Optimization candidates:**
- **(A) Cache `scale_tensor` as a struct field** initialized in `new()`:
  ```rust
  pub struct FlashAttentionV2 {
      scale: f32,
      scale_tensor: Tensor,  // initialized once in new()
      ...
  }
  ```
- **(B) Use `Tensor::affine(scale, 0.0)`** instead of
  `tensor.broadcast_mul(&scale_tensor)?` — fuses scaling into the
  next op without materializing the scalar tensor.
- **(C) Use a closure or scalar kernel** if candle exposes one.

### 6. **[MEDIUM] `softmax_last_dim` allocates 4 intermediate tensors** — `kernel.rs` (util.rs:32-38)

**Pattern:**
```rust
pub fn softmax_last_dim(t: &Tensor) -> Result<Tensor> {
    let max_vals = t.max_keepdim(shape.len() - 1)?;       // alloc #1
    let t_shifted = t.broadcast_sub(&max_vals)?;          // alloc #2
    let exp = t_shifted.exp()?;                           // alloc #3
    let sum = exp.sum_keepdim(shape.len() - 1)?;          // alloc #4
    exp.broadcast_div(&sum)                               // alloc #5
}
```

**5 tensor allocations per softmax call.** Called in every forward path
(`SDPA::forward` l.448, `SDPA::compute_tiled` l.401,
`FlashAttentionV2::forward_standard` l.90, per-block in v2).

**Why it's slow:**
- Per-softmax overhead in the hot loop
- For tiled SDPA: 32 softmax calls per head = **160 allocations per head**
  just for softmax

**Optimization candidates:**
- **(A) Use `candle_nn::ops::softmax`** (already in the dependency tree).
  Same math, possibly better-fused.
- **(B) Fuse softmax with the prior matmul** via online softmax (see
  hotspot #1).
- **(C) Reduce intermediate allocations** via in-place ops if candle
  exposes them.

### 7. **[MEDIUM] Per-iteration `narrow + squeeze` × 6** — `kernel.rs:102-104, 109-111, 203-205, 210-212, 383-385`

**Pattern:**
```rust
let q_b = q.narrow(0, b, 1)?.squeeze(0)?;
let k_b = k.narrow(0, b, 1)?.squeeze(0)?;
let v_b = v.narrow(0, b, 1)?.squeeze(0)?;
// ... then per head ...
let q_h = q_b.narrow(0, h, 1)?.squeeze(0)?;
let k_h = k_b.narrow(0, h % num_heads_k, 1)?.squeeze(0)?;
let v_h = v_b.narrow(0, h % num_heads_k, 1)?.squeeze(0)?;
```

For `B=4, H=14`: **6 view ops × 56 (B×H) iterations = 336 view ops per
forward**. Each is a view, not a copy, but each adds dispatch overhead
and tensor metadata tracking.

**Why it's slow:**
- View ops are cheap individually but compound at scale
- The `(h % num_heads_k)` modulo implies GQA — but if `num_heads_k ==
  num_heads_q` the modulo is a no-op (this is the MHA case)

**Optimization candidates:**
- **(A) Skip the outer batch loop entirely:** flatten `(B, H, S, D)` into
  `(B*H, S, D)` once at the top, run the per-head logic over the
  flattened batch, reshape at the end. Eliminates 3 narrow+squeeze ops
  per (B, H) iteration.
- **(B) Use `slice` or stride-based views** that don't allocate view
  metadata.

### 8. **[LOW] `forward_standard` returns a non-contiguous output** — `kernel.rs:86-92`

**Pattern:**
```rust
fn forward_standard(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let qk = q.matmul(&k.t()?)?;                          // alloc
    let scale_tensor = Tensor::new(self.scale, q.device())?;
    let qk_scaled = qk.broadcast_mul(&scale_tensor)?;     // alloc
    let attn = softmax_last_dim(&qk_scaled)?;             // alloc
    attn.matmul(v)                                         // alloc, but output may be non-contig
}
```

The final `attn.matmul(v)` returns a tensor whose layout depends on v's
strides. No explicit `.contiguous()` is called — caller may pay for a
forced copy downstream. (Note: candle matmul generally returns
contiguous output, so this is a latent concern, not a confirmed hit.)

**Why it's slow (latent):**
- If a caller downstream does `.contiguous()?` on the result, that's an
  unexpected full-tensor copy
- Adding an explicit `.contiguous()?` at the end of `forward_standard`
  makes the contract clear (but adds an unconditional copy cost)

**Optimization candidates:**
- **(A) Document the contiguity contract** in the doc comment; rely on
  callers to handle it.
- **(B) Add `.contiguous()?`** at the end if downstream callers always
  need it — measure before deciding.

---

## Architectural observation (not a hotspot, but worth flagging)

The current `FlashAttentionV2::compute_flash_attention_block` is
**algorithmically online-softmax (correct), but operationally a tiled
SDPA** — it materializes the full QK block per iteration rather than
keeping QK in registers/SRAM as in the original FlashAttention 2 paper.

This means:
- **Memory bandwidth wins (the main FA-2 advantage) are NOT realized**
  on GPU hardware
- The implementation will show **~similar perf to a tiled SDPA** on GPU,
  not the 2-4× speedup claimed by FlashAttention 2
- To realize true FA-2 speedup, a custom CUDA kernel using shared memory
  tiling is needed (out of scope for this static analysis report)

The `FlashAttentionKernel::forward` API surface is well-named and
maintainable; the algorithmic gap is in the kernel implementation, not
the API.

---

## No-issue areas

- **`apply_rope`, projection, output proj:** Not in this file.
- **No `.contiguous()` calls in the entire kernel.rs** (confirmed by
  `rg "\.contiguous\(\)" crates/model/src/kernels/flash_attention/kernel.rs`
  returning 0 matches). All strided matmul is used correctly.
- **`softmax_last_dim` (util.rs:32-38):** Mathematically correct
  numerically-stable softmax. Hotspot #6 is about allocation count, not
  correctness.
- **Config helpers (`select_tile_size`, `should_use_tiled` in
  config.rs:69,84):** Pure functions, no allocations.

---

## Comparison with GQA (H-8) and MLA

**Recurring hotspots — same root cause across all three:**

| Hotspot | GQA | MLA | FlashAttn | Shared fix |
|---------|-----|-----|-----------|------------|
| `Tensor::new(scale, device).broadcast_*` per call | `gqa.rs:179-180` (HIGH) | `mla.rs:210` (HIGH) | `kernel.rs:88, 131, 232, 376, 446` (MED) | Cache `scale_tensor` as struct field |
| `softmax().contiguous()` redundant | `gqa.rs:181` (LOW) | `mla.rs:213` (LOW) | n/a (no .contiguous() in flash) | Drop the .contiguous()? |
| Vec::push + Tensor::cat in tiled paths | `util.rs:208, 211` (MED) | n/a (no tiled path) | `kernel.rs:404, 407` (HIGH) | Pre-allocate output buffer |
| Multiple `.contiguous()` after transpose+reshape | `gqa.rs:154, 176` (HIGH) | `mla.rs:111, 121, 131, 192, 205, 206` (HIGH) | n/a | Layout refactor |
| Causal mask built via Vec<f32> + from_slice | `util.rs:96-106` (MED, uses arange pattern) | n/a | `kernel.rs:285-308` (HIGH, uses Vec push) | Use arange + broadcast_lt |
| Per-iteration narrow+squeeze loops | `util.rs:108-114` (MED) | n/a | `kernel.rs:102-104, 109-111, 203-205, 210-212` (MED) | Flatten (B,H) once |

**Key differences:**

| Area | GQA | MLA | FlashAttn |
|------|-----|-----|-----------|
| `expand_kv` repeat | HIGH | n/a | n/a |
| QK-norm transpose pair | MED (Qwen3+) | n/a | n/a |
| Online softmax (per-block) | n/a (FlashV3 path only) | n/a | **HIGH** (FlashV2 path) |
| Per-block softmax tensor explosion | n/a | n/a | **HIGH** (~10 alloc/block) |

**Key takeaway:** The flash attention kernel shares the **scale-tensor
allocation pattern** and **tiled Vec::push+cat pattern** with GQA, but has
its **own per-block online-softmax allocation explosion** that GQA/MLA
avoid. Flash attention is the most allocation-heavy of the three.

---

## Recommended H-12 optimization targets

| Rank | Target | File:line | Estimated speedup | Risk | Notes |
|------|--------|-----------|-------------------|------|-------|
| **1 (Primary)** | Replace `create_causal_mask` Vec<f32> push loop with arange+broadcast_lt (mirror GQA's `causal_mask` helper at `util.rs:96-106`) | kernel.rs:285-308 | **5-15% on causal FlashV2 path; 20-40% on long sequences** | Low | Same pattern as the existing GQA causal_mask helper; could share the helper directly. Correctness trivial to verify (existing tests cover numerical parity). |
| **2 (Secondary)** | Cache `scale_tensor` as struct field on `FlashAttentionV2` and `ScaledDotProductAttention`; use `Tensor::affine` instead of `broadcast_mul(scale_tensor)` | kernel.rs:88, 131, 232, 376, 446 | **2-5% on smoke; up to 10% on long sequences** (per-call allocation reduction) | Low | Pure refactor; same fix as GQA #1 and MLA #1. Could be a single shared helper. |
| **3 (Stretch)** | Fuse per-block online softmax allocations in `compute_flash_attention_block` — reduce 11 tensors/block → ~4 by reusing buffers across blocks | kernel.rs:124-179, 225-283 | **10-30% on FlashV2 path; depends heavily on seq_len** | Medium | Largest perf win but touches kernel-internal buffer management; needs careful numerical validation against `test_flash_attention_v2_consistency_with_sdpa` (l.741). |

**Suggested order:** #1 first (lowest risk, mirrors existing pattern,
biggest single-fix win on causal path), then #2 (mechanical caching),
then #3 (largest win but needs buffer-management care).

**Out of scope for H-12 (consider H-13+ or a separate kernel task):**
- True FlashAttention 2 GPU kernel via shared-memory tiling (the
  architectural gap noted above) — requires custom CUDA kernel work via
  `kernels/cuda.rs` integration
- Fusion of `forward_flash_v2` outer loops into a single batched matmul
  (hotspot #3) — touches kernel-internal layout
- Replacing `softmax_last_dim` with `candle_nn::ops::softmax` (hotspot
  #6) — depends on whether candle's softmax is materially better-fused
  on CPU/GPU

---

## Note on CPU vs GPU

Flash attention speedups are primarily a **GPU optimization** (memory
bandwidth wins via tiling + SRAM register reuse). The CPU flash
attention path has **limited headroom** because:

1. CPU doesn't have a fast shared memory equivalent — tile-to-tile
   transfer is still DRAM-bandwidth-bound
2. The CPU matmul kernel already does its own tiling via
   `candle_core::matmul` (BLAS-backed)
3. The online-softmax recurrence has more scalar overhead on CPU than
   GPU

Optimizations targeting kernel fusion or memory access patterns (#1, #3)
may show **small CPU wins** but **big GPU wins** (need GPU validation
to confirm). Optimizations targeting per-call allocations (#2) show
**similar wins on both CPU and GPU**.

The CPU smoke number (~11 µs at b1_h2_s16_d32) is dominated by tensor
dispatch overhead, not by the matmul/softmax cost. Real GPU perf gains
will require a GPU runner with `perf_event_paranoid<=0` for flamegraph
validation.

---

## Limitations

- **Static analysis cannot measure actual CPU time per function.** All
  hotspot rankings are based on call-graph shape (allocation count,
  tensor materialization count, loop bodies), not measured wall-clock
  time per function. Real flamegraph data on GPU hardware is required
  to confirm.
- **CPU smoke numbers (~11 µs at b1_h2_s16_d32) are dominated by
  overhead**, not by the flash attention math.
- **H-12 optimizers should re-bench after each change** with
  `just bench-model-one flash_attention` and confirm the existing flash
  tests still pass (`test_flash_attention_*` in `kernel.rs:534+`).
- **GPU profiling deferred.** Re-run this analysis with
  `cargo flamegraph --bench flash_attention` on a GPU runner with
  `perf_event_paranoid<=0` to get real self-time numbers.

---

## Files

- Flash attention kernel: `crates/model/src/kernels/flash_attention/kernel.rs`
- Config: `crates/model/src/kernels/flash_attention/config.rs`
- Util (softmax): `crates/model/src/kernels/flash_attention/util.rs`
- H-4 baseline: `docs/perf/v27-baseline.md` (flash_attention_smoke row)
- H-8 (GQA) reference: `docs/perf/v27-profile-gqa.md`
- H-9 (MLA) reference: `docs/perf/v27-profile-mla.md`
- Plan: `docs/superpowers/plans/2026-06-28-v27-performance.md` (Task H-9)
