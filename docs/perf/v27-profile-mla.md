# MLA Profile (v27.0 H-9)

**Date:** 2026-06-28
**Target:** `MlaAttention::forward` (`crates/model/src/components/attention/mla.rs:156`)
**Helpers:** `split_q` (l.87), `concat_q_nope_rope` (l.104), `reshape_k` (l.118),
`reshape_v` (l.128), `reshape_q_nope_for_attn` (l.183),
`attention_with_compressed_kv` (l.195)
**Method:** Static code analysis (same as H-8)
**Branch:** main (no worktree, per AGENTS.md)
**Harness:** 128-core x86_64 (Linux product-1-23 5.15.0-179-generic), rustc 1.96.0

---

## Environment constraint

Identical to H-8: `cargo-flamegraph` is installed but CPU sampling is blocked
by `perf_event_paranoid=4`. Real profiling deferred to GPU runner. Static
analysis only — hotspot rankings are based on call-graph shape (allocation
count, tensor materialization count, loop bodies), not measured wall-clock.

See `docs/perf/v27-profile-gqa.md` for the full environment note.

---

## MLA forward structure (line-by-line, `crates/model/src/components/attention/mla.rs`)

```text
forward(x: (B, S, H_hidden), positions: &[i64]) @ mla.rs:156
├─ q_compressed = q_proj(x)                                # (B, S, q_lora_rank)
├─ (q_nope, q_rope) = split_q(q_compressed, S)             # split along last dim (l.87-98)
│   ├─ reshape -> (B, S, q_nope + q_rope)
│   └─ narrow × 2  (view only)
├─ q_rope_4d = reshape_q_rope_for_rope(q_rope)            # (B, S, H, qk_rope_dim) view (l.134-141)
├─ q_rope_rotated_4d = apply_rope(q_rope_4d, positions)    # ROPE op (l.164)
├─ q_rope_rotated = reshape((B, S, H*qk_rope_dim))         # view (l.165-166)
├─ kv_compressed = kv_proj(x)                              # (B, S, kv_lora_rank) (l.168)
├─ k_decompressed = k_decompress(kv_compressed)            # Linear (l.169)
├─ k = reshape_k(k_decompressed, B, S)                     # reshape + transpose + CONTIGUOUS (l.170, 118-122)
├─ v_decompressed = v_decompress(kv_compressed)            # Linear (l.171)
├─ v = reshape_v(v_decompressed, B, S)                     # reshape + transpose + CONTIGUOUS (l.172, 128-132)
├─ q_nope_4d = reshape_q_nope_for_attn(q_nope, B, S)       # reshape + transpose + CONTIGUOUS (l.174, 183-193)
├─ attn_output = attention_with_compressed_kv(q_nope_4d, k, v)
│   ├─ q = q_nope.contiguous()?                            # already contiguous; redundant (l.205)
│   ├─ k_t = k.transpose(2,3)?.contiguous()?               # (B, n_kv, v_head_dim, S) — forced copy (l.206)
│   ├─ qk = q.matmul(&k_t)                                 # (B, H, S, S) (l.207)
│   ├─ scale_tensor = Tensor::new(&[scale], device)?       # NEW scalar tensor + broadcast (l.209-210)
│   │                .broadcast_as(qk.dims())?
│   ├─ qk = qk.mul(&scale_tensor)                          # broadcast mul (l.211)
│   ├─ attn_weights = softmax(qk, 3).contiguous()?         # forced copy (l.213)
│   ├─ attn_output = attn_weights.matmul(&v.contiguous()?) # V ALREADY contiguous from reshape_v (l.215)
│   ├─ attn_output.transpose(1,2)                          # view (l.216)
│   └─ reshape((B, S, H*v_head_dim))                       # view (l.217-218)
├─ q_concat = Tensor::cat(&[attn_output, q_rope_rotated], 2)  # CAT allocation (l.177)
└─ o = o_proj(q_concat)                                    # (B, S, H_hidden)
```

**Two public API pathways:**

| Helper | Causal mask? | Where called |
|--------|--------------|--------------|
| `forward()` (direct) | **NO** (`attention_with_compressed_kv` omits mask; softmax at l.213) | H-3 bench (`crates/model/benches/mla_forward.rs`) |
| `Qwen3MlaAttention` wrapper (`crates/model/src/qwen3/mla_attention.rs:50`) | **NO** (no production model imports it) | None — MLA is not wired into any registered model |

The lack of causal masking is by-design (low-level primitive; caller is
responsible), per the H-8 followup investigation
(`docs/perf/v27-correctness-investigation.md` lines 14-26, 82-97). It is a
contract issue, not a production bug. Not a perf hotspot.

---

## Baseline (from H-3 + re-run on 2026-06-28)

Criterion `--bench mla_forward_smoke/cpu_smoke --sample-size 10`
(Run output: `/tmp/v27_h3_baseline.txt`, recorded in
`docs/perf/v27-baseline.md` line 150)

| Path | seq_len | Dims | ns/iter (median) | Source |
|------|---------|------|------------------|--------|
| `mla_forward_smoke/cpu_smoke` | 16 | hidden=64, q_lora=16, kv_lora=16, h=2, h_kv=1, d_nope=8, d_rope=8, d_v=8 | **43,367 ns (~43 µs)** | H-3 baseline |
| `mla_forward` | 128 | hidden=896, q_lora=1536, kv_lora=512, h=14, h_kv=2, d_nope=128, d_rope=64, d_v=128 | TBD (GPU required) | `docs/perf/v27-baseline.md` |
| `mla_forward` | 512 | (same dims) | TBD | `docs/perf/v27-baseline.md` |
| `mla_forward` | 2048 | (same dims) | TBD | `docs/perf/v27-baseline.md` |

**CPU smoke is for path-correctness only.** At seq_len=16, hidden=64 the
projection overhead dominates and the attention math is amortized over
very few elements. Realistic qwen3-7B MLA on CPU would take ~minutes per
forward; the smoke number is a sanity check, not a meaningful baseline.

---

## Suspected hotspots (priority order)

### 1. **[HIGH] `Tensor::new(&[scale], device).broadcast_as(qk.dims())?`** — `mla.rs:210`

**Pattern (identical to GQA hotspot #2):**
```rust
let scale = 1.0 / (self.v_head_dim as f32).sqrt();
let scale_tensor = Tensor::new(&[scale], q.device())?.broadcast_as(qk.dims())?;
let qk = qk.mul(&scale_tensor)?;
```

`scale` is a function of `v_head_dim` only — fixed at instance construction.
Yet it's re-allocated as a 1-element tensor AND broadcast to the full qk
shape on every forward call.

**Why it's slow:**
- Two allocations per call (the scalar tensor + the broadcast view)
- The broadcast is **O(B × H × S × S)** f32 cells — for qwen3-7B MLA at
  prefill (`seq_len=2048, batch=4, h=14`), that's `4 × 14 × 2048 × 2048 × 4B
  = 1.84 GB` of broadcast metadata read on every forward
- The actual `mul` is then an O(B·H·S²) elementwise kernel that touches
  every qk cell again

**Optimization candidates:**
- **(A) Cache `scale` as a struct field** initialized in `new()`. Eliminates
  per-call allocation.
- **(B) Use `Tensor::affine(scale, 0.0)`** instead of `mul(broadcast(scale))` —
  fuses the scaling into a single kernel without materializing a broadcast
  tensor. Already the recommended fix for GQA hotspot #2.
- **(C) Fuse scale into the prior matmul** via `qk * scale` written as
  `qk.broadcast_mul(&cached_scale)` once scale is cached.

### 2. **[HIGH] Redundant `.contiguous()` materialization chain** — `mla.rs:111, 121, 131, 192, 205, 206, 213, 215`

**Pattern (counted):**

| Line | Operation | Source | Allocates? |
|------|-----------|--------|------------|
| 111 | `q.contiguous()` | `concat_q_nope_rope` (post-cat) | YES (post-cat tensor may be non-contig) |
| 121 | `k.contiguous()` | `reshape_k` (post-transpose) | YES (forced copy) |
| 131 | `v.contiguous()` | `reshape_v` (post-transpose) | YES (forced copy) |
| 192 | `q_nope_transposed.contiguous()` | `reshape_q_nope_for_attn` | YES (forced copy) |
| 205 | `q = q_nope.contiguous()?` | `attention_with_compressed_kv` | YES (q_nope already contiguous from l.192; REDUNDANT) |
| 206 | `k_t = k.transpose(2,3)?.contiguous()?` | `attention_with_compressed_kv` | YES (matmul needs K_t contiguous) |
| 213 | `softmax(&qk, 3)?.contiguous()?` | `attention_with_compressed_kv` | YES (softmax already returns contiguous; REDUNDANT) |
| 215 | `attn_weights.matmul(&v.contiguous()?)` | `attention_with_compressed_kv` | YES (v already contiguous from `reshape_v`; REDUNDANT) |

**Eight `.contiguous()` materializations** in the MLA forward path, of
which **three (l.205, l.213, l.215) are no-ops** because the source tensor
was just made contiguous. Each redundant call still pays the
"check-strides + maybe-copy" cost.

**Why it's slow:**
- Memory bandwidth bound: each copy reads + writes the full tensor
- For seq_len=2048, hidden=896, F32: each contiguous copy is ~7 MB; 8 copies
  = ~56 MB of redundant traffic per forward
- The contiguous calls are forced by downstream ops that don't accept
  strided inputs (notably the `q.matmul(&k_t)` at l.207 requires `k_t`
  contiguous, and `attn_weights.matmul(&v)` at l.215 needs V contiguous)

**Optimization candidates:**
- **(A) Drop redundant `.contiguous()` after a just-contiguous source:**
  l.205 (q_nope just made contiguous at l.192), l.215 (v just made
  contiguous at l.121 in `reshape_v`). These are pure overhead — saves
  2 forced copies per forward.
- **(B) Drop redundant `.contiguous()` after softmax** at l.213: candle's
  `softmax_last_dim` returns a fresh tensor that's already contiguous; the
  explicit `.contiguous()?` forces a redundant check + possible copy.
- **(C) Reorganize layout to keep tensors in `(B, H, S, D)` from projection
  onward:** reshape projections to `(B, S, H, D)` directly and only do one
  transpose+contiguous at the boundary. Saves ~3-4 copies per forward.

### 3. **[HIGH] `k_decompress` + `v_decompress` two separate matmuls on `kv_compressed`** — `mla.rs:169-171`

**Pattern:**
```rust
let kv_compressed = self.kv_proj.forward(x)?;           # (B, S, kv_lora_rank)
let k_decompressed = self.k_decompress.forward(&kv_compressed)?;
let v_decompressed = self.v_decompress.forward(&kv_compressed)?;
```

`kv_compressed` is the bottleneck intermediate (output of `kv_proj` at
`kv_lora_rank` dim, which is much smaller than the final head dim).
Both `k_decompress` and `v_decompress` are independent matmuls on the
same compressed input — a textbook fusion opportunity.

**Why it's slow:**
- Two separate matmuls on the same input → 2× launch overhead
- Two separate output tensor allocations
- No data dependency between them — can run in parallel on GPU

**Optimization candidates:**
- **(A) Fuse `k_decompress` and `v_decompress` into one matmul**:
  - `kv_decompressed = concat([k_decompress.weight, v_decompress.weight], dim=0).matmul(kv_compressed.t()).split(...)`
  - Or build a single `kv_decompress: Linear(kv_lora_rank, 2 * out_dim)` that
    splits internally.
  - Saves 1 launch overhead, 1 input broadcast overhead.

### 4. **[MEDIUM] `Tensor::cat` allocations on the output side** — `mla.rs:105, 177`

**Pattern:**
```rust
// concat_q_nope_rope (called outside forward, but is a public helper):
let q = Tensor::cat(&[q_nope, q_rope], 2)?;            // l.105
// then transpose+contiguous again at l.110-111

// Inside forward:
let q_concat = Tensor::cat(&[&attn_output, &q_rope_rotated], 2)?;  // l.177
```

`Tensor::cat` allocates a fresh output tensor and copies both inputs
into it. For MLA the rope-portion is `B × S × H × qk_rope_dim`; for
qwen3-7B at prefill that's `4 × 2048 × 14 × 64 × 4B = 28 MB` of redundant
allocation + copy on every forward.

**Why it's slow:**
- Allocates a new output tensor
- Two memcpys (one per input)
- For prefill, this is a non-trivial memory bandwidth hit

**Optimization candidates:**
- **(A) Pre-allocate `q_concat` buffer** of shape `(B, S, H, d_nope + d_rope)`
  in `MlaAttention::new()` (or lazily on first forward). Use narrow+write
  to fill `q_nope` and `q_rope_rotated` slices directly. Eliminates the
  cat allocation.
- **(B) Fuse the concat into `o_proj`:** pass attn_output and q_rope_rotated
  separately to `o_proj` via two matmuls summed, or pre-concatenate in
  a fused kernel.

### 5. **[MEDIUM] `v.contiguous()?` called inside matmul argument** — `mla.rs:215`

**Pattern:**
```rust
let attn_output = attn_weights.matmul(&v.contiguous()?)?;
```

`v` was just made contiguous by `reshape_v` at line 172 (l.128-132
ends with `.contiguous()`). The `.contiguous()?` here is a **redundant
no-op call**, but candle still pays for the stride check.

**Why it's slow (minor on its own, compounds with hotspot #2):**
- Stride-check overhead per call
- If `v`'s layout were ever changed to non-contiguous upstream, this
  call would silently allocate a full V copy (the bug-in-waiting)

**Optimization candidates:**
- **(A) Drop the `.contiguous()?`** at l.215 entirely. `v` from
  `reshape_v` is provably contiguous.
- **(B) Add `debug_assert!(v.is_contiguous())` if defensive checks are
  desired.**

### 6. **[MEDIUM] `apply_rope` on `q_rope` only — `k` rope missing from attention path** — `mla.rs:164`

**Pattern:** `q_rope_rotated` is computed and concatenated to `attn_output`
**after** the attention matmul, then passed through `o_proj`. The `k`
tensor used inside `attention_with_compressed_kv` (l.195-221) is the
**bare `k_decompressed` with no rope applied**.

**Why this is a correctness concern (out of scope for H-9 but worth flagging):**
- In real DeepSeek-V2 MLA, K also has a rope component (k_rope) that is
  concatenated to k_nope **before** the attention matmul
- This implementation appears to fuse the rope into the output projection
  rather than into the attention, which would produce numerically
  different attention weights
- `docs/perf/v27-correctness-investigation.md` lines 82-97 confirms
  no production model imports MLA, so this divergence is latent

**Optimization candidates (correctness first, then perf):**
- **(A) Fix correctness** by adding a `k_rope` projection and applying
  rope to K before `attention_with_compressed_kv` (matches DeepSeek-V2
  paper).
- **(B) Once correctness is fixed**, the existing rope perf hotspots in
  `crates/model/src/components/positional/rope.rs` apply — see
  follow-up profile work.

### 7. **[LOW] Two separate RoPE/reshape helper functions on the q_rope path** — `mla.rs:104-112, 134-141`

**Pattern:** `concat_q_nope_rope` reshapes (B,S,nope+rope) → (B,S,H,d),
then transposes+contiguous. `reshape_q_rope_for_rope` does a separate
reshape on the same data. These two reshape ops could be unified.

**Why it's slow:** Minor — each is a view op, not a copy. The
`.contiguous()` at l.111 is the only allocation.

**Optimization candidates:**
- **(A) Inline `reshape_q_rope_for_rope`** into `concat_q_nope_rope` —
  removes one reshape call. Trivial diff.

---

## No-issue areas

- **`q_proj` (l.160), `kv_proj` (l.168):** Direct `Linear::forward` —
  minimal overhead. Projections could be fused into one matmul (input →
  [q_lora_rank + kv_lora_rank]) but the savings are small since they're
  independent inputs to independent paths.
- **`apply_rope` (l.164):** External helper, perf-tested in H-2 baseline.
  Not a target here.
- **`o_proj` (l.178):** Single matmul, no obvious wins.
- **No `Vec::push` patterns** in MLA forward — confirmed by
  `rg "Vec::new|\.push\(" crates/model/src/components/attention/mla.rs`
  returning **0 matches** in production code (only test code). MLA
  avoids the GQA tiled_attention Vec::push issue entirely.

---

## Comparison with GQA (H-8)

**Recurring hotspots — same root cause, same fix:**

| Hotspot | GQA | MLA | Shared fix |
|---------|-----|-----|------------|
| `Tensor::new(&[scale], ...)` per-call broadcast | `gqa.rs:179-180` (HIGH) | `mla.rs:210` (HIGH) | Cache as struct field; or use `Tensor::affine` |
| Redundant `.contiguous()` after softmax | `gqa.rs:181` (LOW) | `mla.rs:213` (LOW) | Drop the `.contiguous()?` |
| Multiple `.contiguous()` after transpose+reshape | `gqa.rs:154, 157-158, 176, 183` (HIGH) | `mla.rs:111, 121, 131, 192, 205, 206, 215` (HIGH) | Reorganize layout to `(B, H, S, D)` from projection onward |
| `Tensor::cat` allocations | `flash_attention_v3.rs:268-270` (MED) | `mla.rs:105, 177` (MED) | Pre-allocate output buffer; narrow+write |
| `causal_mask` allocations | `util.rs:96-106` (MED) | n/a (no causal mask in MLA forward) | n/a |

**Differences from GQA:**

| Area | GQA | MLA |
|------|-----|-----|
| `expand_kv` repeat materialization | HIGH | **n/a** (MLA uses `num_kv_heads` directly, no repeat) |
| `tiled_attention` Vec::push + Tensor::cat | MED | **n/a** (MLA has no tiled path) |
| `apply_q_norm`/`apply_k_norm` redundant transpose | MED | **n/a** (MLA does not have QK-norm in this impl) |
| Two separate K/V decompress matmuls | n/a | MED (HIGH if GPU-bound) |

**Key takeaway:** MLA inherits the GQA `.contiguous()` and scale-broadcast
problems but **avoids** the `expand_kv` repeat and tiled Vec::push issues
because MLA uses a smaller latent KV representation throughout.

---

## Recommended H-12 optimization targets

| Rank | Target | File:line | Estimated speedup | Risk | Notes |
|------|--------|-----------|-------------------|------|-------|
| **1 (Primary)** | Cache `scale` tensor; use `affine` instead of `mul(broadcast(scale))` | mla.rs:209-211 | **3-8% on smoke; up to 10% on full qwen3-7B MLA** (broadcast is O(S²) of waste) | Low | Pure refactor; same fix as GQA #1 — could be a single shared helper. Correctness trivial to verify (existing tests cover numerical parity). |
| **2 (Secondary)** | Drop redundant `.contiguous()` at l.205, l.213, l.215; reorganize layout to `(B, H, S, D)` from projection | mla.rs:205, 213, 215, plus 111/121/131/192 | **5-10% on smoke; ~5% on full MLA** (memory bandwidth) | Medium | Requires verifying matmul accepts strided for the (B,S,H,D)→(B,H,S,D) path. Use existing tests to validate. |
| **3 (Stretch)** | Fuse `k_decompress` + `v_decompress` into one matmul; pre-allocate `q_concat` buffer | mla.rs:169-171, 177 | **2-5% on smoke; ~3% on full MLA** (launch overhead reduction; GPU-bound) | Medium | Single-line concat at the matmul output dimension; needs a `kv_decompress: Linear(kv_lora_rank, 2 * out_dim)` or equivalent. |

**Suggested order:** #1 first (lowest risk, easiest to validate, identical
to GQA #1 fix), then #2 (structural but bounded), then #3 (matmul fusion).
Each step should re-bench with `just bench-model-one mla_forward` and run
`just nextest` to confirm no regressions.

**Out of scope for H-12 (consider H-13+ or a separate task):**
- Correctness fix for missing K-rope (hotspot #6) — needs a design
  discussion; this is a deviation from DeepSeek-V2 spec
- Generic layout refactor for `(B, H, S, D)` across all attention
  modules — affects GQA, MLA, GQA-Flash, FlashV2; multi-module work

---

## Correctness note (carried forward from H-8 followup)

`MlaAttention::forward` does **NOT** apply causal masking
(`docs/perf/v27-correctness-investigation.md` lines 82-97). This is a
documented contract: the method is a low-level primitive; callers must
apply masking themselves. **No production model currently imports
`MlaAttention` or `Qwen3MlaAttention`**, so this is a latent foot-gun, not
a production bug.

The K-rope omission (hotspot #6 above) is a separate concern: even when
masking is added, the attention math will be incorrect until K also gets
a rope component.

---

## Limitations

- **Static analysis cannot measure actual CPU time per function.** All
  hotspot rankings are based on call-graph shape (allocation count,
  tensor materialization count, loop bodies), not measured wall-clock
  time per function. Real flamegraph data on GPU hardware is required to
  confirm.
- **CPU smoke numbers (~43 µs at seq_len=16, hidden=64) are dominated
  by overhead, not by the MLA math.** Per-tensor allocations and
  `candle_core::Tensor` dispatch dominate at this scale; on GPU with
  larger seq_len, the matmul/softmax cost would dominate and our
  allocation hotspots would be relatively smaller.
- **H-12 optimizers should re-bench after each change** with
  `just bench-model-one mla_forward` and confirm the existing MLA tests
  still pass (`test_mla_attention_*` in `mla.rs:275+`).
- **GPU profiling deferred.** Re-run this analysis with
  `cargo flamegraph --bench mla_forward` on a GPU runner with
  `perf_event_paranoid<=0` to get real self-time numbers.

---

## Files

- MLA source: `crates/model/src/components/attention/mla.rs`
- MLA bench: `crates/model/benches/mla_forward.rs`
- H-3 baseline: `docs/perf/v27-baseline.md` (mla_forward_smoke rows)
- H-8 (GQA) reference: `docs/perf/v27-profile-gqa.md`
- H-8 followup (causal contract): `docs/perf/v27-correctness-investigation.md`
- Plan: `docs/superpowers/plans/2026-06-28-v27-performance.md` (Task H-9)
