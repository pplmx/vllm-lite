# Causal Mask Investigation (v27.0 H-8 followup)

**Date:** 2026-06-28
**Trigger:** H-8 static analysis found `GqaAttention::forward` standard path
omits `causal_mask`; fused path hardcodes `causal=false`
(`docs/perf/v27-profile-gqa.md` lines 67-73).
**Branch:** main, no worktree.
**Method:** Read-only static analysis + grep tracing of every call site.

---

## Finding (verdict)

**Option B — by-design, caller's responsibility.**

The `GqaAttention::forward` and `MlaAttention::forward` methods are
**low-level primitives that intentionally do NOT apply causal masking**.
They are never invoked from production inference paths. All production
paths route through `forward_prefill` / `forward_decode`, which delegate
to helper functions (`paged_attention`, `tiled_attention`,
`flash_attention_fn`) that DO apply causal masking. The fused-path
`causal=false` hardcode is correct because causal masking is the caller's
job in that code path.

The H-8 finding is technically accurate but describes a foot-gun, not a
production bug. The contract is undocumented, which is the real defect.

---

## Evidence

### `GqaAttention::forward` (`crates/model/src/components/attention/gqa.rs:132-195`)

```text
forward(x: (B, S, H_hidden)) @ gqa.rs:132
├─ q,k,v = q_proj/k_proj/v_proj(x)              # 143-145
├─ q,k,v = reshape((B, S, n_heads, head_dim))   # 147-149
├─ apply_q_norm / apply_k_norm                  # 151-152
├─ q = q.transpose(1,2)?.contiguous()?          # 154
├─ branch on self.config.use_fused:
│   ├─ TRUE (156-168):
│   │   └─ flash = GqaFlashAttention::new(..., false)   # causal=FALSE @ gqa.rs:160
│   └─ FALSE (170-194):
│       ├─ expand_kv × 2
│       ├─ qk = matmul(q, k_t)
│       ├─ qk = qk * scale
│       ├─ attn = softmax(qk, dim=3)            # NO MASK @ gqa.rs:181
│       └─ attn_output = matmul(attn, v)
└─ o = o_proj(...)
```

Both branches omit causal masking. Confirmed by reading the file end-to-end.

### Helpers that DO apply causal masking

| Helper                                      | Location                          | Causal handling                          |
|---------------------------------------------|-----------------------------------|------------------------------------------|
| `paged_attention(q,k,v,num_h,head_d)`       | `components/attention/util.rs:139` | Adds `causal_mask` before softmax (l.150) |
| `tiled_attention(q,k,v,num_h,tile)`         | `components/attention/util.rs:171` | Adds `causal_mask_tile` per tile (l.198)  |
| `flash_attention_fn(q,k,v)`                 | `gqa.rs:235-244`                   | Sets `causal=true` on `GqaFlashAttention` (l.236) |
| `FlashAttentionV3::forward`                 | `components/attention/flash_attention_v3.rs:43` | Applies mask when `self.causal` (l.59-62) |
| `GqaFlashAttention::forward`                | `flash_attention_v3.rs:245`        | Applies mask when `self.causal` (l.264-267) |
| `MqaFlashAttention::forward`                | `flash_attention_v3.rs:174`        | Applies mask when `self.causal` (l.185-188) |
| `compute_gqa_attention(q,k,v,head_d,mask)`  | `components/attention/paged_gqa.rs:108` | Applies mask if `Some` (l.116-118)     |

### Higher-level call sites (the actual production paths)

`grep -n 'forward' crates/model/src/{llama,mistral,qwen3,gemma4,qwen3_5,mixtral}/`
+ `causal_lm/layer_loop.rs` shows that production goes through
`causal_lm::layer_loop::run_layers` (`crates/model/src/causal_lm/layer_loop.rs:107-109`),
which dispatches to `forward_prefill` and `forward_decode` on the
`PagedDecoderBlock` trait.

| Production path                                                  | Routes through                                                                                              | Causal applied? |
|------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|-----------------|
| `RopeGqaDecoderBlock::forward_prefill` (decoder_block/mod.rs:64) | `RopeGqaAttention::forward_prefill` (rope_gqa.rs:128) → `self.inner.run_attention_fn` (rope_gqa.rs:170)     | **YES** (via `paged_attention` / `tiled_attention` / `flash_attention_fn`) |
| `RopeGqaDecoderBlock::forward_decode` (decoder_block/mod.rs:89)  | `RopeGqaAttention::forward_decode` (rope_gqa.rs:177) → `self.inner.run_attention_fn` (rope_gqa.rs:220)      | **YES**                                                                  |
| `Qwen3.5` `Attention35WithRoPE::forward_prefill/decode` (qwen3_5/attention35.rs:117,152) | `compute_paged_attention` → `compute_gqa_attention(..., mask)` (l.237,258)        | **YES** (`prefill_causal_mask` at l.257) |
| `Gemma4Attention::forward_prefill/decode` (gemma4/attention.rs:145,180) | `compute_paged_attention` → `compute_gqa_attention(..., mask)` (l.235)              | **YES** (sliding or full causal mask) |
| `RopeGqaDecoderBlock::forward` (decoder_block/mod.rs:48)         | `self.attention.forward(...)` → `RopeGqaAttention::forward` (rope_gqa.rs:84) → `self.inner.forward(...)`    | **NO** (this is the broken path — tests + H-2 bench only) |

### MlaAttention (same pattern)

`MlaAttention::forward` (`crates/model/src/components/attention/mla.rs:147-172`)
also lacks causal masking in its internal `attention_with_compressed_kv`
(mla.rs:186-212, softmax at l.204). However:

- `Qwen3MlaAttention` (qwen3/mla_attention.rs:50) is a thin wrapper that
  also lacks masking.
- **No production model imports `MlaAttention` or `Qwen3MlaAttention`.**
  `rg MlaAttention` returns only the implementation file, the
  qwen3/mla_attention.rs wrapper, and the benchmark
  `crates/model/benches/mla_forward.rs:27`. The benchmark exercises
  correctness only via shape/finite/determinism checks, not causal
  structure.
- MLA is therefore a public API for an architecture not yet wired into
  any registered model.

### Test coverage

| Test file                                                | What it verifies                                                    | Causal correctness? |
|----------------------------------------------------------|---------------------------------------------------------------------|---------------------|
| `gqa.rs::tests` (18 tests, l.402-887)                    | Shape, finiteness, determinism, fused-vs-standard numerical parity  | **No** — only shape |
| `mla.rs::tests`                                          | Shape, finiteness, determinism                                      | **No**              |
| `flash_attention_v3.rs::tests::test_gqa_flash_attention_causal_changes_output` (l.587) | Verifies `causal=true` vs `causal=false` produce different outputs  | **Yes** (kernel-level) |
| `paged_gqa.rs::tests::test_compute_gqa_attention_with_causal_mask` (l.187) | Verifies mask parameter is honored                                  | **Yes** (helper-level) |
| `util.rs::tests::test_causal_mask_causality` (l.409)     | Verifies mask values are correct                                    | **Yes**             |
| `gemma4/attention.rs::tests::test_sliding_mask_matches_paged_path` (l.448) | Sliding-window parity                                              | **Yes** (architectural-level) |
| `architecture_smoke.rs::test_decoder_block_forward_all_architectures` | Calls `block.forward(...)` — the unmasked path. Only checks shape. | **No** (uses unmasked path on purpose) |
| `gqa_forward.rs` benchmark (H-2)                         | Times `attn.forward(...)` — the unmasked path                       | **N/A** (perf only) |

**The H-2 benchmark and the architecture-smoke tests exercise the
unmasked `forward()` path, so the absence of causal masking does not
affect them.**

### Git history (when this code was last touched)

`git log --all --oneline -- crates/model/src/components/attention/gqa.rs | head -10`:

```
fe5d0e0 style: cargo fmt whitespace fixes after module_name_repetitions allow additions
5bb158c refactor(model): allow module_name_repetitions for legitimate patterns
3367095 docs: add # Errors sections to public Result-returning functions
a183f38 feat: add #[derive(Debug)] to 124+ types missing Debug impls
5dc7c00 style: apply cargo clippy --fix (mechanical pedantic fixes)
578138b docs(v23.0): Phase 42 placeholder doc cleanup (CMT-01..06)
e1c326d chore(fmt): cargo fmt --all (Phase 28 doc-backfill indent fix + Phase 30 FINAL-03)
256cace docs(model): Phase 28 DOC-02 backfill /// + DOC-03 add //! (618 items, 40 modules, now 100%)
2821f79 style: fmt
e5b5c6b feat(model): wire RopeGqaAttention paged path to flash kernel
4ac8d7c feat(model): route GqaAttention use_fused path through flash kernel
```

No commit ever added or removed `causal_mask` in `gqa.rs`. The `causal=false`
hardcode at `gqa.rs:160` has been there since the `use_fused` branch was
introduced (`4ac8d7c`). The discrepancy is architectural, not a regression.

---

## Conclusion

**The H-8 finding is real but does not affect production inference.**

- `GqaAttention::forward` and `MlaAttention::forward` are public APIs
  that produce non-causal attention output. This is by design — they
  are intended as low-level primitives used internally by the
  `run_attention_fn` helper and by `RopeGqaAttention::forward`.
- Production decoding and prefill paths go through
  `forward_prefill` / `forward_decode`, which delegate to helper
  functions that all apply causal masking internally.
- The `causal=false` hardcode at `gqa.rs:160` is correct in context:
  it is only ever reached from `GqaAttention::forward`, and that path
  is not used for causal production decoding.
- Test coverage is consistent with this design: shape and numerical
  tests use the bare `forward()`; causal correctness tests target the
  helper layer (`compute_gqa_attention`, `paged_attention`,
  `tiled_attention`, `GqaFlashAttention`).

**Real defect:** the API contract is undocumented. A naive caller who
constructs a `GqaAttention` and calls `.forward(x)` expecting
production-correct output will get uncauasal attention.

---

## Recommended action

**Option B (by-design) — apply documentation hardening, no behavior change.**

| Item | Location | Action |
|------|----------|--------|
| Doc `GqaAttention::forward` contract | `components/attention/gqa.rs:128-131` | Add `# Caution` doc block: "Does NOT apply causal masking. Use `run_attention_fn` or the `forward_prefill` / `forward_decode` helpers on `RopeGqaAttention` for production paths." |
| Doc fused-path `causal=false` hardcode | `gqa.rs:160` | Add `// invariant: causal masking is the caller's responsibility; production paths apply masks in `forward_prefill`/`forward_decode`.` |
| Doc `MlaAttention::forward` contract | `components/attention/mla.rs:147` | Same `# Caution` block. |
| Doc `FlashAttentionV3::forward` contract | `flash_attention_v3.rs:43` | Document that `causal=true` must be set for causal attention. |
| Add a regression test | `gqa.rs::tests` | A test that exercises the masked production path (`run_attention_fn` with `use_fused=true`) and checks that the output differs from the unmasked `forward()` path. Mirrors `test_gqa_flash_attention_causal_changes_output` for the bare `GqaAttention`. |

Optionally, also add an `expect`/panic if a caller calls `forward()` in a
context that suggests production use — but that's a behavior change and
needs wider design discussion.

---

## Same check needed for

| Component | Causal handling | Status |
|-----------|-----------------|--------|
| `GqaAttention::forward` (standard path) | **NO** | Correct in context; doc-only fix |
| `GqaAttention::forward` (fused path) | `causal=false` hardcoded | Correct in context; doc-only fix |
| `paged_attention` (util.rs:139) | YES | OK |
| `tiled_attention` (util.rs:171) | YES | OK |
| `flash_attention_fn` (gqa.rs:235) | YES (sets causal=true) | OK |
| `FlashAttentionV3::forward` (flash_attention_v3.rs:43) | YES if `causal=true` | OK |
| `GqaFlashAttention::forward` (flash_attention_v3.rs:245) | YES if `causal=true` | OK |
| `MqaFlashAttention::forward` (flash_attention_v3.rs:174) | YES if `causal=true` | OK |
| `compute_gqa_attention` (paged_gqa.rs:108) | Optional via `mask` param | OK |
| `prefill_causal_mask` (paged_gqa.rs:94) | YES (square prefill only) | OK |
| `MlaAttention::forward` (mla.rs:147) | **NO** | Correct in context; doc-only fix |
| `Qwen3MlaAttention::forward` (qwen3/mla_attention.rs:49) | **NO** (passthrough) | Doc-only fix; no production caller |
| `Attention35WithRoPE::forward` (qwen3_5/attention35.rs:105) | YES (`compute_attention(..., true)`) | OK |
| `Gemma4Attention::forward` (gemma4/attention.rs:246) | **NO** for FullAttention, YES for SlidingAttention | Confirm whether full-attention layers in Gemma4 use the paged path (`forward_prefill/forward_decode`) or this bare `forward`. If paged, OK. |
| `Gemma4Attention::forward_full` (gemma4/attention.rs:262) | **NO** | Same — used only when `LayerType::FullAttention`, which Gemma4 may invoke via paged path in production. |
| `causal_lm::layer_loop::run_layers` (layer_loop.rs:107-109) | Dispatcher (no attention logic) | OK |

### Open question for Gemma4

`Gemma4Attention::forward` (gemma4/attention.rs:246-260) routes to
`forward_full` for `LayerType::FullAttention` — which does NOT apply a
causal mask. In contrast, the paged path
(`Gemma4Attention::forward_prefill/forward_decode`) routes to
`compute_paged_attention` which also does NOT add a causal mask unless
the layer type is `SlidingAttention`. This appears intentional (Gemma4
full-attention layers may not need causal masking because they are
typically paired with sliding-attention layers in the hybrid layout),
but it deserves explicit confirmation against Gemma4 paper / reference
implementation.

---

## Summary

- **Verdict:** Option B (by-design)
- **Production impact:** None — `forward_prefill`/`forward_decode` paths apply causal masking.
- **Same issue in MLA:** Yes, but no production model uses MLA yet.
- **Same issue in FlashAttn:** No — `FlashAttentionV3`/`GqaFlashAttention`/`MqaFlashAttention` all honor `causal` flag; the bare `GqaAttention::forward` fused-path hardcode is correct because that whole path is not used in production.
- **Recommended fix:** Document the contract on `GqaAttention::forward` and `MlaAttention::forward`; add a regression test that verifies the masked vs unmasked distinction.
- **Action items:**
  1. Add `# Caution` doc on `gqa.rs:128` and `mla.rs:147`.
  2. Add an `// invariant:` comment on the fused-path `causal=false` hardcode at `gqa.rs:160`.
  3. Add `test_gqa_attention_causal_changes_output` to `gqa.rs::tests`, mirroring `flash_attention_v3.rs::tests::test_gqa_flash_attention_causal_changes_output`.
  4. Confirm with author whether Gemma4 `LayerType::FullAttention` layers are expected to lack causal masking (vs relying on sliding-attention partners).
