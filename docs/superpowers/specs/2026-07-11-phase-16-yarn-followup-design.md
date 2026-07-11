# Phase 16 — YaRN Follow-up Design

**Date:** 2026-07-11
**Status:** Approved
**Supersedes:** Phase 15 deferred items (CHANGELOG "Deferred (out of scope for Phase 15)")
**Author:** design session with user

---

## 1. Context & Motivation

Phase 15 (commit `4fba186`, CHANGELOG) implemented long-context support via
YaRN/Linear RoPE scaling in `crates/model/src/components/positional/rope.rs`.
The implementation was deliberately **minimal and additive**: it introduced
the `apply_with_scaling` / `forward_with_scaling` methods, the
`RopeScalingContext` bundle, and the `Default`/`Linear`/`Yarn` inverse-
frequency formulas, but it explicitly **did not**:

1. Migrate any production caller (`rope_gqa`, `mla`) to use the new APIs.
2. Implement `Dynamic` or `Su` algorithms (`Su`/`Dynamic` fall through to
   `Default` with a TODO comment).
3. Wire `attn_factor` into the attention kernel — it is stored on `RoPE`
   but never consumed (documented as future work).

This left the long-context feature in a half-shipped state: the config
parsing works, the algorithms exist in isolation, but configuring a
`Qwen3Config` with `rope_scaling.rope_type: "yarn"` does not change model
behaviour because no caller routes through `apply_with_scaling`. This
phase closes the loop.

**Scope:** vllm-core + vllm-model only. No public API breaking changes
(additive only). All work is in `crates/model/src/`.

---

## 2. Goals

| ID | Goal | Acceptance |
|----|------|------------|
| G1 | `rope_gqa::RopeGqaAttention` routes through `apply_with_scaling` for both `forward_prefill` and `forward_decode` | Existing rope_gqa tests pass unchanged + new regression test verifies Default is a no-op |
| G2 | `mla` attention routes through `apply_with_scaling` | Existing mla tests pass unchanged + new regression test |
| G3 | `gemma4` RoPE uses `apply_with_scaling` if its rope has scaling config | Existing gemma4 rope tests pass |
| G4 | `Dynamic` NTK implemented per HF Transformers formula | Unit tests: `Dynamic` at `cur <= orig_max` matches Default; `Dynamic` at `cur > orig_max` differs from Default; output is finite at large `cur` |
| G5 | `Su` RoPE implemented per Su et al. paper with `short_factor` / `long_factor` arrays | Unit tests: dim-by-dim scaling; `short_factor == long_factor == ones` matches Default; shape preservation; serde round-trip for new fields |
| G6 | `attn_factor` applied to attention scores in the standard `forward()` path | New test: `attn_factor=1.0` matches Default; `attn_factor=0.5` produces output scaled by sqrt(0.5) on attention scores (post-softmax-equivalent) |
| G7 | Public API additions only; no breaking changes to `RoPE::apply`, `apply_rope`, `RopeScaling` existing fields | All 392+ vllm-model tests pass; clippy + fmt clean |

**Out of scope:**
- Wiring `attn_factor` into `paged_attention_fn` / `tiled_attention_fn` /
  `flash_attention_fn` (those are optimization paths; only the standard
  `forward()` softmax gets the scaling).
- Migrating `rope_gqa` callers in `qwen3` / `llama` model configs to pass
  `rope_scaling` from config — this is a wiring concern that touches
  `Block::new` constructors and is best done as a separate phase.
- Implementing attention-temperature scaling for `Linear` /
  non-YaRN algorithms (only YaRN defines `attn_factor` semantically).

---

## 3. Design

### 3.1 Decision Log

Decisions confirmed with user 2026-07-11:

| ID | Decision | Rationale |
|----|----------|-----------|
| **A1** | `Dynamic` NTK formula = HF / Yarn style | Matches `transformers.modeling_qwen2.RotaryEmbedding._dynamic_frequency_update` and Qwen2.5-Qwen2.5-style open-source impls. Simple, one `max(1, …)` formula. |
| **B2** | `Su` RoPE = paper-original (`short_factor` + `long_factor` arrays) | One-time config schema change (`RopeScaling.short_factor` / `long_factor`) gives a faithful implementation; avoiding half-measures. |
| **C1** | `attn_factor` applied to attention **scores** before softmax | Matches YaRN paper §3.3 formula and HF/Qwen reference impls. |
| **D1** | All callers (`rope_gqa`, `mla`, `gemma4`) migrate to `apply_with_scaling` | Default behaviour is a no-op (already covered by `test_apply_with_scaling_default_matches_unscaled`); one consistent path. |

### 3.2 API Changes

#### 3.2.1 `RoPE` struct visibility

Change `apply_with_scaling` / `forward_with_scaling` from `pub(crate)` →
`pub` so production callers (`RopeGqaAttention`, `mla`) can use them.
Remove the `#[allow(dead_code)]` markers.

```rust
// crates/model/src/components/positional/rope.rs
impl RoPE {
    // Before:
    #[allow(dead_code)]
    pub(crate) fn apply_with_scaling(&self, x: &Tensor, positions: &[i64]) -> Result<Tensor> { … }

    #[allow(dead_code)]
    pub(crate) fn forward_with_scaling(&self, q: &Tensor, k: &Tensor, position: i64)
        -> Result<(Tensor, Tensor)> { … }

    // After:
    pub fn apply_with_scaling(&self, x: &Tensor, positions: &[i64]) -> Result<Tensor> { … }

    pub fn forward_with_scaling(&self, q: &Tensor, k: &Tensor, position: i64)
        -> Result<(Tensor, Tensor)> { … }
}
```

#### 3.2.2 `RopeScalingContext` gains `cur_seq_len` derivation

`Dynamic` NTK needs to know the current sequence length at forward time.
Two options:

1. Pass `cur_seq_len` explicitly through `apply_rope_with_scaling`.
2. Derive from `positions: &[i64]` via `positions.iter().max().map(|m| m + 1)`.

**Choice:** derive from positions. Pros: no API change; matches HF behaviour
(positions are typically contiguous 0..N, max+1 == seq_len). Cons: callers
must pass positions that cover the full sequence (already true in
`rope_gqa` / `mla` since they pass all position_ids).

Implementation:

```rust
pub fn apply_rope_with_scaling(
    query: &Tensor,
    positions: &[i64],
    theta: f32,
    scaling: RopeScalingContext,
) -> Result<Tensor> {
    let inv_freq = match scaling.rope_type {
        RopeType::Default => compute_inv_freq_default(query, theta),
        RopeType::Linear => compute_inv_freq_linear(query, theta, scaling.scaling_factor),
        RopeType::Yarn => compute_inv_freq_yarn(query, theta, scaling.scaling_factor),
        RopeType::Dynamic => {
            let cur_seq_len = derive_seq_len(positions);
            compute_inv_freq_dynamic(query, theta, scaling, cur_seq_len)
        }
        RopeType::Su => {
            let cur_seq_len = derive_seq_len(positions);
            compute_inv_freq_su(query, theta, scaling, cur_seq_len)
        }
        RopeType::Other => compute_inv_freq_default(query, theta),
    };
    apply_rope_with_inv_freq(query, positions, &inv_freq)
}

fn derive_seq_len(positions: &[i64]) -> usize {
    positions.iter().copied().max().map_or(0, |m| (m + 1) as usize)
}
```

The `RoPE::apply_with_scaling` struct method already calls into the same
free function with `scaling_ctx()`, so struct callers inherit Dynamic /
Su support for free.

#### 3.2.3 `RopeScaling` gains `short_factor` and `long_factor`

For `Su` (paper-original). Backward compatible (both default to `None`).

```rust
// crates/model/src/qwen3/config/rope.rs
pub struct RopeScaling {
    // ... existing fields ...
    /// Su RoPE "short factor" array (length head_dim/2). Per-dim scaling
    /// for short-context dimensions; defaults to all-ones.
    #[serde(default)]
    pub short_factor: Option<Vec<f32>>,
    /// Su RoPE "long factor" array (length head_dim/2). Per-dim scaling
    /// for long-context dimensions; defaults to all-ones.
    #[serde(default)]
    pub long_factor: Option<Vec<f32>>,
}
```

`RopeScalingContext::from(&RopeScaling)` extended to copy the new fields.

#### 3.2.4 `RoPE::new_with_config` accepts `original_max_position`

Already in scope (Phase 15 added it). No change.

#### 3.2.5 `compute_inv_freq_dynamic` (new)

```rust
/// HF Transformers / YaRN-style Dynamic NTK.
///
/// At each forward, recompute the scaling factor based on the current
/// sequence length:
///
/// ```text
/// scale = max(1, factor × (cur / orig_max) - (factor - 1))
/// ```
///
/// If `cur <= orig_max`, fall back to the default inv_freq table
/// (no scaling). Otherwise apply the YaRN NTK formula with the
/// dynamic `scale`.
fn compute_inv_freq_dynamic(
    query: &Tensor,
    theta: f32,
    scaling: RopeScalingContext,
    cur_seq_len: usize,
) -> Vec<f32> {
    let (_b, _s, _h, head_dim) = query.dims4().expect("dims4");
    let orig_max = scaling.original_max_position.unwrap_or(scaling.scaling_factor as usize);
    if cur_seq_len <= orig_max || scaling.scaling_factor == 1.0 {
        return compute_inv_freq_for_head_dim(head_dim, theta);
    }
    let factor = scaling.scaling_factor;
    let dynamic_scale = factor * (cur_seq_len as f32 / orig_max as f32) - (factor - 1.0);
    // dynamic_scale is guaranteed >= 1.0 by the cur > orig_max branch
    compute_inv_freq_yarn(query, theta, dynamic_scale)
}
```

Note: `compute_inv_freq_yarn` is reused, so Dynamic and Yarn share the
NTK correction formula — only the `scale` argument differs.

#### 3.2.6 `compute_inv_freq_su` (new)

Per the Su RoPE paper / HuggingFace reference impl
(`transformers.models.llama.modeling_llama.LlamaRotaryEmbedding`):

```rust
/// Su RoPE: per-dimension scaling using `short_factor` (high-frequency
/// dims) and `long_factor` (low-frequency dims).
///
/// Su RoPE re-parameterises `inv_freq` as:
///
/// ```text
/// inv_freq[i] = (short_factor[i] if i < boundary else long_factor[i])
///               / theta^(2i / d)
/// ```
///
/// where `boundary` is the smallest `i` whose wavelength exceeds
/// `original_max_position_embeddings`. This is the "Su RoPE" /
/// "RoPE in any precision" formulation (Su et al. 2024).
fn compute_inv_freq_su(
    query: &Tensor,
    _theta: f32,
    scaling: RopeScalingContext,
    _cur_seq_len: usize,
) -> Vec<f32> {
    let (_b, _s, _h, head_dim) = query.dims4().expect("dims4");
    let half_dim = head_dim / 2;
    let orig_max = scaling
        .original_max_position
        .expect("Su RoPE requires original_max_position_embeddings");

    let short_factor = scaling
        .short_factor
        .as_deref()
        .unwrap_or_else(|| vec![1.0; half_dim].as_slice());
    let long_factor = scaling
        .long_factor
        .as_deref()
        .unwrap_or_else(|| vec![1.0; half_dim].as_slice());
    assert_eq!(
        short_factor.len(),
        half_dim,
        "short_factor length must match head_dim/2"
    );
    assert_eq!(
        long_factor.len(),
        half_dim,
        "long_factor length must match head_dim/2"
    );

    // Compute the default inv_freq first.
    // Compute the default inv_freq first, then apply per-dim factors.
    // `theta` is passed as a free-function parameter (not in RopeScalingContext);
    // we use it directly here.
    let base_inv_freq = compute_inv_freq_for_head_dim(head_dim, theta);

    // Find boundary: smallest i such that base wavelength > orig_max.
    // wavelength[i] = 2π / base_inv_freq[i]. Su treats short_factor and
    // long_factor as 1.0 for the boundary calculation — only the per-dim
    // scaling *after* the boundary decision uses the actual factors.
    let wavelength_at = |i: usize| 2.0 * std::f32::consts::PI / base_inv_freq[i];
    let boundary = (0..half_dim)
        .find(|&i| wavelength_at(i) > orig_max as f32)
        .unwrap_or(half_dim);

    (0..half_dim)
        .map(|i| {
            let default_factor = 1.0;
            let factor = if i < boundary {
                short_factor.get(i).copied().unwrap_or(default_factor)
            } else {
                long_factor.get(i).copied().unwrap_or(default_factor)
            };
            base_inv_freq[i] / factor
        })
        .collect()
}
```
    let (_, _, _, head_dim) = query.dims4().expect("dims4");
    let half_dim = head_dim / 2;
    let orig_max = scaling
        .original_max_position
        .expect("Su RoPE requires original_max_position_embeddings");

    let short_factor = scaling
        .short_factor
        .as_deref()
        .unwrap_or_else(|| &[]);   // empty = no per-dim scaling for high-freq
    let long_factor = scaling
        .long_factor
        .as_deref()
        .unwrap_or_else(|| &[]);   // empty = no per-dim scaling for low-freq

    let base_inv_freq = compute_inv_freq_for_head_dim(head_dim, theta);
    let wavelength_at = |i: usize| 2.0 * std::f32::consts::PI / base_inv_freq[i];
    let boundary = (0..half_dim)
        .find(|&i| wavelength_at(i) > orig_max as f32)
        .unwrap_or(half_dim);

    (0..half_dim)
        .map(|i| {
            let default_factor = 1.0;
            let factor = if i < boundary {
                short_factor.get(i).copied().unwrap_or(default_factor)
            } else {
                long_factor.get(i).copied().unwrap_or(default_factor)
            };
            base_inv_freq[i] / factor
        })
        .collect()
}
```

#### 3.2.7 `RopeGqaAttention` field change

`theta: f32` → `rope: RoPE`. Update both `new` / `new_with_weights`
constructors and the `forward_prefill` / `forward_decode` call sites.

```rust
// crates/model/src/components/attention/rope_gqa.rs
pub struct RopeGqaAttention {
    inner: SharedGqaAttention,
    rope: RoPE,
}

impl RopeGqaAttention {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        theta: f32,
        vb: Option<candle_nn::VarBuilder<'_>>,
        config: AttentionConfig,
        has_qk_norm: bool,
    ) -> Result<Self> {
        let inner = SharedGqaAttention::new(...)?;
        let rope = RoPE::new(head_dim, /* max_position= */ 4096, theta, &inner.device());
        Ok(Self { inner, rope })
    }

    // forward_prefill: replace
    //     let q = apply_rope(&q, &position_ids, self.theta)?;
    //     let k = apply_rope(&k, &position_ids, self.theta)?;
    // with
    //     let q = self.rope.apply_with_scaling(&q, &position_ids)?;
    //     let k = self.rope.apply_with_scaling(&k, &position_ids)?;
    // (same for forward_decode)
}
```

`RoPE::new` takes a `Device`. The inner `GqaAttention` exposes
`device()` (already exists per `gqa/mod.rs`). Use that.

#### 3.2.8 `attn_factor` wiring into attention

Add `attn_factor: Option<f32>` to `GqaAttention` (the inner struct).
Modify `forward()` (the standard attention path in `gqa/forward.rs`):

```rust
// crates/model/src/components/attention/gqa/forward.rs
let scale = 1.0 / (self.head_dim as f32).sqrt();
// attn_factor: when set, additionally divide scores by attn_factor
// (YaRN §3.3 attention-temperature scaling). attn_factor = 1.0 is a no-op.
let attn_scale = self.attn_factor.unwrap_or(1.0) * scale;
let qk = qk.affine(f64::from(attn_scale), 0.0)?;
let attn_weights = candle_nn::ops::softmax(&qk, 3)?;
```

For `paged_attention_fn` / `tiled_attention_fn` / `flash_attention_fn`:
- These are optimization paths that delegate to lower-level kernels.
- For Phase 16, document that `attn_factor` is **only honoured by the
  standard `forward()` path**; production configs that select
  `paged`/`tiled`/`flash` attention get the same numerical behaviour as
  without `attn_factor` (i.e. attn_factor is silently ignored).
- Follow-up phase can wire it into the other paths.

`RopeGqaAttention::new` / `new_with_weights` plumb `attn_factor`
through to the inner `GqaAttention::new` / `new_with_weights`.

---

## 4. Implementation Order (Commits)

Each commit is independently buildable + testable. The user-facing diff
grows incrementally.

| # | Commit | Subject | LOC delta (approx) |
|---|--------|---------|--------------------|
| 1 | `refactor(rope): make apply_with_scaling and forward_with_scaling pub` | Remove `pub(crate)` + `#[allow(dead_code)]` from `apply_with_scaling` / `forward_with_scaling`. Add a doc test that `apply_with_scaling` from a non-crate caller compiles. | +5/-5 |
| 2 | `refactor(rope_gqa): route forward_prefill/decode through apply_with_scaling` | Replace `theta: f32` with `rope: RoPE`. Update both forward paths to call `self.rope.apply_with_scaling`. Update `RopeGqaAttention::new` / `new_with_weights`. All existing tests pass (Default is no-op). | +30/-15 |
| 3 | `refactor(mla): route forward through apply_with_scaling` | Mirror the rope_gqa migration for mla. Tests pass. | +20/-10 |
| 4 | `refactor(gemma4): route rope through apply_with_scaling` | gemma4 has its own `RoPE`-like struct; same pattern. | +15/-5 |
| 5 | `feat(rope): implement Dynamic NTK scaling` | Add `compute_inv_freq_dynamic` + tests. `Dynamic` at `cur <= orig_max` matches Default; `Dynamic` at `cur > orig_max` differs from Default. | +60 |
| 6 | `feat(config): add short_factor/long_factor to RopeScaling` | Extend `RopeScaling` with optional `short_factor` / `long_factor` arrays. Add serde tests. | +30 |
| 7 | `feat(rope): implement Su RoPE per-dim scaling` | Add `compute_inv_freq_su` + tests. Boundary computed from base wavelength vs `original_max_position_embeddings`. | +80 |
| 8 | `feat(gqa): apply attn_factor to attention scores in standard forward` | Plumb `attn_factor: Option<f32>` through `GqaAttention` and `RopeGqaAttention`. Modify standard `forward()` softmax to apply `attn_factor` multiplicatively. Document that paged/tiled/flash paths ignore it. | +40/-10 |
| 9 | `docs(changelog): record Phase 16 YaRN follow-up` | Add CHANGELOG entry. | +30 |

**Total:** ~9 commits, ~+310 / -55 LOC.

---

## 5. Testing Strategy

### 5.1 Unit tests (rope.rs / tests.rs)

Add to existing `crates/model/src/components/positional/rope/tests.rs`:

```rust
#[test]
fn test_dynamic_scaling_matches_default_below_orig_max() { … }

#[test]
fn test_dynamic_scaling_differs_above_orig_max() { … }

#[test]
fn test_dynamic_scaling_at_orig_max_boundary() { … }

#[test]
fn test_su_scaling_with_identity_factors_matches_default() { … }

#[test]
fn test_su_scaling_with_short_factor_modifies_high_freq_dims() { … }

#[test]
fn test_su_scaling_with_long_factor_modifies_low_freq_dims() { … }

#[test]
fn test_su_scaling_factor_array_length_mismatch_panics() { … }

#[test]
fn test_derive_seq_len_handles_empty_positions() { … }

#[test]
fn test_attn_factor_one_is_noop_in_apply() { … }  // for completeness; attn is tested separately
```

### 5.2 Integration tests (rope_gqa/tests.rs, mla/tests.rs)

Add to `crates/model/src/components/attention/rope_gqa/tests.rs`:

```rust
#[test]
fn rope_gqa_default_scaling_matches_unscaled() {
    // Construct two RopeGqaAttention instances with identical theta,
    // one with default RoPE and one explicitly constructed with
    // RopeType::Default + scaling_factor=1.0. Forward the same input
    // through both, assert outputs match to 1e-5.
}

#[test]
fn rope_gqa_attn_factor_affects_attention_output() {
    // Construct RopeGqaAttention with attn_factor=0.5. Run forward on
    // a fixed input. Compare against the same instance with
    // attn_factor=1.0. Outputs MUST differ.
}
```

### 5.3 Config tests

Add to `crates/model/src/qwen3/config/rope.rs`:

```rust
#[test]
fn rope_scaling_short_factor_round_trips() { … }

#[test]
fn rope_scaling_long_factor_round_trips() { … }
```

### 5.4 Regression coverage

- All 392+ existing vllm-model tests must pass unchanged.
- Default `apply_with_scaling` test already exists and verifies no-op
  (`test_apply_with_scaling_default_matches_unscaled`) — this is the
  canary for the migration.

### 5.5 Pre-commit verification

Per `CLAUDE.md` workflow:

```bash
cargo fmt --all
cargo clippy --workspace --all-features -- -D warnings
cargo test --all-features --workspace
cargo doc --workspace --no-deps
```

---

## 6. Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Migrating `RopeGqaAttention` from `theta: f32` to `rope: RoPE` breaks callers that construct it positionally | M | M | All callers go through `RopeGqaAttention::new(...)` which is keyword-style already. Search + verify before committing. |
| `Dynamic` formula mismatch vs HF reference impl | L | H | Compare against HF Transformers Qwen2 implementation line-by-line during dev. Add test that compares our output to a known HF fixture. |
| `Su` boundary calculation off-by-one | M | L | Multiple tests covering boundary positions; visual inspection of base wavelength curve. |
| `attn_factor` ignored by paged/tiled/flash paths silently | L | M | Document in CHANGELOG + module doc; add debug assertion that warns when `attn_factor.is_some()` and the chosen attention path isn't `Standard`. |
| New `RopeScaling` fields break serde deserialisation of existing configs | L | M | Both fields default to `None`; existing configs (no `short_factor`/`long_factor`) continue to deserialize. Test: deserialize HF-style config that lacks these fields. |
| `positions.iter().max()` empty case | L | L | `derive_seq_len` returns 0; `compute_inv_freq_dynamic` / `_su` fall through to Default when `cur_seq_len == 0`. Defensive, no panic. |

---

## 7. Follow-up Phases (Out of Scope)

- Wire `attn_factor` into `paged_attention_fn`, `tiled_attention_fn`,
  `flash_attention_fn` (currently silently ignored).
- Migrate `Block::new` constructors in `qwen3`, `llama` to thread
  `rope_scaling` from `Qwen3Config` / `LlamaConfig` into `RoPE::new_with_config`.
- Real-model integration test: load a Qwen2.5-7B-Instruct with
  `rope_scaling.rope_type: "yarn"` and verify per-token logit
  equivalence to a reference run.

---

## 8. Open Questions

None. All four key design decisions (A1, B2, C1, D1) were resolved in
the brainstorm session.

---

*Spec author: design session with user 2026-07-11.*
*Next step: writing-plans skill to produce the implementation plan.*
