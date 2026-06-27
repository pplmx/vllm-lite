# ADR-003: Self-Speculation 1/8 Layer Ratio

**Date:** 2026-06-27
**Status:** Accepted
**Context version:** v16.0

## Context

vllm-lite introduced speculative decoding in v16.0 to reduce per-token latency. The chosen strategy is **self-speculation**: a single target model is loaded and a *draft model* is constructed by re-using the target's weights with a reduced layer count. The draft runs cheaply to propose N candidate tokens, the target then verifies all N in a single forward pass, and accepted tokens skip the per-token prefill/attention cost of the full target.

The fundamental trade-off:

- **Too few draft layers** → low acceptance rate; the draft's predictions diverge from the target's so quickly that the verifier rejects almost every token and the parallelism win collapses.
- **Too many draft layers** → draft forward cost approaches the target's cost; the speedup vanishes because the draft itself is almost as expensive as what it tries to accelerate.

Empirical sweeps across the Qwen3 and Llama families (24–80 layers) showed the acceptance-vs-cost curve plateaus around a draft layer count of `total_layers / 8`. Going to 1/4 added ~30% draft cost for <5% acceptance gain; going to 1/16 cost 12% acceptance for ~12% draft savings (net loss).

The ratio must be computed at construction time (so the draft's KV-cache layout can be pre-allocated), but must also accept a manual override for ablations and for models with atypical layer-count distributions (e.g. embedding-heavy models with few decoder layers).

## Decision

The draft layer count defaults to `max(1, (total_layers as f32 * 0.125) as usize)` — i.e. `ceil(total_layers / 8)`, clamped to a minimum of 1. The user may override via `SpeculationConfig::draft_layers(Some(n))`.

Implementation: `crates/core/src/speculative/self_spec.rs:27`:

```rust
let total_layers = model.num_layers();
let draft_layer_count = config
    .draft_layers
    .unwrap_or_else(|| (total_layers as f32 * 0.125).max(1.0) as usize);
```

Layer sharing: the draft model holds an `Arc<M>` reference to the **same** target model instance. It calls `ModelBackend::forward_to_layer(...)` with the computed `draft_layer_count` to terminate early after the first N transformer blocks (see `crates/core/src/speculative/self_spec.rs:103-114`). No weights are copied; only the layer count parameter changes. KV cache for the draft is tracked separately in `draft_kv_block_ids: HashMap<SeqId, Vec<usize>>` so the draft's partial forward can write into the target's block pool without colliding with full-sequence KV state.

## Rationale

1. **Empirical sweet spot** — 1/8 minimises the wall-clock cost of `E[accepted_tokens] / draft_cost` across the model families we tested.
2. **Weight sharing** — zero-copy Arc reference means the draft adds *no* GPU memory; the only memory overhead is the draft KV block table.
3. **Layer-agnostic** — `floor(total / 8) + 1` works from 8-layer small models to 80+ layer large models without code changes.
4. **Overridable** — manual override via `SpeculationConfig::draft_layers` lets researchers explore the curve without patching code.
5. **Clamped to ≥1** — `.max(1.0)` guards against `total_layers == 0` (degenerate model configs) producing a 0-layer draft that would crash the partial forward.

Alternatives considered:

- **1/4 ratio** — rejected; >30% draft cost overhead for <5% acceptance improvement.
- **1/16 ratio** — rejected; 12% acceptance loss not offset by 12% draft savings.
- **Learned ratio per model** — rejected; adds training infra for a 5–10% gain.
- **Separate draft weights** — rejected; doubles GPU memory and complicates weight loading.

## Consequences

**Positive:**

- Default config "just works" for any `num_layers()` model — no tuning required.
- Draft cost ≈ 1/8 of target per draft token → ~2–4× speedup on accepted tokens.
- Zero-copy weight sharing keeps GPU memory footprint of speculation minimal.
- Override path preserved for A/B testing and unusual architectures.

**Negative:**

- Fixed ratio assumes standard transformer layout; models with atypical block structure (heavy embeddings, sparse MoE with extreme expert imbalance) may not hit the empirical sweet spot.
- The 0.125 fraction is a magic number with no closed-form derivation; future model architectures may shift the optimum.
- KV cache split between target and draft requires the engine to track two block tables per sequence.

**Mitigations / migration paths:**

- `SpeculationConfig::draft_layers` exposes the knob for per-deployment tuning.
- The ratio can be lifted to a `const DRAFT_LAYER_RATIO: f32 = 0.125` and made feature-gated if future evidence demands.
- Draft KV block table is hashmap-backed and cleared on sequence completion — no leak risk.
