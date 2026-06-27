# ADR-015: vllm-dist Investment vs Deprecation Decision

**Date:** 2026-06-27
**Status:** Accepted (keep, with feature-gate)
**Context version:** v21.4 / Phase 34 / DOC-02

## Context

`vllm-dist` is a ~1,600 LOC crate implementing multi-node distributed inference
primitives (tensor-parallel, pipeline-parallel, gRPC coordination, distributed
KV cache). It is feature-gated behind `--features multi-node` per ADR-008
(v20.1 outcome) but currently has no production consumers.

During the v19.0 audit, three strategic options were raised:

1. **Deprecate entirely** — delete the crate; remove dead code; restart when
   multi-node work actually begins (estimated 6-12 months away).
2. **Keep as feature-gated** — preserve the existing implementation; let
   consumers opt-in via `--features multi-node` when ready.
3. **Invest heavily** — wire it into the serving stack now; justify the
   surface area with production usage.

The audit recommended **option 2** (keep, feature-gated) as the lowest-cost
path that preserves optionality.

## Decision

**Keep `vllm-dist` feature-gated under `--features multi-node`. Do not
deprecate. Do not invest in production wiring until external demand materializes.**

Rationale:

1. **Sunk cost is sunk.** The 1,600 LOC represents tribal knowledge about
   tensor-parallel and pipeline-parallel design patterns. Deleting and
   re-implementing later would cost more than keeping (even unused).
2. **Compilation cost is minimal.** With feature-gating, default builds
   exclude `vllm-dist`. Only `--features multi-node` builds incur the
   cost. This was the primary motivator for ADR-008.
3. **Surface area is bounded.** The crate exports only ~40 public symbols
   (mostly trait definitions + thin wrappers). Even if unused, the
   maintenance burden is low — ~5 files, well-documented.
4. **External demand is possible.** Enterprise customers evaluating
   `vllm-lite` for high-availability multi-node deployments may surface
   in the next 6-12 months. Feature-gating preserves the option to
   reactivate without a major refactor.
5. **Wire-up risk is high.** Wiring `vllm-dist` into the serving stack
   now would require:
   - Server-side multi-node routing logic
   - Distributed KV cache consistency protocols
   - Cross-node health checks and leader election (already partially
     present in `vllm-core::ha`)
   - Operational tooling (deployment, monitoring)
   This is multi-month work and would compete with single-node optimization
   priorities.

## Alternatives Considered

### Deprecate entirely

- **Pro:** Removes ~1,600 LOC of "dead" code; reduces confusion
- **Con:** Loses tribal knowledge; future multi-node work starts from zero
- **Verdict:** Rejected. Sunk cost is low relative to future re-implementation cost.

### Invest heavily now

- **Pro:** Multi-node ready sooner; competitive advantage
- **Con:** Competes with single-node optimization; multi-month effort
  without validated demand
- **Verdict:** Deferred. Re-evaluate when external demand materializes.

### Keep as feature-gated (chosen)

- **Pro:** Lowest cost; preserves optionality; no behavioral change
- **Con:** Maintenance burden even if unused (mitigated by ADR-008)
- **Verdict:** Accepted.

## Consequences

### Positive

- `vllm-dist` remains available for future multi-node work without a
  re-implementation cost.
- Default builds are unaffected (feature-gated).
- Documentation, ADRs, and the codebase treat `vllm-dist` as a first-class
  but optional crate.

### Negative

- The crate remains unmaintained until external demand materializes.
- Tests for `vllm-dist` are limited (no integration tests against real
  multi-node setup).
- Some abstractions (e.g., `TensorParallelError`) live in `vllm-traits`
  rather than `vllm-dist` due to dependency direction (see ARCH-F-13
  resolution in Phase 31).

## Re-evaluation Triggers

Re-evaluate this decision when:

- External customer requests multi-node deployment (re-evaluate option 3).
- `vllm-dist` causes a compile-time or maintenance burden that exceeds
  re-implementation cost (re-evaluate option 1).
- A community contributor offers to maintain the crate (re-evaluate option 3).
- More than 12 months pass without any consumer or roadmap item touching
  `vllm-dist` (re-evaluate option 1).

## Related

- ADR-008: Why vllm-dist is Feature-Gated (technical decision)
- ARCH-F-17: Most of `vllm-dist` is publicly exported but never used (P1)
- ARCH-F-13: `TensorParallelError` location smell (P2, partially resolved in Phase 31)
- DOCS-F-22: vllm-dist underuse decision (this ADR)
