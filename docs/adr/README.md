# Architecture Decision Records (ADRs)

This directory contains ADRs documenting significant architectural
decisions in vllm-lite.

## Index

### Foundational

- [ADR-001](ADR-001-component-sharing-strategy.md) — Component sharing strategy
- [ADR-002](ADR-002-feature-flag-design.md) — Feature flag design
- [ADR-005](ADR-005-kv-cache-split.md) — KV cache split
- [ADR-013](ADR-013-paged-kv-cache.md) — Paged KV cache
- [ADR-014](ADR-014-architecture-registry.md) — Architecture registry

### Compute & Models

- [ADR-003](ADR-003-self-speculation-1-8-layer-ratio.md) — Self-speculation 1:8 layer ratio
- [ADR-004](ADR-004-fp8-e4m3-kv-cache.md) — FP8 E4M3 KV cache
- [ADR-009](ADR-009-fp8-quantizer-orphan-decision.md) — FP8 quantizer orphan decision
- [ADR-010](ADR-010-cuda-graph-feature-gating.md) — CUDA Graph feature gating

### Speculative Decoding

- [ADR-006](ADR-006-speculative-decoding-architecture.md) — Speculative decoding architecture
- [ADR-007](ADR-007-per-request-draft-routing.md) — Per-request draft routing

### Distribution

- [ADR-008](ADR-008-vllm-dist-feature-gated.md) — vllm-dist feature-gated
- [ADR-015](ADR-015-vllm-dist-investment-decision.md) — vllm-dist investment vs deprecation

### Scheduling

- [ADR-012](ADR-012-continuous-batching.md) — Continuous batching

### Cross-cutting

- [ADR-011](ADR-011-cross-crate-error-boundaries.md) — Cross-crate error boundaries

### Test Strategy (v30.0+)

- [ADR-016](ADR-016-proptest-strategy.md) — Property-based testing (proptest)
- [ADR-017](ADR-017-fuzz-strategy.md) — Fuzz testing (cargo-fuzz)
- [ADR-018](ADR-018-mutation-testing.md) — Mutation testing (cargo-mutants)

### Documentation (v31.0+)

- [ADR-019](ADR-019-documentation-standards.md) — Documentation and comment standards

## Conventions

Each ADR follows the format:

```
# ADR-NNN: Title

**Date:** YYYY-MM-DD
**Status:** Accepted | Superseded | Deprecated
**Context version:** vXX.Y

## Context
## Decision
## Consequences
## Alternatives considered
## See also
```

ADRs are **immutable once accepted**. Superseding decisions write a new
ADR that references the old one.

## Adding a new ADR

1. Pick the next available number (`ls docs/adr/ | grep -E '^ADR-' | sort`)
2. Copy an existing ADR as a template
3. Fill in all sections
4. Add an entry to this index under the appropriate group
5. Commit: `docs(adr): ADR-NNN <topic>`

## See also

- `docs/architecture.md` — high-level architecture overview
- `docs/superpowers/specs/` — design specs for major milestones
