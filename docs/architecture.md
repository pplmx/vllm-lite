# vLLM-lite Architecture (v23.0)

This document describes the v23.0 architecture of vLLM-lite: how the six crates
interact, how the engine orchestrates scheduling and model execution, and where
to extend the system when adding new model architectures or scheduling
strategies.

For design decisions and trade-offs, see the [ADRs](./adr/).

## Overview

vLLM-lite is a Rust workspace implementing key vLLM innovations: continuous
batching, paged KV cache, prefix caching, speculative decoding, and an
OpenAI-compatible HTTP API. The codebase is organized into six crates with
strict layering: `vllm-traits` (interface types) is depended on by everyone;
`vllm-core` (engine/scheduler) depends on `vllm-traits`; `vllm-model`
(implementations) depends on `vllm-traits`; `vllm-server` (HTTP) depends on
`vllm-core` and `vllm-model`; `vllm-dist` (distributed primitives) is
feature-gated behind `--features multi-node`; `vllm-testing` (test harness)
provides reusable fixtures.

```text
              vllm-server ─────┐
                 │             │
                 ▼             ▼
   vllm-core ◀────────────► vllm-model
        │                       │
        └──────► vllm-traits ◀──┘
                    ▲
                    │
                vllm-testing (test harness)

   vllm-dist (feature-gated: --features multi-node)
        └──────► vllm-traits
```

See [ADR-008](./adr/ADR-008-vllm-dist-feature-gated.md) for the rationale on
feature-gating `vllm-dist`.

## Engine Orchestration

The `Engine` lives in `crates/core/src/engine.rs` and orchestrates the
inference loop. Since v22.0, the engine is decomposed into focused
sub-modules:

- `engine.rs` — top-level `Engine` struct + `step()` entry point
- `engine/spec_dispatch/` — speculative decoding routing (v18.0+; unified
  tree post-v22.0 ARF-07)
- `engine/sequence.rs` — sequence state management
- `engine/batch.rs` — batch composition
- `engine/speculative.rs` — speculative decode coordination (draft model +
  verification)

The engine uses `Box<dyn ModelBackend>` (non-generic) for model abstraction,
so it can hold heterogeneous target/draft models at runtime. Construction
goes through `Engine::with_config(...)` or `Engine::with_drafts(...)`.

For the refactor rationale, see [ADR-001](./adr/ADR-001-component-sharing-strategy.md)
and the v22.0 phase 39 plan.

## Scheduler Split

The scheduler (`crates/core/src/scheduler/`) was decomposed from a single
file into focused sub-modules during v20.0-v22.0:

- `queue/` — waiting/running/finished sequence queues
- `preemption/` — preemption logic (when memory pressure forces preemption)
- `eviction/` — KV cache eviction policies (LRU, etc.)
- `batch/` — batch composition (which sequences to run in the next step)
- `policy/` — scheduling policies (`FcfsPolicy`, `SjfPolicy`, `PriorityPolicy`)
- `memory/` — block allocator + prefix cache

Each sub-module has a single responsibility; the `SchedulerEngine` orchestrates
them via well-defined interfaces. See
[ADR-012](./adr/ADR-012-continuous-batching.md) for the continuous batching
design.

## KV Cache Layering

The KV cache has a two-layer split:

1. **Logical KV cache** (`crates/core/src/kv_cache/`) — block-based view from
   the scheduler's perspective. Operates on `BlockId`s and token counts; does
   not know about tensor layouts.

2. **Physical KV cache** (`crates/model/src/paged_tensor/`) — actual tensor
   storage. Maps `BlockId`s to tensor slices in the underlying Candle
   storage. Handles quantization (Q4_K_M via GGUF) and precision conversions.

This separation lets the scheduler reason about memory in abstract terms
while the model layer handles the messy tensor details. See
[ADR-005](./adr/ADR-005-kv-cache-split.md) and
[ADR-013](./adr/ADR-013-paged-kv-cache.md).

## Architecture Registry Pattern

Adding a new model architecture (e.g., a new LLaMA variant) requires only:

1. Implement `Architecture` trait in `crates/model/src/{newarch}/arch.rs`
2. Register in `register_all_archs()` in
   `crates/core/src/arch/registry.rs` (or `crates/model/src/arch/registry.rs`)
3. Define model-specific weight remapping and forward pass

The registry detects the architecture from `config.json`'s `model_type` field
and dispatches to the correct implementation. This pattern replaced the
previous enum + match pattern in v18.0+ and is documented in
[ADR-014](./adr/ADR-014-architecture-registry.md).

Stub architectures (`gemma3`, `llama4`, `phi4`, `mistral_small`) are
registered for testing but rejected at load time unless the `allow_stub`
capability is granted.

## Multi-Model Spec Flow (Speculative Decoding)

Speculative decoding uses a draft model to propose tokens that a target model
verifies in batch. The flow:

1. **Registration** — `DraftModelRegistry` (in
   `crates/core/src/speculative/registry/`) holds draft specs (Unloaded)
2. **Loading** — `DraftLoader::attach_loaded` transitions specs to Loaded,
   triggered by `Engine::set_draft_loader`
3. **Routing** — `Engine::step()` dispatches via `engine/spec_dispatch/` to
   the named draft backend
4. **Verification** — target model scores draft tokens; accepted tokens are
   emitted, rejected tokens trigger rollback
5. **Lifecycle** — refcount-driven eviction via `MemoryBudget`

For the speculative architecture, see
[ADR-006](./adr/ADR-006-speculative-decoding-architecture.md) and
[ADR-007](./adr/ADR-007-per-request-draft-routing.md).

## ADR References

- [ADR-001](./adr/ADR-001-component-sharing-strategy.md) — Component sharing strategy
- [ADR-002](./adr/ADR-002-feature-flag-design.md) — Feature flag design
- [ADR-005](./adr/ADR-005-kv-cache-split.md) — KV cache split (logical vs physical)
- [ADR-006](./adr/ADR-006-speculative-decoding-architecture.md) — Speculative decoding architecture
- [ADR-007](./adr/ADR-007-per-request-draft-routing.md) — Per-request draft routing
- [ADR-008](./adr/ADR-008-vllm-dist-feature-gated.md) — vllm-dist feature-gated
- [ADR-010](./adr/ADR-010-cuda-graph-feature-gating.md) — CUDA graph feature-gating
- [ADR-011](./adr/ADR-011-cross-crate-error-boundaries.md) — Cross-crate error boundaries
- [ADR-012](./adr/ADR-012-continuous-batching.md) — Continuous batching
- [ADR-013](./adr/ADR-013-paged-kv-cache.md) — Paged KV cache
- [ADR-014](./adr/ADR-014-architecture-registry.md) — Architecture registry

## Extending the System

| To add...                | Touch these files                                                |
|--------------------------|------------------------------------------------------------------|
| A new model architecture | `crates/model/src/{newarch}/arch.rs`, register in `arch/registry.rs` |
| A new scheduling policy  | Implement `SchedulingPolicy` in `crates/core/src/scheduler/policy/` |
| A new KV cache eviction  | Implement `EvictionPolicy` in `crates/core/src/scheduler/eviction.rs` |
| A new error variant      | Add to the relevant `thiserror::Error` enum; add `#[error("...")]` |
| A new API endpoint       | Add route handler in `crates/server/src/openai/`                 |
| A new draft model kind   | Implement `Architecture` trait + register                        |

For code conventions, see `CLAUDE.md`. For error type conventions, see the
"Error Type Conventions" section in `CLAUDE.md` / `AGENTS.md`.
