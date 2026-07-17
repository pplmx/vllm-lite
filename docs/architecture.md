# vLLM-lite Architecture

> Single source of truth for system design. Last updated: v31.0 planning cycle.

## Overview

vLLM-lite is a Rust LLM inference engine implementing vLLM innovations:
**Continuous Batching**, **Paged KV Cache**, **Prefix Caching**, and
**Speculative Decoding**, exposed via an **OpenAI-compatible HTTP API**.

```mermaid
flowchart TB
    subgraph clients [Clients]
        HTTP[HTTP / SSE]
        CLI[vllm CLI]
    end

    subgraph server [vllm-server]
        AX[Axum Router]
        OAI[OpenAI API]
        SEC[Security Middleware]
        BOOT[bootstrap/]
    end

    subgraph core [vllm-core]
        ENG[Engine Actor]
        SCH[SchedulerEngine]
        SPEC[Speculative Decoding]
        MET[Metrics]
    end

    subgraph traits [vllm-traits]
        MB[ModelBackend]
        CG[CudaGraphExecutor]
    end

    subgraph model [vllm-model]
        ARCH[Architecture Registry]
        COMP[components/]
        KERN[kernels/]
        PKV[paged_tensor/]
    end

    subgraph dist [vllm-dist — multi-node]
        DKV[DistributedKVCache]
        GRPC[gRPC]
    end

    HTTP --> AX --> OAI --> ENG
    ENG --> SCH --> PKV
    ENG --> MB --> ARCH
    ENG -.->|multi-node| DKV
    ENG -.->|cuda-graph| CG
```

## Crate Responsibilities

| Crate | Role | Key Types |
|-------|------|-----------|
| `vllm-traits` | Zero-dep interfaces | `ModelBackend`, `Batch`, `CudaGraphExecutor` |
| `vllm-core` | Engine + scheduler | `Engine`, `SchedulerEngine`, `RadixTree` |
| `vllm-model` | ML implementations | `CausalLm`, `GqaAttention`, `PagedKvCache` |
| `vllm-server` | HTTP layer | `ApiState`, OpenAI handlers |
| `vllm-dist` | Multi-node (feature-gated) | `DistributedKVCache`, gRPC |
| `vllm-testing` | Test harness | `FakeModel`, `BatchBuilder` |

## Request Lifecycle

```mermaid
sequenceDiagram
    participant C as Client
    participant S as Server
    participant E as Engine
    participant SCH as Scheduler
    participant M as ModelBackend

    C->>S: POST /v1/chat/completions
    S->>E: EngineMessage::AddRequest
    E->>SCH: add_request (prefix lookup)
    loop Step Loop
        E->>SCH: build_batch()
        E->>M: forward(batch)
        E->>C: stream tokens (SSE)
    end
```

The `Engine` uses an **actor pattern**: a dedicated worker thread owns the GPU
model; external callers communicate via `EngineMessage` over mpsc channels.

## Scheduler Internals

```mermaid
flowchart LR
    RQ[RequestQueue] --> PS[PhaseScheduler]
    PS --> BC[BatchComposer]
    BC --> MM[MemoryManager]
    MM --> RT[RadixTree Prefix Cache]
    BC --> POL[FCFS / SJF / Priority]
```

**Chunked Prefill**: long prompts are split into memory-bounded chunks.
Continuation chunks use `forward_prefill_continue` (reads existing KV prefix,
writes new tokens at global positions, applies rectangular causal mask).

## KV Cache Layers

| Layer | Location | Responsibility |
|-------|----------|----------------|
| Logical | `core/scheduler/memory/` | Block allocation, LRU eviction |
| Prefix | `core/scheduler/radix_cache/` | O(k) longest-prefix match |
| Physical | `model/paged_tensor/` | Tensor storage, FP8 quant |
| Distributed | `dist/distributed_kv/` | Cross-node hash metadata (multi-node) |

## Feature Flags

The high-level one-row-per-flag summary is below. **For per-crate
detail, cross-crate propagation, and the recommended combinations
for common deployment shapes, see
[`docs/reference/feature-matrix.md`](./reference/feature-matrix.md)** —
the single source of truth for the workspace's feature model. The
matrix doc complements this section and must be updated together
with it when a feature changes.

| Flag | Crate | Description |
|------|-------|-------------|
| `cuda` | model | Candle CUDA backend |
| `gguf` | model | GGUF weight loading |
| `cuda-graph` | core, server | CUDA Graph capture/replay |
| `multi-node` | core, model, testing | Enable `vllm-dist` |
| `full` | model | `cuda` + `gguf` |
| `candle` | traits | Expose `candle-core` `Tensor` to `ModelBackend` (always-on via core/model/server dep) |
| `kernels` | traits | Gate the CUDA-graph kernel module (always-on via core/model dep) |

`vllm-dist` exposes **no Cargo features** itself — see
[`ADR-008`](./adr/ADR-008-vllm-dist-feature-gated.md) for the
rationale. `vllm-traits` has 2 additional features (`candle` +
`kernels`) that the architecture summary table previously omitted; the
new feature matrix doc closes that gap.

## Architecture Registry

New models are registered in 3 steps:

1. Implement `Architecture` trait in `model/src/<arch>/arch.rs`
2. Add `register.rs` calling `registry.register::<T>()`
3. Wire into `register_all_archs()` in `arch/registry.rs`

Stub architectures (Gemma3, Llama4, Phi4, Mistral Small) share
`StubArchitecture` in `arch/stub.rs`.

## Testing Strategy

| Tier | Command | Scope |
|------|---------|-------|
| Fast | `just nextest` | Unit + integration (skips `#[ignore]`) |
| Checkpoint | `just nextest-checkpoint` | Real weight tests |
| Full | `just nextest-all` | All tests including slow |
| Fuzz | `just fuzz-smoke` | 7 fuzz targets |
| Mutation | `just mutants MODULE` | cargo-mutants (907 mutants, 100%) |

## Related Documents

- [ROADMAP.md](../ROADMAP.md) — feature roadmap
- [CHANGELOG.md](../CHANGELOG.md) — release history (most authoritative)
- [OPERATIONS.md](../OPERATIONS.md) — deployment & troubleshooting
- [docs/adr/](./adr/) — Architecture Decision Records
- [docs/tutorial/](./tutorial/) — onboarding guides
- [AGENTS.md](../AGENTS.md) — development conventions
