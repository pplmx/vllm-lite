# Architecture Audit Report — vllm-lite (v19.0)

**Generated:** 2026-06-27
**Scope:** Crate dependencies, module boundaries, circular deps, layering, test architecture
**Constraint:** Pure audit — no code changes
**Workspace root:** `/workspace/vllm-lite`

## Executive Summary

The vllm-lite workspace consists of **6 crates** (not 7 as the planning docs claim; there is no separate `benches` crate — benchmarks live inside `vllm-core/benches/`). The planned layering rule `traits ← core ← {model, server, dist}` is mostly respected, with **2 documented exceptions** (ARCH-F-11, ARCH-F-12) and 1 unintended divergence in dist surface usage (ARCH-F-17). The crate dependency graph has **no compile-time cycles**.

Findings summary:

| Severity | Count |
|----------|-------|
| P0       | 2     |
| P1       | 3     |
| P2       | 8     |
| P3       | 4     |
| **Total** | **17** |

Highest-impact items:

1. **ARCH-F-11 (P0):** `vlll-model` depends on `vllm-dist` — sibling-tier violation.
2. **ARCH-F-12 (P0):** `vllm-core` has optional downward dependency on `vllm-model` via `cuda-graph` feature.
3. **ARCH-F-17 (P1):** Most of `vllm-dist` (distributed_kv, grpc, pipeline modules) is publicly exported but never used outside the crate — only `TensorParallelConfig` is consumed externally.

---

## 1. Crate Dependency Graph (ARCH-01)

### 1.1 Workspace members

| # | Crate (Cargo name) | Crate path | LOC (src/) | One-line purpose |
|---|--------------------|------------|-----------:|------------------|
| 1 | `vllm-traits` | `crates/traits/` | 286 | Interface definitions: `ModelBackend` trait, types (`Batch`, `BatchOutput`, `SeqId`, `TokenId`, `BLOCK_SIZE`, `TensorParallelError`), `kernels` module re-exports. |
| 2 | `vllm-core` | `crates/core/` | 12 855 | Engine, scheduler, KV cache, metrics, HA, circuit-breaker, speculative decoding glue. |
| 3 | `vllm-model` | `crates/model/` | 21 035 | Per-architecture model implementations (Llama, Mistral, Qwen2/3, Qwen3.5, Gemma3/4, Phi-4, Llama-4, Mistral-Small, Mixtral), kernels, loader, tokenizer. |
| 4 | `vllm-server` | `crates/server/` | 5 343 | HTTP/axum server, OpenAI-compatible API, CLI, config, auth, security, debug endpoints. |
| 5 | `vllm-dist` | `crates/dist/` | 2 319 | Tensor-parallel primitives, gRPC scaffolding, distributed KV cache scaffolding, pipeline parallelism scaffolding. |
| 6 | `vllm-testing` | `crates/testing/` | 1 507 | TestHarness, mocks (`StubModel`, `FakeModel`, `IncrementModel`, `SlowModel`, …), `TestFixtures`, builders. |

> **Note:** The plan and `20-CONTEXT.md` mention 7 crates including `benches`. There is **no** `benches` crate in the workspace; benchmarks are `[[bench]]` entries inside `vllm-core/benches/` (8 files). See ARCH-F-03.

### 1.2 Dependency table (rows = source, cols = target; "—" = none)

| From \ To | `traits` | `core` | `model` | `server` | `dist` | `testing` |
|-----------|:--------:|:------:|:-------:|:--------:|:------:|:---------:|
| `traits`  | —        | —      | —       | —        | —      | —         |
| `core`    | ✓        | —      | ✓ (opt, `cuda-graph` feature) | — | — | — (dev-dep) |
| `model`   | ✓        | —      | —       | —        | ✓      | — (dev-dep) |
| `server`  | ✓        | ✓      | ✓       | —        | —      | —         |
| `dist`    | ✓        | —      | —       | —        | —      | —         |
| `testing` | ✓        | ✓      | —       | —        | —      | —         |

### 1.3 Detailed dependency list (from `cargo metadata --format-version 1 --no-deps` + Cargo.toml)

| Source | Target | Kind | Optional | Source location |
|--------|--------|------|----------|-----------------|
| `vllm-core` | `vllm-traits` | normal | no | `crates/core/Cargo.toml:8` |
| `vllm-core` | `vllm-model`  | normal | **yes** (feature `cuda-graph`) | `crates/core/Cargo.toml:25-29` |
| `vllm-core` | `vllm-testing` | dev | no | `crates/core/Cargo.toml:33` |
| `vllm-model` | `vllm-traits` | normal | no | `crates/model/Cargo.toml:8` |
| `vllm-model` | `vllm-dist`   | normal | no | `crates/model/Cargo.toml:9` |
| `vllm-model` | `vllm-testing` | dev | no | `crates/model/Cargo.toml:28` |
| `vllm-server` | `vllm-core`   | normal | no | `crates/server/Cargo.toml:20` |
| `vllm-server` | `vllm-model`  | normal | no | `crates/server/Cargo.toml:22` |
| `vllm-server` | `vllm-traits` | normal | no | `crates/server/Cargo.toml:21` |
| `vllm-dist`   | `vllm-traits` | normal | no | `crates/dist/Cargo.toml:22-23` |
| `vllm-testing` | `vllm-traits` | normal | no | `crates/testing/Cargo.toml:7` |
| `vllm-testing` | `vllm-core`   | normal | no | `crates/testing/Cargo.toml:8` |

### 1.4 Findings

- **ARCH-F-11 (P0) — `vllm-model` depends on `vllm-dist` (`crates/model/Cargo.toml:9`).** Per the planned rule `traits ← core ← {model, server, dist}`, model and dist are **siblings**. model→dist creates a sibling-tier edge. Realized in: `crates/model/src/qwen3/block.rs:11`, `crates/model/src/qwen3/model.rs:8`, `crates/model/src/qwen3/tp.rs:8` (each `use vllm_dist::TensorParallelConfig;`).
- **ARCH-F-12 (P0) — `vllm-core` has optional downward dependency on `vllm-model` (`crates/core/Cargo.toml:25-29`).** Feature-gated by `cuda-graph`. Source: `crates/core/src/engine.rs:27` (`#[cfg(feature = "cuda-graph")] use vllm_model::kernels::BatchCudaGraphExecutor;`). Per layering, core should not depend on model. Default feature set **does not** activate this; it activates when a downstream crate enables `cuda-graph`.
- **ARCH-F-03 (P2) — Stale documentation.** `.planning/PROJECT.md`, `AGENTS.md`, `REQUIREMENTS.md`, and `20-CONTEXT.md` all state "7 crates" including `benches`. Actual workspace has 6 crates; benchmarks are `[[bench]]` entries inside `vllm-core`. Source: `Cargo.toml:2` lists only 6 members.

---

## 2. Module Boundaries (ARCH-02)

### 2.1 Per-crate module inventory (selected; full file count above)

#### `vllm-traits` (4 src files, 286 LOC)

| File | LOC | Single-responsibility one-liner |
|------|----:|----------------------------------|
| `src/lib.rs` | 9 | Crate root — re-exports of `kernels`, `model`, `types`. |
| `src/model.rs` | ~50 | `ModelBackend` trait + `ModelError`. |
| `src/types.rs` | ~200 | Public data types: `Batch`, `BatchOutput`, `BatchPhase`, `SeqId`, `TokenId`, `BlockId`, `BLOCK_SIZE`, `TensorParallelError`. |
| `src/kernels.rs` | ~30 | `CudaGraphConfig`, `ModelGraphConfig`, `GraphExecutionError`. |

#### `vllm-core` (59 src files, 12 855 LOC; 40 files contain `#[cfg(test)]`)

| File / module | LOC | Single-responsibility one-liner |
|---------------|----:|----------------------------------|
| `src/engine.rs` | **1 038** | Engine orchestration: scheduler wiring, response channels, optional CUDA-graph path, error tracking, draft resolver glue. |
| `src/engine/speculative.rs` | 880 | Step-loop integration of speculative decoding (engine↔scheduler↔verifier). |
| `src/scheduler/mod.rs` | 133 | Scheduler module facade: 12 sub-modules, ASCII diagram in `//!` doc. |
| `src/scheduler/engine.rs` | 785 | `SchedulerEngine` orchestration. |
| `src/scheduler/batch_composer.rs` | 602 | `BatchComposer` — phase-aware batch construction. |
| `src/scheduler/policy/{fcfs,sjf,priority,trait_def,tests}.rs` | ~600 total | Scheduling policies (FCFS, SJF, Priority) + trait + tests. |
| `src/scheduler/memory/{allocator,eviction,mod}.rs` | ~400 | Block allocator + LRU eviction. |
| `src/scheduler/radix_cache/{mod,node,tree}.rs` | ~400 | Radix-tree prefix cache. |
| `src/scheduler/{packing,phase_scheduler,preemption,predictive_batching,observer,stats,request_queue,cuda_graph,batch,batch_planner}.rs` | various | Individual scheduler responsibilities; each has a clear single concern. |
| `src/speculative/mod.rs` | 28 | Facade: 9 sub-modules (`adaptive`, `config`, `draft_registry`, `draft_resolver`, `memory_budget`, `model`, `self_spec`, `strategy`, `verifier`). |
| `src/speculative/draft_registry.rs` | **929** | `DraftModelRegistry` — ID allocation, lazy loader, refcount, unload, plus 200+ LOC of test code inline. |
| `src/speculative/draft_resolver.rs` | ~300 | Per-request `DraftResolver` (v18.0). |
| `src/speculative/{adaptive,model,self_spec,strategy,verifier,memory_budget,config}.rs` | 200–700 | Speculative decoding strategies. |
| `src/kv_cache/mod.rs` | ~100 | Block allocator + prefix cache helpers (`pub use vllm_traits::BLOCK_SIZE;`). |
| `src/metrics/{mod,collector,exporter,lock_free,types}.rs` | ~900 | Metrics collection + Prometheus exporter + lock-free counters. |
| `src/ha/{mod,failover,leader_election}.rs` | ~500 | HA primitives (leader election, failover). |
| `src/circuit_breaker/{mod,breaker,strategy}.rs` | ~400 | Circuit-breaker (resilience). |
| `src/{beam,sampling,sync,types,error/{mod,recovery}}.rs` | various | Beam search, sampling, locking helpers, common types, error definitions. |
| `src/routing/{mod,hash_router}.rs` | ~200 | Consistent-hash request router (multi-node). |

#### `vllm-model` (117 src files, 21 035 LOC; 70 files contain `#[cfg(test)]`)

| File / module | LOC | Single-responsibility one-liner |
|---------------|----:|----------------------------------|
| `src/lib.rs` | 29 | Crate root — exposes 21 sub-modules. |
| `src/qwen3_config.rs` | **487** | Qwen3/Qwen3.5 shared config types (`Qwen3Config`, `TextConfig`, `RopeParameters`, `RopeScaling`). **Misplaced at top level** — only used by `qwen3/*` and `qwen3_5/*`. |
| `src/causal_lm/{mod,block_wrapper,hybrid_lm,layer_loop,model,weights}.rs` | ~600 | Causal LM abstraction wrapping per-architecture impls. |
| `src/arch/{mod,registry,capabilities}.rs` | ~250 | Architecture registry — runtime discovery of model impls. |
| `src/config/{mod,architecture,hyperparams,model_config}.rs` | ~500 | Per-model `Architecture` enum + config plumbing. |
| `src/loader/{mod,builder,checkpoint,format,io}.rs` | ~1 200 | Checkpoint loading (safetensors + GGUF). |
| `src/paged_tensor/{mod,tensor_store,quant,quantization}.rs` | ~1 100 | Physical paged KV cache + quantization. |
| `src/components/attention/{mod,gqa,mla,flash,flash_v3,rope_gqa,paged_gqa}.rs` | ~3 200 | Attention variants (GQA, MLA, FlashAttention v2/v3, paged, RoPE). |
| `src/components/attention/mod.rs` | 455 | Module facade + utility funcs (`expand_kv`, `causal_mask`, `paged_attention`, `tiled_attention`) + 270 LOC of inline tests. |
| `src/components/{block,decoder_block/{mod,factory},gated_delta/mod,kv_cache_fp8,mlp/{mod,swiglu},norm/{mod,rms_norm,layer_norm},positional/{mod,rope,mrope},ssm,vision}.rs` | ~2 300 | Shared model components (blocks, RoPE, RMSNorm, MLP, SSM, gated-delta, etc.). |
| `src/kernels/{mod,flash_attention,fused_mlp,cuda_graph.rs,cuda_graph/{config,executor}}.rs` | ~1 700 | Kernel implementations (FA, fused MLP, CUDA Graph). |
| `src/quantize/{mod,types,gguf}.rs` | ~700 | Quantization (GGUF Q4_K_M, FP8, INT). |
| `src/tokenizer.rs` | ~300 | Tokenizer wrapper (tiktoken + tokenizers). |
| `src/{llama,llama4,mistral,mistral_small,mixtral,qwen3,qwen3_5,phi4,gemma3,gemma4}/{arch,block,model,register,mod,mlp,attention,rope,…}.rs` | ~8 500 | Per-architecture implementations, each with `arch.rs`, `register.rs`, `mod.rs`. |
| `src/qwen3_5/{hybrid,attention35,ssm,gated_delta,block/{mod,full,linear},config,weights}.rs` | ~1 400 | Qwen3.5 hybrid (SSM + attention) specializations. |
| `src/qwen3/{model_tests.rs}`, `src/qwen3_5/{model_tests.rs,speculative_tests.rs}` | 960 | Test files referenced via `#[path = "..."]` directive — see ARCH-F-09. |

#### `vllm-server` (30 src files, 5 343 LOC; 20 files contain `#[cfg(test)]`)

| File | LOC | Single-responsibility one-liner |
|------|----:|----------------------------------|
| `src/lib.rs` | 47 | Crate root + `ApiState` definition; exposes `pub mod test_fixtures; #[doc(hidden)]`. |
| `src/main.rs` | ~300 | Server binary entry. |
| `src/bin/vllm.rs` | ~? | Alternative CLI binary. |
| `src/cli.rs` | 532 | Clap CLI argument parsing. |
| `src/config.rs` | 441 | `ServerConfig` + draft-loader config (v18.0). |
| `src/api.rs` | ~300 | Engine handle abstraction + Axum routes. |
| `src/auth.rs` | ~200 | Auth middleware. |
| `src/backpressure.rs` | ~150 | Backpressure handling. |
| `src/debug.rs` | ~200 | `/debug/*` endpoints. |
| `src/draft_loader.rs` | ~250 | Production `DraftLoader` impl (v18.0). |
| `src/health.rs` | ~150 | Health probes. |
| `src/logging.rs` | ~200 | Tracing setup. |
| `src/openai/{mod,chat,chat_template,completions,embeddings,models,types}.rs` | ~1 700 | OpenAI-compatible handlers. |
| `src/openai/batch/{mod,handler,manager,types}.rs` | ~700 | Batch API handlers. |
| `src/security/{mod,audit,correlation,jwt,rbac,tls}.rs` | ~900 | AuthN/AuthZ/TLS (v13/v15 hardening). |
| `src/test_fixtures.rs` | 64 | Test fixtures (`api_state`, `spawn_mock_engine`, `api_state_with_mock_engine`) — exposed as `pub mod` in `lib.rs`. See ARCH-F-14. |

#### `vllm-dist` (14 src files, 2 319 LOC; 10 files contain `#[cfg(test)]`)

| File / module | LOC | Single-responsibility one-liner |
|---------------|----:|----------------------------------|
| `src/lib.rs` | 21 | Crate root — exposes 5 sub-modules + 15 pub re-exports. |
| `src/types.rs` | ~30 | `TensorParallelConfig` (only externally-used type from this crate). |
| `src/tensor_parallel/{mod,all_reduce,device_mesh,parallel_linear}.rs` | ~700 | TP primitives (all-reduce, device mesh, parallel linear). |
| `src/distributed_kv/{mod,cache,protocol}.rs` | ~600 | Distributed KV cache scaffolding. **No external usage.** |
| `src/grpc.rs` | 160 | gRPC state + `tonic::include_proto!("vllm.distributed")`. **No external usage.** |
| `src/pipeline/{mod,pipeline,stage}.rs` | ~500 | Pipeline parallelism scaffolding. **No external usage.** |
| `src/generated/vllm.distributed.rs` | 574 | Generated protobuf bindings. |

#### `vllm-testing` (8 src files, 1 507 LOC; 7 files contain `#[cfg(test)]`)

| File | LOC | Single-responsibility one-liner |
|------|----:|----------------------------------|
| `src/lib.rs` | 30 | Crate root + `prelude` module. |
| `src/harness.rs` | ~250 | `TestHarness` — test environment setup. |
| `src/request_factory.rs` | ~150 | `RequestFactory` — generates test requests. |
| `src/mocks/mod.rs` | 477 | Mock `ModelBackend` impls (`StubModel`, `FakeModel`, `IncrementModel`, `ConstModel`, `NeverProgressModel`). |
| `src/slow_model.rs` | ~150 | `SlowModel` — controllable-latency mock for backpressure tests. |
| `src/fixtures/mod.rs` | 149 | `TestFixtures` struct — `SchedulerConfig` defaults. |
| `src/builders/mod.rs` | ~200 | `BatchBuilder`, `RequestBuilder`. |
| `src/utils/mod.rs` | ~50 | `assert_batch_consistency`, `create_simple_batch`, `generate_random_tokens`. |

### 2.2 God-module detection (≥1000 LOC)

| File | LOC | Pub items | Severity |
|------|----:|----------:|----------|
| `crates/core/src/engine.rs` | **1 038** | 12 (struct + 7 pub methods + others) | **P1** (God module by LOC) |

Files at 700–1000 LOC (borderline, monitor; **not flagged** unless review finds evidence of mixed responsibilities):

| File | LOC |
|------|----:|
| `crates/core/src/speculative/draft_registry.rs` | 929 |
| `crates/core/src/engine/speculative.rs` | 880 |
| `crates/model/src/kernels/flash_attention.rs` | 899 |
| `crates/model/src/components/attention/gqa.rs` | 829 |
| `crates/model/src/paged_tensor/tensor_store.rs` | 825 |
| `crates/core/src/scheduler/engine.rs` | 785 |
| `crates/model/src/components/attention/mla.rs` | 657 |
| `crates/model/src/components/attention/flash_v3.rs` | 646 |

No file exceeds 30 pub items; largest is `crates/core/src/scheduler/mod.rs` with 26.

### 2.3 Findings

- **ARCH-F-04 (P1) — `crates/core/src/engine.rs` at 1 038 LOC is the only file over the 1 000 LOC God-module threshold.** Single-responsibility is "Engine orchestration", but the file contains: scheduler wiring, response-channel management, optional CUDA-graph integration (10 lines `#[cfg(feature = "cuda-graph")]` paths), speculative-decode glue (resolver wiring, draft loop integration), error tracking (`error_count`, `last_error`), and an actor loop (`run`). Splitting `cuda_graph` and `speculative_resolver` glue into separate sub-modules would reduce coupling.
- **ARCH-F-05 (P2) — `crates/core/src/speculative/draft_registry.rs` at 929 LOC is borderline.** Single-responsibility is "DraftModelRegistry" but the file combines: ID allocation (`DraftId`), lazy weight loading (delegated to `DraftLoader`), refcount semantics, unload paths (`unload`, `force_unload`), memory tracking. A `registry/loader.rs` split would improve testability.
- **ARCH-F-06 (P2) — `crates/core/src/engine.rs` + `crates/core/src/engine/speculative.rs` together account for 1 918 LOC.** Engine declares `mod speculative;` (line 1) to scope the speculative glue, but Engine itself still imports speculative internals directly (e.g., `crates/core/src/engine.rs:8-12`: `use crate::speculative::{AdaptiveSpeculativeDecoder, DraftModelRegistry, DraftRegistryError, DraftSpec}; use crate::speculative::draft_registry::DraftId, DraftModelRegistry, DraftRegistryError, DraftSpec; use crate::speculative::draft_resolver::{DraftLoader, DraftResolver, NoopLoader}; use crate::speculative::memory_budget::MemoryBudget;`). The `mod speculative;` split is incomplete.
- **ARCH-F-07 (P2) — `crates/model/src/qwen3_config.rs` (487 LOC) is a top-level crate file but contains types only used by qwen3/qwen3_5 modules.** The directory layout implies `qwen3/` is self-contained, but `qwen3_config.rs` lives next to `qwen3/` itself. Should logically be at `crates/model/src/qwen3/config.rs` (note: `qwen3_5/config.rs` already follows this convention for hybrid-specific config). Source: `crates/model/src/lib.rs:20` declares `pub mod qwen3_config;`.
- **ARCH-F-08 (P2) — `crates/model/src/components/attention/mod.rs` (455 LOC, 17 pub items) mixes module-coordinator role with utility role.** It contains substantial utility functions (`expand_kv`, `causal_mask`, `causal_mask_tile`, `paged_attention`, `tiled_attention`, `AttentionConfig`) alongside `pub use` re-exports. The utilities (180+ LOC of standalone logic) overlap with `paged_gqa.rs` and `flash.rs` which implement their own variants. Single-responsibility is ambiguous: is this the "attention module facade" or the "attention utilities module"?
- **ARCH-F-09 (P3) — Non-idiomatic test-file naming using `#[path = "..."]` directive.** Three files use this pattern:
  - `crates/model/src/qwen3/model_tests.rs` (554 LOC, included at `crates/model/src/qwen3/model.rs:51-53`)
  - `crates/model/src/qwen3_5/model_tests.rs` (131 LOC, included at `crates/model/src/qwen3_5/model.rs:120-122`)
  - `crates/model/src/qwen3_5/speculative_tests.rs` (275 LOC, included at `crates/model/src/qwen3_5/model.rs:124-126`)
  
  Rust-idiomatic pattern is `mod tests { ... }` (or a single `tests.rs` file in the same directory). The `_tests` plural suffix with separate-file-via-`#[path]` is unusual but functional. Note: this pattern is only used inside the `qwen3` family; other crates use inline `#[cfg(test)] mod tests { ... }`.
- **ARCH-F-18 (P3) — `crates/model/src/qwen3_config.rs` does not declare a module-level `//!` doc comment** (file starts directly with `use serde::Deserialize;`). Most other top-level model files do; this is a doc-coverage inconsistency to be raised in DOCS audit (Phase 22).

---

## 3. Circular Dependency Scan (ARCH-03)

### 3.1 Method

1. Built workspace dependency graph from `cargo metadata --format-version 1 --no-deps` (output: `/tmp/meta.json`, 6 packages).
2. For each crate, listed normal-dep edges (excluding `kind = "dev"` because dev-deps do not propagate transitively and never create compile cycles).
3. Ran DFS-based cycle detection. If a back-edge to an ancestor in the DFS tree is found, a cycle exists.
4. Separately verified dev-dep relationships do not introduce cycles (cargo's feature resolver excludes dev-deps from dependents' graphs).

### 3.2 Normal-dependency graph (only `kind = null` and `kind = "normal"`)

```text
traits    → ∅
core      → {traits, model(opt)}
model     → {traits, dist}
dist      → {traits}
testing   → {core, traits}
server    → {core, model, traits}
```

Adjacency matrix (source → target):

| Source | traits | core | model | server | dist | testing |
|--------|:------:|:----:|:-----:|:------:|:----:|:-------:|
| traits | — | — | — | — | — | — |
| core   | ✓ | — | ✓ (opt) | — | — | — |
| model  | ✓ | — | — | — | ✓ | — |
| server | ✓ | ✓ | ✓ | — | — | — |
| dist   | ✓ | — | — | — | — | — |
| testing | ✓ | ✓ | — | — | — | — |

### 3.3 DFS results

For each starting node:

- **traits**: no outbound edges → terminal.
- **core** → traits → terminal; → model (optional) → traits → terminal; → dist → traits → terminal. **No cycle.**
- **model** → traits → terminal; → dist → traits → terminal. **No cycle.**
- **dist** → traits → terminal. **No cycle.**
- **testing** → core → … (no back edge); → traits → terminal. **No cycle.**
- **server** → core → … (no back edge); → model → … (no back edge); → traits → terminal. **No cycle.**

### 3.4 Dev-dependency edges (informational, do not create compile cycles)

| Source | Dev-dep on |
|--------|------------|
| `vllm-core` | `vllm-testing` |
| `vllm-model` | `vllm-testing` |

The `vllm-core ↔ vllm-testing` relationship via `(core → testing-dev) ↔ (testing → core-normal)` looks cyclic, but cargo's feature resolver excludes dev-deps from transitive closure. **No compile cycle.**

### 3.5 Results

**No circular dependencies detected.**

### 3.6 Findings

- **ARCH-F-10 (P3) — Lemon pair `vllm-core` ↔ `vllm-testing`.** While not a cycle, this pattern (core uses testing as dev-dep, testing uses core as normal dep) tightly couples the two crates. Consider splitting `vllm-testing` into `vllm-testkit` (trait-only helpers, no core dep) and `vllm-harness` (core-aware test orchestration).

---

## 4. Layering Consistency Matrix (ARCH-04)

### 4.1 Documented layering rule

```text
vllm-traits   (leaf — no inbound)
vllm-core     ← vllm-traits
vllm-model    ← {vllm-core, vllm-traits}        (sibling tier: server, dist)
vllm-server   ← {vllm-core, vllm-model, vllm-traits}
vllm-dist     ← {vllm-core, vllm-traits}        (sibling tier: model, server)
vllm-testing  ← {vllm-core, vllm-model, vllm-traits}
```

Allowed inbound (who may depend on me):

| Crate | Allowed inbound from |
|-------|---------------------|
| `traits` | nobody |
| `core` | `traits` |
| `model` | `core`, `traits` |
| `server` | `core`, `model`, `traits` |
| `dist` | `core`, `traits` |
| `testing` | `core`, `model`, `traits` |

### 4.2 Actual inbound matrix (✓ = actual dep, ✗ = allowed-and-present, ✗✗ = forbidden-and-present)

| From \ To | `traits` | `core` | `model` | `server` | `dist` | `testing` |
|-----------|:--------:|:------:|:-------:|:--------:|:------:|:---------:|
| `traits`  | —        | —      | —       | —        | —      | —         |
| `core`    | ✓        | —      | **✗✗ (opt)** | — | — | — |
| `model`   | ✓        | —      | —       | —        | **✗✗** | — |
| `server`  | ✓        | ✓      | ✓       | —        | —      | —         |
| `dist`    | ✓        | —      | —       | —        | —      | —         |
| `testing` | ✓        | ✓      | —       | —        | —      | —         |

### 4.3 Cross-layer import violations

Verified via `grep -rn "use vllm_" crates/` for cross-crate imports:

#### 4.3.1 ARCH-F-11 (P0) — `vllm-model → vllm-dist`

| File | Line | Import |
|------|-----:|--------|
| `crates/model/src/qwen3/block.rs` | 11 | `use vllm_dist::TensorParallelConfig;` |
| `crates/model/src/qwen3/model.rs` | 8 | `use vllm_dist::TensorParallelConfig;` |
| `crates/model/src/qwen3/tp.rs` | 8 | `use vllm_dist::TensorParallelConfig;` |

Per the rule `traits ← core ← {model, server, dist}`, model and dist are siblings — neither should depend on the other. This is a **hard layering violation** at the Cargo.toml level (`crates/model/Cargo.toml:9`). Real-world impact: any change to `vllm-dist`'s public surface forces `vllm-model` recompilation; offline-model builds that want to skip dist are impossible.

#### 4.3.2 ARCH-F-12 (P0) — `vllm-core → vllm-model` (optional, feature-gated)

| File | Line | Import |
|------|-----:|--------|
| `crates/core/src/engine.rs` | 27 | `#[cfg(feature = "cuda-graph")] use vllm_model::kernels::BatchCudaGraphExecutor;` |
| `crates/core/src/engine.rs` | 30 | `#[cfg(feature = "cuda-graph")] use vllm_traits::kernels::CudaGraphConfig;` |

Source: `crates/core/Cargo.toml:25-29` — `cuda-graph = ["dep:vllm-model"]`. Default feature set (`default = ["prometheus"]`) does **not** activate this; only when a downstream crate enables `cuda-graph`. Real-world impact: when `vllm-server` builds with `cuda-graph` enabled (`crates/server/Cargo.toml:20: vllm-core = { path = "../core", features = ["cuda-graph"] }`), the dependency becomes active, creating a real `core → model` edge. The fact that it is feature-gated does not eliminate the layering violation; it only defers its activation.

#### 4.3.3 ARCH-F-13 (P2) — `TensorParallelError` lives in `vllm-traits` but is semantically a `vllm-dist` type

| File | Line | Definition / re-export |
|------|-----:|------------------------|
| `crates/traits/src/types.rs` | 79 | `pub enum TensorParallelError { ... }` |
| `crates/traits/src/lib.rs` | 8 | `pub use types::{..., TensorParallelError, ...};` |
| `crates/dist/src/tensor_parallel/mod.rs` | 8 | `pub use vllm_traits::TensorParallelError;` |
| `crates/dist/src/lib.rs` | 19 | `pub use tensor_parallel::TensorParallelError;` |

`TensorParallelError` is conceptually a tensor-parallel error but is defined in the leaf `traits` crate because traits needs to expose a stable error type for the `ModelBackend` trait. Moving it to `dist::types` would require `traits` to depend on `dist` (worse layering). The current arrangement is a layering smell — the type lives in the wrong crate but cannot easily be moved.

### 4.4 ARCH-F-17 (P1) — Most of `vllm-dist` is publicly exported but never used outside the crate

Verified via `grep -rn "vllm_dist::" crates/` (excluding `crates/dist/`):

```text
crates/model/src/qwen3/block.rs:11:    use vllm_dist::TensorParallelConfig;
crates/model/src/qwen3/model.rs:8:     use vllm_dist::TensorParallelConfig;
crates/model/src/qwen3/tp.rs:8:        use vllm_dist::TensorParallelConfig;
```

The only externally-consumed item from `vllm-dist` is `TensorParallelConfig` (in `dist::types`). The following are exposed via `pub use` in `crates/dist/src/lib.rs:7-11` but never imported outside `dist`:

- `CacheConfig`, `CacheMessage`, `DistributedKVCache`, `NodeId` (`distributed_kv` module)
- `GrpcState` (`grpc` module)
- `PipelineStage`, `PipelineError`, `PipelineParallel`, `Result`, `PipelineStageConfig`, `StageInput`, `StageOutput` (`pipeline` module)

This represents ~1 600 LOC of publicly-exported API surface that is effectively dead code (kept alive by inline `#[cfg(test)]` tests within `dist` itself). Either the planned multi-node / gRPC / pipeline parallelism work is not yet wired into the serving stack, or these modules should be feature-gated / private until then.

---

## 5. Test Architecture (ARCH-05)

### 5.1 Test boundaries

| Crate | Unit tests (`#[cfg(test)]` in `src/`) | Integration tests (`tests/`) | Benches (`benches/`) |
|-------|---------------------------------------:|-----------------------------:|---------------------:|
| `vllm-traits`  | 0 files | 1 effective file (`tests/model_backend.rs`); `tests/mod.rs` is dead code (see ARCH-F-15) | 0 |
| `vllm-core`    | 40 files | 21 files (`adaptive_speculative.rs`, `beam.rs`, `cuda_graph_integration.rs`, `e2e_concurrent.rs`, `e2e_error_recovery.rs`, `e2e_graceful_shutdown.rs`, `e2e_lifecycle.rs`, `engine_trace.rs`, `engine_v18_wiring.rs`, `error_handling.rs`, `integration.rs`, `multi_draft_integration.rs`, `observer.rs`, `packing_integration.rs`, `prefix_cache.rs`, `resource_limits.rs`, `sampling.rs`, `scheduler_integration.rs`, `scheduler.rs`, `speculative_kv_cache.rs`, `speculative_memory_overhead.rs`) | 8 files (`radix_cache`, `latency_percentiles`, `speculative_vs_baseline`, `multi_draft_speculative`, `optimization_benchmarks`, `prefix_cache_benchmarks`, `scheduler_benchmarks`, `scheduler`) |
| `vllm-model`   | 70 files | 16 + `support/` sub-module (`arch_checkpoint_smoke.rs`, `architecture_smoke.rs`, `attention_batch_benchmark.rs`, `attention.rs`, `checkpoint_loading_tests.rs`, `fake_model.rs`, `gqa_shape_tests.rs`, `kv_cache_batch.rs`, `logits.rs`, `qwen3_config.rs`, `qwen3_integration.rs`, `qwen3_rope.rs`, `qwen3_token_pipeline.rs`, `ssm_optimization_tests.rs`, `tiled_attention.rs`, `tokenizer_verification.rs`; `support/{mod,on_disk,qwen3,tokenizer}.rs`) | 0 |
| `vllm-server`  | 20 files | 2 files (`chat_integration_test.rs`, `models_handler_test.rs`) | 0 |
| `vllm-dist`    | 10 files | 0 | 0 |
| `vllm-testing` | 7 files | 0 | 0 |
| **Total** | **147 files** | **40 files** (+ 1 dead) | **8 files** |

Total `.rs` files (excluding generated): **285**. Total LOC: **54 146**.

### 5.2 `vllm-testing` reuse

`vllm-testing` is used by:

- `vllm-core/tests/` (12 files): `adaptive_speculative.rs`, `e2e_concurrent.rs`, `e2e_error_recovery.rs`, `e2e_graceful_shutdown.rs`, `e2e_lifecycle.rs`, `integration.rs`, `prefix_cache.rs`, `speculative_kv_cache.rs`, `speculative_memory_overhead.rs` use `TestFixtures`, `IncrementModel`, `ConstModel`, `StubModel`.
- `vllm-core/benches/` (3 files): `optimization_benchmarks.rs`, `latency_percentiles.rs`, `speculative_vs_baseline.rs` use `TestFixtures`.
- `vllm-core/src/engine.rs:760` uses `vllm_testing::StubModel` in a `#[cfg(test)]` block.
- `vllm-model/tests/fake_model.rs:3` uses `vllm_testing::FakeModel`.

**NOT used by `vllm-server`.** Server has its own `test_fixtures.rs` (see ARCH-F-14).

Reuse coverage is **good within `vllm-core`** (12 of 21 integration tests + 3 of 8 benches) and **minimal in `vllm-model`** (1 of 16+ integration tests). `vllm-server` has **zero reuse** of `vllm-testing`.

### 5.3 Shared fixture hygiene

| Fixture | Crate | Location | Reused by |
|---------|-------|----------|-----------|
| `TestFixtures` (`SchedulerConfig` defaults) | `vllm-testing` | `src/fixtures/mod.rs` | `vllm-core` tests + benches |
| `StubModel`, `FakeModel`, `IncrementModel`, `ConstModel`, `NeverProgressModel` | `vllm-testing` | `src/mocks/mod.rs` | `vllm-core` tests |
| `SlowModel` | `vllm-testing` | `src/slow_model.rs` | declared but **not used anywhere** outside `vllm-testing` itself (P3 dead-code in testing crate — flag for separate audit) |
| `TestHarness` | `vllm-testing` | `src/harness.rs` | declared, not directly imported in tests (may be wrapped by fixtures — confirm in DOCS audit) |
| `RequestFactory` | `vllm-testing` | `src/request_factory.rs` | declared, not directly imported (likely wrapped) |
| `BatchBuilder`, `RequestBuilder` | `vllm-testing` | `src/builders/mod.rs` | declared, not directly imported |
| `assert_batch_consistency`, `create_simple_batch`, `generate_random_tokens` | `vllm-testing` | `src/utils/mod.rs` | declared, not directly imported |
| `support::` sub-module (qwen3, on_disk, tokenizer, mod) | `vllm-model` | `tests/support/` | 6 model integration tests (`checkpoint_loading_tests.rs`, `architecture_smoke.rs`, `tokenizer_verification.rs`, `arch_checkpoint_smoke.rs`, `qwen3_integration.rs`, `qwen3_token_pipeline.rs`) |
| `test_fixtures.rs` (server-local: `api_state`, `spawn_mock_engine`, `api_state_with_mock_engine`) | `vllm-server` | `src/test_fixtures.rs` | `vllm-server` 3 inline `#[cfg(test)]` blocks + 2 integration tests. **Not `#[cfg(test)]`-gated — ships in production.** |

### 5.4 Findings

- **ARCH-F-14 (P2) — `crates/server/src/test_fixtures.rs` is exposed via `pub mod test_fixtures;` in production code.** Source: `crates/server/src/lib.rs:26` (`#[doc(hidden)] pub mod test_fixtures;`). The module (64 LOC) is not gated by `#[cfg(test)]`, so it ships in every `vllm-server` binary including `vllm-server` and `vllm`. It is used by:
  - `crates/server/src/openai/batch/handler.rs:172` (`#[cfg(test)] crate::test_fixtures::api_state(...)`)
  - `crates/server/src/openai/completions.rs:125`
  - `crates/server/src/openai/embeddings.rs:63`
  - `crates/server/tests/models_handler_test.rs:4`
  - `crates/server/tests/chat_integration_test.rs:18,38`
  
  Recommended remediation: move helpers into `vllm-testing` (so all crates share the same fixture infrastructure) and delete `crates/server/src/test_fixtures.rs`. The `#[doc(hidden)]` attribute hides it from docs but does not prevent it from being publicly accessible via the binary.
- **ARCH-F-15 (P3) — `crates/traits/tests/mod.rs` is dead code.** File contains `mod model_backend;` (1 LOC), but `tests/` files are auto-loaded by cargo as top-level integration tests. The `tests/mod.rs` file is never loaded as a test, so its `mod model_backend;` declaration has no effect. The `tests/model_backend.rs` file is loaded directly. Either delete `tests/mod.rs` or refactor to a true shared module pattern.
- **ARCH-F-16 (P3) — `vllm-server` has zero reuse of `vllm-testing`.** Server-side tests (`chat_integration_test.rs`, `models_handler_test.rs`) use `vllm_server::test_fixtures::*` instead of `vllm_testing::*`. This creates a parallel fixture infrastructure. The duplication is small today (one helper file) but represents a pattern divergence: if `vllm-testing` grows useful server-side helpers (e.g., HTTP harness), server would not benefit.
- **ARCH-F-19 (P3) — Several public items in `vllm-testing` are declared but unused.** `SlowModel`, `TestHarness`, `RequestFactory`, `BatchBuilder`, `RequestBuilder`, `assert_batch_consistency`, `create_simple_batch`, `generate_random_tokens` are re-exported from `lib.rs` but have no `use` import sites outside `vllm-testing` itself. May be intended for downstream consumers / examples / docs — verify in DOCS audit (Phase 22).

---

## 6. Cross-Cutting Findings

These findings span multiple dimensions (ARCH + DOCS/API/NAME):

- **ARCH-F-09 + NAME-04 (cross-cutting) — Non-idiomatic Rust test file naming.** Three files in `vllm-model` use `*_tests.rs` (plural suffix) via `#[path = "..."]` directive. The pattern works but is unusual. See NAME-01 in Phase 21 for renaming recommendations.
- **ARCH-F-14 + DOCS-01 (cross-cutting) — `pub mod test_fixtures;` ships undocumented test code.** Even with `#[doc(hidden)]`, the module surface is publicly accessible. See DOCS audit for `///` doc coverage on fixture APIs.
- **ARCH-F-13 + API-04 (cross-cutting) — `TensorParallelError` lives in `vllm-traits` but is semantically a tensor-parallel concern.** Consider whether trait-layer error types should be opaque `Box<dyn Error>` with concrete error types defined in their owning crate.

---

## 7. Appendix: Methodology

### 7.1 Tools used

- `cargo metadata --format-version 1 --no-deps` (output to `/tmp/meta.json`) — workspace dependency graph.
- `cargo --version` — implicit through cargo metadata.
- `find` + `wc -l` — file inventory and LOC counting.
- `grep -c "^pub "` — pub-item counting (note: matches both `pub fn`, `pub struct`, `pub mod`, `pub use`; not deduplicated, may over-count).
- `grep -l "#\[cfg(test)\]"` — unit-test file detection.
- `grep -rn "use vllm_"` — cross-crate import analysis.

### 7.2 Thresholds

- **God module (P1):** ≥1 000 LOC in a single `.rs` file in `src/`.
- **Borderline (P2):** 700–999 LOC in a single `.rs` file in `src/`.
- **Pub-item count (informational):** ≥30 pub items per file (no file exceeded).
- **Cycle:** Confirmed via DFS over normal-dep edges only; dev-deps excluded from cycle graph (cargo does not propagate them transitively).
- **Layering violation (P0):** Cargo.toml `[dependencies]` introduces an edge that violates the documented tier rule, AND that edge is realized by an actual `use` import in source.
- **Layering smell (P2):** A type lives in the wrong crate but cannot be easily moved without worsening layering.

### 7.3 Manual checks performed

- Read `Cargo.toml` for each of the 6 crates.
- Read `lib.rs` for each crate to understand module surface.
- Spot-checked largest files (`engine.rs`, `scheduler/mod.rs`, `qwen3_5/*`, `attention/mod.rs`) for responsibility statements.
- Verified `tests/mod.rs` in `vllm-traits` is dead code by checking for any other file that does `mod tests::...`.
- Verified `vllm-dist` external usage via `grep -rn "vllm_dist::"` across all crates.
- Verified `vllm-testing` external usage via `grep -rn "vllm_testing"` across all crates.
- Verified all `#[path = "..."] mod ...;` declarations to confirm ARCH-F-09 is realized, not abandoned.

### 7.4 Out-of-scope items (deferred to other audit phases)

- **NAME-01..05** — File/type/function/variable/module naming audit (Phase 21).
- **DOCS-01..05** — Doc-comment coverage, module-level docs, stale comments, external docs, ADRs (Phase 22).
- **API-01..05** — Public API surface, error types, error ergonomics, trait design, deprecation hygiene (Phase 23).
- **SYNTH-01..03** — Cross-dimensional synthesis, prioritized backlog, v20.0+ migration roadmap (Phase 24).

### 7.5 Verification

After writing this report, `git status --short` was executed to confirm that **only** `.planning/audit/architecture/` was modified. No source files were touched.
