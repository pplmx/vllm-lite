# 📋 Changelog

<p align="center">
  <img src="https://img.shields.io/badge/Keep%20a%20Changelog-1.0.0-blue.svg?style=flat-square" alt="Keep a Changelog">
  <img src="https://img.shields.io/badge/Semantic%20Versioning-2.0.0-green.svg?style=flat-square" alt="Semantic Versioning">
</p>

> All notable changes to **vLLM-lite** will be documented in this file.

---

## 📊 Release Statistics

|     版本     |    日期    |          测试          | 覆盖率 |
| :----------: | :--------: | :--------------------: | :----: |
| [Unreleased] |     -      |         1179+          | 97.8%  |
|   [v22.0]    | 2026-06-27 |         1179+          | 97.8%  |
|   [v21.0]    | 2026-06-27 |         1146+          | 97.8%  |
|   [v20.0]    | 2026-06-27 |         1144+          | 97.8%  |
|   [v19.0]    | 2026-06-27 |         1139+          | 97.8%  |
|   [v18.0]    | 2026-06-27 | 277 (vllm-core) + 654+ |  90%+  |

---

## 🚀 [Unreleased]

### Added

- **MambaBlock Weight Loading**
    - Added `MambaBlock::from_weights` method to load SSM layer weights
    - Implemented full weight loading for Qwen3.5 Mamba models
    - Supports fallback for embed_tokens and lm_head weight names
    - Supports tied embeddings (tie_word_embeddings)

### Changed

- **Workspace Lint Policy (v24.0 Phase A)** — established three-tier clippy lint
  configuration in root `Cargo.toml` `[workspace.lints.clippy]`:
    - **deny tier**: `correctness`, `suspicious`, `perf` (breaks CI)
    - **warn tier**: `pedantic`, `nursery`, `missing_errors_doc`, `must_use_candidate`, etc. (visible but not blocking)
    - **allow tier**: `cast_precision_loss`, `similar_names`, `too_many_lines`, `too_many_arguments`, etc. (with rationale)
    - All 6 crates inherit via `[lints] workspace = true`.
    - `just clippy` switched from `-D warnings` to explicit per-group denies so pedantic stays visible without breaking CI.
    - New `just clippy-pedantic` recipe for local pedantic+nursery view.
    - `AGENTS.md` gained a "Lint Policy" section documenting the tier system, local commands, and the rationale for each allow-list entry.

- **Unwrap Cleanup (v24.0 Phase B)** — fixed real bug risk and improved error reporting:
    - **B-1**: `cuda_graph/executor.rs:222` race condition unwrap → typed `GraphNotFound` error via new `lookup_graph` helper
    - **B-2**: 5 production unwraps in `engine.rs`, `main.rs`, `handler.rs` → typed error variants / cleaner panic messages
    - **B-3**: 48 production `// invariant:` comments added across 25+ files documenting legitimate invariants (RwLock/Mutex poison, SystemTime, Tensor allocation, signal handlers, etc.)
    - Baseline audit: spec originally claimed 787 production unwraps; actual was 60 (the 787 figure included inline `#[cfg(test)] mod tests` blocks). Spec target `≤160` was already met; Phase B reduced production unwraps to ~54.
    - New `EngineError::EmptyBeamList` variant added
    - `AGENTS.md` gained an "Invariant comments" section documenting the convention

---

## 🚀 [v18.0] — Multi-Model Speculative Decoding (2026-06-27)

### Added

- **DraftModelRegistry** (`crates/core/src/speculative/draft_registry.rs`)
    - Runtime registry for heterogeneous external draft models
    - Each draft owns a private `ModelBackend` and `BlockAllocator` (KV isolation by construction)
    - Lazy weight loading via `register` (Unloaded) → `attach_loaded` (Loaded) state machine
    - `Engine::with_drafts_boxed` constructor for pre-loading specs at engine startup
    - Loader-agnostic — does not depend on `vllm-model`; caller drives actual ModelLoader

- **MemoryBudget** (`crates/core/src/speculative/memory_budget.rs`)
    - VRAM budget enforcement for target + concurrent drafts
    - Atomic `try_reserve_draft` with structured `MemoryBudgetExceeded { requested_bytes, available_bytes, draft_id }` error
    - Runtime KV-cache growth tracking via `record_draft_kv_growth`
    - Default is `u64::MAX` (unlimited) — existing flows unchanged

- **Refcount-driven lifecycle**
    - `unload` returns `InUse(refcount)` if refcount > 0 (LIFE-02)
    - `force_unload` bypasses refcount for admin/test paths
    - `decrement_ref` auto-unloads when count reaches zero (LIFE-03)
    - Releases budget reservation on unload

- **DraftResolver** (`crates/core/src/speculative/draft_resolver.rs`)
    - Per-request draft selection with FALL-01 fallback semantics
    - `ResolvedDraft::{External, SelfSpec, None}` enum makes outcomes explicit
    - `DraftLoader` trait abstracts actual model loading (no vllm-model coupling)
    - Records metrics for every resolution (external / self_spec / none)

- **Per-request routing**
    - `Request.draft_model_id: Option<DraftId>` + `Request::with_draft_model` builder (RTE-01)
    - `Sequence.draft_model_id` propagated from Request (RTE-02)
    - Per-request resolution enables mixed drafts in one batch (RTE-03)

- **Fallback semantics**
    - FALL-01: load failure / unknown id / budget exceeded → silent fallback to self-spec
    - FALL-02: `Sequence.degraded_draft: bool` sticky flag set on runtime draft errors

- **Metrics** (5 new counters in `EnhancedMetricsCollector`)
    - `draft_resolutions_external_total`
    - `draft_resolutions_self_spec_total`
    - `draft_resolutions_none_total`
    - `draft_load_failures_total`
    - `draft_runtime_errors_total`

- **Integration tests** (`crates/core/tests/multi_draft_integration.rs`)
    - 14 tests covering full lifecycle, budget boundaries, mixed routing, all fallback paths
    - Stub backends with configurable failure injection

- **Benchmark** (`crates/core/benches/multi_draft_speculative.rs`)
    - Criterion benchmark: `no_draft` vs `self_spec` vs `external_draft` (3 configs)
    - Measures orchestration overhead (~1.7-2.1 µs per 16-step iteration)

### Changed

- `Sequence` gained `degraded_draft: bool` and `draft_model_id: Option<DraftId>` fields
- `Request` gained `draft_model_id: Option<DraftId>` field
- `BlockAllocator` gained `bytes_per_block()` and `allocated_bytes()` methods
- `DraftSpec` gained `weight_size_estimate_bytes: u64` field for MEM-02 budget estimation

### Requirements Satisfied

- MMLT-01, MMLT-02, MMLT-03 (multi-model loading)
- LIFE-01, LIFE-02, LIFE-03 (lifecycle management)
- MEM-01, MEM-02, MEM-03 (memory budget)
- RTE-01, RTE-02, RTE-03 (request routing)
- FALL-01, FALL-02 (fallback semantics)

**14/14 requirements passed.** Test count: 209 → 277 (+68).

### Refactored

#### Architecture Refactoring

- **Scheduler Module Split**
    - Split monolithic `scheduler.rs` into focused submodules
    - Created `scheduler/queue.rs` with `RequestQueue` for queue management
    - Created `scheduler/preemption.rs` with `PreemptionManager` for preemption decisions
    - Created `scheduler/eviction.rs` with `EvictionPolicy` for block eviction
    - Fully integrated all modules into `Scheduler` struct

- **KV Cache Layer Separation**
    - Split `core/kv_cache.rs` into `kv_cache/block_allocator.rs` and `kv_cache/prefix_cache.rs`
    - Created `model/paged_tensor/` module (separating logical and physical KV cache)
    - `tensor_store.rs` for GPU KV tensor management
    - `quantization.rs` for INT8/FP8 quantization
    - Added deprecated alias in `kv_cache.rs` for backward compatibility

- **Kernel Layer Extraction**
    - Created `model/kernels/` directory for GPU kernels
    - Moved `flash_attention.rs` → `kernels/flash_attention.rs`
    - Moved `fused_kernel.rs` → `kernels/fused_mlp.rs`
    - Moved `cuda_graph.rs` from core to `model/kernels/cuda_graph.rs`
    - Updated `components/` to use kernels module

#### Architecture Refactor Phase 5 (Qwen3.5 Hybrid 收敛, 2026-06-15)

- Split `qwen3_5/hybrid.rs` (1176 lines) into `block/` + `model.rs` + `weights.rs` + `config.rs`
- Introduce `HybridLm<B, Norm>` shell paralleling `CausalLm<B, N, H>`
- Move `GatedDeltaState` from `qwen3_5::gated_delta` to `components::gated_delta`
- Remove `causal_lm → qwen3_5` reverse dependency (`rg 'use qwen3_5' crates/model/src/causal_lm/` → 0 matches)
- GDN dims now read from `Qwen3Config` (no more hardcoded `(16, 4, 2)`)
- `Qwen35Architecture::capabilities()` upgraded to `PRODUCTION_SPECULATIVE`
- Speculative parity tests in `model_tests.rs` (124 lines) + `speculative_tests.rs` (285 lines)

Refs: `decc8c8`, `73dab5e`, `52f77ce`

#### Adaptive Speculative Decoding Counter Wire-up (Wave 2, 2026-06-26)

- `AdaptiveSpeculativeDecoder::record_verification` now returns `bool` adjustment event
- Engine `step_speculative_inner` calls `MetricsCollector::record_speculative_adjustment()` on actual adjustment
- `speculative_adjustments_total` Prometheus counter now correctly tracks adaptive decoder activity
- 3 new unit tests locking the bool return contract (high acceptance, low acceptance, deadband)
- Documentation: `SPEC-ADAPT-01` / `SPEC-ADAPT-02` marked complete in `.planning/PROJECT.md`

Refs: `docs/superpowers/specs/2026-06-26-wave2-adapt-spec.md`

#### Speculative Warmup Test Coverage (Wave 4, 2026-06-26)

- `Engine::warmup_draft_kv` visibility relaxed from `fn` to `pub(crate) fn` for test access
- New `CounterModel` wrapper in `engine::speculative::tests` mod (counts forward/forward_logits calls via AtomicUsize)
- New fast unit test `test_warmup_draft_kv_invokes_draft_per_sequence` verifies draft model receives exactly N forward() calls for N-seq Prefill batch
- Documentation: `SPEC-WARM-01` marked complete in `.planning/PROJECT.md`

Refs: `docs/superpowers/specs/2026-06-26-wave4-warmup-test.md`

#### Benchmark Suite Closure (Wave 5, 2026-06-26)

- New `crates/core/benches/latency_percentiles.rs` — per-request latency distribution with criterion auto-reported p50/p95/p99 (SPEC-BENCH-01)
- New `crates/core/benches/speculative_vs_baseline.rs` — explicit baseline vs adaptive speculative throughput comparison (SPEC-BENCH-02)
- New `docs/benchmark-suite.md` — suite documentation covering all 9 benchmarks
- New `just bench` recipe — runs all benchmarks with `--output-format bencher`
- Documentation: `SPEC-BENCH-01` / `SPEC-BENCH-02` marked complete in `.planning/PROJECT.md`; v17.0 milestone closes 7/9 SPECs

Refs: `docs/superpowers/specs/2026-06-26-wave5-benchmark-suite.md`

### Added (Phase 4)

#### Phase 4: Performance Optimization

- **Quantization Support**
    - FP16 support
    - INT8 Weight-Only quantization (`QuantizedLinear`, `quantize_2d`)
    - INT8 KV Cache with per-layer scaling
    - QuantizationCalibrator for calibration
- **Compute Optimization**
    - Flash Attention framework with software fallback (`FlashAttention`, `ScaledDotProductAttention`)
    - Sliding window attention support
    - CUDA Graph framework (`CudaGraph`, `CudaGraphExecutor`)
- **Scheduling Optimization**
    - PD Separation (Prefill/Decode separation)
    - Chunked Prefill with configurable chunk size
    - Dynamic Batch Size based on available KV blocks
    - Priority-based scheduling (`Priority`, `enable_priority_scheduling`)
- **Distributed**
    - Multi-GPU Tensor Parallelism (`DeviceMesh`, `ColumnParallelLinear`, `RowParallelLinear`, `AllReduce`)

#### Phase 5: Production Readiness (2025-04-12)

**Observability & Metrics**

- Prometheus-compatible metrics export (`/metrics` endpoint)
- Enhanced metrics collection (CUDA Graph, Sequence Packing, Adaptive Speculative)
- Health check endpoints (`/health`, `/ready`)
- Real-time metrics with `EnhancedMetricsCollector`

**Fault Tolerance**

- Circuit breaker pattern for automatic failure recovery
- Retry strategy with exponential backoff
- Degrade strategy for graceful degradation
- Recovery manager with error severity classification

**Testing**

- 26 E2E integration tests (lifecycle, concurrent, error recovery, graceful shutdown)
- Deterministic mock models for reproducible tests
- Performance regression testing in CI

**Deployment**

- Multi-stage Docker build (`Dockerfile`)
- Docker Compose with Prometheus (`docker-compose.yml`)
- Kubernetes manifests (namespace, deployment, service, HPA)
- CI performance regression workflow (`.github/workflows/benchmark.yml`)

**Core Features**

- Request timeout support (`timeout` parameter)
- Graceful shutdown (SIGINT/SIGTERM handling)
- YAML configuration file support
- Environment variable overrides (`VLLM_HOST`, `VLLM_PORT`, etc.)
- Structured JSON logging with file rotation
- Grafana dashboard (`docs/grafana/dashboard.json`)
- Config validation on startup
- Error retry support (`retries` parameter)

#### Core Features

- Real-time metrics collection with `/v1/stats` and `/metrics` endpoints
- Quantization utilities (`crates/model/src/quantize.rs`)
- Tiled Attention for memory optimization
- INT8 quantization support in KV Cache
- Forward pass with tiled attention strategy
- Comprehensive test suite for tiled attention

### Changed

- Improved documentation structure (README.md, docs/README.md, ROADMAP.md)
- Added detailed development roadmap

### Fixed

- Clippy warnings and code quality improvements
- Test compatibility with new AttentionConfig

## [0.1.0] - 2026-03-31

### Added

- **Continuous Batching** - Dynamic batch scheduling with decode-priority
- **Paged KV Cache** - Memory-efficient cache management with LRU eviction
- **Prefix Caching** - Exact match and prefix hit support
- **Speculative Decoding** - Draft-target verification architecture
- **Qwen3 Model Integration** - Support for Qwen2.5-0.5B model with real weights
- **OpenAI-compatible API** - `/v1/completions`, `/v1/chat/completions`
- **Streaming (SSE)** - Real-time token streaming
- **Sampling** - Temperature, Top-P support
- **Chunked Prefill** - Process long prompts in chunks

### Architecture

- **3-Crate Structure**:
    - `vllm-core`: Scheduler, Engine, KV Cache, Types
    - `vllm-model`: Qwen3, Attention, MLP
    - `vllm-server`: HTTP API (axum)

### Dependencies

- Rust (edition 2021)
- Candle (ML backend)
- Axum (HTTP)
- Tokio (async runtime)
- SafeTensors (weight loading)

---

## Migration Guides

### Upgrading to 0.1.0

No migration needed - initial release.

---

## 🚀 [v22.0] — Production Hardening (2026-06-27)

### Added

- **Security middleware wired (SEC-01..06)** — JWT signature verification (HMAC-SHA256 + RSA/ECDSA), `RbacMiddleware` permission enforcement, `RequestBodyLimitLayer` (configurable max body), audit log integration test, Grafana credentials moved to `.env`, structured TLS error replacing `unwrap()` panic
- **Phase 19 e2e tests un-ignored (OPS-02)** — `Engine::step()` speculative-mode hang fixed (DashMap shard re-entry deadlock in `EnhancedMetricsCollector`); 9 tests across `engine_wiring.rs` and `engine/spec_dispatch/tests.rs` now pass
- **Production polish (OPS-01, RFU-05, PERF-01..03)** — `parking_lot::Mutex` migration (24 sites in scheduler/engine), `MlaKvCache::write_compressed` incremental `slice_assign`, `eq_ignore_ascii_case` in arch detection, `std::sync::LazyLock` migration
- **Engine refactor (ARF-06, ARF-07)** — `engine.rs` God module split into focused sub-modules; `engine/spec_dispatch` tree unified post-Phase-31

### Fixed

- 5 cargo doc broken-link warnings resolved (intra-doc-link fixes in `engine.rs`, `components/attention/mod.rs`, `block.rs`, `decoder_block/mod.rs`, `speculative/registry/`)
- `Engine::step()` speculative-mode determinism bug (was hanging in 9 tests)

### Tech Debt Rolled Forward

- Stub architectures (`gemma3`, `llama4`, `phi4`, `mistral_small`) — policy deferred to v23.0
- `TensorParallelError` manual impl — deferred to v23.0
- `Box<dyn Error>` in `dist/src/grpc.rs` — deferred to v23.0
- Stale `CLAUDE.md`, `README.md`, `CHANGELOG.md`, `MIGRATING.md` — deferred to v23.0
- Dead code (~2000 LOC across scheduler/, routing/, ha/, circuit_breaker/) — deferred to v23.0
- `core → model` upward dep via cuda-graph feature — deferred to v23.0

### Stats

- **Phases:** 4 (Phase 36-39)
- **Test count:** 1179+ (≥ 1146 v21.0 baseline; +33 net)
- **Coverage:** doc 97.8%, clippy/fmt clean, 0 cargo doc warnings
- **ADRs:** 15 (no new ADRs in v22.0)

---

## 🚀 [v21.0] — P2/P3 Backlog Cleanup (2026-06-27)

### Added

- **API/error boundary work** (API-04, API-06, API-08, API-10) — Mutex `.expect()` migrations, `From<E>` impls for cross-crate error conversion, `dyn Trait` compile-only tests for object-safe traits (8 tests in `crates/testing/tests/dyn_safety.rs`)
- **Naming audit compliance** (NAME-F-04) — `qwen3_config` module moved to `qwen3::config`; test files migrated from `src/` to `crates/*/tests/`

### Removed

- **Dead `mod.rs`** in `crates/traits/tests/` (P3-01)

### Tech Debt Rolled Forward

- `mut Prompt::token_ids` non-mutating methods (P2-09) — partial; deferred to v22+
- Multi-node/vllm-dist resurrection — feature-gated only; OPS-05 still deferred
- Real-model benchmark — OPS-04 deferred (no GPU env)

### Stats

- **Phases:** 5 (Phase 31-35)
- **Test count:** 1146+ (1144 baseline + 13 new − 11 dedup)
- **Coverage:** doc 97.8%

---

## 🚀 [v20.0] — Codebase Remediation (2026-06-27)

### Added

- **48 requirements addressed across 6 phases** (Phase 25-30) — code remediation of v19.0 audit findings (architecture, naming, comments/docs, API/errors, tests, benchmarks)
- **12 new ADRs** — component sharing, feature flags, self-speculation, FP8, KV cache split, speculative decoding, per-request draft routing, multi-node feature-gating, FP8 quantizer orphan decision, CUDA graph feature-gating, cross-crate error boundaries, continuous batching

### Stats

- **Phases:** 6 (Phase 25-30)
- **Test count:** 1144+ passed, 0 failed
- **Coverage:** doc 97.8% / 99.6%

---

## 🚀 [v19.0] — Codebase Health Audit (2026-06-27)

### Added

- **5 analysis-only phases** (Phase 20-24) producing `.planning/audit/` directory:
    - Architecture audit (crate deps, module boundaries, layering matrix)
    - Naming audit (NAME-* findings)
    - Comments/docs audit (placeholder doc survey)
    - API/error audit (error type hygiene)
    - Test/benchmark audit
- **No code changes** — analysis-only milestone; findings drive v20.0-v23.0 remediation

### Stats

- **Phases:** 5 (Phase 20-24)
- **Output:** 5 audit reports in `.planning/audit/{architecture,naming,docs,api,benchmark}/`
- **Findings:** 22+ categories, drove 4 remediation milestones (v20.0, v21.0, v22.0, v23.0)

---

## Known Issues

- Long context (>32K) not yet supported (v24+ candidate)
- Multimodal/Vision not yet supported (v24+ candidate)
- Tool calling not yet supported (v24+ candidate)
- Multi-node / vllm-dist resurrection deferred (feature-gated only)

---

## Credits

Thanks to all contributors and the vLLM project for inspiration.
