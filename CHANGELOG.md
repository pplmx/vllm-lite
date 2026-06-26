# 📋 Changelog

<p align="center">
  <img src="https://img.shields.io/badge/Keep%20a%20Changelog-1.0.0-blue.svg?style=flat-square" alt="Keep a Changelog">
  <img src="https://img.shields.io/badge/Semantic%20Versioning-2.0.0-green.svg?style=flat-square" alt="Semantic Versioning">
</p>

> All notable changes to **vLLM-lite** will be documented in this file.

---

## 📊 Release Statistics

|     版本     | 日期  | 测试  | 覆盖率 |
| :----------: | :---: | :---: | :----: |
| [Unreleased] |   -   | 654+  |  90%+  |

---

## 🚀 [Unreleased]

### Added

- **MambaBlock Weight Loading**
    - Added `MambaBlock::from_weights` method to load SSM layer weights
    - Implemented full weight loading for Qwen3.5 Mamba models
    - Supports fallback for embed_tokens and lm_head weight names
    - Supports tied embeddings (tie_word_embeddings)

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

## Known Issues

- Limited model support (Qwen3 only)
- No multi-GPU support
- Quantization in progress

---

## Credits

Thanks to all contributors and the vLLM project for inspiration.
