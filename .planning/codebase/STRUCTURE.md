# Codebase Structure

**Analysis Date:** 2026-05-13

## Directory Layout

```text
vllm-lite/                          # Repository root
├── Cargo.toml                      # Workspace root (7 crates + benches)
├── justfile                        # Build automation (build, test, clippy, bench, ci)
├── AGENTS.md                       # AI agent development guide
├── CLAUDE.md                       # Claude-specific instructions
├── crates/
│   ├── traits/                     # vllm-traits: Interface definitions (no heavy deps)
│   │   └── src/
│   │       ├── lib.rs              # Re-exports: ModelBackend, types, kernels
│   │       ├── model.rs            # ModelBackend trait + ModelError
│   │       ├── types.rs            # Batch, BatchOutput, SeqId, TokenId, BlockId, BLOCK_SIZE
│   │       └── kernels.rs          # CUDA Graph config traits
│   ├── core/                       # vllm-core: Engine, Scheduler, KV cache, Metrics
│   │   └── src/
│   │       ├── lib.rs              # Re-exports: Engine, SchedulerEngine, Metrics, etc.
│   │       ├── engine.rs           # Engine: actor loop, model forward, step orchestration
│   │       ├── engine/
│   │       │   └── speculative.rs  # Speculative/adaptive step dispatch (800 lines)
│   │       ├── types.rs            # Request, Sequence, Status, SchedulerConfig, SamplingParams
│   │       ├── error/
│   │       │   ├── mod.rs          # EngineError enum (thiserror)
│   │       │   └── recovery.rs     # Error recovery strategies
│   │       ├── scheduler/
│   │       │   ├── mod.rs          # Module docs + re-exports
│   │       │   ├── engine.rs       # SchedulerEngine: orchestrates all sub-components
│   │       │   ├── request_queue.rs # O(1) request queue with phase-aware indexing
│   │       │   ├── phase_scheduler.rs # Prefill/decode phase separation
│   │       │   ├── batch_composer.rs  # Batch construction from sequences
│   │       │   ├── batch_planner.rs   # Adaptive batch planning
│   │       │   ├── batch.rs           # Batch data structures
│   │       │   ├── packing.rs         # Sequence packing utilities
│   │       │   ├── packing/           # Sequence packing sub-module
│   │       │   ├── policy/
│   │       │   │   ├── mod.rs         # Re-exports
│   │       │   │   ├── trait_def.rs   # SchedulingPolicy trait + SchedulingContext
│   │       │   │   ├── fcfs.rs        # First-Come-First-Served policy
│   │       │   │   ├── sjf.rs         # Shortest Job First policy
│   │       │   │   ├── priority.rs    # Priority-based scheduling policy
│   │       │   │   └── tests.rs       # Policy tests
│   │       │   ├── memory/
│   │       │   │   ├── mod.rs         # MemoryManager interface
│   │       │   │   ├── allocator.rs   # BlockAllocator with free list
│   │       │   │   └── eviction.rs    # LRU-based eviction policies
│   │       │   ├── cache/             # KV cache management
│   │       │   ├── radix_cache/       # Radix tree for O(k) prefix lookup
│   │       │   ├── preemption.rs      # Request preemption manager
│   │       │   ├── cuda_graph.rs      # CUDA graph capture/replay config
│   │       │   ├── observer.rs        # SchedulerObserver trait + event system
│   │       │   ├── predictive_batching.rs # Predictive batch optimization
│   │       │   └── stats.rs           # Scheduler statistics
│   │       ├── kv_cache/
│   │       │   ├── mod.rs             # Re-exports: BLOCK_SIZE, BlockAllocator, PrefixCache
│   │       │   └── prefix_cache.rs    # Hash-based prefix cache (used by SchedulerEngine)
│   │       ├── metrics/
│   │       │   ├── mod.rs             # Re-exports
│   │       │   ├── collector.rs       # Core metrics collection
│   │       │   ├── enhanced.rs        # EnhancedMetricsCollector
│   │       │   ├── exporter.rs        # Prometheus exporter
│   │       │   ├── legacy.rs          # Legacy metrics support
│   │       │   └── types.rs           # MetricsSnapshot, metric types
│   │       ├── speculative/
│   │       │   ├── mod.rs             # Re-exports
│   │       │   ├── adaptive.rs        # AdaptiveSpeculativeDecoder (draft count tuning)
│   │       │   ├── config.rs          # SpeculationConfig + builder
│   │       │   ├── model.rs           # SpeculativeModel trait
│   │       │   ├── self_spec.rs       # Self-speculation (model predicts own tokens)
│   │       │   ├── strategy.rs        # RejectionStrategy for draft verification
│   │       │   └── verifier.rs        # DraftVerifier + VerificationResult
│   │       ├── circuit_breaker/
│   │       │   ├── mod.rs             # Re-exports
│   │       │   ├── breaker.rs         # Circuit breaker implementation
│   │       │   └── strategy.rs        # Breaker strategies
│   │       ├── ha/
│   │       │   ├── mod.rs             # Re-exports
│   │       │   ├── failover.rs        # FailoverManager
│   │       │   └── leader_election.rs # LeaderElection for HA
│   │       ├── routing/
│   │       │   ├── mod.rs             # Re-exports
│   │       │   └── hash_router.rs     # Hash-based request router
│   │       ├── beam.rs                # Beam search decoding
│   │       ├── sampling.rs            # Token sampling (top-k, top-p, temperature)
│   │       ├── health.rs              # Engine health tracking
│   │       └── tensor_parallel.rs     # TP support (re-exports from vllm-dist)
│   ├── model/                       # vllm-model: Model implementations + components
│   │   └── src/
│   │       ├── lib.rs                # Re-exports: arch, kernels, loader, quantize
│   │       ├── arch/
│   │       │   ├── mod.rs            # Architecture trait definition
│   │       │   └── registry.rs       # ArchitectureRegistry + ARCHITECTURE_REGISTRY global
│   │       ├── components/
│   │       │   ├── mod.rs            # Re-exports: attention, mlp, norm, positional, ssm, vision
│   │       │   ├── block.rs          # StandardBlock + TransformerBlock trait
│   │       │   ├── attention/
│   │       │   │   ├── mod.rs        # AttentionConfig + utility functions (causal_mask, expand_kv, paged_attention, tiled_attention)
│   │       │   │   ├── gqa.rs        # GqaAttention: Grouped-Query Attention
│   │       │   │   ├── mla.rs        # MlaAttention: Multi-head Latent Attention
│   │       │   │   ├── flash.rs      # FlashAttention kernel (v1/v2)
│   │       │   │   └── flash_v3.rs   # FlashAttentionV3 kernel
│   │       │   ├── mlp/
│   │       │   │   ├── mod.rs        # Re-exports
│   │       │   │   └── swiglu.rs     # SwiGLU feed-forward layer
│   │       │   ├── norm/
│   │       │   │   ├── mod.rs        # Re-exports: layer_norm, rms_norm
│   │       │   │   ├── rms_norm.rs   # RMSNorm implementation
│   │       │   │   └── layer_norm.rs # LayerNorm implementation
│   │       │   ├── positional/
│   │       │   │   ├── mod.rs        # Re-exports: RoPE, MRoPE, apply_rope
│   │       │   │   ├── rope.rs       # Standard Rotary Position Embedding
│   │       │   │   └── mrope.rs      # MRoPE (Qwen3.5 hybrid models)
│   │       │   ├── ssm.rs            # SSMLayer, MambaBlock, SSMHarmonicSSMLayer
│   │       │   ├── vision.rs         # VisionEncoder (placeholder)
│   │       │   └── kv_cache_fp8.rs   # FP8 KV cache compression
│   │       ├── llama/                # Llama architecture
│   │       │   ├── mod.rs, arch.rs, block.rs, model.rs, register.rs
│   │       ├── mistral/              # Mistral architecture
│   │       │   ├── mod.rs, arch.rs, block.rs, model.rs, register.rs
│   │       ├── qwen3/                # Qwen2/3 architecture
│   │       │   ├── mod.rs, arch.rs, attention.rs, block.rs, mla_attention.rs, model.rs, register.rs
│   │       ├── qwen3_5/              # Qwen3.5 Mamba SSM Hybrid
│   │       │   ├── mod.rs, arch.rs, hybrid.rs, model.rs, register.rs, ssm.rs
│   │       ├── gemma3/               # Gemma3 architecture
│   │       ├── gemma4/               # Gemma4 (Hybrid Attention)
│   │       ├── llama4/               # Llama4 architecture
│   │       ├── mistral_small/        # Mistral Small architecture
│   │       ├── mixtral/              # Mixtral (Sparse MoE)
│   │       ├── phi4/                 # Phi-4 architecture
│   │       ├── config/
│   │       │   ├── mod.rs            # Re-exports
│   │       │   ├── model_config.rs   # ModelConfig struct
│   │       │   └── architecture.rs   # Architecture enum
│   │       ├── loader/
│   │       │   ├── mod.rs            # Re-exports: ModelLoader, ModelLoaderBuilder
│   │       │   ├── builder.rs        # ModelLoaderBuilder (334 lines)
│   │       │   ├── checkpoint.rs     # Checkpoint loading logic
│   │       │   ├── format.rs         # FormatLoader trait + SafetensorsLoader
│   │       │   └── io.rs             # I/O utilities for weight loading
│   │       ├── paged_tensor/
│   │       │   ├── mod.rs            # Re-exports
│   │       │   ├── tensor_store.rs   # Physical KV cache tensor storage
│   │       │   ├── quant.rs          # Quantized tensor types
│   │       │   └── quantization.rs   # Quantization utilities
│   │       ├── kernels/
│   │       │   ├── mod.rs            # Re-exports: FlashAttention, CudaGraph, fused ops
│   │       │   ├── flash_attention.rs # FlashAttention kernel config
│   │       │   ├── fused_mlp.rs      # Fused MLP kernel
│   │       │   ├── cuda_graph.rs     # CUDA graph capture/replay
│   │       │   └── cuda_graph/       # CUDA graph sub-modules
│   │       ├── quantize/
│   │       │   ├── mod.rs            # Re-exports: QuantizationConfig, StorageTensor, etc.
│   │       │   ├── types.rs          # QuantizationFormat enum
│   │       │   └── gguf.rs           # GGUF Q4_K_M loading/dequantization
│   │       ├── tokenizer.rs          # Tokenizer (tiktoken + tokenizers backends)
│   │       ├── qwen3_config.rs       # Qwen3-specific config helpers
│   │       └── kv_cache.rs           # Model-side KV cache helpers
│   ├── server/                       # vllm-server: HTTP API + CLI
│   │   └── src/
│   │       ├── main.rs               # Binary entry point (#[tokio::main] async)
│   │       ├── lib.rs                # Library crate root + ApiState
│   │       ├── cli.rs                # Clap CLI argument parsing (528 lines)
│   │       ├── config.rs             # AppConfig struct + validation
│   │       ├── api.rs                # EngineHandle, health, shutdown, metrics handlers
│   │       ├── auth.rs               # AuthMiddleware (API key auth + rate limiting)
│   │       ├── health.rs             # HealthChecker (liveness/readiness probes)
│   │       ├── logging.rs            # Tracing/logging initialization
│   │       ├── backpressure.rs       # Backpressure manager (buffer limits)
│   │       ├── debug.rs              # Debug endpoints (/debug/metrics, /debug/kv-cache)
│   │       ├── security/             # Security utilities
│   │       ├── openai/
│   │       │   ├── mod.rs            # Module declarations
│   │       │   ├── chat.rs           # Chat completions (SSE streaming, prompt building, 485 lines)
│   │       │   ├── completions.rs    # Text completions endpoint
│   │       │   ├── embeddings.rs     # Embeddings endpoint
│   │       │   ├── models.rs         # /v1/models listing endpoint
│   │       │   ├── types.rs          # OpenAI API types (Usage, ErrorResponse, ChatMessage, etc.)
│   │       │   └── batch/
│   │       │       ├── mod.rs, handler.rs, manager.rs, types.rs
│   │       └── bin/
│   │           └── vllm.rs           # Lightweight CLI binary (no HTTP server)
│   ├── dist/                         # vllm-dist: Tensor/pipeline parallelism + distributed KV
│   │   └── src/
│   │       ├── lib.rs                # Re-exports
│   │       ├── types.rs              # TensorParallelConfig
│   │       ├── grpc.rs               # gRPC service definitions
│   │       ├── generated/            # Prost-generated protobuf code
│   │       ├── tensor_parallel/
│   │       │   ├── mod.rs            # Re-exports
│   │       │   ├── all_reduce.rs     # AllReduce + NcclAllReduce
│   │       │   ├── device_mesh.rs    # DeviceMesh / NodeMesh
│   │       │   └── parallel_linear.rs # ColumnParallelLinear, RowParallelLinear, TensorParallelManager
│   │       ├── pipeline/
│   │       │   ├── mod.rs            # Re-exports
│   │       │   ├── pipeline.rs       # PipelineParallel executor
│   │       │   └── stage.rs          # PipelineStage trait + StageInput/StageOutput
│   │       └── distributed_kv/
│   │           ├── mod.rs            # Re-exports
│   │           ├── cache.rs          # DistributedKVCache
│   │           └── protocol.rs       # Cache protocol messages
│   └── testing/                      # vllm-testing: Shared test infrastructure
│       └── src/
│           ├── lib.rs                # Re-exports + prelude module
│           ├── harness.rs            # TestHarness (scheduler + metrics setup)
│           ├── mocks/                # Mock models (FakeModel, StubModel, ConstModel, etc.)
│           ├── request_factory.rs    # RequestFactory for generating test requests
│           ├── slow_model.rs         # SlowModel (artificially slow for timeout tests)
│           ├── builders/             # Test builders
│           ├── fixtures/             # Test data fixtures
│           └── utils/                # Test utilities
├── benches/                          # Benchmark suite
│   ├── Cargo.toml                    # vllm-lite-benchmarks crate
│   ├── src/
│   │   ├── lib.rs                    # Benchmark library
│   │   └── bin/benchmark.rs          # Benchmark binary
│   ├── integration.rs                # Integration benchmarks
│   ├── attention.rs                  # Attention benchmarks
│   ├── scheduler.rs                  # Scheduler benchmarks
│   └── speculative.rs                # Speculative decoding benchmarks
├── config/
│   └── prometheus.yml                # Prometheus scrape config
├── docs/                             # Documentation
├── tests/                            # Integration tests (currently empty)
├── k8s/                              # Kubernetes deployment manifests
├── scripts/                          # Utility scripts
├── models/                           # Model storage (empty — populated at runtime)
├── .github/                          # GitHub Actions CI workflows
└── docker-compose.yml                # Docker Compose for local deployment
```

## Directory Purposes

**`crates/traits/`:**

- Purpose: Define the `ModelBackend` trait and shared types that all crates depend on
- Contains: 4 source files — trait definition, type aliases, batch/output structs, CUDA graph config traits
- Key files: `lib.rs`, `model.rs`, `types.rs`, `kernels.rs`

**`crates/core/`:**

- Purpose: Inference engine, request scheduling, KV cache memory management, token generation loop
- Contains: 16 top-level modules + `engine/` sub-module; the scheduler alone has 18 files across 10 sub-directories
- Key files: `engine.rs`, `scheduler/engine.rs`, `scheduler/request_queue.rs`, `scheduler/batch_composer.rs`, `scheduler/memory/allocator.rs`, `speculative/mod.rs`

**`crates/model/`:**

- Purpose: Model implementations, architecture registry, shared transformer components, GPU kernels, KV cache tensor storage, tokenization
- Contains: 21 top-level modules including 10 per-architecture modules (`llama/`, `mistral/`, etc.), 5 shared component modules (`attention/`, `mlp/`, `norm/`, `positional/`, `ssm.rs`), plus loader, kernels, paged tensor, quantize
- Key files: `arch/registry.rs`, `loader/builder.rs`, `components/block.rs`, `components/attention/gqa.rs`, `components/attention/mla.rs`, `components/ssm.rs`, `tokenizer.rs`

**`crates/server/`:**

- Purpose: HTTP API server exposing OpenAI-compatible endpoints
- Contains: 13 top-level modules + `bin/` directory; the `openai/` sub-module contains 7 files
- Key files: `main.rs`, `lib.rs`, `cli.rs`, `openai/chat.rs`, `openai/types.rs`, `api.rs`

**`crates/dist/`:**

- Purpose: Multi-GPU and multi-node distributed inference support
- Contains: 7 top-level modules; `tensor_parallel/` (4 files), `pipeline/` (3 files), `distributed_kv/` (3 files)
- Key files: `lib.rs`, `grpc.rs`, `tensor_parallel/parallel_linear.rs`

**`crates/testing/`:**

- Purpose: Reusable test infrastructure consumed as dev-dependency by all other crates
- Contains: 8 top-level modules; mock models, test harness, request factory, slow model
- Key files: `harness.rs`, `mocks/`, `request_factory.rs`

**`benches/`:**

- Purpose: Criterion benchmarks for scheduler, attention, speculative decoding, and integration scenarios
- Contains: 4 benchmark files + binary entry point
- Key files: `integration.rs`, `scheduler.rs`, `attention.rs`, `speculative.rs`

## Key File Locations

**Entry Points:**

- `crates/server/src/main.rs:91` — Main server binary (`#[tokio::main] async fn main()`)
- `crates/server/src/bin/vllm.rs` — Lightweight CLI binary
- `crates/core/src/engine.rs:361` — Engine actor loop (`Engine::run()`)
- `crates/model/src/arch/registry.rs:64` — Lazy global `ARCHITECTURE_REGISTRY`

**Configuration:**

- `Cargo.toml` — Workspace root, version=0.1.0, edition=2024, rust-version=1.85
- `crates/server/src/cli.rs` — Clap CLI arg parsing (~528 lines)
- `crates/server/src/config.rs` — `AppConfig` struct + validation
- `crates/core/src/types.rs:182` — `SchedulerConfig` (default and builder)
- `justfile` — Build automation (build, test, ci, bench, clean, fmt-check, clippy)

**Core Logic:**

- `crates/core/src/engine.rs` — `Engine<M: ModelBackend>` struct + `run()` loop (930 lines)
- `crates/core/src/scheduler/engine.rs` — `SchedulerEngine` (771 lines)
- `crates/core/src/engine/speculative.rs` — Speculative step dispatch (800 lines)
- `crates/core/src/scheduler/batch_composer.rs` — Batch assembly logic
- `crates/core/src/scheduler/memory/allocator.rs` — Block allocation with free list
- `crates/core/src/scheduler/radix_cache/` — Radix tree prefix matching

**Protocol/Trait Definitions:**

- `crates/traits/src/model.rs` — `ModelBackend` trait (128 lines)
- `crates/model/src/arch/mod.rs` — `Architecture` trait (41 lines, plus tests)
- `crates/model/src/components/block.rs` — `TransformerBlock` trait (line 135) + `StandardBlock` (536 lines)
- `crates/core/src/scheduler/policy/trait_def.rs` — `SchedulingPolicy` trait (19 lines)

**OpenAI API Surface:**

- `crates/server/src/openai/chat.rs` — Chat completions with SSE streaming (485 lines)
- `crates/server/src/openai/completions.rs` — Text completions
- `crates/server/src/openai/embeddings.rs` — Embedding endpoint
- `crates/server/src/openai/batch/handler.rs` — Batch API CRUD handlers
- `crates/server/src/openai/types.rs` — OpenAI-format types (Usage, ErrorResponse, ChatMessage; 227 lines)

**Testing Infrastructure:**

- `crates/testing/src/harness.rs` — `TestHarness` and `TestHarnessConfig` (215 lines)
- `crates/testing/src/mocks/` — Mock model implementations
- `crates/testing/src/request_factory.rs` — Test request generation

**Benchmarks:**

- `benches/scheduler.rs` — Scheduler benchmarks
- `benches/attention.rs` — Attention benchmarks
- `benches/speculative.rs` — Speculative decoding benchmarks
- `crates/core/benches/scheduler_benchmarks.rs` — Core scheduler benchmarks (criterion)
- `crates/core/benches/prefix_cache_benchmarks.rs` — Prefix cache benchmarks (criterion)

## Naming Conventions

**Files:**

- `snake_case.rs` — All source files (`batch_composer.rs`, `request_queue.rs`, `phase_scheduler.rs`)
- `mod.rs` — Module directory roots (`scheduler/mod.rs`, `attention/mod.rs`)
- `trait_def.rs` — Trait-only files when separated from implementations (`policy/trait_def.rs`)

**Directories:**

- `snake_case/` — All directory names (`scheduler/`, `kv_cache/`, `paged_tensor/`, `radix_cache/`)
- Per-architecture directories: `llama/`, `mistral/`, `qwen3/`, `qwen3_5/`, `gemma3/`, `gemma4/`, `llama4/`, `mixtral/`, `mistral_small/`, `phi4/`

**Crates:**

- `kebab-case` — Crate names (`vllm-core`, `vllm-model`, `vllm-server`, `vllm-traits`, `vllm-dist`, `vllm-testing`)
- `vllm-lite-benchmarks` — Benches crate

**Per-architecture module pattern:**
Each architecture directory contains the same 5-file layout:

```text
{arch}/
├── mod.rs         # Module declaration + re-exports
├── arch.rs        # Architecture trait implementation (detect + create_model)
├── block.rs       # TransformerBlock trait implementation
├── model.rs       # ModelBackend trait implementation
└── register.rs    # Registry registration function
```

## Where to Add New Code

**New Model Architecture (e.g., "Falcon"):**

- Primary code: `crates/model/src/falcon/` (5 files: `mod.rs`, `arch.rs`, `block.rs`, `model.rs`, `register.rs`)
- Registration: Call `crate::falcon::register::register(registry)` in `crates/model/src/arch/registry.rs:77`'s `register_all_archs()`
- Components: If needed, add new attention/norm/positional variants in `crates/model/src/components/`

**New Scheduler Policy (e.g., "RoundRobinPolicy"):**

- Implementation: `crates/core/src/scheduler/policy/round_robin.rs`
- Register: Export from `crates/core/src/scheduler/policy/mod.rs`
- Tests: `crates/core/src/scheduler/policy/tests.rs`

**New HTTP Endpoint:**

- Handler: `crates/server/src/openai/{endpoint}.rs`
- Route: Add to the `Router` builder in `crates/server/src/main.rs` (around line 230)
- Types: Add request/response types to `crates/server/src/openai/types.rs`

**New Scheduler Component:**

- Implementation: `crates/core/src/scheduler/{component}.rs`
- Integration: Wire into `SchedulerEngine::new()` in `crates/core/src/scheduler/engine.rs`

**Utilities / Shared Helpers:**

- Engine utilities: `crates/core/src/` (e.g., `beam.rs`, `sampling.rs`)
- Model utilities: `crates/model/src/components/` (for attention, norm, positional, etc.)
- Test utilities: `crates/testing/src/`

**Benchmarks:**

- Core benchmarks: `crates/core/Cargo.toml` (`[[bench]]` sections) or `benches/`
- New benchmark: Add `[[bench]]` entry to relevant `Cargo.toml` + create benchmark file

## Special Directories

**`crates/dist/src/generated/`:**

- Purpose: Prost-generated Rust code from protobuf definitions for gRPC services
- Generated: Yes (via `tonic-build` in `crates/dist/build.rs`)
- Committed: Yes (checked into version control)

**`config/`:**

- Purpose: Infrastructure configuration files (currently `prometheus.yml` for metrics scraping)
- Generated: No
- Committed: Yes

**`models/`:**

- Purpose: Runtime model storage directory (populated at deployment, not in repo)
- Generated: No
- Committed: No (empty directory in repo)

**`target/`:**

- Purpose: Cargo build output; generated at build time
- Generated: Yes
- Committed: No (in `.gitignore`)

**`.planning/`:**

- Purpose: GSD planning artifacts (codebase maps, implementation plans)
- Generated: Yes (by GSD commands)
- Committed: Yes

**`.rumdl_cache/`:**

- Purpose: Cached results for `rumdl` (Rust markdown linter)
- Generated: Yes
- Committed: No (in `.gitignore`)

---

*Structure analysis: 2026-05-13*
