# Project Structure

**Last updated:** 2026-05-09
**Focus:** Architecture

## Top-Level Layout

```
vllm-lite/
├── Cargo.toml              # Workspace root (7 members)
├── justfile                # Task automation
├── AGENTS.md               # Development guide
├── .planning/              # Project planning artifacts
│   ├── codebase/           # Codebase mapping docs (this directory)
│   ├── milestones/         # Milestone definitions
│   ├── phases/             # Phase plans
│   ├── config.json         # GSD config
│   ├── PROJECT.md          # Project overview
│   ├── ROADMAP.md          # Development roadmap
│   ├── REQUIREMENTS.md     # Requirements spec
│   ├── STATE.md            # Project state tracking
│   └── RETROSPECTIVE.md    # Retrospective notes
├── crates/
│   ├── traits/             # Interface definitions (no deps)
│   ├── core/               # Engine + Scheduler
│   ├── model/              # Model architectures + components
│   ├── server/             # HTTP API server
│   ├── dist/               # Distributed inference
│   ├── testing/            # Test utilities + mocks
│   └── benches/            # Benchmarks
├── tests/                  # Integration tests (top-level)
└── images/                 # Architecture diagrams
```

## Crate Structure

### `crates/traits/` (vllm-traits)
Core trait definitions with minimal dependencies:
```
src/
├── lib.rs          # Re-exports
├── model.rs        # ModelBackend trait, ModelError
├── types.rs        # Batch, BatchOutput, SeqId, TokenId, BlockId, BatchPhase
└── kernels.rs      # CudaGraphConfig, GraphExecutionError
tests/
├── mod.rs
└── model_backend.rs
```

### `crates/core/` (vllm-core)
Inference engine and scheduler:
```
src/
├── lib.rs                     # Module exports
├── engine.rs                  # Main Engine (actor loop, step, beam search)
├── types.rs                   # Request, Sequence, EngineMessage, SchedulerConfig
├── sampling.rs                # Token sampling strategies
├── beam.rs                    # Beam search
├── tensor_parallel.rs         # Tensor parallel integration
├── health.rs                  # Health check
├── circuit_breaker/
│   ├── mod.rs
│   ├── breaker.rs             # Circuit breaker pattern
│   ├── strategy.rs            # Configurable strategies
├── engine/
│   └── speculative.rs         # Speculative decoding in engine
├── error/
│   ├── mod.rs                 # EngineError enum
│   └── recovery.rs            # Error recovery
├── ha/
│   ├── mod.rs
│   ├── failover.rs            # HA failover
│   └── leader_election.rs     # Leader election
├── kv_cache/
│   ├── mod.rs
│   └── prefix_cache.rs        # Prefix cache helper
├── metrics/
│   ├── mod.rs
│   ├── collector.rs           # MetricsCollector
│   ├── enhanced.rs            # EnhancedMetricsCollector
│   ├── exporter.rs            # Metrics export
│   ├── legacy.rs              # Legacy metrics
│   └── types.rs               # Metrics types
├── routing/
│   ├── mod.rs
│   └── hash_router.rs         # Request routing
├── scheduler/
│   ├── mod.rs                 # Module exports
│   ├── engine.rs              # SchedulerEngine (componentized)
│   ├── request_queue.rs       # Phase-aware request queue
│   ├── phase_scheduler.rs     # Prefill/decode phase mgmt
│   ├── batch_composer.rs      # Batch construction
│   ├── batch.rs               # Batch types
│   ├── batch_planner.rs       # Batch planning
│   ├── predictive_batching.rs # Predictive batching
│   ├── preemption.rs          # Request preemption
│   ├── packing.rs             # Token packing
│   ├── cuda_graph.rs          # CUDA graph batch types
│   ├── observer.rs            # Scheduler observers
│   ├── stats.rs               # Scheduler stats
│   ├── memory/
│   │   ├── mod.rs
│   │   ├── allocator.rs       # Block allocator
│   │   └── eviction.rs        # Block eviction
│   ├── cache/
│   │   ├── mod.rs
│   │   └── prefix_cache.rs    # Prefix cache (legacy)
│   ├── poolicy/
│   │   ├── mod.rs
│   │   ├── trait_def.rs       # SchedulingPolicy trait
│   │   ├── fcfs.rs            # First-come first-served
│   │   ├── sjf.rs             # Shortest job first
│   │   ├── priority.rs        # Priority scheduling
│   │   └── tests.rs           # Policy tests
│   └── radix_cache/
│       ├── mod.rs
│       ├── node.rs            # Radix tree node
│       └── tree.rs            # Radix tree implementation
└── speculative/
    ├── mod.rs
    ├── adaptive.rs            # Adaptive speculative decoding
    ├── config.rs              # Speculation config
    ├── model.rs               # SpeculativeModel
    ├── self_spec.rs           # Self-speculative decoding
    ├── strategy.rs            # Rejection strategies
    └── verifier.rs            # Draft verification
benches/                       # Core benchmarks
tests/                         # Integration tests (16 files)
```

### `crates/model/` (vllm-model)
Model architectures and neural network components:
```
src/
├── lib.rs                     # Module exports
├── qwen3_config.rs            # Qwen3 config parsing
├── tokenizer.rs               # Tokenizer wrapper (tiktoken + tokenizers)
├── kv_cache.rs                # KV cache struct
├── arch/
│   ├── mod.rs                 # Architecture trait
│   └── registry.rs            # ArchitectureRegistry + register_all_archs
├── config/
│   └── architecture.rs        # Architecture config detection
├── components/
│   ├── mod.rs
│   ├── block.rs               # TransformerBlock trait
│   ├── ssm.rs                 # SSMLayer, MambaBlock, HarmonicSSM
│   ├── vision.rs              # Vision encoder (placeholder)
│   ├── kv_cache_fp8.rs        # FP8 KV cache
│   ├── attention/
│   │   ├── mod.rs             # Module + utility functions
│   │   ├── gqa.rs             # GQA attention (725 lines)
│   │   ├── mla.rs             # MLA attention (657 lines)
│   │   ├── flash.rs           # Flash attention
│   │   └── flash_v3.rs        # Flash attention v3
│   ├── mlp/
│   │   ├── mod.rs
│   │   └── swiglu.rs          # SwiGLU MLP
│   ├── norm/
│   │   ├── mod.rs
│   │   ├── rms_norm.rs        # RMSNorm
│   │   └── layer_norm.rs      # LayerNorm
│   └── positional/
│       ├── mod.rs
│       ├── rope.rs            # Rotary Position Embedding
│       └── mrope.rs           # MRoPE (Qwen3.5)
├── kernels/
│   ├── mod.rs
│   ├── flash_attention.rs     # Flash attention kernel
│   ├── fused_mlp.rs           # Fused MLP kernel
│   ├── cuda_graph.rs          # CUDA graph kernel
│   └── cuda_graph/
│       ├── config.rs          # CUDA graph config
│       └── executor.rs        # CUDA graph executor
├── loader/
│   ├── mod.rs
│   ├── builder.rs             # ModelLoaderBuilder
│   ├── checkpoint.rs          # Checkpoint loading
│   ├── format.rs              # Format detection
│   └── io.rs                  # File I/O
├── paged_tensor/
│   ├── mod.rs
│   ├── tensor_store.rs        # KV tensor store
│   ├── quantization.rs        # Quantization schemes
│   └── quant.rs               # Quantized tensor types
├── quantize/
│   ├── mod.rs
│   ├── types.rs               # QuantizationFormat, StorageTensor
│   └── gguf.rs                # GGUF quantization support
├── llama/
│   ├── mod.rs, arch.rs, block.rs, model.rs, register.rs
├── mistral/
│   ├── mod.rs, arch.rs, block.rs, model.rs, register.rs
├── qwen3/
│   ├── mod.rs, arch.rs, attention.rs, mla_attention.rs, block.rs, model.rs, register.rs
├── qwen3_5/
│   ├── mod.rs, arch.rs, hybrid.rs, model.rs, ssm.rs, register.rs
├── gemma3/
│   ├── (mod.rs, arch.rs, model.rs, register.rs)
├── gemma4/
│   ├── mod.rs, arch.rs, attention.rs, block.rs, mlp.rs, model.rs, register.rs, rope.rs
├── llama4/
│   ├── mod.rs, arch.rs, register.rs
├── mistral_small/
│   ├── mod.rs, arch.rs, register.rs
├── mixtral/
│   ├── mod.rs, arch.rs, block.rs, model.rs, register.rs, sparse_moe.rs
├── phi4/
│   ├── mod.rs, arch.rs, register.rs
tests/                         # Model tests (11 files)
benches/
└── attention.rs               # Attention benchmarks
```

### `crates/server/` (vllm-server)
HTTP API server:
```
src/
├── main.rs                    # Entry point
├── lib.rs                     # ApiState, module re-exports
├── api.rs                     # Health, metrics, shutdown endpoints
├── cli.rs                     # CLI arg parsing (clap)
├── config.rs                  # Server configuration
├── auth.rs                    # Auth middleware
├── backpressure.rs            # Backpressure management
├── debug.rs                   # Debug utilities
├── health.rs                  # Health check types
├── logging.rs                 # Logging configuration
├── bin/
│   └── vllm.rs                # Alternative binary entry
├── openai/
│   ├── mod.rs
│   ├── chat.rs                # Chat completions (SSE streaming)
│   ├── completions.rs         # Text completions
│   ├── embeddings.rs          # Embeddings
│   ├── models.rs              # Model listing
│   ├── types.rs               # OpenAI API types
│   └── batch/
│       ├── mod.rs
│       ├── handler.rs         # Batch request handler
│       ├── manager.rs         # Batch manager
│       └── types.rs           # Batch types
├── security/
│   ├── mod.rs
│   ├── audit.rs               # Audit logging
│   ├── correlation.rs         # Correlation IDs
│   ├── jwt.rs                 # JWT validation
│   ├── rbac.rs                # RBAC
│   └── tls.rs                 # TLS listener
└── tests/
    └── models_handler_test.rs
```

### `crates/dist/` (vllm-dist)
Distributed inference:
```
src/
├── lib.rs                     # Re-exports
├── grpc.rs                    # gRPC state management
├── types.rs                   # TensorParallelConfig
├── generated/
│   └── vllm.distributed.rs    # Generated protobuf code
├── distributed_kv/
│   ├── mod.rs
│   ├── cache.rs               # Distributed KVCache
│   └── protocol.rs            # Cache protocol messages
├── pipeline/
│   ├── mod.rs
│   ├── pipeline.rs            # Pipeline parallel
│   └── stage.rs               # Pipeline stage
├── tensor_parallel/
│   ├── mod.rs
│   ├── device_mesh.rs         # Device mesh topology
│   ├── parallel_linear.rs     # Column/Row parallel linear
│   └── all_reduce.rs          # All-reduce (NCCL stub)
└── build.rs                   # tonic-build protobuf compilation
```

### `crates/testing/` (vllm-testing)
Test utilities:
```
src/
├── lib.rs
├── harness.rs                 # Test harness
├── request_factory.rs         # Request builder for tests
├── slow_model.rs              # Slow model simulation
├── builders/
│   └── mod.rs                 # Builder utilities
├── fixtures/
│   └── mod.rs                 # Test fixtures
├── mocks/
│   └── mod.rs                 # Mock implementations
└── utils/
    └── mod.rs                 # Test utilities
```

## Naming Conventions

- **Crates**: kebab-case (`vllm-core`, `vllm-model`)
- **Modules**: snake_case (`scheduler/engine.rs`, `attention/gqa.rs`)
- **Types**: PascalCase (`SchedulerEngine`, `GqaAttention`)
- **Functions**: snake_case (`add_request`, `build_batch`)
- **Test files**: snake_case (`scheduler_integration.rs`, `e2e_concurrent.rs`)
- **Arch directories**: flat structure per model (`llama/`, `qwen3/`, `mixtral/`)
