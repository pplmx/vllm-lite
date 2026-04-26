# Codebase Structure

**Analysis Date:** 2026-04-26

## Directory Layout

```
vllm-lite/
├── Cargo.toml              # Workspace root (7 crates)
├── justfile                # Build automation (fmt, clippy, test, ci)
├── AGENTS.md               # Development guide for AI agents
│
├── crates/                 # Primary source crates
│   ├── traits/             # Core interfaces (no dependencies)
│   ├── core/               # Engine, scheduler, KV cache, metrics
│   ├── model/              # Model implementations, kernels, components
│   ├── server/             # HTTP API server (OpenAI-compatible)
│   ├── dist/               # Tensor parallel support
│   └── testing/            # Shared test utilities and mocks
│
├── benches/                # Benchmark suite
├── tests/                  # Integration tests (currently empty)
├── config/                 # Configuration files
├── docs/                   # Documentation
├── models/                 # Model storage
├── scripts/                # Build/dev scripts
├── k8s/                    # Kubernetes manifests
└── .planning/codebase/     # This document
```

## Crate Purposes

### `crates/traits` - Interface Definitions

**Purpose:** Core abstractions and shared types with zero dependencies
**Contains:**
- `model.rs` - `ModelBackend` trait, `ModelError`
- `types.rs` - `Batch`, `BatchOutput`, `SeqId`, `TokenId`, `BlockId`
- `kernels.rs` - CUDA graph configuration types

**Key Files:**
- `src/lib.rs` - Public re-exports

### `crates/core` - Inference Engine Core

**Purpose:** Request scheduling, batch management, KV cache, metrics
**Contains:**
- `engine/` - Engine implementation with speculative decoding
- `scheduler/` - Componentized scheduler (queue, batch, memory, cache)
- `kv_cache/` - Block allocation and prefix caching
- `metrics/` - Prometheus metrics collection
- `sampling.rs` - Sampling strategies

**Key Files:**
- `src/lib.rs` - Module declarations, public re-exports
- `src/types.rs` - `Request`, `Sequence`, `Status`, `SchedulerConfig`

### `crates/model` - ML Model Implementations

**Purpose:** Transformer model implementations, GPU kernels, components
**Contains:**
- `arch/` - Architecture registry and trait
- `llama/`, `mistral/`, `qwen3/`, `qwen3_5/`, `gemma4/`, `mixtral/` - Model architectures
- `components/` - Shared components (attention, mlp, norm, positional, ssm)
- `kernels/` - GPU kernels (flash attention, fused MLP, CUDA graph)
- `paged_tensor/` - Physical KV cache management
- `loader/` - Model loading from checkpoints
- `tokenizer.rs` - Tokenizer implementation

**Key Files:**
- `src/lib.rs` - Module declarations, public re-exports

### `crates/server` - HTTP API Server

**Purpose:** OpenAI-compatible REST API
**Contains:**
- `main.rs` - Entry point, Axum router setup
- `openai/` - OpenAI API implementations (chat, completions, embeddings, batches)
- `api.rs` - Generic API handlers (health, metrics, shutdown)
- `auth.rs` - API key authentication middleware
- `config.rs` - Server configuration
- `health.rs` - Health checker implementation

**Key Files:**
- `src/lib.rs` - `ApiState` struct

### `crates/dist` - Distributed Tensor Parallel

**Purpose:** Multi-GPU tensor parallel support
**Contains:**
- `tensor_parallel/` - All-reduce and collective operations
- `types.rs` - Tensor parallel error types

### `crates/testing` - Test Utilities

**Purpose:** Shared mocks and test utilities
**Contains:**
- `mocks/` - `FakeModel`, `StubModel`, `ConstModel`
- `builders/` - Test builders
- `fixtures/` - Test fixtures
- `utils/` - Test utilities

## Key Module Locations

### Entry Points

| Entry Point | Purpose | File |
|-------------|---------|------|
| Server start | HTTP API server | `crates/server/src/main.rs` |
| Engine loop | Inference orchestration | `crates/core/src/engine/speculative.rs` |
| Model loading | Checkpoint loading | `crates/model/src/loader/mod.rs` |

### Core Logic

| Component | Purpose | File |
|-----------|---------|------|
| SchedulerEngine | Request scheduling | `crates/core/src/scheduler/engine.rs` |
| ArchitectureRegistry | Model registration | `crates/model/src/arch/registry.rs` |
| ModelBackend | Model abstraction | `crates/traits/src/model.rs` |
| BatchComposer | Batch building | `crates/core/src/scheduler/batch_composer.rs` |

### Configuration

| Config | Purpose | File |
|--------|---------|------|
| SchedulerConfig | Batch/memory limits | `crates/core/src/types.rs:175` |
| AppConfig | Full server config | `crates/server/src/config.rs` |
| ModelConfig | Model architecture | `crates/model/src/config/model_config.rs` |

## Subdirectory Purposes

### Scheduler Subsystem (`crates/core/src/scheduler/`)

```
scheduler/
├── engine.rs              # SchedulerEngine main implementation
├── mod.rs                 # Module docs and re-exports
├── batch.rs               # Batch data structures
├── batch_composer.rs      # Phase-specific batch construction
├── batch_planner.rs       # Adaptive batch planning
├── cache/                 # KV cache integration
├── cuda_graph.rs          # CUDA graph configuration
├── memory/                # Block allocation and eviction
│   ├── allocator.rs       # Free list block allocation
│   └── eviction.rs        # LRU eviction policy
├── observer.rs            # Event observer system
├── packing/               # Sequence packing utilities
├── phase_scheduler.rs     # Prefill/decode phase selection
├── policy/                # Scheduling policies (FCFS, SJF, Priority)
├── preemption.rs          # Request preemption
├── radix_cache/           # Radix tree prefix cache
│   └── mod.rs             # RadixTree implementation
├── request_queue.rs       # O(1) request queue with phase indexing
└── stats.rs               # Scheduling statistics
```

### Model Components (`crates/model/src/components/`)

```
components/
├── attention/
│   ├── mod.rs             # GqaAttention, MlaAttention, utilities
│   ├── gqa.rs             # Grouped-query attention implementation
│   ├── mla.rs             # Multi-head Latent Attention (DeepSeek-V3)
│   └── flash.rs           # Flash attention kernel wrapper
├── mlp/
│   └── swiglu.rs          # SwiGLU feed-forward implementation
├── norm/
│   ├── rms_norm.rs        # RMSNorm implementation
│   └── layer_norm.rs      # LayerNorm implementation
├── positional/
│   ├── rope.rs            # Standard RoPE implementation
│   └── mrope.rs           # MRoPE (Qwen3.5) implementation
├── block.rs               # TransformerBlock trait
├── ssm.rs                 # SSMLayer, MambaBlock, SSMHarmonicSSMLayer
└── vision.rs              # VisionEncoder (placeholder)
```

### Model Architectures (`crates/model/src/`)

```
├── llama/
│   ├── arch.rs            # LlamaArchitecture implementation
│   ├── block.rs           # LlamaBlock implementation
│   ├── model.rs           # LlamaModel implementation
│   ├── register.rs        # Registration entry point
│   └── mod.rs             # Module re-exports
├── mistral/               # Mistral model implementation
├── qwen3/                 # Qwen2/3 model implementation
├── qwen3_5/               # Qwen3.5 hybrid attention+SSM
├── gemma4/                # Gemma4 model implementation
├── mixtral/               # Mixtral MoE implementation
```

### Server API (`crates/server/src/openai/`)

```
openai/
├── chat.rs                # /v1/chat/completions handler
├── completions.rs         # /v1/completions handler
├── embeddings.rs          # /v1/embeddings handler
├── models.rs              # /v1/models handler
├── types.rs               # OpenAI API type definitions
├── mod.rs                 # Module re-exports
└── batch/                 # Batch API implementation
    ├── handler.rs         # Batch CRUD endpoints
    └── manager.rs         # Batch lifecycle management
```

## Where to Add New Code

### New Model Architecture (3 steps)

1. **Create directory**: `crates/model/src/newarch/`
2. **Implement architecture**:
   - `arch.rs` - Implement `Architecture` trait
   - `block.rs` - Implement `TransformerBlock` trait
   - `model.rs` - Implement `ModelBackend` trait
   - `register.rs` - Registration function
   - `mod.rs` - Module exports

3. **Register architecture** in `crates/model/src/arch/registry.rs:register_all_archs()`:
```rust
pub fn register_all_archs(registry: &ArchitectureRegistry) {
    // ... existing registrations
    crate::newarch::register::register(registry);
}
```

### New Scheduling Policy

1. **Implement trait** in `crates/core/src/scheduler/policy/`:
   - Create `newpolicy.rs`
   - Implement `SchedulingPolicy` trait
   - Add module to `policy/mod.rs`

2. **Configure engine** to use policy:
```rust
scheduler_engine.set_policy(Box::new(NewPolicy::new()));
```

### New Attention Mechanism

1. **Implement kernel** in `crates/model/src/kernels/`
2. **Create attention wrapper** in `crates/model/src/components/attention/`
3. **Integrate** into model block implementations

### New Server Endpoint

1. **Add handler** in `crates/server/src/openai/` or `api.rs`
2. **Register route** in `crates/server/src/main.rs:app` router

### New GPU Kernel

1. **Add CUDA kernel** in `crates/model/src/kernels/cuda_graph/` or new subdirectory
2. **Export via** `crates/model/src/kernels/mod.rs`

## Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Crate names | kebab-case | `vllm-core`, `vllm-model` |
| Modules | snake_case | `request_queue`, `batch_composer` |
| Types/Structs | PascalCase | `SchedulerEngine`, `BatchComposer` |
| Functions | snake_case | `build_batch`, `add_request` |
| Constants | SCREAMING_SNAKE_CASE | `BLOCK_SIZE`, `MAX_BATCH_SIZE` |
| Enums | PascalCase | `Status`, `Phase` |
| Enum variants | PascalCase | `BatchPhase::Prefill` |
| Traits | PascalCase | `ModelBackend`, `TransformerBlock` |

## Special Directories

### `benches/`

**Purpose:** Cargo benchmark suite
**Location:** Project root
**Contains:** Criterion benchmarks
**Generated:** `target/` directory (not committed)

### `tests/`

**Purpose:** Integration tests
**Location:** Project root
**Contains:** Currently empty (integration tests in crate-level test modules)
**Note:** Unit tests colocated in `#[cfg(test)]` modules

### `.planning/`

**Purpose:** Architecture planning documents
**Location:** Project root
**Contains:** `codebase/` subdirectory with architecture documents
**Generated:** By `/gsd-map-codebase` command

### `config/`

**Purpose:** Configuration file templates
**Generated:** No
**Contains:** Configuration examples and defaults

---

*Structure analysis: 2026-04-26*
