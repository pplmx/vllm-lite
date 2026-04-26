# Architecture

**Analysis Date:** 2026-04-26

## System Overview

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                          HTTP Server (Axum)                                  │
│                      `crates/server/src/main.rs`                             │
├─────────────────────────────────┬───────────────────────────────────────────┤
│   OpenAI API Layer              │   Health/Metrics                          │
│   ├── chat/completions          │   ├── /health                             │
│   ├── completions               │   ├── /ready                              │
│   ├── embeddings                │   ├── /metrics                            │
│   └── batches                   │   └── /shutdown                           │
└─────────────────────┬───────────┴───────────────────────────────────────────┘
                      │ mpsc channel
                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Engine (`crates/core/src/engine/`)                    │
│  ┌──────────────────┬──────────────────┬──────────────────────────────────┐ │
│  │ SchedulerEngine  │   MetricsCollector  │    Speculative Decoder        │ │
│  │ scheduling/      │   metrics/        │    speculative.rs              │ │
│  │ engine.rs        │                   │    (draft + target model)       │ │
│  └────────┬─────────┴──────────────────┴──────────────────────────────────┘ │
└─────────────────────┬───────────────────────────────────────────────────────┘
                      │ ModelBackend trait
                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Model Implementations (`crates/model/`)                  │
│  ┌────────────────────┬─────────────────────┬─────────────────────────────┐ │
│  │ Architecture       │  Shared Components  │  GPU Kernels                │ │
│  │ Registry           │  ├── attention/     │  ├── flash_attention.rs      │ │
│  │ arch/registry.rs   │  │   gqa.rs         │  ├── fused_mlp.rs            │ │
│  │                    │  │   mla.rs         │  └── cuda_graph.rs           │ │
│  │ ├── llama/         │  ├── mlp/           │                              │ │
│  │ ├── mistral/       │  │   swiglu.rs      │  Paged KV Cache              │ │
│  │ ├── qwen3/         │  ├── norm/          │  paged_tensor/               │ │
│  │ ├── qwen3_5/       │  │   rms_norm.rs    │  └── kv_cache.rs             │ │
│  │ ├── gemma4/        │  ├── positional/    │                              │ │
│  │ └── mixtral/       │  │   rope.rs        │  Model Loader                │ │
│  │                    │  └── ssm.rs         │  loader/                     │ │
│  └────────────────────┴─────────────────────┴─────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

| Component | Responsibility | Location |
|-----------|----------------|----------|
| `SchedulerEngine` | Request queuing, batch building, memory allocation | `crates/core/src/scheduler/engine.rs` |
| `RequestQueue` | O(1) lookup/removal with phase-aware indexing | `crates/core/src/scheduler/request_queue.rs` |
| `PhaseScheduler` | Prefill/Decode phase separation | `crates/core/src/scheduler/phase_scheduler.rs` |
| `BatchComposer` | Phase-specific batch construction | `crates/core/src/scheduler/batch_composer.rs` |
| `MemoryManager` | Block allocation and eviction | `crates/core/src/scheduler/memory/` |
| `RadixTree` | Prefix caching for prompt reuse | `crates/core/src/scheduler/radix_cache/` |
| `ModelBackend` | ML model inference abstraction | `crates/traits/src/model.rs` |
| `ArchitectureRegistry` | Dynamic model architecture registration | `crates/model/src/arch/registry.rs` |
| `Engine` | Main inference loop orchestration | `crates/core/src/engine/speculative.rs` |

## Pattern Overview

**Overall:** Dynamic Registry + Trait-Based Plugin Architecture with Componentized Scheduler

### Key Characteristics

- **Trait-based abstraction**: `ModelBackend` trait in `crates/traits/` allows pluggable model implementations
- **Dynamic registration**: `ArchitectureRegistry` enables adding new architectures without modifying core code
- **Componentized scheduler**: SchedulerEngine composes multiple specialized components (RequestQueue, PhaseScheduler, BatchComposer, MemoryManager, RadixTree)
- **Channel-based IPC**: Async message passing between HTTP server and inference engine via tokio mpsc

## Layers

### HTTP Layer (Server)

**Purpose:** OpenAI-compatible REST API endpoint
- **Location:** `crates/server/src/`
- **Contains:** Axum router, OpenAI API handlers, auth middleware
- **Depends on:** vllm-core (Engine), vllm-model (Tokenizer)
- **Used by:** External HTTP clients

### Core Engine Layer

**Purpose:** Request scheduling, batch management, inference orchestration
- **Location:** `crates/core/src/`
- **Contains:** SchedulerEngine, MemoryManager, MetricsCollector, KV Cache
- **Depends on:** vllm-traits (types, ModelBackend trait)
- **Used by:** Server (via mpsc channel), vllm-model (for forward passes)

### Model Layer

**Purpose:** ML model implementations and GPU kernels
- **Location:** `crates/model/src/`
- **Contains:** Architecture implementations (Llama, Mistral, Qwen, etc.), attention, MLP, normalization, positional encoding, SSM components
- **Depends on:** vllm-traits, candle (ML framework)
- **Used by:** vllm-core (Engine)

### Traits Layer

**Purpose:** Core interfaces and shared types
- **Location:** `crates/traits/src/`
- **Contains:** ModelBackend trait, Batch/SeqId/TokenId types, BlockSize constant
- **Depends on:** None (no internal crate dependencies)
- **Used by:** All crates

### Tensor Parallel Layer (Distributed)

**Purpose:** Multi-GPU model parallelism
- **Location:** `crates/dist/src/`
- **Contains:** Tensor parallel utilities, all-reduce operations
- **Depends on:** vllm-traits
- **Used by:** vllm-core (optional feature)

## Data Flow

### Primary Request Path

1. **HTTP Request** → `crates/server/src/openai/chat.rs:chat_completions()`
   - Parse OpenAI chat completion request
   - Encode text to tokens via Tokenizer

2. **Message Send** → `crates/server/src/main.rs:ApiState.engine_tx`
   - Send `EngineMessage::AddRequest` via mpsc channel

3. **Engine Loop** → `crates/core/src/engine/speculative.rs:Engine::run()`
   - Receive message, add to scheduler via `SchedulerEngine::add_request()`

4. **Scheduling** → `crates/core/src/scheduler/engine.rs:SchedulerEngine::build_batch()`
   - PhaseScheduler selects Prefill or Decode
   - BatchComposer builds batch respecting memory/token limits
   - Prefix cache checked for prompt reuse

5. **Model Forward** → `crates/traits/src/model.rs:ModelBackend::forward()`
   - LlamaModel/MistralModel/etc. performs actual inference
   - KV cache updated via PagedKvCache

6. **Scheduler Update** → `crates/core/src/scheduler/engine.rs:SchedulerEngine::update()`
   - Process output tokens, update sequence status
   - Allocate/release KV blocks
   - Check completion, add finished prompts to prefix cache

7. **Response** → Token sent via `response_tx` channel back to client

### Speculative Decoding Path

1. **Draft Generation** → `crates/core/src/engine/speculative.rs:generate_draft_tokens()`
   - Optional smaller draft model generates candidate tokens

2. **Verification** → `crates/core/src/engine/speculative.rs:verify_draft_tokens()`
   - Target model verifies draft tokens in parallel
   - Accepted tokens emitted, rejected tokens replaced with target output

3. **Adaptive Adjustment** → `crates/core/src/engine/speculative.rs:step_adaptive_speculative()`
   - Track acceptance rate, dynamically adjust max draft tokens

**State Management:**
- Sequences tracked in `SchedulerEngine.running: Vec<Sequence>`
- RequestQueue holds waiting sequences by phase
- `Sequence.kv_blocks: Arc<Vec<BlockId>>` for efficient sharing

## Key Abstractions

### ModelBackend Trait

**Purpose:** Unified interface for all ML model implementations
- **File:** `crates/traits/src/model.rs`
- **Examples:** `crates/model/src/llama/model.rs:LlamaModel`, `crates/model/src/mistral/model.rs:MistralModel`
- **Pattern:** Strategy pattern with trait object returns (`Box<dyn ModelBackend>`)

```rust
pub trait ModelBackend: Send + Sync {
    fn forward(&mut self, seq_ids: &[SeqId], ...) -> Result<BatchOutput>;
    fn forward_logits(&mut self, seq_ids: &[SeqId], ...) -> Result<Vec<Vec<f32>>>;
    fn embed(&mut self, input_tokens: &[Vec<TokenId>], ...) -> Result<Vec<Vec<f32>>>;
}
```

### Architecture Trait

**Purpose:** Dynamic model architecture registration and instantiation
- **File:** `crates/model/src/arch/mod.rs`
- **Examples:** `crates/model/src/llama/arch.rs:LlamaArchitecture`
- **Pattern:** Factory pattern with detection logic

```rust
pub trait Architecture: Send + Sync + 'static {
    fn name(&self) -> &'static str;
    fn detect(&self, config_json: &Value) -> bool;
    fn create_model(...) -> Result<Box<dyn ModelBackend>>;
}
```

### SchedulerPolicy Trait

**Purpose:** Pluggable scheduling algorithms
- **File:** `crates/core/src/scheduler/policy/mod.rs`
- **Examples:** `FcfsPolicy`, `SjfPolicy`, `PriorityPolicy`

### TransformerBlock Trait

**Purpose:** Unified transformer layer interface
- **File:** `crates/model/src/components/block.rs`
- **Examples:** `LlamaBlock`, `MistralBlock`, `Qwen3Block`

## Entry Points

### Server Entry Point

**Location:** `crates/server/src/main.rs:main()`
- **Triggers:** `cargo run -p vllm-server`
- **Responsibilities:** Parse CLI args, load model, initialize Engine, start Axum HTTP server

### Engine Loop

**Location:** `crates/core/src/engine/speculative.rs:Engine<M>::run()`
- **Triggers:** Spawned as separate thread from main()
- **Responsibilities:** Process EngineMessage channel, call step methods, manage lifecycle

### Test Entry Points

**Location:** Various `#[cfg(test)]` modules throughout crates
- **Triggers:** `cargo test --package vllm-core -- test_name`
- **Responsibilities:** Unit test individual components

## Crate Dependencies

```
vllm-traits   → (no dependencies)
vllm-core     → vllm-traits, [vllm-model with cuda-graph feature]
vllm-model    → vllm-traits, candle, candle-nn
vllm-server   → vllm-core, vllm-model, tokio, axum
vllm-dist     → vllm-traits
vllm-testing  → vllm-traits, vllm-core, vllm-model
benches       → vllm-traits, vllm-core, vllm-model
```

## Architectural Constraints

- **Threading:** Single-threaded inference loop spawned on `std::thread::spawn`. Model operations use internal Rayon parallelism for tensor ops.
- **Global state:** `ARCHITECTURE_REGISTRY` is a global `Lazy<ArchitectureRegistry>` at `crates/model/src/arch/registry.rs:64`
- **Circular imports:** None detected. Layer hierarchy strictly enforced.
- **Blocking:** Engine loop uses `mpsc::UnboundedReceiver` for async message handling

## Anti-Patterns

### ModelBackend Lock Contention

**What happens:** `self.target_model.lock().unwrap()` called frequently in speculative decoding path
**Why it's wrong:** Mutex lock overhead on every forward pass, potential contention
**Do this instead:** Use `parking_lot::RwLock` or redesign to avoid per-call locking

### Batch Composition Duplication

**What happens:** Two `build_batch` methods exist: `build_batch()` and `build_batch_with_graph()`
**Why it's wrong:** Code duplication and potential inconsistency between code paths
**Do this instead:** Consolidate into single method with optional graph routing parameter

## Error Handling

**Strategy:** `thiserror` enums with `?` propagation

**Patterns:**
- `ModelError` in `crates/traits/src/model.rs` wraps candle errors
- `EngineError` via `thiserror` in engine modules
- `Result<T>` type aliases for fallible functions
- Global `tracing` for observability (info, debug, warn, error levels)

## Cross-Cutting Concerns

**Logging:** `tracing` crate with structured spans. Levels: ERROR (2 logs), WARN (7), INFO (18), DEBUG (35), TRACE (20).

**Validation:** Config validation in `crates/server/src/config.rs:AppConfig::validate()`

**Authentication:** API key validation via `crates/server/src/auth.rs:AuthMiddleware`

---

*Architecture analysis: 2026-04-26*
