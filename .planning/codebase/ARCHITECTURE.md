# Architecture

**Last updated:** 2026-05-09
**Focus:** Architecture

## System Architecture Overview

vLLM-lite uses a modular, trait-based architecture with clear separation of concerns across 7 crates. The core design follows the **actor model** with message-passing concurrency.

```
┌────────────────────────────────────────────────────────────┐
│                    vllm-server (HTTP API)                   │
│  axum Router → OpenAI endpoints → EngineHandle (mpsc tx)   │
└──────────────────────────┬─────────────────────────────────┘
                           │ EngineMessage (IPC)
┌──────────────────────────▼─────────────────────────────────┐
│                      vllm-core (Engine)                     │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │Scheduler │  │ModelBackend  │  │Speculative Decoding  │  │
│  │Engine    │──┤(trait)       │  │(Adaptive/Self/Std)   │  │
│  │          │  │              │  │                      │  │
│  ├─RequestQ │  │BoxedModelBknd│  ├─SpeculativeModel     │  │
│  ├─PhaseSc  │  │              │  ├─DraftVerifier        │  │
│  ├─MemMgr   │  │              │  └─RejectionStrategy    │  │
│  ├─BatchCmp │  │              │                          │  │
│  └─Prefix   │  └──────────────┘                          │  │
│    Cache    │                                            │  │
└─────────────┴────────────────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
  vllm-model    vllm-dist     vllm-traits
  (Candle ops)  (gRPC/TP)     (interfaces)
```

## Core Architecture Pattern: Trait-Based Abstraction

### ModelBackend Trait (`crates/traits/src/model.rs`)
The central abstraction point. All model architectures implement this trait:

```rust
pub trait ModelBackend: Send + Sync {
    fn forward(...) -> Result<BatchOutput>;
    fn forward_logits(...) -> Result<Vec<Vec<f32>>>;
    fn embed(...) -> Result<Vec<Vec<f32>>>;
    fn vocab_size(&self) -> usize;
    fn num_layers(&self) -> usize;
    fn num_heads(&self) -> usize;
}
```

### Architecture Trait (`crates/model/src/arch/mod.rs`)
Factory pattern for dynamic model architecture registration:

```rust
pub trait Architecture: Send + Sync + 'static {
    fn name(&self) -> &'static str;
    fn detect(&self, config_json: &Value) -> bool;
    fn create_model(...) -> Result<Box<dyn ModelBackend>>;
}
```

### Architecture Registry (`crates/model/src/arch/registry.rs`)
Lazy-initialized singleton that auto-detects model architecture from config.json. Uses `RwLock<HashMap<String, ArchFactory>>`.

## Engine Architecture

### Engine (`crates/core/src/engine.rs`)
- Actor-based: receives `EngineMessage` via `mpsc::UnboundedReceiver`
- Main loop in `run()`: processes messages, steps scheduler + model forward
- Supports both generic `Engine<M>` and type-erased `Engine<BoxedModelBackend>`

### SchedulerEngine (`crates/core/src/scheduler/engine.rs`)
Componentized scheduler with 6 sub-components:
1. **RequestQueue** — Phase-aware request indexing (prefill/decode)
2. **PhaseScheduler** — Strict prefill/decode separation with configurable switch policies
3. **BatchComposer** — Phase-specific batch construction with token packing
4. **MemoryManager** — Block allocation and eviction for KV cache
5. **RadixTree** — Prefix caching for reusable prompt computations
6. **SchedulingPolicy** — Pluggable policies (FCFS, SJF, Priority)

### Sampling (`crates/core/src/sampling.rs`)
- Token selection strategies (greedy, temperature, top-k, top-p)
- Pluggable sampling strategy pattern

## Model Layer Architecture

### Shared Components (`crates/model/src/components/`)
Reusable neural network components:
- `attention/` — GQA (`gqa.rs`), MLA (`mla.rs`), Flash Attention (`flash.rs`, `flash_v3.rs`)
- `mlp/` — SwiGLU feed-forward
- `norm/` — RMSNorm, LayerNorm
- `positional/` — RoPE, MRoPE (for Qwen3.5)
- `ssm.rs` — SSM, Mamba blocks, HarmonicSSM for hybrid models

### Model Architectures (10 supported)
Each in its own module with `arch.rs` + `register.rs`:
- `llama/` — Llama (RMSNorm, RoPE, SwiGLU)
- `mistral/` — Mistral (Sliding Window, GQA)
- `qwen3/` — Qwen2/3 (GQA, MLA, RoPE, QK-Norm)
- `qwen3_5/` — Qwen3.5 (Mamba SSM Hybrid, HarmonicSSM)
- `gemma4/` — Gemma4 (Hybrid Attention)
- `mixtral/` — Mixtral (Sparse MoE)
- `llama4/` — Llama4
- `gemma3/` — Gemma3
- `mistral_small/` — Mistral Small
- `phi4/` — Phi-4

### KV Cache (`crates/model/src/kv_cache.rs`)
Paged attention KV cache with block-level management. Two tensor storage strategies:
- `paged_tensor/` — Physical KV cache with block IDs and quantization
- `tensor_store.rs` — KV cache read/write with logging

### Model Loading (`crates/model/src/loader/`)
- `builder.rs` — `ModelLoaderBuilder` and `ModelLoader`
- `format.rs` — Auto-detection of safetensors vs GGUF
- `checkpoint.rs` — Weight loading from sharded checkpoints
- `io.rs` — File I/O utilities

## Distributed Architecture (`crates/dist/`)

### Tensor Parallel (`crates/dist/src/tensor_parallel/`)
- `device_mesh.rs` — Device/node topology management
- `parallel_linear.rs` — Column/Row parallel linear layers
- `all_reduce.rs` — NCCL-based all-reduce (stub)

### Pipeline Parallel (`crates/dist/src/pipeline/`)
- `pipeline.rs` — Pipeline parallelism orchestrator
- `stage.rs` — Pipeline stage definition

### Distributed KV Cache (`crates/dist/src/distributed_kv/`)
- `cache.rs` — Distributed KV cache with message protocol
- `protocol.rs` — Message types for cache operations

## Server Architecture (`crates/server/`)

### API Layer (`crates/server/src/openai/`)
- `chat.rs` — Chat completions (streaming SSE + non-streaming)
- `completions.rs` — Text completions
- `embeddings.rs` — Embedding endpoints
- `models.rs` — Model listing
- `types.rs` — OpenAI-compatible request/response types

### Security Layer (`crates/server/src/security/`)
- JWT validation, RBAC, TLS, audit logging, correlation IDs

### Backpressure (`crates/server/src/backpressure.rs`)
- Dynamic backpressure based on queue depth and latency

## Data Flow

### Request Lifecycle
```
Client → HTTP POST /chat/completions
  → axum handler (chat.rs)
  → Build EngineMessage::AddRequest
  → mpsc channel → Engine::run() loop
  → SchedulerEngine::add_request()
  → Engine::step() loop:
      → SchedulerEngine::build_batch()
      → ModelBackend::forward() (CUDA/CPU)
      → SchedulerEngine::update()
      → Send tokens via mpsc response channel
  → SSE stream back to client
```

### Batch Construction
```
AddRequest → RequestQueue (phase-aware)
  → PhaseScheduler (select phase)
  → MemoryManager (allocate KV blocks)
  → BatchComposer (build vllm_traits::Batch)
  → ModelBackend::forward()
  → Process outputs → Update scheduler state
```

## Key Design Decisions

1. **Actor model** for engine concurrency — single-threaded engine loop with message passing
2. **Type-erased models** via `BoxedModelBackend` — allows runtime model selection
3. **Dynamic architecture detection** — registry pattern, no enum matching
4. **Paged KV cache** — block-based memory management with prefix caching
5. **Continuous batching** — dynamic batch composition each step
6. **Feature-gated CUDA** — `cuda` and `cuda-graph` features, graceful CPU fallback
