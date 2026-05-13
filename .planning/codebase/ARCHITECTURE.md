<!-- refreshed: 2026-05-13 -->
# Architecture

**Analysis Date:** 2026-05-13

## System Overview

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                          vllm-server (HTTP Layer)                             │
│  `crates/server/` — axum router, OpenAI-compatible API, SSE streaming        │
├──────────────────────┬──────────────────────┬────────────────────────────────┤
│   openai/chat.rs     │   openai/completions  │  openai/embeddings.rs          │
│   openai/models.rs   │   openai/batch/       │  api.rs (health, metrics)      │
└──────────┬───────────┴──────────┬───────────┴───────────────┬────────────────┘
           │ mpsc channels         │                          │
           ▼                       ▼                          ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                          vllm-core (Engine Layer)                             │
│  `crates/core/` — actor-pattern inference engine, scheduling, memory mgmt    │
├──────────────────────┬──────────────────────┬────────────────────────────────┤
│  Engine (engine.rs)  │  SchedulerEngine     │  Speculative Decoder            │
│  Actor loop on       │  scheduler/engine.rs │  speculative/                   │
│  dedicated thread    │  request_queue,      │  adaptive.rs, verifier.rs       │
│                      │  phase_scheduler,    │  model.rs, self_spec.rs         │
│                      │  batch_composer,     │                                │
│                      │  memory/allocator,   │                                │
│                      │  radix_cache/        │                                │
└──────────┬───────────┴──────────┬───────────┴───────────────┬────────────────┘
           │                      │                           │
           ▼                      ▼                           ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                       vllm-traits (Interface Layer)                           │
│  `crates/traits/` — ModelBackend trait, Batch/BatchOutput types, SeqId/TokenId│
└──────────────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                          vllm-model (Model Layer)                             │
│  `crates/model/` — architecture registry, attention, SSM, KV cache, kernels  │
├──────────────────────┬──────────────────────┬────────────────────────────────┤
│  Architecture Registry│  components/         │  paged_tensor/                 │
│  arch/registry.rs     │  attention/gqa.rs    │  tensor_store.rs               │
│  llama/, mistral/,    │  attention/mla.rs    │  quant.rs                      │
│  qwen3/, qwen3_5/,    │  ssm.rs, mlp/        │  quantization.rs               │
│  gemma3/, gemma4/,    │  norm/, positional/  │                                │
│  llama4/, phi4/,      │  block.rs            │                                │
│  mixtral/, mistral_   │                      │                                │
│  small/               │                      │                                │
└───────────────────────┴──────────────────────┴────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│              candle-core / candle-nn (Tensor Computation)                     │
│  External — GPU-accelerated tensor operations, neural network primitives      │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

| Component                | Responsibility                                                                                                                         | File                                           |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| **Engine**               | Actor-loop inference orchestrator; mpsc message handling; model forward dispatch; step control for speculative/CUDA graph/normal paths | `crates/core/src/engine.rs`                    |
| **SchedulerEngine**      | Componentized scheduler: request queuing, batch composition, phase management, memory allocation, prefix caching                       | `crates/core/src/scheduler/engine.rs`          |
| **RequestQueue**         | O(1) request insertion/removal with phase-aware indexing; tracks waiting, running, finished sequences                                  | `crates/core/src/scheduler/request_queue.rs`   |
| **PhaseScheduler**       | Strict prefill/decode separation; configurable via `PhaseSwitchPolicy`; tracks consecutive decode rounds                               | `crates/core/src/scheduler/phase_scheduler.rs` |
| **BatchComposer**        | Phase-specific batch construction; respects token budgets and batch size limits                                                        | `crates/core/src/scheduler/batch_composer.rs`  |
| **MemoryManager**        | Block allocation (`allocator.rs`) and LRU-based eviction (`eviction.rs`); front-end for KV cache block management                      | `crates/core/src/scheduler/memory/`            |
| **RadixTree**            | O(k) prefix match lookup; enables prompt reuse across requests via radix tree                                                          | `crates/core/src/scheduler/radix_cache/`       |
| **ModelBackend** (trait) | Core model interface: `forward()`, `forward_logits()`, `embed()` with KV block IDs                                                     | `crates/traits/src/model.rs`                   |
| **Architecture** (trait) | Factory trait for model creation: `detect()`, `create_model()`, `create_block()`                                                       | `crates/model/src/arch/mod.rs`                 |
| **ArchitectureRegistry** | Dynamic architecture registration and format detection; lazy-static global singleton                                                   | `crates/model/src/arch/registry.rs`            |
| **ModelLoader**          | Builder-pattern loader; load, detect architecture, register, create model                                                              | `crates/model/src/loader/`                     |
| **Tokenizer**            | Text ↔ token encoding/decoding; supports tiktoken + tokenizers backends                                                                | `crates/model/src/tokenizer.rs`                |
| **GqaAttention**         | Grouped-Query Attention; flash attention + tiled attention + paged attention                                                           | `crates/model/src/components/attention/gqa.rs` |
| **MlaAttention**         | Multi-head Latent Attention (32x KV cache compression for DeepSeek-style models)                                                       | `crates/model/src/components/attention/mla.rs` |
| **Speculative Decoder**  | Draft-then-verify token generation; adaptive draft count tuning; self-speculation support                                              | `crates/core/src/speculative/`                 |
| **MetricsCollector**     | Token throughput, latency, KV cache usage metrics; Prometheus exporter                                                                 | `crates/core/src/metrics/`                     |
| **Distributed KV Cache** | gRPC-based distributed KV cache protocol for multi-node deployment                                                                     | `crates/dist/src/distributed_kv/`              |
| **Tensor Parallel**      | Column-parallel and row-parallel linear layers; AllReduce via NCCL                                                                     | `crates/dist/src/tensor_parallel/`             |
| **Pipeline Parallel**    | Pipeline stage trait and parallel executor for model sharding across devices                                                           | `crates/dist/src/pipeline/`                    |

## Pattern Overview

**Overall:** Actor pattern + Componentized scheduler + Architecture registry

**Key Characteristics:**

- **Actor-based engine:** The `Engine` runs on its own dedicated `std::thread`, receiving commands via `mpsc::UnboundedReceiver<EngineMessage>`. All external communication flows through mpsc channels — no shared mutable state between engine and server.
- **Dynamic architecture dispatch:** Models are loaded through the `ArchitectureRegistry` — a lazy-static `RwLock<HashMap<String, ArchFactory>>` that detects the model format (via config JSON) and creates the correct model backend at runtime. No enum+match required.
- **Componentized scheduler:** The `SchedulerEngine` composes seven sub-components (`RequestQueue`, `PhaseScheduler`, `BatchComposer`, `MemoryManager`, `RadixTree`, `SchedulingPolicy`, `SchedulerObservers`) each behind traits or clear interfaces.
- **Paged KV cache:** Block-based memory management (`BLOCK_SIZE=16` tokens per block). Logical block allocation in `vllm-core/kv_cache`, physical tensor storage in `vllm-model/paged_tensor/tensor_store.rs`.
- **Speculative decoding:** Optional draft model (lighter model that generates candidate tokens) + target model verification. Adaptive draft count tuning via `AdaptiveSpeculativeDecoder`.
- **Separation of concerns:** `vllm-traits` is the zero-dependency interface layer; `vllm-core` depends on traits but NOT on model implementations; `vllm-model` implements the trait; `vllm-server` orchestrates everything.

## Layers

**Interface Layer (`vllm-traits`):**

- Purpose: Define shared types and trait contracts with zero or minimal dependencies
- Location: `crates/traits/src/`
- Contains: `ModelBackend` trait, `Batch`/`BatchOutput` types, `SeqId`/`TokenId`/`BlockId` type aliases, `TensorParallelError`, CUDA graph config traits
- Depends on: `serde`, `serde_json`, optional `candle-core`
- Used by: Every other crate in the workspace

**Engine Layer (`vllm-core`):**

- Purpose: Orchestrate inference: scheduling, batching, memory management, token generation loop
- Location: `crates/core/src/`
- Contains: `Engine` (actor loop), `SchedulerEngine` and sub-components, KV cache (`kv_cache/`), metrics (`metrics/`), speculative decoding (`speculative/`), sampling (`sampling.rs`), beam search (`beam.rs`), circuit breaker, health checks, routing
- Depends on: `vllm-traits`, optional `vllm-model` (for CUDA graph feature)
- Used by: `vllm-server`, `vllm-testing`, `benches`

**Model Layer (`vllm-model`):**

- Purpose: Model loading, architecture implementations, transformer components, GPU kernels, KV cache tensor storage, tokenization
- Location: `crates/model/src/`
- Contains: Architecture registry (`arch/`), per-model implementations (`llama/`, `mistral/`, `qwen3/`, `qwen3_5/`, `gemma3/`, `gemma4/`, `llama4/`, `mixtral/`, `mistral_small/`, `phi4/`), shared components (`components/`), kernels (`kernels/`), paged tensor storage (`paged_tensor/`), model loader (`loader/`), quantization (`quantize/`), tokenizer
- Depends on: `vllm-traits`, `vllm-dist`, `candle-core`, `candle-nn`
- Used by: `vllm-server`, `vllm-core` (optional), `benches`

**Server Layer (`vllm-server`):**

- Purpose: HTTP API server with OpenAI-compatible endpoints
- Location: `crates/server/src/`
- Contains: axum router, OpenAI API handlers (`openai/chat.rs`, `openai/completions.rs`, `openai/embeddings.rs`, `openai/batch/`, `openai/models.rs`), CLI (`cli.rs`), config (`config.rs`), auth (`auth.rs`), health checks (`health.rs`), logging, backpressure, debug endpoints
- Depends on: `vllm-core`, `vllm-traits`, `vllm-model`, `axum`, `clap`, `tracing-subscriber`
- Used by: End users (binary `vllm-server`)

**Distribution Layer (`vllm-dist`):**

- Purpose: Multi-GPU and multi-node support: tensor parallelism, pipeline parallelism, distributed KV cache
- Location: `crates/dist/src/`
- Contains: gRPC service definitions (`grpc.rs`), distributed KV cache protocol (`distributed_kv/`), tensor parallel linear layers (`tensor_parallel/`), pipeline stage execution (`pipeline/`)
- Depends on: `vllm-traits`, `candle-core`, `tonic`/`prost`
- Used by: `vllm-model`

**Testing Layer (`vllm-testing`):**

- Purpose: Shared test infrastructure: mock models, test harness, request factories
- Location: `crates/testing/src/`
- Contains: `TestHarness`, mock models (`FakeModel`, `StubModel`, `ConstModel`, `IncrementModel`, `NeverProgressModel`), `RequestFactory`, `SlowModel`
- Depends on: `vllm-traits`, `vllm-core`
- Used by: All crates' `dev-dependencies`

## Data Flow

### Primary Request Path (Chat Completion)

1. **HTTP POST `/v1/chat/completions`** → `chat_completions()` handler in `crates/server/src/openai/chat.rs:99`
2. **Build prompt** from messages → `build_prompt_from_messages()` at `crates/server/src/openai/chat.rs:26`
3. **Tokenize** prompt via `Tokenizer` at `crates/model/src/tokenizer.rs`
4. **Send to engine** via `EngineHandle` (mpsc::UnboundedSender) with `EngineMessage::AddRequest { request, response_tx }` → `crates/core/src/types.rs:282`
5. **Engine.run() loop** receives message at `crates/core/src/engine.rs:361-401`
6. **Engine.add_request()** delegates to `SchedulerEngine.add_request()` at `crates/core/src/scheduler/engine.rs`
7. **Scheduling loop:** When `has_pending()`, builds batch via `build_batch()` → `RequestQueue` selects sequences → `PhaseScheduler` picks phase → `BatchComposer` assembles batch → `MemoryManager` allocates blocks → `RadixTree` checks prefix hits
8. **Model forward** via `forward_batch()` → calls `target_model.lock().unwrap().forward(...)` at `crates/core/src/engine.rs:595-599`
9. **Process output** via `process_output()` → sends tokens through `response_tx` channels → `SchedulerEngine.update()` updates sequence state at `crates/core/src/engine.rs:619-667`
10. **SSE streaming** in chat handler iterates response channel, formats SSE events, returns `Sse` response to client

### Speculative Decoding Path

1. Engine checks `adaptive_decoder.is_some()` or `speculative_mode` in run loop at `crates/core/src/engine.rs:406-414`
2. `step_with_draft(Some(max_draft))` dispatches to `step_speculative_inner()` at `crates/core/src/engine/speculative.rs`
3. Draft model runs forward to generate candidate tokens
4. Target model verifies draft tokens via `DraftVerifier` at `crates/core/src/speculative/verifier.rs`
5. Accepted tokens streamed to clients; rejected tokens discarded
6. Adaptive decoder tunes draft count based on acceptance rate history

### Embedding Path

1. **HTTP POST `/v1/embeddings`** → `embeddings()` handler at `crates/server/src/openai/embeddings.rs`
2. Tokenize input text
3. Send `EngineMessage::GetEmbeddings { input_tokens, response_tx }`
4. Engine directly calls `target_model.embed(&input_tokens, &positions)` at `crates/core/src/engine.rs:386-390`
5. Returns embedding vectors through response channel

**State Management:**

- Engine state lives entirely within the `Engine` struct on the dedicated engine thread
- Server state (`ApiState`) is `Clone` (all fields `Arc`-wrapped or `mpsc::Sender`)
- Model state (weights, KV cache) lives inside the `Box<dyn ModelBackend>` behind `Arc<Mutex<dyn ModelBackend>>`
- Only the engine thread holds the `Mutex` lock on the model; server never touches it directly

## Key Abstractions

**`ModelBackend` trait:**

- Purpose: Abstract interface for all model implementations; defines `forward()`, `forward_logits()`, `embed()`, `vocab_size()`, `num_layers()`, `num_heads()`, `forward_to_layer()`
- Examples: `llama/model.rs`, `mistral/model.rs`, `qwen3/model.rs`, `qwen3_5/model.rs`, `gemma3/`, `gemma4/`, `llama4/`, `mixtral/`, `phi4/`, `mistral_small/`
- Pattern: Each architecture implements `ModelBackend` on a struct containing layers + KV cache; forward receives `seq_ids`, `input_tokens`, `positions`, `kv_block_ids`, `num_computed_tokens`, `is_prefill`

**`Architecture` trait:**

- Purpose: Factory for creating model backends from config + weights; enables dynamic dispatch without enum matching
- Examples: `qwen3/arch.rs`, `llama/arch.rs`, `mistral/arch.rs`
- Pattern: Implement `detect(config_json)` to identify model type, `create_model()` to build the backend, `create_block()` to build individual transformer blocks

**`TransformerBlock` trait:**

- Purpose: Abstract transformer block; supports attention + MLP forward, KV cache interaction
- Location: `crates/model/src/components/block.rs:135`
- Pattern: `forward(&mut self, hidden_states, positions, kv_block_ids, num_computed, is_prefill)`

**`SchedulingPolicy` trait:**

- Purpose: Pluggable scheduling algorithms; computes priority scores for sequences
- Location: `crates/core/src/scheduler/policy/trait_def.rs:16`
- Patterns: `FcfsPolicy`, `SjfPolicy`, `PriorityPolicy`

**`FormatLoader` trait:**

- Purpose: Automatic checkpoint format detection and loading
- Location: `crates/model/src/loader/format.rs`
- Patterns: `SafetensorsLoader`, GGUF loader (feature-gated)

**`BoxedModelBackend`:**

- Purpose: Type-erased wrapper around `Box<dyn ModelBackend>`; enables `Engine<BoxedModelBackend>` to work with any model
- Location: `crates/core/src/engine.rs:23`
- Pattern: Newtype with `Deref`/`DerefMut` to `dyn ModelBackend`; blanket impl of `ModelBackend` that delegates

## Entry Points

**`vllm-server` binary:**

- Location: `crates/server/src/main.rs:91` (`#[tokio::main] async fn main()`)
- Triggers: CLI invocation; loads model, spawns engine thread, starts axum HTTP server
- Responsibilities: Parse CLI args → load model via `ModelLoader` → create `Engine<BoxedModelBackend>` → spawn engine thread → build axum router → serve

**`Engine.run()` actor loop:**

- Location: `crates/core/src/engine.rs:361`
- Triggers: Called on the dedicated engine thread after server spawn
- Responsibilities: Loop: drain mpsc messages → if pending work, execute step (speculative/CUDA graph/normal) → sleep with adaptive backoff

**`Engine.step()` / `step_with_draft()` / `step_with_graph()`:**

- Location: `crates/core/src/engine.rs` (step defined in `engine/speculative.rs:13`)
- Triggers: Called by `run()` loop when work is pending
- Responsibilities: Build batch → model forward → process output → stream tokens → update state

**OpenAI-compatible HTTP endpoints** → `crates/server/src/main.rs:230-253`:

- `POST /v1/chat/completions` — SSE streaming chat
- `POST /v1/completions` — text completions
- `POST /v1/embeddings` — embedding vectors
- `GET /v1/models` — model listing
- `POST /v1/batches` — batch API (create)
- `GET /v1/batches/:id` — batch API (retrieve)
- `GET /health/live`, `GET /health/ready`, `GET /health` — K8s health probes
- `GET /metrics` — Prometheus metrics
- `GET /debug/metrics`, `GET /debug/kv-cache`, `GET /debug/trace` — debug endpoints

**`vllm` helper binary:**

- Location: `crates/server/src/bin/vllm.rs`
- Purpose: Lightweight CLI for quick inference without HTTP server

**`ARCHITECTURE_REGISTRY` global:**

- Location: `crates/model/src/arch/registry.rs:64`
- Triggers: Initialized lazily on first access; `register_all_archs()` is called during loading

## Architectural Constraints

- **Threading:** Engine runs on a single dedicated `std::thread` with interior mutability (`Arc<Mutex<dyn ModelBackend>>`). Server runs on the tokio multi-threaded runtime. Engine communicate via mpsc channels — no shared mutable state across the thread boundary.
- **Global state:** `ARCHITECTURE_REGISTRY` is a `Lazy<RwLock<HashMap<...>>>` global in `crates/model/src/arch/registry.rs`. `candle_core` uses GPU device indices as global state.
- **Circular imports:** `vllm-core` optionally depends on `vllm-model` (feature `cuda-graph`), but `vllm-model` does NOT depend on `vllm-core`. No circular crate dependencies.
- **Block size:** `BLOCK_SIZE = 16` tokens per KV cache block, defined in `crates/traits/src/types.rs:3` and re-exported from `crates/core/src/kv_cache/mod.rs`.
- **Batch composition:** Mixed prefill+decode batches are only assembled when `enable_pd_separation = false` in `SchedulerConfig`. Otherwise, batches are strictly prefill OR decode, never mixed.

## Anti-Patterns

### Global Mutable State in Architecture Registry

**What happens:** `ARCHITECTURE_REGISTRY` is a module-level `Lazy<ArchitectureRegistry>` static accessed globally throughout the model crate.
**Why it's wrong:** Makes testing harder (shared mutable state), prevents having multiple registry instances, creates hidden dependency between module loading order and registration.
**Do this instead:** Pass `ArchitectureRegistry` explicitly as a parameter. Use dependency injection for the registry in `ModelLoaderBuilder`.

### Engine Thread Spawning from Server with Raw std::thread

**What happens:** Server `main()` spawns `std::thread::spawn(move || { engine.run(msg_rx); })` at `crates/server/src/main.rs:179`.
**Why it's wrong:** No structured concurrency. The thread runs detached and there's no cleanup mechanism beyond the mpsc shutdown message. Panic in the engine thread goes unobserved by the server (no join handle stored).
**Do this instead:** Store the `JoinHandle` and join on it during graceful shutdown. Consider using `tokio::task::spawn_blocking` for integration with the async runtime.

### BoxedModelBackend Wraps Arc<Mutex<...>> Delegation

**What happens:** `BoxedModelBackend` is a newtype around `Box<dyn ModelBackend>` but the engine holds it behind `Arc<Mutex<dyn ModelBackend>>`. All model access goes through `target_model.lock().unwrap()` with no timeout or deadlock detection.
**Why it's wrong:** If a panic poisons the mutex, all subsequent model accesses panic. Long forward passes hold the lock for extended periods, blocking concurrent reads.
**Do this instead:** Use `try_lock()` with timeout or consider a read/write lock if forward passes can be parallelized.

## Error Handling

**Strategy:** Layered error types using `thiserror`:

- `vllm_traits::ModelError` — simple string-based error for model operations (`crates/traits/src/model.rs:3`)
- `vllm_core::error::EngineError` — enum with `SeqNotFound`, `InvalidRequest`, `ModelError`, `SamplingError` variants (`crates/core/src/error/mod.rs:4`)
- `vllm_dist::PipelineError` — pipeline-specific errors (`crates/dist/src/pipeline/`)
- `vllm_traits::TensorParallelError` — tensor parallel errors (`crates/traits/src/types.rs:78`)

**Patterns:**

- `?` operator for propagation within a crate
- `From<vllm_traits::ModelError>` impl for `EngineError` bridges trait layer errors
- `Result<T> = std::result::Result<T, EngineError>` type alias in core crate
- Errors are logged via `tracing::error!()` but NOT propagated to clients (tokens just stop streaming on error)

## Cross-Cutting Concerns

**Logging:** Structured 5-level logging via `tracing` crate. Console output (formatted) + optional JSON file output. Levels: ERROR (system failures), WARN (degradation), INFO (lifecycle), DEBUG (internal flow), TRACE (token-level details). Configuration via `RUST_LOG` env var or `--log-dir` CLI flag. Implementation in `crates/server/src/logging.rs`.

**Validation:** `SchedulerConfig::new()` asserts invariants at construction (`crates/core/src/types.rs:211-258`). Server config validated in `crates/server/src/config.rs` via `app_config.validate()`.

**Authentication:** Optional API key auth via `AuthMiddleware` in `crates/server/src/auth.rs`. Supports rate limiting. Enabled only when `app_config.auth.api_keys` is non-empty. Applied as axum middleware layer conditionally.

**Health Checks:** `HealthChecker` in `crates/server/src/health.rs` provides liveness (`/health/live`) and readiness (`/health/ready`) probes following K8s conventions.

**Circuit Breaker:** `Breaker` in `crates/core/src/circuit_breaker/breaker.rs` for fault isolation; strategies in `strategy.rs`.

**Backpressure:** `BackpressureConfig` and `BackpressureManager` in `crates/server/src/backpressure.rs` protect against overload by tracking buffer usage with high/low watermarks.

---

*Architecture analysis: 2026-05-13*
