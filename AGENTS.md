# vLLM-lite Development Guide

This guide helps AI agents work effectively with the vllm-lite codebase.

## ⚠️ IMPORTANT: No Worktrees

**DO NOT use git worktrees for local development.**

Single-developer local worktrees cause unnecessary merge conflicts and add complexity without benefit. Just work directly on the current branch.

---

## Quick Commands

```bash
# Build
cargo build --workspace

# Run all tests (skips #[ignore] slow tests)
just nextest

# Run all tests including slow/ignored ones
just nextest-all

# Run single test (by name)
cargo test -p vllm-core test_engine_streaming
cargo test -p vllm-model -- attention

# Run clippy (required before commit)
cargo clippy --workspace -- -D warnings
cargo clippy --all-targets --workspace --all-features -- -D warnings

# Format check
cargo fmt --all --check

# Full CI check
just ci

# Quick: auto-fix + doc check + test
just quick

# Clean build
just clean
```

### Test Notes

- Slow tests are marked with `#[ignore]` and skipped by default
- Use `just nextest-all` or `cargo test -- --ignored` to run them

---

## Project Structure

```text
vllm-lite/
├── Cargo.toml              # Workspace root (7 crates: traits, core, model, server, dist, testing, benches)
├── justfile                # Build automation
├── crates/
│   ├── traits/             # Interface definitions (ModelBackend trait)
│   ├── core/               # Engine, Scheduler, KV cache, Metrics
│   │   └── src/
│   │       ├── scheduler/  # Scheduler modules (queue, preemption, eviction, batch)
│   │       └── kv_cache/   # Logical KV cache (block_allocator, prefix_cache)
│   ├── model/              # Model implementations
│   │   └── src/
│   │       ├── kernels/    # GPU kernels (flash_attention, fused_mlp, cuda_graph)
│   │       ├── paged_tensor/ # Physical KV cache (tensor_store, quantization)
│   │       └── components/ # Model components (attention, mlp, norm, positional)
│   ├── dist/               # Tensor Parallel support
│   └── server/             # HTTP API (OpenAI compatible)
└── tests/                  # Integration tests
```

---

## Checkpoint Loading

The `ModelLoader` supports multiple checkpoint formats with automatic format detection:

- **Safetensors** (`.safetensors`, sharded: `model-00001-of-00002.safetensors`)
- **GGUF** (`.gguf`) - with Q4_K_M quantization support (dequantizes to FP16)

### Usage

```rust
use vllm_model::loader::ModelLoader;
use candle_core::Device;

let device = Device::Cpu;
let loader = ModelLoader::builder(device)
    .with_model_dir("path/to/model".to_string())
    .with_kv_blocks(1024)
    .build()?;

let model = loader.load()?;
```

### Format Detection

The `FormatLoader` trait provides automatic format detection:

```rust
use vllm_model::loader::format::load_checkpoint;
use std::path::Path;

let weights = load_checkpoint(Path::new("model.gguf"), &device)?;
```

---

## Quantization

Supported quantization formats:

- GGUF Q4_K_M (loads and dequantizes to FP16)

The `StorageTensor` abstraction supports multiple storage strategies:

- `Quantized(QuantizedTensor)` - Keep in quantized form (memory efficient)
- `Fp16(Tensor)` - Dequantize to FP16 (balanced)
- `Fp32(Tensor)` - Dequantize to FP32 (highest precision)

### Future Support

The `QuantizationFormat` enum is designed for future support of:

- GPTQ
- AWQ
- Custom quantization schemes

---

## SSM Performance

The `MambaBlock` uses optimized sequential processing with pre-allocated buffers for improved performance on medium to long sequences.

Key optimizations:

- Pre-allocated output buffers
- Minimized tensor allocations in the forward loop
- Local variable caching for frequently accessed dimensions

---

## Code Style Guidelines

### Imports

- Use absolute imports: `use crate::types::Request;`
- External crates: `use vllm_traits::{ModelBackend, SeqId};`
- Group in order: std → external → crate
- Use `super` for sibling module access

### Formatting

- Run `cargo fmt` before committing
- 4-space indentation (Rust standard)
- Max line length: 100 characters (soft limit)
- Use `#![allow(clippy::too_many_arguments)]` when needed

### Naming Conventions

| Type                | Convention           | Example                        |
| ------------------- | -------------------- | ------------------------------ |
| Types               | PascalCase           | `SchedulerEngine`, `Status`    |
| Functions/Variables | snake_case           | `add_request`, `running_count` |
| Constants           | SCREAMING_SNAKE_CASE | `BLOCK_SIZE`, `MAX_BATCH_SIZE` |
| Modules             | snake_case           | `queue_manager`, `eviction`    |
| Crate names         | kebab-case           | `vllm-core`                    |

### Types

- Use `usize` for sizes/lengths, `u64` for IDs (`SeqId`, `TokenId`)
- Prefer explicit type annotations in function signatures
- Use `&T` for read-only, `&mut T` for mutable references
- Use `Option<T>` for nullable values, never `null`

### Error Handling

Use `thiserror` for error enums:

```rust
#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("sequence {id} not found")]
    SeqNotFound { id: u64 },
    #[error("model forward failed: {0}")]
    ModelError(String),
}
pub type Result<T> = std::result::Result<T, EngineError>;
```

- Return `Result<T>` from fallible functions
- Use `?` operator for error propagation

### Tests

- Add to `#[cfg(test)]` module in the same file
- Use `FakeModel` or `StubModel` for mocking
- Naming: `test_<function>_<expected_behavior>`
- Run: `cargo test -p vllm-core -- test_name`

### Documentation

- Document public APIs with `///` doc comments
- Add examples for complex functions
- Use `//` for implementation comments, avoid `/**`

---

## Commit Message Format

```text
<type>(<scope>): <subject>
```

| Type       | Description           |
| ---------- | --------------------- |
| `feat`     | New feature           |
| `fix`      | Bug fix               |
| `refactor` | Code restructuring    |
| `test`     | Adding/updating tests |
| `docs`     | Documentation         |
| `chore`    | Maintenance           |

**Examples**:

```bash
git commit -m "feat(model): add forward_prefill to GqaAttention"
git commit -m "fix(core): resolve prefix cache eviction bug"
git commit -m "test(core): add prefix cache hit test"
```

---

## Verification (Required Before Commit)

```bash
just fmt-check    # Format validation
just clippy       # Code quality
just test         # Run tests
just ci           # Full CI
```

---

## Key Design Patterns

| Pattern                     | Description                            |
| --------------------------- | -------------------------------------- |
| **ModelBackend trait**      | Abstracts ML model implementations     |
| **Paged KV Cache**          | Block-based KV memory management       |
| **Prefix Caching**          | Reuse KV for repeated prompts          |
| **Speculative Decoding**    | Draft-then-verify token generation     |
| **Continuous Batching**     | Dynamic batch scheduling with fairness |
| **Architecture Registry**   | Dynamic model architecture registration |

---

## Architecture Registry System

The project uses a dynamic registration system for model architectures, replacing the previous enum + match pattern.

### Core Components

```rust
// Architecture trait - implement for each model
pub trait Architecture: Send + Sync + 'static {
    fn name(&self) -> &'static str;
    fn detect(&self, config_json: &serde_json::Value) -> bool;
    fn create_model(...) -> Result<Box<dyn ModelBackend>>;
}

// Registry - manages architecture instances
pub struct ArchitectureRegistry {
    architectures: RwLock<HashMap<String, Box<dyn Architecture>>>,
}
```

### Adding a New Architecture (3 steps)

1. **Create `arch.rs`** in the model module:
```rust
pub struct NewModelArchitecture;
impl Architecture for NewModelArchitecture { ... }
```

2. **Create `register.rs`** to register the architecture:
```rust
use crate::arch::{Architecture, ArchitectureRegistry};
pub fn register(registry: &ArchitectureRegistry) {
    registry.register::<NewModelArchitecture>();
}
```

3. **Update `register_all_archs()`** in `arch/registry.rs`:
```rust
pub fn register_all_archs(registry: &ArchitectureRegistry) {
    // ... existing registrations
    crate::newmodel::register::register(registry);
}
```

### Supported Architectures

| Architecture | Directory | Features |
| ------------ | --------- | -------- |
| Llama | `model/src/llama/` | RMSNorm, RoPE, SwiGLU |
| Mistral | `model/src/mistral/` | Sliding Window, GQA |
| Qwen2/3 | `model/src/qwen3/` | GQA, RoPE, QK-Norm |
| Qwen3.5 | `model/src/qwen3_5/` | Mamba SSM Hybrid |
| Gemma4 | `model/src/gemma4/` | Hybrid Attention |
| Mixtral | `model/src/mixtral/` | Sparse MoE |

### Benefits

- **Extensibility**: New architectures without modifying core code
- **Separation**: Each architecture in its own module
- **Testability**: Independent testing per architecture
- **Type Safety**: Compile-time checks via trait bounds

---

## Crate Dependencies

```text
vllm-traits   → (no deps)
vllm-core     → vllm-traits [(optional) vllm-model with cuda-graph feature]
vllm-model    → vllm-traits, candle
vllm-server   → vllm-core, vllm-model, tokio
vllm-dist     → vllm-traits
vllm-testing  → vllm-traits, vllm-core, vllm-model
benches       → vllm-traits, vllm-core, vllm-model
```

---

## Shared Components Layer

The project uses a shared components architecture to reduce code duplication:

```
crates/model/src/components/
├── attention/
│   ├── mod.rs         # GqaAttention, utility functions
│   ├── gqa.rs         # Grouped-query attention implementation
│   └── flash.rs       # Flash attention placeholder
├── mlp/
│   ├── mod.rs
│   └── swiglu.rs      # SwiGLU feed-forward
├── norm/
│   ├── mod.rs
│   ├── rms_norm.rs    # RMSNorm
│   └── layer_norm.rs  # LayerNorm
├── positional/
│   ├── mod.rs
│   ├── rope.rs        # Standard RoPE
│   └── mrope.rs       # MRoPE (Qwen3.5)
└── block.rs           # TransformerBlock base class
```

### Feature Flags

| Feature | Description              |
| ------- | ------------------------ |
| `cuda`  | Candle CUDA support      |
| `gguf`  | GGUF model loading       |
| `full`  | All features (cuda+gguf) |

Note: Tokenizer (tiktoken, tokenizers) is always enabled as it's required for model inference.

---

## Async Patterns

### Engine Initialization

```rust
// Async engine setup
let engine = Engine::new(config).await?;
```

### Request Processing

```rust
// Add request, returns SeqId
let seq_id = engine.add_request(request);

// Build batch
let batch = engine.build_batch();

// Process batch
let output = model.forward(&batch).await?;

// Update engine state
engine.update(&batch.seq_ids, &output.tokens, &input_counts);
```

---

## Testing Guidelines

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scheduler_add_request() {
        let mut engine = SchedulerEngine::default();
        let id = engine.add_request(Request::new(0, vec![1, 2, 3], 10));
        assert!(id > 0);
    }
}
```

### Integration Tests

- Test end-to-end request flow
- Use realistic model configurations
- Verify metrics collection
- Test error handling paths

---

## Debugging

```bash
# Run with verbose logging
RUST_LOG=debug cargo run -p vllm-server

# Run specific test with output
cargo test -p vllm-core test_name -- --nocapture

# Check memory usage
cargo build --release && valgrind ./target/release/vllm-server
```

---

## Logging System

vLLM-lite provides a structured 5-level logging system with dual output (console formatted + JSON file).

### Log Levels

| Level | Usage | Coverage |
|-------|-------|----------|
| **ERROR** | System failures (config, model loading) | 2 logs |
| **WARN** | Degradation (CUDA Graph disabled) | 7 logs |
| **INFO** | Lifecycle (startup, request start/end) | 18 logs |
| **DEBUG** | Internal flow (scheduling, batching) | 35 logs |
| **TRACE** | Verbose (token, KV cache, attention) | 20 logs |

### Configuration

```bash
# Default (info level)
cargo run -p vllm-server

# Enable debug logs
RUST_LOG=debug cargo run -p vllm-server

# Enable trace logs (all details)
RUST_LOG=trace cargo run -p vllm-server

# Enable file logging (JSON format)
cargo run -p vllm-server -- --log-dir ./logs
```

### Log Fields Standard

| Field | Type | Description |
|-------|------|-------------|
| `request_id` | string | Request tracking ID |
| `prompt_tokens` | usize | Input token count |
| `output_tokens` | usize | Output token count |
| `duration_ms` | u64 | Operation duration |
| `seq_id` | SeqId | Sequence ID |
| `batch_size` | usize | Batch size |
| `phase` | Phase | Prefill/Decode |

### Adding Logs

```rust
use tracing::{info, debug, trace, warn, error};

// Info: lifecycle events
info!(request_id = %id, prompt_tokens = 150, "Request started");
info!(output_tokens = 42, duration_ms = 1234, "Request completed");

// Debug: internal flow
debug!(batch_size = 4, phase = ?batch.phase, "Batch built");
debug!(running = 10, waiting = 5, "Scheduling decision");

// Trace: verbose details
trace!(seq_id = %seq_id, token = %token, "Token generated");
trace!(layer_idx = 12, block_ids = ?blocks, "KV cache read");

// Warn: degradation
warn!("CUDA Graph disabled, falling back");

// Error: failures
error!(error = %e, "Model forward failed");
```

### Key Log Locations

| Component | Level | Events |
|-----------|-------|--------|
| `server/main.rs` | info | Startup, model load, shutdown |
| `server/openai/chat.rs` | info | Request start/complete |
| `core/engine.rs` | debug | Model forward, token output |
| `core/scheduler/engine.rs` | debug | Scheduling decision, batch build |
| `core/scheduler/batch_composer.rs` | debug | Batch composition |
| `core/scheduler/memory/allocator.rs` | debug/trace | Block allocation/free |
| `core/sampling.rs` | trace | Sampling strategy |
| `model/components/attention/gqa.rs` | trace | Attention layer |
| `model/paged_tensor/tensor_store.rs` | trace | KV cache read/write |
| `core/kv_cache/prefix_cache.rs` | trace | Prefix cache hit/miss |
