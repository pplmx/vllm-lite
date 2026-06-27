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
├── Cargo.toml              # Workspace root (6 crates: traits, core, model, server, testing, dist)
│                           # dist is feature-gated behind --features multi-node (Phase 26 MT-07)
├── justfile                # Build automation
├── crates/
│   ├── traits/             # Interface definitions (ModelBackend trait, kernel traits)
│   ├── core/               # Engine, Scheduler, KV cache, Metrics
│   │   └── src/
│   │       ├── scheduler/  # Scheduler modules (queue, preemption, eviction, batch)
│   │       └── kv_cache/   # Logical KV cache (block_allocator, prefix_cache)
│   ├── model/              # Model implementations
│   │   └── src/
│   │       ├── kernels/    # GPU kernels (flash_attention, fused_mlp, cuda_graph)
│   │       ├── paged_tensor/ # Physical KV cache (tensor_store, quantization)
│   │       ├── components/ # Shared components (attention, mlp, norm, positional)
│   │       ├── llama/      # Llama architecture
│   │       ├── qwen3/      # Qwen3 architecture (GQA + MLA)
│   │       └── qwen3_5/    # Qwen3.5 architecture (Mamba SSM Hybrid)
│   ├── dist/               # Tensor Parallel support (feature-gated: --features multi-node)
│   ├── server/             # HTTP API (OpenAI compatible)
│   └── testing/            # Test harness, factories, slow-model stubs
└── scripts/                # Utility scripts (doc_coverage.sh, etc.)
```

Integration tests live in `crates/*/tests/` (per crate), not in a top-level `tests/` directory.

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

The `SSMLayer`, `MambaBlock`, and `SSMHarmonicSSMLayer` (for Qwen3.5 hybrid models) use optimized sequential processing.

Key optimizations:

- Pre-allocated output buffers
- Minimized tensor allocations in the forward loop
- Local variable caching for frequently accessed dimensions

### SSM Types

| Type                      | Location            | Use Case                              |
| ------------------------- | ------------------- | ------------------------------------- |
| `SSMLayer` + `MambaBlock` | `components/ssm.rs` | Standard Mamba (Qwen3.5 Mamba-only)   |
| `SSMHarmonicSSMLayer`     | `components/ssm.rs` | Hybrid attention+SSM (Qwen3.5 hybrid) |

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

#### Verb Policy (read/load/get/create/build)

Use consistent verb prefixes so readers can predict semantics:

| Prefix        | Use for                                                              | Examples                                  |
| ------------- | -------------------------------------------------------------------- | ----------------------------------------- |
| `get_*`       | In-memory accessors (sync, no I/O)                                   | `get_block`, `get_token_id`               |
| `load_*`      | Resource acquisition (file/IO, deserialization)                      | `load_checkpoint`, `load_gguf_tensors`    |
| `read_*`      | Streamed/buffered I/O with explicit cursor/position semantics        | `read_kv`, `read_request`, `read_kv_batch`|
| `set_*`       | Mutator for owned state                                               | `set_running`, `set_priority`             |
| `write_*`     | Streamed/buffered write-back                                          | `write_kv`, `write_compressed`            |
| `create_*`    | One-shot resource construction (non-builder)                          | `create_engine`, `create_request`         |
| `build_*`     | Builder-pattern finalization step                                    | `BatchBuilder::build`, `ConfigBuilder::build` |
| `add_*`       | Collection append (no key/ID allocation concerns)                    | `add_request`, `add_to_batch`             |
| `register_*`  | Insert into a registry/lazy initialization map                       | `register_architecture`, `register_kernel`|
| `forward`     | ML forward pass (no prefix)                                           | `ModelBackend::forward`                   |

If a function does both acquisition and construction, prefer `load_*` over `create_*`.

#### Suffix Conventions

| Suffix        | When required                                                       | Example                      |
| ------------- | ------------------------------------------------------------------- | ---------------------------- |
| `*Manager`    | Type owns and coordinates a concrete resource                       | `BatchManager`, `PreemptionManager` |
| `*Info`       | Type is metadata-only (no behavior); bare name would be ambiguous  | `NodeInfo` (vs graph `Node`) |
| `*Data`       | Avoid unless matching an external API spec field name               | (avoid; prefer bare names)   |
| `*Factory`    | Builder-pattern factory that produces one type                      | `RequestFactory`             |
| `*Config`     | Immutable configuration bundle                                       | `SchedulerConfig`            |

The `*Data` suffix is **discouraged** because output types already imply data.
Keep it only when matching external API names (e.g., OpenAI's `data` field).

#### Single-letter variables

Single-letter variables are allowed **only** for:

1. Loop indices (`for i in 0..n`).
2. Tensor-math conventions in attention / SSM / MLP code:
   - `q`, `k`, `v`, `o` (query/key/value/output projections)
   - `b`, `c`, `h`, `d`, `x`, `z` (state/batch/head/dimension/SSM variables)
   - `g`, `r` (gating/routing in MoE / gated nets)
3. Trigonometric placeholders (`pi`, `e`).

Outside these contexts (scheduler, sampling, server handlers), use descriptive names
(`priority_a`, `priority_b`, `random_threshold`, etc.). See audit `NAME-F-18` for
historical exceptions and the rationale for the exemption.

#### Test file location

| Test type           | Location                                              | Discovery         |
| ------------------- | ----------------------------------------------------- | ----------------- |
| Unit tests          | `#[cfg(test)] mod tests {}` block in the source file  | `cargo test`      |
| Integration tests   | `crates/<crate>/tests/<topic>.rs`                     | `cargo test`      |
| Cross-crate e2e     | `crates/<crate>/tests/integration.rs` or similar      | `cargo test`      |

**Do not** place `.rs` test files in `src/` directories outside `mod tests` blocks.
They will not be auto-discovered and become dead code (audit `NAME-F-04`).

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

## Lint Policy

Workspace-wide clippy configuration lives in the root `Cargo.toml` under
`[workspace.lints.clippy]`. Every crate inherits via `[lints] workspace = true`.

### Tiers

| Tier   | Lints                                                          | Effect                              |
| ------ | -------------------------------------------------------------- | ----------------------------------- |
| deny   | `correctness`, `suspicious`, `perf`                            | Breaks `just clippy` (CI blocks)    |
| warn   | `pedantic`, `nursery`, `missing_errors_doc`, `must_use_candidate`, etc. | Visible in `cargo clippy`, not blocking |
| allow  | `cast_precision_loss`, `too_many_lines`, `too_many_arguments`, etc. | Explicitly silenced with rationale |

### Local commands

```bash
# Standard CI check (deny-tier only)
just clippy

# Show pedantic+nursery warnings without breaking
just clippy-pedantic
```

### Adding a new lint

1. Identify which tier it belongs to (correctness/suspicious/perf → deny; otherwise → warn first, promote to deny later)
2. Add to `[workspace.lints.clippy]` in root `Cargo.toml`
3. Run `just clippy` to verify
4. If deny-tier and existing code violates it, fix the violations in the same PR

### Invariant comments

Production `.unwrap()` / `.expect()` calls that legitimately cannot fail must be
preceded by a `// invariant:` comment explaining why. This applies to:

- `RwLock` / `Mutex` `.expect("...poisoned")` — lock is only held for sync field access
- `SystemTime::now().duration_since(UNIX_EPOCH).unwrap()` — monotonic clock cannot underflow
- `.expect("duplicate <key>")` after a pre-check — programmer error path
- Tensor allocations with statically-known shapes
- Cargo env vars (always set by Cargo during build)
- Signal handler installation
- Serialize of known-good structs
- HashMap access immediately after `insert()`

If a `.unwrap()` / `.expect()` call is in production code and has no `// invariant:`
comment, treat it as a bug and convert it to typed error propagation. See Phase B
audit at `/tmp/phase_b_audit/SUMMARY.md` for the full inventory.

### Rationales for current allow list

| Lint                      | Why allowed                                                                |
| ------------------------- | -------------------------------------------------------------------------- |
| `cast_precision_loss`     | Model dim casts (`usize as f32`) are intentional for tensor math          |
| `cast_possible_truncation`| Same as above                                                              |
| `cast_possible_wrap`      | Same as above                                                              |
| `cast_sign_loss`          | Same as above                                                              |
| `similar_names`           | Tensor-math conventions (`q`, `k`, `v`, `b`, `c`, `h`, `d`)               |
| `too_many_lines`          | Phase D will refactor oversized files; lint enforced after that           |
| `too_many_arguments`      | Phase C builder API work will reduce; lint enforced after that            |
| `multiple_crate_versions` | Dependency cleanup tracked separately                                      |

---

## API Conventions

These conventions ensure a uniform API surface across all crates. **All new public types and trait extensions must follow them**; existing types that violate them should be migrated opportunistically.

### Builder vs Struct Literal

For public types with **more than 2 optional fields**, use a builder pattern instead of struct literals. Builders compose cleanly, document fields with `with_*` methods, and support future field additions without breaking changes.

**Use a builder when:**

- The struct has >2 `Option<T>` or `Default`-able fields
- Fields are commonly set non-default in tests and user code
- Future field additions are likely (avoid API churn)

**Use a struct literal when:**

- The struct has ≤2 fields with obvious semantics
- All fields are required (no defaults)
- The type is a value object with no expected evolution (e.g., coordinate pair)

**Examples (good builders):**

```rust
// vllm_core::speculative::config::SpeculationConfig
let cfg = SpeculationConfig::builder()
    .with_max_draft_tokens(5)
    .with_acceptance_threshold(0.7)
    .build();

// vllm_testing::BatchBuilder
let batch = BatchBuilder::new()
    .with_seq_id(1)
    .with_tokens(vec![10, 20, 30])
    .build();

// vllm_core::speculative::registry::DraftSpec
let spec = DraftSpec::new("qwen-small", model_dir, 1024)
    .with_arch_hint("qwen3")
    .with_weight_size(2_000_000_000);
```

### Crate-Root Re-exports

Every crate MUST re-export its most-used public types at the crate root, even if they live in submodules. This avoids forcing callers to navigate the module hierarchy for common imports.

**Convention:** `lib.rs` ends with a `pub use` block listing the top-level re-exports. Internal modules use `pub(crate)` for things that should NOT be re-exported.

**Example pattern (vllm-core/src/lib.rs):**

```rust
pub use crate::error::{EngineError, Result};
pub use crate::scheduler::{SchedulerEngine, Request, SchedulerConfig};
pub use crate::speculative::{
    DraftId, DraftModelRegistry, DraftSpec, DraftState, LoadedDraft,
};
```

**Crate-root trait re-exports:** When a trait is fundamental and used across many call sites, re-export it from the consuming crate's root (not just from the implementing module). Example: `pub use vllm_traits::ModelBackend;` if you have a wrapper that adds context.

### Error Type Conventions

All error enums follow these rules (Phase 30 / v20.6 invariants):

- Use `#[derive(thiserror::Error)]` — no hand-written `Display`/`Error` impls
- Every variant has `#[error("...")]` for user-facing messages
- When wrapping another error, use `#[source]` to preserve the chain:

```rust
#[derive(Debug, thiserror::Error)]
pub enum DraftRegistryError {
    #[error("draft load failed: {message}")]
    LoadFailedWithSource {
        message: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}
```

- Provide `From<E>` impls for cross-crate error conversion in `error/mod.rs`
- Engine-level errors include a `with_request_id(u64) -> Self` helper for attaching log-correlation context post-construction
- **Never** use `Box<dyn std::error::Error>` in public APIs — always use a typed enum (see Phase 32 / API-03)

### Sync vs Async Trait Splits

Traits that wrap generic operations should be split into sync + async variants when both code paths exist:

```rust
// Pure-computational fallback — sync only
pub trait FallbackStrategy {
    fn execute<T, E>(&self, op: fn() -> Result<T, E>) -> Result<T, E>;
}

// I/O-bound fallback — async
#[async_trait::async_trait]
pub trait AsyncFallbackStrategy {
    async fn execute<F, Fut, T, E>(&self, op: F) -> Result<T, E>
    where
        F: Fn() -> Fut + Send,
        Fut: std::future::Future<Output = Result<T, E>> + Send;
}
```

Callers explicitly pick the trait that matches their runtime. This avoids forcing async onto purely-computational call sites and vice versa.

### Default for Object-Safe Traits

Object-safe traits (no generic methods, no `Self: Sized`) MUST provide `Default` impls when reasonable, so `Arc<dyn Trait>` consumers can construct empty instances. See `DraftVerifier` and `SchedulerObserver` in `vllm-core` for the pattern.

Compile-only `dyn Trait` tests live in `crates/testing/tests/dyn_safety.rs` and verify every public trait compiles as `dyn Trait`.

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

| Pattern                   | Description                             |
| ------------------------- | --------------------------------------- |
| **ModelBackend trait**    | Abstracts ML model implementations      |
| **Paged KV Cache**        | Block-based KV memory management        |
| **Prefix Caching**        | Reuse KV for repeated prompts           |
| **Speculative Decoding**  | Draft-then-verify token generation      |
| **Continuous Batching**   | Dynamic batch scheduling with fairness  |
| **Architecture Registry** | Dynamic model architecture registration |

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

| Architecture | Directory            | Features                      |
| ------------ | -------------------- | ----------------------------- |
| Llama        | `model/src/llama/`   | RMSNorm, RoPE, SwiGLU         |
| Mistral      | `model/src/mistral/` | Sliding Window, GQA           |
| Qwen2/3      | `model/src/qwen3/`   | GQA, MLA, RoPE, QK-Norm       |
| Qwen3.5      | `model/src/qwen3_5/` | Mamba SSM Hybrid, HarmonicSSM |
| Gemma4       | `model/src/gemma4/`  | Hybrid Attention              |
| Mixtral      | `model/src/mixtral/` | Sparse MoE                    |

### Attention Mechanisms

| Type              | Class               | Location                      | Description                                            |
| ----------------- | ------------------- | ----------------------------- | ------------------------------------------------------ |
| GQA               | `GqaAttention`      | `components/attention/gqa.rs` | Grouped-query attention                                |
| MLA               | `MlaAttention`      | `components/attention/mla.rs` | Multi-head Latent Attention (32x KV cache compression) |
| Qwen3Attention    | `Qwen3Attention`    | `qwen3/attention.rs`          | GQA + RoPE + QK-Norm wrapper                           |
| Qwen3MlaAttention | `Qwen3MlaAttention` | `qwen3/mla_attention.rs`      | MLA wrapper for Qwen3                                  |

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

```text
crates/model/src/components/
├── attention/
│   ├── mod.rs         # GqaAttention, MlaAttention, utility functions
│   ├── gqa.rs         # Grouped-query attention implementation
│   └── mla.rs         # Multi-head Latent Attention (DeepSeek-V3)
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
├── block.rs           # StandardBlock (unused, model-specific blocks preferred)
├── ssm.rs             # SSMLayer, MambaBlock, SSMHarmonicSSMLayer
└── vision.rs          # VisionEncoder (placeholder)
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

| Level     | Usage                                   | Coverage |
| --------- | --------------------------------------- | -------- |
| **ERROR** | System failures (config, model loading) | 2 logs   |
| **WARN**  | Degradation (CUDA Graph disabled)       | 7 logs   |
| **INFO**  | Lifecycle (startup, request start/end)  | 18 logs  |
| **DEBUG** | Internal flow (scheduling, batching)    | 35 logs  |
| **TRACE** | Verbose (token, KV cache, attention)    | 20 logs  |

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

| Field           | Type   | Description         |
| --------------- | ------ | ------------------- |
| `request_id`    | string | Request tracking ID |
| `prompt_tokens` | usize  | Input token count   |
| `output_tokens` | usize  | Output token count  |
| `duration_ms`   | u64    | Operation duration  |
| `seq_id`        | SeqId  | Sequence ID         |
| `batch_size`    | usize  | Batch size          |
| `phase`         | Phase  | Prefill/Decode      |

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

| Component                            | Level       | Events                           |
| ------------------------------------ | ----------- | -------------------------------- |
| `server/main.rs`                     | info        | Startup, model load, shutdown    |
| `server/openai/chat.rs`              | info        | Request start/complete           |
| `core/engine.rs`                     | debug       | Model forward, token output      |
| `core/scheduler/engine.rs`           | debug       | Scheduling decision, batch build |
| `core/scheduler/batch_composer.rs`   | debug       | Batch composition                |
| `core/scheduler/memory/allocator.rs` | debug/trace | Block allocation/free            |
| `core/sampling.rs`                   | trace       | Sampling strategy                |
| `model/components/attention/gqa.rs`  | trace       | GQA attention layer              |
| `model/components/attention/mla.rs`  | trace       | MLA attention layer              |
| `model/paged_tensor/tensor_store.rs` | trace       | KV cache read/write              |
| `core/kv_cache/prefix_cache.rs`      | trace       | Prefix cache hit/miss            |
