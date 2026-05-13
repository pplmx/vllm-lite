# Coding Conventions

**Analysis Date:** 2026-05-13

## Naming Patterns

**Files:**

- Rust sources: `snake_case.rs` (e.g., `engine.rs`, `batch_composer.rs`, `rms_norm.rs`)
- Module directories: `snake_case/` (e.g., `scheduler/`, `components/positional/`)
- Crate names: `kebab-case` (`vllm-core`, `vllm-model`, `vllm-server`)

**Types (structs, enums, traits):**

- `PascalCase` — `SchedulerEngine`, `BatchPhase`, `ModelBackend`, `CudaGraphConfig`, `GraphExecutionError`

**Functions and Methods:**

- `snake_case` — `add_request`, `build_batch`, `is_empty`, `vocab_size`
- Builder methods: `with_*` (e.g., `with_kv_blocks`, `with_model_dir`, `with_flash`)
- Test functions: `test_<component>_<expected_behavior>` (e.g., `test_engine_add_request`, `test_prefill_batch_includes_all_prompt_tokens`)

**Variables:**

- `snake_case` — `seq_id`, `num_computed_tokens`, `max_batch_size`

**Constants:**

- `SCREAMING_SNAKE_CASE` — `BLOCK_SIZE`, `MAX_BATCH_SIZE`

**Lifetimes:**

- Single-char when obvious (`'a`), meaningful names for clarity. Not widely used — this codebase mostly avoids explicit lifetime annotations.

## Code Style

**Formatting:**

- Tool: `rustfmt` (run via `cargo fmt --all` or `just fmt-check`)
- Indentation: 4 spaces (Rust standard)
- Max line length: 100 characters (soft limit)
- No `rustfmt.toml` custom config — project uses Rust defaults
- Trailing commas encouraged in struct literals and match arms

**Linting:**

- Tool: `clippy` with deny-warnings: `cargo clippy --all-targets --workspace -- -D warnings`
- CI runs `cargo clippy --workspace -- -D warnings`
- Allowed exceptions use explicit `#[allow(...)]` attributes:
    - `#[allow(clippy::too_many_arguments)]` — used on `ModelBackend::forward_to_layer` in `crates/traits/src/model.rs:107`
    - `#![allow(dead_code)]` — used on module-level for work-in-progress code (e.g., `crates/core/src/sampling.rs:1`, `crates/server/src/logging.rs:1`)
    - `#![allow(unused_variables)]` — used sparingly (e.g., `crates/core/src/sampling.rs:2`)

**Edition:**

- Rust edition 2024 (specified in workspace `Cargo.toml`)

**Cargo Workspace:**

- 2nd-gen resolver (`resolver = "2"`)
- 7 workspace members: `crates/core`, `crates/model`, `crates/server`, `crates/traits`, `crates/dist`, `crates/testing`, `benches`
- Release profile: `opt-level = 3`, `lto = "fat"`, `codegen-units = 1`, `panic = "abort"`

## Import Organization

**Ordering (std → external → crate → super):**

```rust
// 1. Standard library
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// 2. External crates
use tokio::sync::mpsc;
use tracing::{error, trace};
use serde::{Deserialize, Serialize};

// 3. Workspace crates (absolute paths)
use vllm_traits::{BatchOutput, BatchPhase, ModelBackend, Result as ModelResult, SeqId, TokenId};
use vllm_model::kernels::BatchCudaGraphExecutor;

// 4. Current crate (absolute paths preferred)
use crate::scheduler::engine::SchedulerEngine;
use crate::types::{EngineMessage, Request, SchedulerConfig};

// 5. Sibling modules (super, used sparingly)
use super::*;  // Only in #[cfg(test)] modules
```

**Key conventions observed:**

- Always use absolute crate paths: `use vllm_traits::...` not relative paths
- Prefer `crate::` for intra-crate references
- `use super::*` used almost exclusively within `#[cfg(test)] mod tests` blocks
- Group imports from the same module: `use tracing::{error, trace};`
- Consolidate multi-line imports with `use crate::scheduler::{A, B, C};` style

**Feature-gated imports:**

```rust
#[cfg(feature = "cuda-graph")]
use vllm_model::kernels::BatchCudaGraphExecutor;

#[cfg(feature = "candle")]
use candle_core::Tensor;
```

**Path Aliases:**

- Not detected — no `#[path]` attributes or module aliasing

## Error Handling

**Primary Framework: `thiserror` (version "2")**

Used for structured error enums with `#[derive(Debug, thiserror::Error)]`:

```rust
// Example from crates/core/src/error/mod.rs:3
#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("sequence {id} not found")]
    SeqNotFound { id: u64 },

    #[error("invalid request: {0}")]
    InvalidRequest(String),

    #[error("model forward failed: {0}")]
    ModelError(String),

    #[error("sampling failed: {0}")]
    SamplingError(String),
}
pub type Result<T> = std::result::Result<T, EngineError>;
```

**Other thiserror usages:**

- `GraphExecutionError` in `crates/traits/src/kernels.rs:64` — CUDA graph errors
- `ExporterError` in `crates/core/src/metrics/exporter.rs:12`
- `CircuitBreakerError` in `crates/core/src/circuit_breaker/breaker.rs:35`
- `PipelineError` in `crates/dist/src/pipeline/mod.rs:9`
- Various security errors in `crates/server/src/security/`

**Manual Error Pattern (without thiserror):**

- `ModelError` in `crates/traits/src/model.rs:4` — manual `Display` + `Error` impl
- `TensorParallelError` in `crates/traits/src/types.rs:78` — manual `Display` + `Error` impl

**Error Conversion:**
Use `From<T>` trait for automatic error conversion:

```rust
// crates/core/src/error/mod.rs:18
impl From<vllm_traits::ModelError> for EngineError {
    fn from(err: vllm_traits::ModelError) -> Self {
        EngineError::ModelError(err.to_string())
    }
}
```

**Type Aliases:**

- Each crate/module defines its own `pub type Result<T> = std::result::Result<T, XxxError>;`
- Examples: `crates/core/src/error/mod.rs:24`, `crates/traits/src/model.rs:29`

**Propagation:**

- Use `?` operator for error propagation
- Return `Result<T>` from fallible functions — never use `unwrap()` or `expect()` in production code paths
- `unwrap()` used only in test code and configuration initialization

## Logging

**Framework: `tracing`**

No direct `log` crate usage — everything goes through `tracing` macros.

**Levels and Usage:**

| Level    | Usage                                   | Example           |
| -------- | --------------------------------------- | ----------------- |
| `error!` | System failures (config, model loading) | `tracing::error!` |
| `warn!`  | Degradation (CUDA Graph disabled)       | `tracing::warn!`  |
| `info!`  | Lifecycle (startup, request start/end)  | `tracing::info!`  |
| `debug!` | Internal flow (scheduling, batching)    | `tracing::debug!` |
| `trace!` | Verbose (token details, KV cache)       | `tracing::trace!` |

**Structured Logging:**

```rust
// Key-value structured fields
trace!(seq_id = %seq_id, token = %token, "Token generated");
trace!(layer_idx = 12, block_ids = ?blocks, "KV cache read");
info!(request_id = %id, prompt_tokens = 150, "Request started");
debug!(batch_size = 4, phase = ?batch.phase, "Batch built");
error!(error = %e, "Model forward failed");
```

**Import pattern:**

```rust
use tracing::{error, trace};  // Only import needed macros
```

**Configuration:**

- Controlled via `RUST_LOG` environment variable (default: `info`)
- Server-side dual output: console (compact format) + file (JSON format) in `crates/server/src/logging.rs:7`
- File logging uses `tracing_appender` with daily rotation
- Log directory specified via `--log-dir` CLI flag

## Comments

**Documentation Comments:**

- `///` for public API documentation on types, functions, structs
- `//!` for module-level documentation (e.g., `crates/testing/src/lib.rs:1`, `crates/testing/src/harness.rs:1`)
- Doc examples use `/// # Example` sections with ```` ```rust,ignore ```` blocks
- `/** ... */` style explicitly avoided per `AGENTS.md`

**Implementation Comments:**

- `//` for inline explanations
- Comments describe *why* not *what* — behavior descriptions are in docs

**When to Comment:**

- Public API items MUST have `///` doc comments
- Complex algorithms get inline comments explaining approach
- Test files include comments describing expected flow (e.g., `crates/core/tests/integration.rs:28-30`)

**Dead Code Annotations:**

- `#[allow(dead_code)]` on modules still under construction (not a permanent fix)
- Found in `crates/core/src/sampling.rs`, `crates/server/src/logging.rs`

## Function Design

**Size:**

- Most functions are compact (10-40 lines of logic)
- Large functions (100+ lines) exist in complex orchestrators like `Engine::step()` in `crates/core/src/engine.rs`
- Helper functions extracted for test setup (e.g., `make_sequence`, `create_test_engine`)

**Constructors:**

- `new()` — primary constructor, often with minimal config
- `from_config(...)` — alternative constructor (e.g., `TestHarness::from_config`)
- `default()` via `Default` trait where sensible

**Builder Pattern:**
Widely used for configuration objects. Methods take `mut self` and return `Self`:

```rust
// crates/testing/src/harness.rs:31
pub fn kv_blocks(mut self, n: usize) -> Self {
    self.kv_blocks = n;
    self
}
```

Examples:

- `TestHarnessConfig` in `crates/testing/src/harness.rs`
- `RequestConfig` in `crates/testing/src/request_factory.rs`
- `ModelLoader::builder()` in `crates/model/src/loader/builder.rs`
- `BatchBuilder` / `RequestBuilder` in `crates/testing/src/builders/mod.rs`

**Parameters:**

- Use `&T` for read-only access, `&mut T` for mutation
- `Option<T>` for nullable values — never use raw pointers or null
- Explicit type annotations in function signatures
- `#[allow(clippy::too_many_arguments)]` on functions needing many params

**Return Values:**

- `Result<T, Error>` for fallible operations
- `Option<T>` only when "not found" is expected and non-error
- Prefer returning owned values over references where semantics allow

## Module Design

**Module Structure:**

- `mod.rs` files used for module roots (not `module_name.rs` at crate root level)
- Submodules declared within `mod.rs`: `pub mod submodule;`
- Separate `tests.rs` file for extracted test modules (e.g., `crates/core/src/scheduler/packing/tests.rs`, `crates/core/src/scheduler/policy/tests.rs`)

**Exports (Barrel Pattern):**

- `pub use` re-exports in `lib.rs` for public API surface:

```rust
// crates/traits/src/lib.rs
pub use kernels::{CudaGraphConfig, GraphExecutionError, ModelGraphConfig};
pub use model::{ModelBackend, ModelError, Result};
pub use types::{BLOCK_SIZE, Batch, BatchOutput, BatchPhase, BlockId, SeqId, TensorParallelError, TokenId};
```

**Prelude Pattern (for testing crate):**

```rust
// crates/testing/src/lib.rs
pub mod prelude {
    pub use super::{
        ConstModel, FakeModel, IncrementModel, NeverProgressModel, RequestFactory, SlowModel,
        StubModel, TestHarness,
    };
}
```

**Visibility:**

- Items are private by default
- `pub(crate)` for internal crate APIs
- `pub` only for public API surface and trait methods

## Type Conventions

| Domain        | Type              | Location                       |
| ------------- | ----------------- | ------------------------------ |
| Sequence IDs  | `SeqId = u64`     | `crates/traits/src/types.rs:6` |
| Token IDs     | `TokenId = u32`   | `crates/traits/src/types.rs:5` |
| Block IDs     | `BlockId = usize` | `crates/traits/src/types.rs:4` |
| Sizes/Lengths | `usize`           | Convention                     |
| Counters      | `usize` or `u64`  | Context-dependent              |

## Derive Macros

Commonly applied derive macros (in typical order):

```rust
#[derive(Clone, Debug, Serialize, Deserialize)]           // Config structs
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]  // Enums
#[derive(Debug, thiserror::Error)]                        // Error types
#[derive(Debug, Clone)]                                   // Simple types
```

## Security Patterns

**Secrets handling:**

- Environment variables via `std::env::var()` — used for configuration (e.g., `crates/traits/src/kernels.rs:41`)
- JWT authentication in `crates/server/src/security/jwt.rs`
- TLS configuration in `crates/server/src/security/tls.rs`
- RBAC in `crates/server/src/security/rbac.rs`
- Audit logging in `crates/server/src/security/audit.rs`

## Dependency Management

- All dependencies versioned in workspace `Cargo.toml` under `[workspace.dependencies]`
- Individual crates reference workspace deps: `vllm-traits = { workspace = true }`
- Feature flags: `cuda` (Candle CUDA), `gguf` (GGUF loading), `full` (cuda+gguf)
- `candle-core` gated behind `#[cfg(feature = "candle")]`

---

*Convention analysis: 2026-05-13*
