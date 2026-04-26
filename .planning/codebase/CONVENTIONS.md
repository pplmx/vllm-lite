# Coding Conventions

**Analysis Date:** 2026-04-26

## Naming Patterns

| Type | Convention | Example |
|------|------------|---------|
| Types | PascalCase | `SchedulerEngine`, `Status` |
| Functions/Variables | snake_case | `add_request`, `running_count` |
| Constants | SCREAMING_SNAKE_CASE | `BLOCK_SIZE`, `MAX_BATCH_SIZE` |
| Modules | snake_case | `queue_manager`, `eviction` |
| Crate names | kebab-case | `vllm-core` |

## Code Style

**Formatting:**
- Tool: `cargo fmt` (Rust standard)
- 4-space indentation
- Max line length: 100 characters (soft limit)

**Linting:**
- Tool: `cargo clippy`
- CI enforcement: `cargo clippy --all-targets --workspace -- -D warnings`

**Allowances:**
- Use `#![allow(clippy::too_many_arguments)]` when needed for function signatures

## Import Organization

Imports are grouped in this order:

1. **Standard library** (`std::`, `core::`)
2. **External crates** (e.g., `tokio::`, `tracing::`, `candle_core::`)
3. **Crate local** (`crate::`, `super::`, `use vllm_traits::`)

```rust
// Example from `crates/core/src/engine.rs`
mod speculative;

use crate::beam::BeamSequence;
use crate::error::Result;
use crate::metrics::{EnhancedMetricsCollector, MetricsCollector};
use crate::scheduler::engine::SchedulerEngine;
use std::collections::HashMap;
use std::marker::PhantomData;
use tokio::sync::mpsc;
use tracing::{error, trace};
use vllm_traits::{BatchOutput, BatchPhase, ModelBackend, Result as ModelResult, SeqId, TokenId};

#[cfg(feature = "cuda-graph")]
use vllm_model::kernels::BatchCudaGraphExecutor;

#[cfg(feature = "cuda-graph")]
use vllm_traits::kernels::CudaGraphConfig;
```

**Key patterns:**
- Use `use crate::types::Request` for absolute crate paths
- Use `use super::sibling::Module` for sibling module access
- Use `vllm_traits::{Type1, Type2}` for external trait crate imports

## Error Handling

**Pattern: Use `thiserror` for error enums**

```rust
// From `crates/core/src/error/mod.rs`
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

impl From<vllm_traits::ModelError> for EngineError {
    fn from(err: vllm_traits::ModelError) -> Self {
        EngineError::ModelError(err.to_string())
    }
}

pub type Result<T> = std::result::Result<T, EngineError>;
```

**Key conventions:**
- Derive `Debug` and `thiserror::Error`
- Use `#[error("...")]` with field interpolation for descriptive messages
- Implement `From` traits for automatic error conversion
- Define a type alias `pub type Result<T> = std::result::Result<T, ErrorEnum>`
- Return `Result<T>` from all fallible functions, use `?` for propagation

## Logging Conventions

**Framework:** `tracing` crate with structured logging

**Log Levels:**

| Level | Usage | Example |
|-------|-------|---------|
| `error!` | System failures (config, model loading) | `error!(error = %e, "Model forward failed")` |
| `warn!` | Degradation (CUDA Graph disabled) | `warn!("CUDA Graph support not enabled")` |
| `info!` | Lifecycle (startup, request start/end) | `info!(address = %addr, "Server listening")` |
| `debug!` | Internal flow (scheduling, batching) | `debug!(batch_size = 4, phase = ?batch.phase, "Batch built")` |
| `trace!` | Verbose (token, KV cache, attention) | `trace!(seq_id = %seq_id, token = %token, "Token generated")` |

**Structured Fields:**
```rust
// Include relevant context as structured fields
tracing::info!(
    request_id = %id,
    prompt_tokens = 150,
    "Request started"
);

tracing::debug!(
    running = 10,
    waiting = 5,
    "Scheduling decision"
);

tracing::trace!(
    layer_idx = 12,
    block_ids = ?blocks,
    "KV cache read"
);
```

**Environment Configuration:**
```bash
RUST_LOG=debug cargo run -p vllm-server  # Enable debug logs
RUST_LOG=trace cargo run                 # Enable trace logs
```

## Documentation Standards

**Public APIs:** Use `///` doc comments

```rust
/// Core inference engine managing requests, scheduling, and model execution.
///
/// The Engine orchestrates the entire inference pipeline:
/// - Receives requests via `add_request`
/// - Schedules batches via the Scheduler
pub struct Engine { ... }
```

**Implementation comments:** Use `//` for inline notes

```rust
// Step 1: Process prompt token
engine.step().unwrap();
assert!(rx.try_recv().is_ok(), "should get first token");
```

**Module-level docs:** Use `//!` at top of module files

```rust
//! Unified mock implementations for testing.
//!
//! This module consolidates all mock ModelBackend implementations
//! previously scattered across the codebase.
```

**Documentation CI Check:**
```bash
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --document-private-items --workspace --all-features
```

## Type Conventions

| Type | Usage |
|------|-------|
| `usize` | Sizes, lengths, counts |
| `u64` | IDs (`SeqId`, `TokenId`) |
| `&T` | Read-only references |
| `&mut T` | Mutable references |
| `Option<T>` | Nullable values (never `null`) |

## Function Design

**Size:** Keep functions focused; use `#![allow(clippy::too_many_arguments)]` when needed

**Return types:** Always return `Result<T>` for fallible operations

**Parameters:** Prefer explicit type annotations in function signatures

## Commit Message Format

```text
<type>(<scope>): <subject>
```

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `refactor` | Code restructuring |
| `test` | Adding/updating tests |
| `docs` | Documentation |
| `chore` | Maintenance |

**Examples:**
```bash
git commit -m "feat(model): add forward_prefill to GqaAttention"
git commit -m "fix(core): resolve prefix cache eviction bug"
git commit -m "test(core): add prefix cache hit test"
```

---

*Convention analysis: 2026-04-26*
