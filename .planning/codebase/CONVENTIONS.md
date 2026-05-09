# Coding Conventions

**Last updated:** 2026-05-09
**Focus:** Quality

## Code Style

### Imports
- Absolute imports preferred: `use crate::types::Request;`
- External crates: `use vllm_traits::{ModelBackend, SeqId};`
- Group order: std → external → crate
- `super` for sibling module access

### Formatting
- 4-space indentation (Rust standard)
- 100-character soft line limit
- `cargo fmt` required before commits
- `cargo clippy --all-targets --workspace -- -D warnings` required

### Lint Suppressions
- `#![allow(clippy::too_many_arguments)]` — common in attention layers (`gqa.rs:1`, `mla.rs:1`)
- `#![allow(dead_code)]` — used in WIP code (`chat.rs:68`, `completions.rs:26`, `embeddings.rs:9`)
- `#[allow(clippy::derivable_impls)]` — server config (`config.rs`, `cli.rs`)
- `#[allow(clippy::should_implement_trait)]` — RBAC (`rbac.rs:13`)

## Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Crates | kebab-case | `vllm-core`, `vllm-model` |
| Modules | snake_case | `scheduler/engine.rs`, `attention/gqa.rs` |
| Types | PascalCase | `SchedulerEngine`, `GqaAttention` |
| Traits | PascalCase | `ModelBackend`, `SchedulingPolicy` |
| Functions | snake_case | `add_request`, `build_batch` |
| Variables | snake_case | `running_count`, `max_batch_size` |
| Constants | SCREAMING_SNAKE_CASE | `BLOCK_SIZE`, `MAX_BATCH_SIZE` |
| Error types | PascalCase | `EngineError`, `VerifierError` |

## Error Handling

### Pattern: `thiserror` enums with `Result<T>` type aliases
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

### Error type locations:
- `crates/core/src/error/mod.rs` — `EngineError`
- `crates/traits/src/model.rs` — `ModelError`
- `crates/traits/src/types.rs` — `TensorParallelError`
- `crates/core/src/speculative/verifier.rs` — `VerifierError`
- `crates/core/src/circuit_breaker/breaker.rs` — `CircuitBreakerError`
- `crates/core/src/metrics/exporter.rs` — `MetricsError`
- `crates/server/src/security/jwt.rs` — `JwtError`
- `crates/server/src/security/tls.rs` — `TlsError`
- `crates/model/src/components/ssm.rs` — `SSMError`
- `crates/dist/src/pipeline/mod.rs` — `PipelineError`

### Error conversion:
```rust
impl From<vllm_traits::ModelError> for EngineError {
    fn from(err: vllm_traits::ModelError) -> Self {
        EngineError::ModelError(err.to_string())
    }
}
```

## Type Conventions
- `usize` for sizes/lengths
- `u64` for IDs (`SeqId`, `TokenId`)
- `u32` for token values
- `Option<T>` for nullable values (never `null`)
- `&T` for read-only, `&mut T` for mutable references

## Logging

5-level structured logging with `tracing` crate:

```rust
use tracing::{info, debug, trace, warn, error};

// Structured fields with key=value syntax
info!(request_id = %id, prompt_tokens = 150, "Request started");
debug!(batch_size = 4, phase = ?batch.phase, "Batch built");
trace!(seq_id = %seq_id, token = %token, "Token generated");
warn!("CUDA Graph disabled, falling back");
error!(error = %e, "Model forward failed");
```

### Log level distribution:
- **ERROR** — System failures (config, model loading) — 2 logs
- **WARN** — Degradation (CUDA Graph disabled) — 7 logs
- **INFO** — Lifecycle (startup, request start/end) — 18 logs
- **DEBUG** — Internal flow (scheduling, batching) — 35 logs
- **TRACE** — Verbose (token, KV cache, attention) — 20 logs

## Common Patterns

### Feature-gated code
```rust
#[cfg(feature = "cuda-graph")]
pub fn step_with_graph(&mut self) -> Result<Vec<(SeqId, TokenId)>> { ... }

#[cfg(not(feature = "cuda-graph"))]
pub fn step_with_graph(&mut self) -> Result<Vec<(SeqId, TokenId)>> {
    tracing::warn!("CUDA Graph support not enabled, using regular step");
    self.step()
}
```

### Architecture registration pattern
```rust
// In register.rs
pub fn register(registry: &ArchitectureRegistry) {
    registry.register("llama", || Box::new(LlamaArchitecture));
}
```

### Test pattern: inline `#[cfg(test)]` modules
- Tests are placed in the same file as implementation
- Use `StubModel` or `FakeModel` for mocking `ModelBackend`

### Unsafe code
Minimal: only in `crates/model/src/loader/io.rs` (memory-mapped file I/O with `std::slice::from_raw_parts`) and `crates/model/src/kernels/cuda_graph.rs` (unsafe impl Send for CudaGraph).

## Documentation
- Public APIs documented with `///` doc comments (some areas sparse)
- AGENTS.md serves as primary developer onboarding document
- README not present (project uses `.planning/` for docs)
