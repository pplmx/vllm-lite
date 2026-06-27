# ADR-011: Cross-Crate Error Type Boundaries

**Date:** 2026-06-27
**Status:** Accepted
**Context version:** v20.3 outcome

## Context

The v19.0 audit (API + error handling theme) catalogued 13 distinct error enums across the workspace (`EngineError`, `ModelError`, `CudaGraphError`, `DraftRegistryError`, `MemoryBudgetExceeded`, `CircuitBreakerError`, `MetricsError`, ...). Most used `thiserror`'s `#[derive(thiserror::Error)]` for ergonomic `Display` and `source()` chains, but the audit also found:

- **`Result<_, String>` anti-patterns** in 10+ locations — functions returning `Result<T, String>` with hand-rolled error messages, losing the ability to programmatically match on error variants.
- **Inconsistent `From` impls** — some cross-crate errors mapped into `EngineError`, others were boxed as `Box<dyn Error>` or wrapped in `anyhow::Error` ad hoc.
- **Missing variants** — `EngineError` lacked `Timeout`, `Cancelled`, `ResourceExhausted`, `BackendUnavailable` until v20.3 added them (`crates/core/src/error/mod.rs:24-34`).
- **Inconsistent conversion paths** — the same foreign error type was converted differently in different modules.

The cross-crate boundary question: when `vllm-core` calls into `vllm-model` and the model returns `ModelError`, what does `vllm-core`'s caller see? Three options:

1. **Single global error type** — everything is `anyhow::Error` or one giant enum. Loses type information; forces callers to downcast.
2. **Each error leaks across the boundary** — `vllm-core` returns `Result<T, ModelError | EngineError | ...>`. Verbose, error-prone.
3. **Each crate owns its error type; cross-crate errors map via `From`** — `vllm-core` sees `EngineError::ModelError(String)` for any model failure, but the original `ModelError` is preserved in the `source()` chain.

## Decision

Each crate owns its error enum using `#[derive(thiserror::Error)]`. Cross-crate `From` impls map foreign errors into the local enum. Binary entry points (`vllm-server/src/main.rs`) use `anyhow::Result` for top-level ergonomics.

Pattern, with `vllm-core` as the canonical example (`crates/core/src/error/mod.rs`):

```rust
// vllm-core owns EngineError
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
    #[error("internal lock poisoned")]
    LockPoisoned,
    #[error("operation '{op}' timed out after {ms} ms")]
    Timeout { op: String, ms: u64 },
    #[error("request {request_id} was cancelled")]
    Cancelled { request_id: u64 },
    #[error("resource exhausted: {resource}")]
    ResourceExhausted { resource: String },
    #[error("backend unavailable: {backend}")]
    BackendUnavailable { backend: String },
}

// Cross-crate From impls — module/file-local to error/mod.rs
impl From<vllm_traits::ModelError> for EngineError {
    fn from(err: vllm_traits::ModelError) -> Self {
        EngineError::ModelError(err.to_string())
    }
}

impl From<crate::speculative::DraftRegistryError> for EngineError {
    fn from(err: crate::speculative::DraftRegistryError) -> Self {
        EngineError::ModelError(format!("draft registry: {}", err))
    }
}

impl From<crate::speculative::memory_budget::MemoryBudgetExceeded> for EngineError {
    fn from(err: crate::speculative::memory_budget::MemoryBudgetExceeded) -> Self {
        EngineError::ResourceExhausted { resource: format!("memory_budget: {}", err)) }
    }
}

impl From<crate::circuit_breaker::breaker::CircuitBreakerError> for EngineError {
    fn from(err: crate::circuit_breaker::breaker::CircuitBreakerError) -> Self {
        EngineError::BackendUnavailable { backend: format!("circuit_breaker: {}", err) }
    }
}

impl From<crate::metrics::exporter::MetricsError> for EngineError {
    fn from(err: crate::metrics::exporter::MetricsError) -> Self {
        EngineError::ModelError(format!("metrics: {}", err))
    }
}

pub type Result<T> = std::result::Result<T, EngineError>;
```

The `?` operator then "just works" across crate boundaries: `let x = model.forward(...)?;` in `vllm-core` automatically converts `ModelError` to `EngineError::ModelError` via the `From` impl.

Same pattern repeats in `vllm-model` (`crates/model/src/error.rs` → `ModelError`), `vllm-server`, and `vllm-dist`. Each crate has exactly one primary error enum.

Binary entry points (`crates/server/src/main.rs`) use `anyhow::Result<T>` so setup failures (config parse, model load, network bind) can be reported without ceremony.

## Rationale

1. **Type safety with low friction** — the `?` operator converts at the boundary; callers don't write explicit `.map_err`.
2. **Caller-visible categorisation** — `EngineError::Timeout`, `Cancelled`, `ResourceExhausted`, `BackendUnavailable` are first-class variants the HTTP layer can map to specific status codes (504, 499, 503, 502 respectively).
3. **Source chain preserved** — `EngineError::ModelError(msg)` contains a `String` representation, but the original `ModelError`'s full chain is reachable via the conversion site's log line. (`String` was chosen over wrapping the original to avoid leaking `vllm_traits::ModelError` into `vllm-core`'s public surface.)
4. **Pattern-matchable** — every variant is a struct or tuple variant; no `Box<dyn Error>` to downcast.
5. **Consistent style** — every crate uses `thiserror::Error` + `#[error("...")]`; no mixing of hand-rolled `Display` + manual `Error::source()`.
6. **`anyhow` only at the top** — `main.rs` uses `anyhow::Result` because setup failures are diverse and not worth cataloguing; internal APIs use `Result<T, CrateError>`.

Alternatives considered:

- **Single global error enum** — rejected; would force `vllm-core` to depend on `vllm-dist`'s error types and vice versa, recreating the layering violations v20.0 was eliminating.
- **`Box<dyn Error>` everywhere** — rejected; loses pattern-matchability, makes HTTP status mapping guesswork.
- **Hand-rolled `impl Display + impl Error` without `thiserror`** — rejected; `thiserror` removes boilerplate and standardises the `source()` chain.
- **Preserve original error types across the boundary via `From<ModelError> for EngineError` that wraps the original** — rejected; the original error's variants would leak into `vllm-core`'s public API. Converting to a `String` representation at the boundary is the cleanest split.
- **Only use `anyhow`** — rejected; loses programmatic error matching in the engine core where it matters.

## Consequences

**Positive:**

- The `?` operator works seamlessly across crate boundaries.
- HTTP layer can pattern-match on `EngineError` variants to produce correct status codes (timeout → 504, cancelled → 499, exhausted → 503, unavailable → 502).
- Each crate's public API surfaces only its own error enum; no transitive error-type dependencies.
- `thiserror`'s `#[source]` attribute (when needed) preserves the original error chain in logs.
- New error variants are added in one place (`EngineError`) and are immediately usable across the crate.

**Negative:**

- Converting foreign errors to `String` at the boundary loses structured information (e.g. `ModelError::OutOfMemory { requested_bytes }` becomes `"out of memory"`). Callers can't programmatically react to specific failure modes.
- `From` impl proliferation — every cross-crate error type needs an `impl From<X> for Y`. Adding a new error type requires updating multiple `From` blocks.
- `EngineError::ModelError(String)` is a catch-all — it's easy for new error variants to be smuggled in as `String` payloads, bypassing the typed enum.
- Tests that assert on specific error variants must either use the local error type or parse the `String` representation.

**Mitigations / migration paths:**

- For high-value foreign errors, wrap rather than stringify: `EngineError::Model(ModelError)` instead of `EngineError::ModelError(String)`. The current codebase chose `String` for simplicity; revisit when a specific cross-crate error variant needs structured handling.
- A central `From` impl registry in each crate's `error/mod.rs` makes the conversion surface auditable.
- The audit-style review (v19.0) can re-check that no new `Result<_, String>` anti-patterns have crept in; CI grep for `Result<.*String>` in non-test code is a cheap guard.
