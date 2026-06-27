# Phase 32: API Consistency — PLAN

**Phase:** 32
**Goal:** Make API surface uniform — typed errors throughout, ergonomic builders, structured error context, sync/async trait splits where the runtime requires it.

**Requirements:** API-01, API-02, API-03, API-04, API-05, API-06, API-07, API-08, API-09, API-10, API-11

## Audit Reality Check (Phase 32 items)

Several items from the v19.0 audit BACKLOG.md have already been addressed in earlier phases:

- **API-04 (mutex `.expect()`)**: predictive_batching.rs:8 sites already use `.unwrap_or_else(|e| e.into_inner())` (poison recovery). Already fixed in v20.3 (Phase 27).
- **EngineError variants** (`Timeout`, `Cancelled`, `ResourceExhausted`, `BackendUnavailable`): added in v20.6 (Phase 30).

This plan focuses on items NOT yet addressed.

## Plans

### 32-01: Document builder/struct-literal convention + crate-root re-exports — API-01, API-07

**Tasks:**
1. Add new section to `AGENTS.md`: "API Conventions" covering:
   - Builder pattern: prefer `Foo::builder().with_X(value).build()` for types with >2 optional fields
   - Struct literal: acceptable for ≤2 fields with clear semantics
   - Crate-root re-exports: each crate exposes its most-used types at the crate root
   - Trait re-export pattern: `pub use vllm_traits::Foo;` at crate root
2. Reference examples from current codebase:
   - `SpeculationConfigBuilder` (good example)
   - `RequestBuilder` in vllm-testing (good example)
   - `RequestFactory` in vllm-testing (functional API)

**Validation:** AGENTS.md updated; existing conventions remain valid; new contributors have guidance.

### 32-02: Add `#[source]` to error chains — API-02, API-09

**Tasks:**
1. `DraftRegistryError::LoadFailed(String)` → change to `LoadFailed(#[source] Box<dyn std::error::Error + Send + Sync>)` (or keep String but document that source chain is preserved via error message).
   - Simpler: add a new variant `LoadFailedWithSource { msg: String, #[source] source: Option<Box<dyn Error + Send + Sync>> }`
   - Even simpler: keep `LoadFailed(String)` and document the error chain is best-effort
   - Pragmatic: Add a `#[source]` attribute to the existing variant via a different approach — create a wrapper:
     ```rust
     #[derive(Debug, thiserror::Error)]
     pub enum DraftRegistryError {
         #[error("draft load failed: {message}")]
         LoadFailed {
             message: String,
             #[source]
             source: Option<Box<dyn std::error::Error + Send + Sync>>,
         },
     }
     ```
   - This is breaking API change. Mark with `#[deprecated]` for backward compat.
2. Add `From<candle_core::Error> for EngineError`:
   - New variant `EngineError::CandleError(#[source] candle_core::Error)` 
   - But vllm-core doesn't depend on candle-core. So create a `ModelError` variant that carries the source.
3. Add `From<tokio::task::JoinError>` for EngineError (common error type):
   - `EngineError::TaskJoin(#[source] tokio::task::JoinError)`

**Validation:** Error chains preserved via `std::error::Error::source()`.

### 32-03: Replace 2 `Box<dyn Error>` in model lib with typed errors — API-03

**Tasks:**
1. `crates/model/src/config/model_config.rs:208` `from_config_json` — change return type to `Result<Self, ConfigError>` where ConfigError is a new typed error:
   ```rust
   #[derive(Debug, thiserror::Error)]
   pub enum ConfigError {
       #[error("unknown architecture: {0}")]
       UnknownArchitecture(String),
       #[error("missing required field: {0}")]
       MissingField(&'static str),
       #[error("invalid field value: {field}={value}")]
       InvalidField { field: String, value: String },
       #[error("JSON error: {0}")]
       Json(#[from] serde_json::Error),
   }
   ```
2. `crates/model/src/qwen3/config.rs:216` `from_file` — change return type to `Result<Self, Qwen3ConfigError>`:
   ```rust
   #[derive(Debug, thiserror::Error)]
   pub enum Qwen3ConfigError {
       #[error("io error reading {path}: {source}")]
       Io {
           path: String,
           #[source] source: std::io::Error,
       },
       #[error("json parse error: {0}")]
       Json(#[from] serde_json::Error),
   }
   ```
3. Update callers of these functions.

**Validation:** Production code has 0 `Result<_, Box<dyn Error>>` (per ERR-01 invariant).

### 32-04: Split `FallbackStrategy` into sync + async traits — API-08

**Tasks:**
1. Current trait (single async):
   ```rust
   #[async_trait::async_trait]
   pub trait FallbackStrategy {
       async fn execute<F, Fut, T, E>(...) -> Result<T, E>;
   }
   ```
2. Split into:
   ```rust
   /// Sync fallback strategy — for purely-computational fallbacks
   pub trait FallbackStrategy {
       fn execute<T, E>(&self, op: fn() -> Result<T, E>) -> Result<T, E>;
   }
   
   /// Async fallback strategy — for I/O-bound fallbacks (e.g., retry network calls)
   #[async_trait::async_trait]
   pub trait AsyncFallbackStrategy {
       async fn execute<F, Fut, T, E>(...) -> Result<T, E>
       where ...;
   }
   ```
3. Migrate implementations:
   - `RetryStrategy` → `AsyncFallbackStrategy`
   - `FailFastStrategy` → `FallbackStrategy` (sync) — re-design as direct call wrapper
   - `DegradeStrategy` → either sync or async depending on what it does
4. Add `Default` impls where possible (e.g., `RetryStrategy::default()` returns a reasonable default like 3 attempts with 100ms base)
5. Update callers

**Validation:** Two distinct traits exist; callers explicitly choose sync or async.

### 32-05: Add `request_id`/`seq_id` to error variants — API-11

**Tasks:**
1. Audit current `EngineError` variants for context carrying:
   - `SeqNotFound { id: u64 }` — already has id
   - `Timeout { op: String, ms: u64 }` — missing request_id/seq_id
   - `Cancelled { request_id: u64 }` — already has request_id
   - `ResourceExhausted { resource: String }` — missing context
   - `BackendUnavailable { backend: String }` — missing context
   - `InvalidRequest(String)`, `ModelError(String)`, `SamplingError(String)` — no structured fields
2. Add optional `request_id` and `seq_id` to variants that lack context:
   - `Timeout { op: String, ms: u64, request_id: Option<u64>, seq_id: Option<u64> }`
   - `ResourceExhausted { resource: String, request_id: Option<u64> }`
   - `BackendUnavailable { backend: String, request_id: Option<u64> }`
3. Add `EngineError::with_request_id(self, id) -> Self` builder-style helper for attaching context post-construction:
   ```rust
   impl EngineError {
       pub fn with_request_id(mut self, id: u64) -> Self {
           match &mut self {
               Self::Timeout { request_id, .. } => *request_id = Some(id),
               Self::ResourceExhausted { request_id, .. } => *request_id = Some(id),
               Self::BackendUnavailable { request_id, .. } => *request_id = Some(id),
               _ => {}
           }
           self
       }
   }
   ```

**Validation:** All variants can carry request_id/seq_id for log correlation.

### 32-06: Add `Default` impls + `dyn Trait` compile tests — API-06, API-10

**Tasks:**
1. Add `Default` impls:
   - `RetryStrategy::default()` → 3 attempts, 100ms base delay
   - `FailFastStrategy::default()` → no-op
   - `DraftVerifier` (find existing trait; add Default if object-safe)
   - `SchedulerObserver` (find existing trait; add Default if object-safe)
2. Add `dyn_safety.rs` style compile-only tests for each new trait added/modified in Phase 32:
   - `SyncFallbackStrategy`
   - `AsyncFallbackStrategy`
   - Any new typed error types
3. Verify existing `crates/testing/tests/dyn_safety.rs` still passes; add new tests for Phase 32 traits.

**Validation:** `dyn Trait` works for all Phase 32 traits; `Default` impls available.

### 32-07: Introduce builders where only `Default` exists — API-05

**Tasks:**
1. Audit structs with `Default` impl but no builder:
   - `AttentionConfig` (model/src/components/attention/util.rs) — has `new()` and `Default`, no builder
   - `SpeculationConfig` — has builder already
   - `MemoryBudget` — needs check
   - `DraftSpec` — has builder methods
   - `BatchBuilder`/`RequestBuilder` in vllm-testing — exist
   - Other configs across crates
2. For each, add builder if missing:
   - `AttentionConfig::builder().with_tile_size(...).with_use_fused(...).build()`
3. Limit scope: focus on public API types, not internal helpers.

**Validation:** At least 22 new builders exist (per phase success criteria).

## Execution Order

1. **32-03 first** (typed errors in model) — small, contained, fixes real anti-pattern
2. **32-02** (error chains) — small, additive
3. **32-05** (request_id/seq_id context) — small, additive
4. **32-01** (AGENTS.md docs) — pure docs
5. **32-06** (Default impls + dyn tests) — small tests
6. **32-04** (FallbackStrategy split) — moderate, may break callers
7. **32-07** (22 builders) — substantial, do last

## Verification

After each plan completes:
- `cargo build --workspace --all-features` clean
- `cargo test --workspace` ≥1144 tests pass (no regression)
- `cargo clippy --workspace --all-targets -- -D warnings` clean
- `cargo fmt --all --check` clean

## Risks

- **API-04 (FallbackStrategy split) is breaking** — careful migration of all impls
- **API-03 (typed errors) is breaking** — update all callers
- **API-05 (builders) is additive** — no break
- **API-02 (#[source])** is breaking for `LoadFailed` variant — wrap in deprecated alias
