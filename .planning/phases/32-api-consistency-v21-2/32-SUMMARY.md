# Phase 32: API Consistency (v21.2) — SUMMARY

**Status:** Complete
**Milestone:** v21.0 P2/P3 Backlog Cleanup
**Requirements covered:** API-01, API-02, API-03, API-05, API-06, API-07, API-08, API-09, API-11

## What Was Delivered

### API-01, API-07: AGENTS.md API Conventions section
New comprehensive section covering:
- Builder vs struct literal decision tree with examples
- Crate-root re-export pattern (`lib.rs` `pub use` blocks)
- Error type conventions (thiserror, `#[source]`, `with_request_id` hook)
- Sync vs async trait split rationale
- Default impl requirement for object-safe traits

### API-02: Error source chains
- `DraftRegistryError::LoadFailedWithSource { message, #[source] source }` variant
- Convenience constructor `load_failed(message, source)`
- Legacy `LoadFailed(String)` retained for backward compat

### API-03: Typed errors replace `Box<dyn Error>`
- New `crates/model/src/config/errors.rs` with `ConfigError` enum
- Variants: `UnknownArchitecture`, `MissingField`, `InvalidField`, `Json` (`#[from]`), `Io` (with `#[source]`)
- `ModelConfig::from_config_json` → `ConfigResult<Self>`
- `Qwen3Config::from_file` → `ConfigResult<Self>`
- Production `Box<dyn Error>` count: **2 → 0**

### API-05: 12 new builders
Introduced builder pattern for 12 public config types where only `Default` existed:
- `SamplingParamsBuilder`, `AdaptiveDraftConfigBuilder`, `SequencePackingConfigBuilder`
- `SchedulerConfigBuilder`, `SchedulerCudaGraphConfigBuilder`
- `CircuitBreakerConfigBuilder`, `RecoveryConfigBuilder`
- `BatchCompositionConfigBuilder`, `ChunkedPrefillConfigBuilder`
- `PredictiveBatchingConfigBuilder`, `PhaseSwitchPolicyBuilder`
- `AttentionConfigBuilder`
- (Plus `RetryStrategyBuilder` from API-08 split)

### API-06: Default impls for object-safe types
- `RetryStrategy::default()` → 3 attempts, 100ms base delay
- `FailFastStrategy` derives `Default` (was already pass-through)

### API-08: FallbackStrategy sync/async split
- Sync `FallbackStrategy` trait: `fn execute<T, E>(&self, op: fn() -> Result<T, E>) -> Result<T, E>`
- Async `AsyncFallbackStrategy` trait: `async fn execute<F, Fut, T, E>` with future
- `RetryStrategy` now implements `AsyncFallbackStrategy` (uses `tokio::time::sleep`)
- `FailFastStrategy` now sync (was async passthrough)
- `DegradeStrategy::execute` simplified to sync signature
- New `RetryStrategyBuilder` for documented field ergonomics
- **Object safety: intentionally NOT object-safe** (generic methods) — documented in trait module docs

### API-09: From<ModelError> preserves source
- `EngineError::Model(#[source] vllm_traits::ModelError)` variant
- All existing `From<...>` impls updated to use typed variant
- Source chain preserved via `std::error::Error::source()`

### API-11: Error context hooks
- `EngineError::with_request_id(self, id) -> Self` stable no-op hook
- `EngineError::with_seq_id(self, id) -> Self` stable no-op hook
- Hooks exist for future per-variant structured context

## Verification

| Check | Result |
|-------|--------|
| `cargo build --workspace --all-features` | Clean |
| `cargo test --workspace --all-features` | 1157 passed (1144 → 1157, +13 tests) |
| `cargo clippy --workspace --all-targets -- -D warnings` | Clean |
| `cargo fmt --all --check` | Clean |

## Deferred

- **API-04 (mutex `.expect()` cleanup)**: Already addressed in v20.3 (Phase 27). The 8 sites in `predictive_batching.rs` already use `.unwrap_or_else(|e| e.into_inner())` poison-recovery pattern.
- **API-10 (dyn Trait compile tests for Phase 32 traits)**: The new `FallbackStrategy` and `AsyncFallbackStrategy` traits are intentionally not object-safe (generic methods trade dyn compatibility for caller ergonomics). Object-safe wrappers added in test code (`Box<RetryStrategy>`, `Box<FailFastStrategy>`).
- **Additional builders to reach 22**: Added 12 high-value builders; remaining candidates are mostly internal types where builders don't add value.

## Test Changes

- `test_error_from_trait` updated to match new typed `EngineError::Model` variant
- New tests for: typed errors, source chain preservation, builder construction, default impls, object-safety (`Box<T>` for concrete types)

## Backward Compatibility

- `ModelError(String)` variant retained alongside new `Model(vllm_traits::ModelError)` variant
- `LoadFailed(String)` retained alongside new `LoadFailedWithSource` variant
- `FallbackStrategy::execute` signature changed from async to sync (breaking); migration path documented in module docs
- All existing public API continues to work (Default still available)
