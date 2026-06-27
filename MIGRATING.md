# MIGRATING.md — vllm-lite Version Changelog

This document tracks breaking changes and migration paths for vllm-lite. For each
version, affected APIs and how to update your code are listed.

## v21.0 (2026-06-27) — P2/P3 Backlog Cleanup

### Module Layout (v21.1)

#### `qwen3_config` module moved to `qwen3::config`

The `qwen3_config` module at `vllm_model::qwen3_config` has been moved to
`vllm_model::qwen3::config`. The old path is preserved as a `#[deprecated]`
re-export shim for one minor release.

```rust
// Before
use vllm_model::qwen3_config::Qwen3Config;

// After (preferred)
use vllm_model::qwen3::config::Qwen3Config;

// Or via flattened re-export
use vllm_model::qwen3::Qwen3Config;
```

#### `speculative::draft_registry` reorganized into `speculative::registry`

The `DraftModelRegistry` and related types have been split into focused
sub-modules under `vllm_core::speculative::registry/`. The old path is
preserved as a `#[deprecated]` re-export shim.

```rust
// Before
use vllm_core::speculative::draft_registry::DraftModelRegistry;

// After (preferred)
use vllm_core::speculative::registry::DraftModelRegistry;
```

#### `engine::spec_dispatch` sub-tree

The single-file `engine/speculative.rs` has been split into a focused
sub-tree at `engine/spec_dispatch/`. Renamed from `engine/speculative` to
avoid namespace conflict with `crate::speculative` (the broader speculative
module). No public API changes.

#### `TensorParallelError` canonical home

The canonical import path for `TensorParallelError` is now
`vllm_dist::error::TensorParallelError`. The previous path
`vllm_traits::TensorParallelError` remains the technical definition site
(for dependency direction reasons) and continues to work.

```rust
// Before
use vllm_traits::TensorParallelError;

// After (canonical)
use vllm_dist::error::TensorParallelError;
```

### API Consistency (v21.2)

#### `ConfigError` replaces `Box<dyn std::error::Error>`

`ModelConfig::from_config_json` and `Qwen3Config::from_file` now return
typed `ConfigResult<Self>` instead of `Result<Self, Box<dyn std::error::Error>>`.

```rust
// Before
fn from_config_json(value: &Value) -> Result<Self, Box<dyn std::error::Error>>;

// After
fn from_config_json(value: &Value) -> ConfigResult<Self>;
```

#### Error source chain preservation

`EngineError::Model` (new typed variant) carries the underlying
`vllm_traits::ModelError` via `#[source]`. The legacy `EngineError::ModelError(String)`
variant is retained for free-form cases.

```rust
// New: typed variant with preserved source chain
let inner = vllm_traits::ModelError::new("candle backend crashed");
let err = EngineError::from(inner);
// std::error::Error::source(&err) returns Some(inner)
```

`DraftRegistryError::LoadFailedWithSource` provides the same pattern for
draft load failures.

#### `FallbackStrategy` split into sync + async

`FallbackStrategy` is now a sync trait. Async behavior moved to the new
`AsyncFallbackStrategy` trait.

```rust
// Before
#[async_trait::async_trait]
impl FallbackStrategy for MyStrategy {
    async fn execute<F, Fut, T, E>(&self, operation: F) -> Result<T, E>
    where F: Fn() -> Fut + Send, ...;
}

// After (sync)
impl FallbackStrategy for MyStrategy {
    fn execute<T, E>(&self, op: fn() -> Result<T, E>) -> Result<T, E>;
}

// Or (async)
#[async_trait::async_trait]
impl AsyncFallbackStrategy for MyStrategy {
    async fn execute<F, Fut, T, E>(&self, operation: F) -> Result<T, E>
    where F: Fn() -> Fut + Send, ...;
}
```

#### New builders (12 added)

Builder pattern now available for: `SamplingParams`, `AdaptiveDraftConfig`,
`SequencePackingConfig`, `SchedulerConfig`, `SchedulerCudaGraphConfig`,
`CircuitBreakerConfig`, `BatchCompositionConfig`, `ChunkedPrefillConfig`,
`PredictiveBatchingConfig`, `PhaseSwitchPolicy`, `AttentionConfig`,
`RecoveryConfig`. Use `Type::builder().with_field(value).build()` instead of
struct literals.

### Naming Consistency (v21.3)

#### `flash_v3.rs` renamed to `flash_attention_v3.rs`

Module file renamed to match V2 naming pattern. No public type renames.

#### Non-tensor single-letter variables renamed in sampling code

- `r` (random_f32 result) → `random_threshold`
- `k` (top_k clamp value) → `top_k_limit`

Both in `crates/core/src/sampling.rs` and `crates/core/src/engine.rs:get_top_k`.

### External Doc Fixes (v21.4)

No breaking changes; documentation-only.

### P3 + Final Verification (v21.5)

#### `CircuitBreakerError::HalfOpenRejected(u32)` variant added

New error variant emitted when half-open circuit rejects a probe call
because `half_open_max_calls` was already reached.

```rust
// New variant
match circuit_breaker.try_call(|| do_work()) {
    Err(CircuitBreakerError::HalfOpenRejected(max)) => {
        // back off: probe budget exhausted
    }
    // ...
}
```

#### `gemma4::Gemma4Attention::default()` uses documented `.expect()` instead of `.unwrap()`

4 sites in `crates/model/src/gemma4/attention.rs:362-380` replaced `.unwrap()`
with `.expect("...")` carrying descriptive allocation-failure messages.
Per ERR-03: production `.expect()` sites ≤5, each documented.

#### Removed dead `crates/traits/tests/mod.rs`

The `tests/mod.rs` aggregator was creating a duplicate test binary. The
standalone `tests/model_backend.rs` test binary remains and runs all 11 tests.

## v20.0 (2026-06-27) — Codebase Remediation

### Major Changes

- **vllm-dist feature-gated** behind `--features multi-node` (ADR-008)
- **ModelError** converted from struct to enum
- **CudaGraphError** uses `#[derive(thiserror::Error)]`
- 8 non-object-safe traits made dyn-compatible
- Doc coverage: 19.5% → 97.8%
- 12 ADRs created in `docs/adr/`
- 776 `pub` items gained `///` doc comments

### Renames / Deprecations

- `EmbeddingData` → `Embedding` (with `#[deprecated]` alias)
- 7 P1 + 19 P2 naming fixes applied (NAM-01/02)

See commit history and Phase 25-30 SUMMARYs for complete details.

## Earlier Versions

Version history prior to v20.0 is available via git history:

```bash
git log --oneline --grep="^docs: start milestone"
```

## Audit Closure Note

The v21.0 milestone closed 100% of the v19.0 codebase audit backlog (5 P0,
38 P1, 44 P2, 13 P3 = 100 findings). Two findings (ARCH-F-09 `#[path]`
directives, API-F-29 `DraftLoader::load` returning `Box<dyn ModelBackend>`)
were not addressed via dedicated v21.0 plans but were resolved implicitly
by earlier phases' work:

- **ARCH-F-09** — resolved by Phase 26 (test files moved out of `src/`)
- **API-F-29** — resolved by Phase 25 (`ModelBackend` made object-safe)

See `.planning/v21.0-MILESTONE-AUDIT.md` for full audit trail.

---

## v22.0 (2026-06-27) — Production Hardening

This release wires previously-stub security middleware, migrates concurrency
primitives, and adopts modern Rust 1.80+ stdlib features. Most changes are
internal; downstream consumers may need updates only if they touched the
specific APIs listed below.

### Security middleware wired (SEC-01..06)

**Before:** JWT tokens were parsed but not signature-verified. `RbacMiddleware`
was a no-op pass-through. Request body size was unbounded. Audit log was
silent. TLS had `unwrap()` panics on malformed certificates.

**After:** All five middleware paths enforce their policies. The server returns
HTTP 401 on invalid JWT signatures (HMAC-SHA256 or RSA/ECDSA), HTTP 403 on
RBAC denial, HTTP 413 on request body overflow, structured errors on TLS
handshake failure, and emits audit log entries for every authenticated
request.

```rust
// Before (v21.x) — JWT parsed but not verified
let token = parse_jwt(authorization_header)?;

// After (v22.0+) — full signature verification
let claims = verify_jwt(authorization_header, &VerificationKey::from_env()?)?;
// → Returns HTTP 401 on bad signature, expiry, or tampering
```

No code changes required for consumers of `vllm_server` — the wiring is
internal. Consumers using `vllm_server::auth::*` types directly should
re-compile against the v22.0 API.

### `parking_lot::Mutex` migration (RFU-05)

**Before:** Scheduler and engine paths used `std::sync::Mutex` with
`.lock().unwrap()` poison-check calls (24 sites). Poison errors propagated
to engine step failure paths.

**After:** All scheduler and engine paths use `parking_lot::Mutex` (which
does not have a poison concept). The `.lock()` API is unchanged; callers
that explicitly handled `PoisonError` should remove the `.unwrap()` calls.

```rust
// Before
let mut guard = self.scheduler.lock().unwrap();

// After (no change required; just no PoisonError to handle)
let mut guard = self.scheduler.lock();
```

### `std::sync::LazyLock` adoption (PERF-03)

**Before:** Lazy initialization used `once_cell::sync::Lazy`. This required
the `once_cell` crate dependency.

**After:** Lazy initialization uses `std::sync::LazyLock` (Rust 1.80+ stdlib).
The `once_cell` crate may still be present for other uses, but new code
should prefer `std::sync::LazyLock`. The migration is mechanical:

```rust
// Before
use once_cell::sync::Lazy;
static REGISTRY: Lazy<RwLock<HashMap<String, ...>>> = Lazy::new(|| ...);

// After
use std::sync::LazyLock;
static REGISTRY: LazyLock<RwLock<HashMap<String, ...>>> = LazyLock::new(|| ...);
```

Rust toolchain requirement remains `stable` (1.85 in practice); no consumer
toolchain change required.

### Engine signature refactor (ARF-06, ARF-07)

The `Engine` struct was decomposed from a single 1,057-LOC file into focused
sub-modules. The public API surface is unchanged — `Engine::with_config(...)`,
`Engine::step()`, `Engine::with_drafts(...)` all retain the same signatures.
Consumers do not need to update import paths.

### Test count

v22.0 closed with **1179 passing tests** (1146 v21.0 baseline + 33 new).
The Phase 19 e2e tests in `crates/core/tests/engine_wiring.rs` (formerly
`engine_v18_wiring.rs`) and `crates/core/tests/draft_resolver_integration.rs`
that were `#[ignore]`'d for the v18.0 speculative-mode hang now pass.

---

*Last updated: 2026-06-27 — v21.0 P2/P3 Backlog Cleanup milestone*
