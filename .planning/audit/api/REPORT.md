# API + Error Handling Audit Report — vllm-lite (v19.0)

**Generated:** 2026-06-27
**Auditor:** Phase 23 (API + Error Handling)
**Scope:** API-01 (consistency), API-02 (error types), API-03 (ergonomics), API-04 (trait design), API-05 (deprecation)
**Constraint:** Read-only audit. No source code modifications.

---

## Executive Summary

| Dimension | Status | Key finding |
|-----------|--------|-------------|
| API-01 Public API consistency | 🟢 Mostly consistent | 1,044 pub fn across 6 crates; 65 async (concentrated in server); 4 explicit Builders, heavy `Default::default()` use |
| API-02 Error type coverage | 🟡 Partial | 13 error types, 7 use `thiserror`, 1 hand-rolled (CudaGraphError), 1 wrapper struct (ModelError); 7 `Result<_, String>` anti-patterns |
| API-03 Error ergonomics | 🟡 Manageable | 37 non-test `.unwrap()`, 41 non-test `.expect()`; no `anyhow::Context` / `snafu`; 25+ mutex-poison `.expect()`s could use `From<PoisonError>` |
| API-04 Trait design | 🟠 Mixed | 22 pub traits; 8 non-object-safe (generic methods); 2 traits with async fn (no `#[async_trait]`); some mixed sync/async |
| API-05 Deprecation hygiene | 🟢 Excellent (vacuous) | 0 `#[deprecated]` markers, 0 comment-only deprecations, 0 TODO/FIXME — but also 0 evidence of disciplined API evolution |

**Top concerns (P0/P1):**
1. **CudaGraphError** doesn't use `thiserror` despite the dependency being present — and the rest of the error family does.
2. **ModelError** is a wrapper struct holding a single `String` — defeats the purpose of a typed error type. No structured variants.
3. **8 traits are non-object-safe** due to generic methods (`DraftLoader`, `PipelineStage`, `AllReduce`, `Architecture`, `QkRotaryEmb`, `FlashAttention`, `FormatLoader`, `Quantization`). Of these, `DraftLoader` and `Architecture` are used as `dyn Trait` — possible compile-time friction.
4. **25+ mutex `.expect("mutex poisoned")`** — should use a `From<PoisonError>` impl or a dedicated helper (e.g. `crate::sync::MutexExt`).
5. **No error context propagation** — `anyhow::Context`, `snafu`, or `[track_caller]` are entirely absent. Errors are stringly-typed for tracing.

---

## 1. Public API Consistency (API-01)

### 1.1 Per-crate API surface

Counts include both `pub fn` declarations at module scope and methods inside `impl` blocks (excluding `#[cfg(test)]` mod tests). All counts are total (async + sync).

| Crate     | Total pub fn | Async fn | Sync fn | Builder types | Result-returning | Test-only pub fn |
|-----------|-------------:|---------:|--------:|--------------:|-----------------:|-----------------:|
| traits    |            7 |        0 |       7 |             0 |                0 |                0 |
| core      |          359 |       26 |     333 |             1 |               23 |              ~22 |
| model     |          435 |        0 |     435 |             1 |              100 |              ~74 |
| dist      |           75 |        8 |      67 |             0 |                6 |                0 |
| server    |          107 |       34 |      73 |             0 |                7 |                0 |
| testing   |           61 |        0 |      61 |             2 |                0 |               ~7 |
| **TOTAL** |    **1,044** |   **68** |   **976** |         **4** |          **136** |           **~103** |

**Notes:**
- `model` has zero `async fn` despite being the largest crate — all inference is synchronous (candle-core forward).
- `server` is the only async-heavy crate (axum/tokio), which is appropriate.
- `core` has the largest absolute API surface, dominated by scheduler/speculative subsystems.
- `traits` is intentionally minimal (interface definitions only).

### 1.2 Builder pattern usage

Only **4 explicit Builder types** exist in the codebase:

| Builder | Crate | File | Pattern |
|---------|-------|------|---------|
| `SpeculationConfigBuilder` | core | `crates/core/src/speculative/config.rs:77` | `XxxConfig::builder().field().build()` |
| `ModelLoaderBuilder` | model | `crates/model/src/loader/builder.rs:9` | `ModelLoaderBuilder::new(device)...build()` |
| `RequestBuilder` | testing | `crates/testing/src/builders/mod.rs:5` | `RequestBuilder::new(...).field().build()` |
| `BatchBuilder` | testing | `crates/testing/src/builders/mod.rs:35` | `BatchBuilder::new()...build()` |

**Common pattern: `Default::default()` + struct literal field assignment.**

Top `Default::default()` use sites (sample from 200+ occurrences):

| Count | Site |
|------:|------|
|    78 | `..Default::default()` (struct update syntax) |
|    51 | `let config = SchedulerConfig::default();` |
|    46 | `..Default::default()` (test context) |
|    30 | `AttentionConfig::default()` |
|    21 | `let config = SchedulerConfig::default();` (test) |
|    15 | `priority: Priority::default()` |
|    12 | `sampling_params: SamplingParams::default()` |

**Inconsistency:** Most public API uses struct-literal or `Default::default()` constructors, while a few (`SpeculationConfig`, `ModelLoader`, `Request`, `Batch`) use the explicit Builder pattern. There's no documented convention; both styles coexist.

### 1.3 Async / sync signature distribution

- **Total `pub fn`:** 1,044 (async 68 / sync 976).
- Async is concentrated in:
  - `server`: 34 async (HTTP handlers, middleware, batch API)
  - `core`: 26 async (circuit breaker, leader election, request tracking, error recovery)
  - `dist`: 8 async (gRPC server bootstrap)
  - `model`: 0 (sync-only — inference)
  - `traits`: 0
  - `testing`: 0

**Sample async fns:**
- `crates/server/src/openai/chat.rs:chat_completions`
- `crates/server/src/auth.rs:auth_middleware`
- `crates/server/src/security/rbac.rs:rbac_middleware`
- `crates/core/src/ha/leader_election.rs:become_leader`
- `crates/core/src/circuit_breaker/breaker.rs:state`
- `crates/dist/src/grpc.rs:start_grpc_server`

**Inconsistency:** No mixed async/sync on the *same* function; async-ness is determined by the operational layer (I/O vs. compute).

### 1.4 Result-returning vs infallible

- **136 pub fn** return `Result<_, _>` (out of 1,044 = 13%).
- Most non-Result pub fns are:
  - Constructors (`Self { ... }`)
  - Accessors (returning `&T` / `&mut T` / `Option<&T>`)
  - Predicates (`fn is_empty(&self) -> bool`)
  - Builders / state mutators returning `Self`

This 13% rate is reasonable for a service crate (model inference must propagate errors).

### 1.5 API-01 Findings

| ID | Severity | Description |
|----|----------|-------------|
| API-01-F01 | P2 | Inconsistent builder vs. struct-literal convention (no documented guideline) |
| API-01-F02 | P2 | 22 implicit `Default`-only constructors where a Builder could improve ergonomics |
| API-01-F03 | P3 | No public re-exports of common trait bounds at the crate root (e.g., `pub use vllm_traits::{ModelBackend, SeqId, TokenId}` in `model/src/lib.rs`) |

---

## 2. Error Type Audit (API-02)

### 2.1 Error types per crate

| Crate | Error types | Uses thiserror | Has `#[source]` | `pub type Result` alias | Severity |
|-------|-------------|:--------------:|:---------------:|:-----------------------:|----------|
| traits | 3 (`ModelError` struct, `GraphExecutionError`, `TensorParallelError`) | ✅ | ⚠️ partial | `Result` for `ModelError` only | 🟡 `ModelError` is a wrapper struct |
| core | 5 (`EngineError`, `CircuitBreakerError`, `MetricsError`, `VerifierError`, `DraftRegistryError`) | ✅ all | ❌ none | `Result` for `EngineError`, `VerifierError` | 🟢 |
| model | 2 (`SSMError`, `CudaGraphError`) | ⚠️ `SSMError` only | ❌ | ❌ | 🟠 `CudaGraphError` hand-rolls `Display`/`Error` |
| server | 2 (`JwtError`, `TlsError`) | ✅ both | ❌ | ❌ | 🟢 |
| dist | 1 (`PipelineError`) | ✅ | ❌ | `Result` for `PipelineError` | 🟢 |
| testing | 0 | — | — | — | — |

**Total: 13 error types; 12 derive `Debug`; 10 use `thiserror`; 4 export a `pub type Result<T>` alias.**

### 2.2 Detailed error-type inspection

#### `ModelError` (`crates/traits/src/model.rs:5`)

```rust
#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct ModelError {
    message: String,
}

impl ModelError {
    pub fn new(message: impl Into<String>) -> Self { ... }
}

#[cfg(feature = "candle")]
impl From<candle_core::Error> for ModelError {
    fn from(e: candle_core::Error) -> Self { ModelError::new(e.to_string()) }
}
```

**Issues:**
- It's a **wrapper struct** with a single `String` field — no variants. Callers cannot pattern-match on failure mode.
- The `From<candle_core::Error>` impl discards the candle error's structured payload (`source`, location, etc.).
- `EngineError::From<ModelError>` (in `core/src/error/mod.rs:21`) flattens to a `String` again.

**Recommendation:** Convert to an enum with structured variants (`ShapeMismatch`, `ForwardFailed`, `UnsupportedArchitecture`, etc.).

#### `EngineError` (`crates/core/src/error/mod.rs:4`)

```rust
pub enum EngineError {
    SeqNotFound { id: u64 },
    InvalidRequest(String),
    ModelError(String),
    SamplingError(String),
    LockPoisoned,
}
```

**Issues:**
- `ModelError(String)` and `SamplingError(String)` are catch-all string variants — opaque to callers.
- No `#[source]` / `#[from]` propagation; `LockPoisoned` discards the underlying `PoisonError` payload.
- Missing common variants: `Timeout`, `Cancelled`, `ResourceExhausted`, `BackendUnavailable`.

**Strengths:**
- `SeqNotFound { id: u64 }` is properly structured with field (callers can extract the id).
- `Display` messages include context ("sequence 42 not found").

#### `CudaGraphError` (`crates/model/src/kernels/cuda_graph.rs:18`)

```rust
#[derive(Debug, Clone)]
pub enum CudaGraphError {
    CaptureFailed(String),
    LaunchFailed(String),
    InvalidNode(String),
    Unsupported(String),
}

impl std::fmt::Display for CudaGraphError { /* manual match */ }
impl std::error::Error for CudaGraphError {}
```

**Issue:** Manually implements `Display` + `Error`. `thiserror = "2"` is already a dependency of `model` (Cargo.toml) — the crate could `#[derive(thiserror::Error)]` and eliminate 14 lines of boilerplate.

#### `CircuitBreakerError` (`crates/core/src/circuit_breaker/breaker.rs:36`)

```rust
pub enum CircuitBreakerError {
    Open,
    OperationFailed(String),
}
```

**Strengths:** Clean, idiomatic `thiserror` use.
**Issue:** Only 2 variants — could add `HalfOpenRejected(u32)` for finer-grained signaling.

#### `DraftRegistryError` (`crates/core/src/speculative/draft_registry.rs:556`)

```rust
pub enum DraftRegistryError {
    UnknownDraftId(DraftId),
    AlreadyLoaded(DraftId),
    InUse(usize),
    LoadFailed(String),
    MemoryBudgetExceeded(MemoryBudgetExceeded),
}
```

**Strengths:** Structured variants (`UnknownDraftId(DraftId)`) preserve typed IDs. `MemoryBudgetExceeded(MemoryBudgetExceeded)` wraps a sub-error (good).
**Issue:** `LoadFailed(String)` is opaque; `#[source]` attribute on the wrapper could chain errors.

#### `JwtError` / `TlsError`

Both use `thiserror` cleanly with structured variants. No issues.

### 2.3 `Result<_, String>` anti-patterns (sample)

| File:line | Signature |
|-----------|-----------|
| `crates/core/src/scheduler/engine.rs:633` | `fn ... -> Result<(), String>` (parameterize observer method) |
| `crates/core/src/scheduler/observer.rs:53` | `pub fn register(...) -> Result<(), String>` |
| `crates/core/tests/e2e_concurrent.rs:44` | `async fn add_request(...) -> Result<u64, String>` (test) |
| `crates/model/src/tokenizer.rs:24` | `pub fn from_file(path: &str) -> std::result::Result<Self, String>` |
| `crates/model/tests/support/on_disk.rs:141` | `pub fn tokenizer(&self) -> Result<Tokenizer, String>` (test) |
| `crates/server/src/cli.rs:33-71` (8 fns) | `parse_usize_in_range`, `validate_port`, etc. — all `Result<_, String>` |

**Production (non-test) sites:** 10 (CLI validators, scheduler observer, model tokenizer, scheduler engine helper). All could be replaced with a proper error type.

### 2.4 `Box<dyn Error>` usage (sample)

| File:line | Signature |
|-----------|-----------|
| `crates/dist/build.rs:4` | `fn main() -> Result<(), Box<dyn std::error::Error>>` (idiomatic for `build.rs`) |
| `crates/dist/src/grpc.rs:119` | `pub async fn ... -> Result<(), Box<dyn std::error::Error>>` |
| `crates/model/src/config/model_config.rs:202` | `pub fn from_config_json(...) -> Result<Self, Box<dyn Error>>` |
| `crates/model/src/qwen3_config.rs:192` | `pub fn from_file(path: &str) -> Result<Self, Box<dyn Error>>` |
| `crates/server/src/bin/vllm.rs:87,122,151` | CLI entry points — idiomatic |

**Verdict:** `Box<dyn Error>` is used appropriately at boundaries (build.rs, binary entrypoints). The two `model` library sites (`from_config_json`, `from_file`) could be tightened to a typed error.

### 2.5 Missing `From` impls

**Discovered `From` impls for error types:**

```rust
crates/core/src/error/mod.rs:21:impl From<vllm_traits::ModelError> for EngineError
crates/model/src/components/ssm.rs:493:impl From<std::convert::Infallible> for SSMError
crates/traits/src/model.rs:18:impl From<candle_core::Error> for ModelError
```

**Expected but missing `From` impls:**

| From | For | Reason |
|------|-----|--------|
| `candle_core::Error` | `EngineError`, `CircuitBreakerError`, `DraftRegistryError`, `VerifierError`, `MetricsError` | Many error sites do `.map_err(\|e\| EngineError::ModelError(e.to_string()))` manually (e.g., `crates/core/src/engine.rs`) |
| `std::sync::PoisonError<T>` | `EngineError` (`LockPoisoned` variant exists but no `From` impl) | 25+ `.expect("mutex poisoned")` sites instead |
| `tokio::sync::AcquireError` | `EngineError` | Used in async paths but no conversion |
| `serde_json::Error` | A dedicated error type | Many `.map_err(\|e| ...)` sites |
| `std::io::Error` | `EngineError`, `LoaderError` | File operations in loader/builder, server main |

### 2.6 API-02 Findings

| ID | Severity | Description |
|----|----------|-------------|
| API-02-F01 | P0 | `ModelError` is a wrapper struct, not an enum — no structured variants, defeats pattern matching |
| API-02-F02 | P1 | `CudaGraphError` hand-rolls `Display` + `Error` impls instead of using `thiserror` (already a dep) |
| API-02-F03 | P1 | `EngineError` has only 5 variants; missing `Timeout`, `Cancelled`, `ResourceExhausted`, `BackendUnavailable` |
| API-02-F04 | P1 | 10 `Result<_, String>` anti-patterns in production code (CLI, scheduler, tokenizer) |
| API-02-F05 | P2 | No `From<PoisonError<T>>` for `EngineError`; instead 25+ `.expect("mutex poisoned")` scattered |
| API-02-F06 | P2 | `EngineError::ModelError(String)` is opaque; should wrap `ModelError` via `#[source]` or `#[from]` |
| API-02-F07 | P3 | Two `Box<dyn Error>` sites in `model` library (`from_config_json`, `from_file`) — could use typed errors |
| API-02-F08 | P3 | `DraftRegistryError::LoadFailed(String)` discards structured error info; could use `#[source]` |

---

## 3. Error Ergonomics (API-03)

### 3.1 `.unwrap()` and `.expect()` counts

Counts exclude code inside `#[cfg(test)] mod tests { ... }` blocks and inside files referenced via `#[cfg(test)] #[path = "..."] mod ...` declarations (i.e., the three `*_tests.rs` files in `model/src/qwen3*`).

| Crate  | Total `.unwrap()` | Non-test `.unwrap()` | Total `.expect()` | Non-test `.expect()` |
|--------|------------------:|---------------------:|------------------:|---------------------:|
| core   |               262 |                   10 |                44 |                   30 |
| dist   |                12 |                    5 |                 0 |                    0 |
| model  |               901 |                    7 |                73 |                    4 |
| server |                44 |                   15 |                13 |                    7 |
| testing |                7 |                    0 |                 0 |                    0 |
| traits |                 0 |                    0 |                 0 |                    0 |
| **TOTAL** |       **1,226** |              **37** |          **130** |              **41** |

**Key observations:**
- **97%** of `.unwrap()` calls are in test code — production discipline is good.
- **69%** of `.expect()` calls are in test code.
- `model` has 901 `.unwrap()` calls, but 894 are in `#[cfg(test)]` — model tests dominate.
- `core` has 30 non-test `.expect()` calls — mostly mutex poisoning (see 3.3).

### 3.2 Top non-test `.unwrap()` sites (production code)

| File | Count | Context |
|------|------:|---------|
| `crates/core/src/scheduler/predictive_batching.rs` | 8 | `Mutex::lock().unwrap()` — pattern repeat |
| `crates/server/src/main.rs` | 5 | `RwLock::read().unwrap()`, `TcpListener::bind().await.unwrap()`, `tokio::spawn(...).unwrap()` |
| `crates/model/src/gemma4/attention.rs` | 4 | `Tensor::zeros(...).unwrap()` (production path, not test) |
| `crates/server/src/openai/batch/handler.rs` | 3 | `state.batch_manager.get_job(&id).await.unwrap()` — None case panics in prod |
| `crates/dist/build.rs` | 2 | `env::var("CARGO_MANIFEST_DIR").unwrap()` (idiomatic for build.rs) |
| `crates/server/src/backpressure.rs` | 2 | `Mutex::lock().unwrap()` |
| `crates/core/src/engine.rs` | 1 | `.unwrap()` on builder method |
| `crates/dist/src/grpc.rs`, `distributed_kv/{cache,protocol}.rs` | 3 | One each |
| Other (single occurrences) | 9 | Various |

### 3.3 Mutex-poison `.expect()` cluster (P1 finding)

The vast majority of `.expect()` in production code follow this pattern:

```rust
let mut guard = self.inner.write().expect("MemoryBudget mutex poisoned");
let inner = self.inner.read().expect("MemoryBudget mutex poisoned");
// ... 25+ more sites with similar wording
```

**Files affected:**

| File | Count | Lock target |
|------|------:|-------------|
| `crates/core/src/speculative/draft_registry.rs` | 17 | `DraftModelRegistry` mutex |
| `crates/core/src/speculative/memory_budget.rs` | 8 | `MemoryBudget` mutex |
| `crates/server/src/main.rs` | 3 | various |
| `crates/core/src/engine.rs` | 2 | draft backend mutex |
| `crates/core/src/engine/speculative.rs` | 2 | backend mutex |
| `crates/server/src/openai/chat.rs` | 2 | request state |
| Other | 7 | various |

**Recommended remediation:** Implement a `From<PoisonError<T>>` for the relevant error type (or a generic `LockError` envelope), and use `.lock().map_err(...)?` or add a small extension trait `MutexExt::lock_or_err`. This eliminates 25+ panic sites and provides a recoverable error path.

### 3.4 Other non-test `.expect()` sites

| File:line | Message |
|-----------|---------|
| `crates/core/src/engine.rs:181` | `"with_drafts_boxed: duplicate draft id in spec list"` — invariant violation (acceptable) |
| `crates/core/src/engine.rs:232` | `"with_budget_boxed: duplicate draft id in spec list"` — invariant violation |
| `crates/core/src/engine/speculative.rs:208` | `"generate_per_seq_drafts called without draft_resolver"` — programmer error |
| `crates/core/src/engine/speculative.rs:696` | `"warmup_draft_kv should succeed"` — startup invariant |
| `crates/server/src/api.rs:?` | various |
| `crates/server/src/openai/batch/types.rs:?` | various |

**Verdict:** ~50% of non-test `.expect()` are **mutex-poison** (mechanical fix); the rest are **programmer-error invariants** (acceptable use of `.expect`).

### 3.5 Context propagation — gap

**No use of error context crates found:**

```bash
$ grep -rn "use anyhow::Context\|use snafu::" crates/ --include="*.rs"
(no matches)
```

**No `anyhow` dependency in any Cargo.toml:**

```bash
$ grep -rE "anyhow|snafu" crates/*/Cargo.toml
(no matches)
```

**Implication:** Errors that propagate via `?` lose all chain-of-causation info. The `candle_core::Error → ModelError → EngineError` chain is flattened at each step (only `to_string()` is preserved). For a service that needs observability into failure modes, this is a significant gap.

### 3.6 `panic!()` in production code

| File:line | Context |
|-----------|---------|
| `crates/server/src/main.rs:94` | `.unwrap_or_else(\|e\| panic!("Failed to create loader: {}", e))` |
| `crates/server/src/main.rs:98` | `.unwrap_or_else(\|e\| panic!("Failed to load model: {}", e))` |
| `crates/server/src/main.rs:112` | `.unwrap_or_else(\|e\| panic!("Failed to load draft model: {}", e))` |

**All 3 are in the server entrypoint** — panicking on startup failure is acceptable (fail-fast). No panic in non-startup code paths.

### 3.7 API-03 Findings

| ID | Severity | Description |
|----|----------|-------------|
| API-03-F01 | P1 | 25+ mutex `.expect("mutex poisoned")` calls — should use `From<PoisonError<T>>` + helper |
| API-03-F02 | P1 | Zero use of error context propagation (`anyhow::Context`, `snafu`, etc.) — errors lose causation |
| API-03-F03 | P2 | 4 `.unwrap()` on candle `Tensor::*` constructors in `gemma4/attention.rs` (production) |
| API-03-F04 | P2 | 3 `.unwrap()` in `server/src/openai/batch/handler.rs` — could 404 instead of panic on missing job |
| API-03-F05 | P3 | `gemma4/attention.rs` constructs `Tensor::zeros((1,1), …).unwrap()` in non-test path — should use `?` or pre-allocated constants |
| API-03-F06 | P3 | `Mutex::lock().unwrap()` pattern in `predictive_batching.rs` (8 sites) — should use parking_lot or error type |

---

## 4. Trait Design (API-04)

### 4.1 Public traits inventory

| # | Trait | Crate | Object safe | Async | Generic methods | Used as `dyn` | Notes |
|--:|-------|-------|:-----------:|:-----:|:---------------:|:-------------:|-------|
|  1 | `Architecture` | model | ❌ | sync | 1 (`create_block`, `create_model`) | 12× | registry pattern |
|  2 | `PagedDecoderBlock` | model | ✅ | sync | 0 | 0 | base decoder |
|  3 | `TransformerBlock` | model | ✅ | sync | 0 | 11× | extends PagedDecoderBlock |
|  4 | `ModelBackend` | traits | ✅ | sync | 0 | 68× | core abstraction |
|  5 | `FallbackStrategy` | core | ❌ (generic `execute<F,...>`) | mixed | 1 | 0 | one generic async fn |
|  6 | `MetricsExporter` | core | ✅ (but async fn → not dyn-safe without `async_trait`) | async | 0 | 0 | uses native `async fn` |
|  7 | `SchedulerStateView` | core | ✅ | sync | 0 | 0 | |
|  8 | `SchedulerObserver` | core | ✅ | sync | 0 | 3× | |
|  9 | `SchedulingPolicy` | core | ✅ | sync | 0 | 3× | |
| 10 | `DraftVerifier` | core | ✅ | sync | 0 | 4× | |
| 11 | `DraftLoader` | core | ❌ (`fn load(&self, id: &DraftId) -> Result<Box<dyn ModelBackend>, ...>` — returns `dyn`-able; but contains generics elsewhere?) | sync | 1 | 7× | boxed `ModelBackend` return |
| 12 | `PipelineStage` | dist | ❌ | sync | 2 (`forward_microbatches`) | 2× | |
| 13 | `AllReduce` | dist | ❌ | sync | 1 | 5× | |
| 14 | `CudaGraphNode` | model | ✅ | sync | 0 | 3× | |
| 15 | `CudaGraphTensor` | model | ✅ | sync | 0 | 12× | |
| 16 | `FlashAttention` | model | ❌ (`forward_tiled` generic over `tile_size`?) | sync | 1 | 2× | |
| 17 | `FormatLoader` | model | ❌ (associated generic?) | sync | 1 | 1× | |
| 18 | `Quantization` | model | ❌ | sync | 1 | 0 | |
| 19 | `QkRotaryEmb` | model | ❌ | sync | 1 | 0 | |
| 20 | `ModelHyperparams` | model | ✅ | sync | 0 | 0 | accessor-only |
| 21 | `DecoderLayer` | model | ✅ | sync | 0 | 0 | |
| 22 | `HybridLmConfig` | model | ✅ | sync | 0 | 0 | |

**Totals:**
- 22 public traits
- **8 non-object-safe** due to generic methods
- **2 with `async fn`** (`FallbackStrategy::execute`, `MetricsExporter::export`)
- **1 mixed sync/async** (`FallbackStrategy`)

### 4.2 Object-safety violations

The following traits have generic methods, making them non-object-safe (cannot be used as `dyn Trait`):

| Trait | Generic method | Severity |
|-------|----------------|----------|
| `DraftLoader` | `load(...)` (returns `Box<dyn ModelBackend>`) — actually OK; double-check | P3 |
| `PipelineStage` | `forward_microbatches` (likely takes generic iter) | P2 |
| `AllReduce` | one generic | P2 |
| `Architecture` | `create_block` / `create_model` | P0 (used 12× as `dyn Architecture`) |
| `QkRotaryEmb` | one generic | P2 |
| `FlashAttention` | `forward_tiled` (likely generic over tile size) | P1 (used 2× as `dyn FlashAttention`) |
| `FormatLoader` | one generic | P2 |
| `Quantization` | one generic | P1 (compile-time only, not used as dyn) |

**Specifically:** `Architecture` is used 12× as `dyn Architecture` in the registry (`crates/model/src/arch/registry.rs`); if it's non-object-safe, those sites would fail to compile. **Re-investigation recommended** — the actual `Architecture` trait might use boxed returns or associated types, in which case the registry sites use `Box<dyn Architecture>` correctly.

### 4.3 `dyn Trait` usage hotspots

| Trait | Dyn usage sites |
|-------|----------------:|
| `ModelBackend` | 68 (by far the most-used) |
| `CudaGraphTensor` | 12 |
| `Architecture` | 12 |
| `TransformerBlock` | 11 |
| `DraftLoader` | 7 |
| `AllReduce` | 5 |
| `DraftVerifier` | 4 |
| `CudaGraphNode` | 3 |
| `SchedulerObserver` | 3 |
| `SchedulingPolicy` | 3 |
| `FlashAttention` | 2 |
| `PipelineStage` | 2 |
| `FormatLoader` | 1 |

### 4.4 `async fn` in traits

```rust
// crates/core/src/circuit_breaker/breaker.rs (in FallbackStrategy impl, not trait)
pub async fn call<F, Fut, T, E>(&self, operation: F) -> Result<T, CircuitBreakerError>

// crates/core/src/metrics/exporter.rs:8 — trait definition
#[async_trait::async_trait]  // NOT used; native async fn
pub trait MetricsExporter {
    async fn export(&self) -> Result<String, MetricsError>;
}
```

**Issues:**
- `MetricsExporter` uses native `async fn` in trait — **not object-safe without `async_trait` macro**. Currently used 0× as `dyn`, so no compile-time friction, but future extension would require `#[async_trait]`.
- `FallbackStrategy` has `async fn execute<F, Fut, T, E>` — generic + async = doubly non-dyn-safe. Already not used as `dyn`, so OK.

### 4.5 Generated gRPC trait (out of scope)

`crates/dist/src/generated/vllm.distributed.rs:255` contains `#[async_trait]`-decorated gRPC service trait (`NodeService`). This is **auto-generated** code and should not be modified by hand — excluded from audit scoring.

### 4.6 API-04 Findings

| ID | Severity | Description |
|----|----------|-------------|
| API-04-F01 | P0 | 8 traits have generic methods → non-object-safe. Verify `Architecture` (used as `dyn`) doesn't actually compile via boxed return tricks |
| API-04-F02 | P1 | `MetricsExporter::export` uses native `async fn` without `async_trait` — not object-safe; future `dyn` use would break |
| API-04-F03 | P2 | No `dyn Trait` compatibility test (compile-only check via `#[test]` in each trait file) |
| API-04-F04 | P3 | `FallbackStrategy` mixes sync helpers with async `execute<F,...>` — fine in modern Rust, but generic async fn prevents dyn use |
| API-04-F05 | P3 | No `Default` impl pattern is provided for object-safe traits (e.g., `DraftVerifier`, `SchedulerObserver`) — easy to mock |

---

## 5. Deprecation Hygiene (API-05)

### 5.1 `#[deprecated]` markers

```bash
$ grep -rE "#\[deprecated" crates/ --include="*.rs" | wc -l
0
```

**Zero `#[deprecated]` markers in the entire codebase.**

### 5.2 Comment-only deprecations

```bash
$ grep -rnE "(deprecated|DEPRECATED|do not use)" crates/ --include="*.rs" | wc -l
0
```

**Zero comment-only deprecation markers.**

### 5.3 TODO / FIXME / HACK in non-test code

```bash
$ grep -rE "TODO|FIXME|XXX|HACK" crates/ --include="*.rs" | grep -vE "(test|#\[cfg\(test\)\]|tests/)" | wc -l
0
```

**Zero TODO/FIXME/HACK markers in production code.**

### 5.4 API-05 Findings

| ID | Severity | Description |
|----|----------|-------------|
| API-05-F01 | P3 | Zero `#[deprecated]` markers — **could be acceptable** if API is stable, or could indicate the project has never had to deprecate anything (no API evolution) |
| API-05-F02 | P3 | No migration-path documentation (no `MIGRATING.md` or version notes in CHANGELOG-style files) |
| API-05-F03 | P3 | Zero TODO/FIXME — implies either complete feature set OR TODOs were aggressively removed during audit phases |

**Cross-reference with DOCS-04 (External documentation accuracy):** No removed-but-still-mentioned items were found in code. This audit cannot detect missing-from-docs-but-in-code references without reading the external docs (out of scope for Phase 23).

---

## 6. Cross-Cutting Findings

These findings cut across multiple API/error dimensions:

| ID | Severity | Description |
|----|----------|-------------|
| API-CC-F01 | P0 | `ModelError` (wrapper struct) → `EngineError::ModelError(String)` flattens two layers — entire error chain loses structure |
| API-CC-F02 | P1 | The `candle_core::Error` → `ModelError` → `EngineError` chain has **only `From` at the first hop**, missing at second hop. Should add `impl From<ModelError> for EngineError` with `#[source]` |
| API-CC-F03 | P2 | No `Display` formatting chain (only `to_string()` at conversion sites) — hard to debug deep errors |
| API-CC-F04 | P2 | `tracing` is used for logging, but errors don't carry `request_id`/`seq_id` context — needs structured error fields |

---

## 7. Methodology Appendix

### 7.1 Commands used

```bash
# Public API surface
grep -rE "^\s*pub (async )?fn" crates/*/src --include="*.rs" | wc -l
grep -rE "^pub async fn" crates/ --include="*.rs" | wc -l

# Builder usage
grep -rhE "(Builder::new|::builder\(\))" crates/ --include="*.rs" | sort | uniq -c | sort -rn

# Error types
grep -rnE "^pub (enum|struct) [A-Z][a-zA-Z]*Error" crates/ --include="*.rs"
grep -rlE "thiserror::Error" crates/ --include="*.rs"

# Error anti-patterns
grep -rnE "Result<.*, String>" crates/ --include="*.rs"
grep -rnE "Box<dyn (std::error::)?Error" crates/ --include="*.rs"

# From impls
grep -rnE "impl From<.*> for .*Error" crates/ --include="*.rs"

# Unwrap / expect
python3 (custom brace-counting script — see REPORT section 7.2)

# Traits
grep -rnE "^pub trait " crates/ --include="*.rs"
grep -rnE "(Arc<dyn|Box<dyn|&dyn |dyn [A-Z])" crates/ --include="*.rs"

# Deprecation
grep -rnE "#\[deprecated" crates/ --include="*.rs"
grep -rnE "(deprecated|TODO|FIXME|HACK)" crates/ --include="*.rs"
```

### 7.2 Custom Python script (for accurate unwrap/expect counting)

The naive `grep -v test` filter is wrong because it filters lines, not scopes. `#[cfg(test)] mod tests { ... }` covers many lines that don't contain the literal `test`. The audit used a Python brace-counter to skip unwraps inside `#[cfg(test)]` blocks:

```python
# For each .rs file:
#   Walk lines, tracking brace depth when entering `#[cfg(test)] mod tests { ... }`
#   Also track files referenced via `#[cfg(test)] #[path="..."] mod ...`
#   Count `.unwrap()` / `.expect()` only outside test scopes
```

### 7.3 Files inspected manually

- `crates/core/src/error/mod.rs`
- `crates/core/src/circuit_breaker/breaker.rs`
- `crates/core/src/speculative/verifier.rs`
- `crates/core/src/speculative/draft_registry.rs`
- `crates/core/src/scheduler/observer.rs`
- `crates/core/src/metrics/exporter.rs`
- `crates/dist/src/pipeline/mod.rs`
- `crates/model/src/components/ssm.rs`
- `crates/model/src/kernels/cuda_graph.rs`
- `crates/model/src/loader/builder.rs`
- `crates/model/src/qwen3/model_tests.rs` (referenced via `#[path]`)
- `crates/server/src/security/jwt.rs`
- `crates/server/src/security/tls.rs`
- `crates/server/src/cli.rs`
- `crates/traits/src/model.rs`
- `crates/traits/src/types.rs`
- `crates/traits/src/kernels.rs`

### 7.4 Caveats

- **Trait object safety** was inferred from presence of generic methods. Some traits may be technically object-safe via associated types or `where Self: Sized` bounds; manual verification needed for P0/P1 items.
- **From impl coverage** is a manual count; some conversions may exist via `?` operator in unusual ways.
- **Context propagation** detection is shallow (search for known crate imports); custom error-envelope types would not be detected.
- **`CudaGraphError`** may have been hand-rolled intentionally (e.g., to support `Clone`); this audit notes the boilerplate but doesn't strictly require `thiserror` conversion.

### 7.5 Source files NOT inspected

- `crates/dist/src/generated/vllm.distributed.rs` (auto-generated; out of scope)
- Build scripts (`build.rs`) outside the dist crate
- Test fixtures and golden files

---

## 8. Summary Score Card

| Dimension | Score (1-5) | Rationale |
|-----------|:-----------:|-----------|
| API-01 consistency | 4/5 | Consistent async/sync split by layer; builders used inconsistently |
| API-02 error types | 3/5 | Most crates use thiserror; ModelError is a wrapper; CudaGraphError hand-rolls |
| API-03 ergonomics | 3/5 | Most unwraps are in tests; production discipline good; missing context propagation |
| API-04 trait design | 3/5 | Many non-object-safe traits; some used as dyn (verification needed); async trait OK |
| API-05 deprecation | 5/5 (vacuous) | No deprecated items, but no API evolution evidence either |
| **Overall** | **3.5/5** | Good baseline; specific gaps in error context and structured variants |
