# API + Error Handling Audit Summary

**Generated:** 2026-06-27
**Milestone:** v19.0 Codebase Health Audit
**Scope:** Phase 23 — API + Error Handling
**Detailed report:** [`REPORT.md`](./REPORT.md)

**Total findings:** 33
- **P0:** 3
- **P1:** 8
- **P2:** 13
- **P3:** 9

---

## Prioritized Findings

| ID | Dim | Description | Severity | Source | Effort |
|----|-----|-------------|----------|--------|--------|
| API-F-01 | API-02 | `ModelError` is a wrapper struct (not enum); defeats pattern matching | **P0** | `crates/traits/src/model.rs:5` | M (introduce enum, update 1+ impls) |
| API-F-02 | API-04 | 8 traits non-object-safe due to generic methods; `Architecture`/`FlashAttention` used as `dyn` — verify compile correctness | **P0** | `Architecture` (12×), `FlashAttention` (2×) | M (introduce associated types or generic-erasure) |
| API-F-03 | API-02 | `CudaGraphError` hand-rolls `Display`/`Error` impls instead of using `thiserror` (already a dep) | **P0** | `crates/model/src/kernels/cuda_graph.rs:18` | XS (delete 14 lines, replace with derive) |
| API-F-04 | API-03 | 25+ mutex `.expect("mutex poisoned")` calls scattered across core; need `From<PoisonError>` + helper | **P1** | `draft_registry.rs` (17), `memory_budget.rs` (8), `engine.rs`, etc. | M (add helper, replace 25+ sites) |
| API-F-05 | API-03 | No error context propagation (`anyhow::Context`, `snafu`, etc.); errors lose causation across `?` boundaries | **P1** | entire codebase | L (introduce `anyhow` dep + usage pattern) |
| API-F-06 | API-02 | `EngineError` has only 5 variants; missing `Timeout`, `Cancelled`, `ResourceExhausted`, `BackendUnavailable` | **P1** | `crates/core/src/error/mod.rs:4` | S (add 4 variants) |
| API-F-07 | API-02 | 10 `Result<_, String>` anti-patterns in production code (CLI validators, scheduler observer, model tokenizer) | **P1** | 5 files | M (introduce typed error per subsystem) |
| API-F-08 | API-02 | `EngineError::From<ModelError>` (and similar) flattens via `to_string()`; loses chain | **P1** | `crates/core/src/error/mod.rs:21` | S (use `#[source]` / structured variants) |
| API-F-09 | API-04 | `MetricsExporter` uses native `async fn` without `async_trait` — not object-safe for `dyn` | **P1** | `crates/core/src/metrics/exporter.rs:8` | XS (add `#[async_trait]`) |
| API-F-10 | API-03 | 4 `.unwrap()` on candle `Tensor::*` constructors in `gemma4/attention.rs` (production) | **P1** | `crates/model/src/gemma4/attention.rs:360-372` | XS (use `?` or pre-allocate) |
| API-F-11 | API-03 | 3 `.unwrap()` in `server/src/openai/batch/handler.rs` — would 404 on missing job instead of panic | **P1** | `crates/server/src/openai/batch/handler.rs:42,45,132` | XS |
| API-F-12 | API-01 | Inconsistent builder vs. struct-literal convention (no documented guideline) | **P2** | across crates | S (write convention doc) |
| API-F-13 | API-02 | No `From<PoisonError<T>>` for any error type (except `EngineError::LockPoisoned` variant without impl) | **P2** | `crates/core/src/error/mod.rs` | S |
| API-F-14 | API-02 | `DraftRegistryError::LoadFailed(String)` discards structured error info | **P2** | `crates/core/src/speculative/draft_registry.rs:564` | S (use `#[source]`) |
| API-F-15 | API-02 | Two `Box<dyn Error>` sites in `model` library (`from_config_json`, `from_file`) — could use typed errors | **P2** | `crates/model/src/config/model_config.rs:202`, `crates/model/src/qwen3_config.rs:192` | S |
| API-F-16 | API-03 | `Mutex::lock().unwrap()` pattern in `predictive_batching.rs` (8 sites) | **P2** | `crates/core/src/scheduler/predictive_batching.rs` | S (use parking_lot or sync helper) |
| API-F-17 | API-03 | 22 implicit `Default`-only constructors where a Builder could improve ergonomics | **P2** | across crates | L (introduce 22 builders) |
| API-F-18 | API-04 | No `dyn Trait` compatibility test (compile-only check via `#[test]` in each trait file) | **P2** | missing | S (add tests) |
| API-F-19 | API-01 | No public re-exports of common trait bounds at crate root (e.g., `pub use vllm_traits::{ModelBackend, SeqId, TokenId}`) | **P2** | `crates/model/src/lib.rs`, `crates/server/src/lib.rs` | XS |
| API-F-20 | API-04 | `FallbackStrategy` mixes sync helpers with async `execute<F,...>` — fine but generic async fn prevents dyn use | **P2** | `crates/core/src/circuit_breaker/strategy.rs` | M (split trait) |
| API-F-21 | API-02 | `candle_core::Error` → `ModelError` → `EngineError` chain missing `From` impls at second hop | **P2** | `crates/core/src/error/mod.rs:21` | XS |
| API-F-22 | API-04 | No `Default` impl pattern for object-safe traits (e.g., `DraftVerifier`, `SchedulerObserver`) — hard to mock | **P2** | `crates/core/src/speculative/verifier.rs`, `crates/core/src/scheduler/observer.rs` | M |
| API-F-23 | API-01 | `model` has 0 async fn; `core` has 26; `server` has 34 — by-layer async-ness is consistent (good) | **P2** | n/a (positive finding) | — |
| API-F-24 | API-03 | `tracing` used for logging, but errors don't carry `request_id`/`seq_id` context | **P2** | various | L (structured error fields) |
| API-F-25 | API-04 | `gemma4/attention.rs` constructs `Tensor::zeros((1,1), …).unwrap()` in non-test path | **P3** | `crates/model/src/gemma4/attention.rs` | XS |
| API-F-26 | API-05 | Zero `#[deprecated]` markers — could indicate stable API or lack of API evolution discipline | **P3** | across crates | — (observation) |
| API-F-27 | API-05 | No `MIGRATING.md` or versioned changelog | **P3** | repo root | S (add file) |
| API-F-28 | API-05 | Zero TODO/FIXME in production code — implies complete feature set OR aggressive cleanup during audits | **P3** | across crates | — (observation) |
| API-F-29 | API-04 | `DraftLoader::load` returns `Box<dyn ModelBackend>` — verify this is intentional (object-safe via boxed return) | **P3** | `crates/core/src/speculative/draft_resolver.rs:57` | XS |
| API-F-30 | API-02 | `DraftRegistryError::MemoryBudgetExceeded(MemoryBudgetExceeded)` — verify the inner error type also derives `Error` | **P3** | `crates/core/src/speculative/draft_registry.rs:566` | XS |
| API-F-31 | API-04 | `CircuitBreakerError` only has 2 variants (`Open`, `OperationFailed`) — could add `HalfOpenRejected(u32)` | **P3** | `crates/core/src/circuit_breaker/breaker.rs:36` | XS |
| API-F-32 | API-01 | `model` crate has highest unwrap count (901 total, 7 non-test) — heavily test-driven; verify production unwraps | **P3** | `crates/model/src/` | S |
| API-F-33 | API-02 | `CudaGraphError` derives `Clone` — verify `Error` types need `Clone` (uncommon) | **P3** | `crates/model/src/kernels/cuda_graph.rs:17` | XS |

---

## Top 3 Action Items

### 1. **Introduce structured error types across the error chain** (addresses API-F-01, F-03, F-06, F-07, F-08, F-13, F-21)

Convert `ModelError` from a struct to an enum with structured variants (e.g., `ShapeMismatch`, `ForwardFailed`, `UnsupportedArchitecture`). Convert `CudaGraphError` to use `thiserror` (delete 14 lines of manual impl). Expand `EngineError` with `Timeout`, `Cancelled`, `ResourceExhausted`, `BackendUnavailable`. Replace 10 `Result<_, String>` sites with typed errors. Add `From<candle_core::Error>` / `From<PoisonError<T>>` impls with `#[source]` attributes.

**Effort:** ~3 days (one engineer).
**Impact:** Eliminates the largest single source of untyped errors; enables pattern matching; surfaces failure modes in logs.

### 2. **Eliminate mutex-poison `.expect()` cluster** (addresses API-F-04, F-13, F-16)

Add a `From<PoisonError<T>>` impl for `EngineError` (and any other relevant error types) using the existing `LockPoisoned` variant. Replace 25+ `.expect("mutex poisoned")` calls with `.map_err(...)?`. Optionally introduce a `LockExt` trait with a `lock_or_err()` method.

**Effort:** ~0.5 days.
**Impact:** Removes 25+ panic sites; makes lock poisoning recoverable; surfaces the failure in the error chain.

### 3. **Verify and fix non-object-safe traits used as `dyn`** (addresses API-F-02, F-09, F-18, F-22)

The audit identified 8 traits with generic methods. Of these, `Architecture` (12× `dyn` usage) and `FlashAttention` (2× `dyn` usage) are actually used as `dyn Trait` — but generic methods should prevent this. Likely the implementations use boxed returns or associated types that make them object-safe; needs verification. Add `#[test]` compile-only tests to each trait file to prevent regressions. Convert `MetricsExporter` to use `#[async_trait]` for future `dyn` safety.

**Effort:** ~1 day (investigation + targeted fixes).
**Impact:** Prevents future compile errors; documents object-safety contract per trait.

---

## Suggested v20.0+ Phase

**Proposed phase name:** `v20.1 Error System Modernization`

**Scope:**
1. Introduce `vllm_core::error::EngineError` v2 with structured variants + `#[source]` chain
2. Replace `ModelError` struct with enum (`ShapeMismatch`, `ForwardFailed`, etc.)
3. Convert `CudaGraphError` to `thiserror`
4. Eliminate 25+ mutex `.expect()` via `From<PoisonError>`
5. Replace `Result<_, String>` anti-patterns with typed errors (10 sites)
6. Add error-context propagation via lightweight `anyhow::Context`-style helper (or adopt `anyhow` directly in server/CLI layers)
7. Add `dyn Trait` compile tests for all public traits
8. Document error-handling convention in `AGENTS.md`

**Estimated effort:** 1-2 weeks (single engineer).
**Dependencies:** None (build-system-only changes).
**Risk:** Low — error refactor is internal; observability improves.
**Suggested priority:** P1 for v20.1, can defer P2/P3 items to v20.2+.

---

## Phase 23 Compliance Check

| Requirement | Status | Notes |
|-------------|:------:|-------|
| API-01: Public API surface consistency | ✅ | 1,044 pub fn across 6 crates analyzed |
| API-02: Error type audit | ✅ | 13 error types reviewed, 7 use thiserror, 6 need fixes |
| API-03: Error ergonomics | ✅ | 37 non-test unwrap, 41 non-test expect, 3 panic inventoried |
| API-04: Trait design | ✅ | 22 traits, 8 non-object-safe, 2 with async fn identified |
| API-05: Deprecation hygiene | ✅ | 0 deprecated items (vacuous positive) |
| REPORT.md ≥ 300 lines | ✅ | 480+ lines |
| SUMMARY.md P0/P1/P2 table | ✅ | 33 findings prioritized |
| No source code changes | ✅ | `git status` clean outside `.planning/` |
