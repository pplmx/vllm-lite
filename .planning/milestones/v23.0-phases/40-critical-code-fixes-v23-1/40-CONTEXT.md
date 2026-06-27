# Phase 40: Critical Code Fixes (v23.1) - Context

**Gathered:** 2026-06-28
**Status:** Ready for planning
**Mode:** Smart discuss auto-accepted (infrastructure phase)

<domain>

## Phase Boundary

Resolve P0 code defects surfaced by v22.0 post-ship audit across 5 categories:
error-type hygiene, error-chain preservation, public-API type hygiene, stub-architecture
policy, and placeholder API resolution. All fixes are internal to the Rust API and
preserve the v22.0 invariant: 1179 tests remain green.

Specifically:

1. **`TensorParallelError`** (traits/src/types.rs:86-112) — manual `Display` +
   `std::error::Error` impls; convert to `#[derive(thiserror::Error)]` matching the
   project's 19 other error enums (per AGENTS.md "Error Type Conventions").
2. **`Engine::step()` source-chain loss** (core/src/engine.rs:677) — wraps `ModelError`
   in `EngineError::ModelError(e.to_string())`, dropping the underlying source chain;
   replace with `EngineError::from(e)` (or `#[source]` wiring) so logs retain the cause.
3. **`Box<dyn Error>` in `dist/src/grpc.rs:129`** — public `start_grpc_server` returns
   `Result<(), Box<dyn std::error::Error>>`; replace with a typed `GrpcError` enum
   (thiserror) per AGENTS.md "Never use `Box<dyn std::error::Error>` in public APIs".
4. **Stub architecture policy** (CODE-04) — `gemma3`/`llama4`/`phi4`/`mistral_small`
   (currently stub-only) need an explicit load-time policy. Phase 43 ARCH-05 will
   collapse them into a single `StubArchitecture`; Phase 40 must enforce the
   runtime policy so non-test builds cannot load them.
5. **`SchedulerEngine::prefix_cache_hit_rate()` placeholder** (core/src/scheduler/engine.rs:555-559)
   — returns `0.0` regardless of state; either wire to existing metrics counters
   (`prefix_cache_hits` / `prefix_cache_queries`) or remove from the public API.

FINAL-01 invariant: All 1179 tests remain green post-fix.

</domain>

<decisions>

## Implementation Decisions

### the agent's Discretion

All implementation choices are at the agent's discretion — pure infrastructure
phase. Use the codebase scout below to drive decisions. Specific guidance:

- **CODE-01 (TensorParallelError):** Use the AGENTS.md prescribed shape:
  `#[derive(thiserror::Error)]`, per-variant `#[error("...")]` strings, `#[source]`
  chains on `AllReduceFailed`/`CudaError` if they wrap another error (currently they
  take `String`, so `#[source]` would require adding an inner-error type — judgement
  call: keep `String` for parity with other project errors like `EngineError::ModelError`
  unless an inner-error type is naturally available). Preserve all 6 existing variants
  to keep call-site compatibility.
- **CODE-02 (Engine source chain):** Replace `EngineError::ModelError(e.to_string())`
  with a new variant or `From<vllm_traits::ModelError>` impl that preserves `#[source]`.
  Check `EngineError` enum in `crates/core/src/error.rs` for the right shape; if no
  `From` exists, add one (matches existing AGENTS.md convention).
- **CODE-03 (GrpcError):** Define a `GrpcError` enum in `crates/dist/src/grpc.rs` (or
  a new `error.rs` module) with variants for the failure modes actually reachable
  from `start_grpc_server`: bind failure, tonic transport failure. Add `From<...>`
  impls for `std::io::Error` and `tonic::transport::Error`. Update
  `crates/dist/src/lib.rs` to re-export it; check downstream callers (`bin/`, tests)
  and update them.
- **CODE-04 (Stub policy):** Define an `allow_stub` capability check. Pattern:
  - Add `LoadError::StubNotAllowed(String)` variant to `crates/model/src/loader/error.rs`
    (or wherever `LoadError` is defined).
  - In each stub architecture's `create_model()` (or unified equivalent), check
    `cfg!(test)` or a runtime capability flag; if not allowed, return `StubNotAllowed`.
  - Add a unit test asserting non-test profile rejects stubs (use `#[cfg(not(test))]`
    or runtime capability flag).
  - Note: full `StubArchitecture` collapse is Phase 43 ARCH-05. Phase 40 establishes
    the policy; ARCH-05 executes the consolidation. Coordinate by adding the policy
    in the unified location now (where Phase 43 will operate) — easiest path is
    a shared `is_stub_allowed()` helper in `crates/model/src/loader/stub_policy.rs`.
- **CODE-05 (prefix_cache_hit_rate):** Inspect metrics counters
  (`crates/core/src/metrics/collector.rs` or similar) for `prefix_cache_hits` and
  `prefix_cache_queries`. Decision tree:
  - If both counters exist as accessible fields → implement as
    `hits as f64 / queries.max(1) as f64` and return `0.0` if queries == 0.
  - If only one exists → implement what we can; document the gap.
  - If neither exists as accessible fields → remove the public API; check
    `grep -r "prefix_cache_hit_rate"` for all callers and update them.
  Prefer removal over a misleading stub return (per audit finding).

</decisions>

<code_context>

## Existing Code Insights

### Reusable Assets

- **`EngineError` enum** (crates/core/src/error.rs) — has variants `ModelError(String)`,
  `SchedulerError`, `KvCacheError`, etc.; check `From` impls for whether `ModelError`
  is already wired (likely not — the audit found `e.to_string()` pattern, indicating
  manual wrapping without source preservation).
- **`LoadError` enum** (crates/model/src/loader/error.rs) — used by `ModelLoader::load()`;
  add `StubNotAllowed` variant here. Stub architectures' `create_model()` are
  registered via `Architecture` trait; check `register_all_archs()` in
  `crates/model/src/arch/registry.rs`.
- **Metrics counters** (crates/core/src/metrics/) — `EnhancedMetricsCollector`
  tracks `prefix_cache_hits` and `prefix_cache_queries` (per v22.0 audit knowledge);
  verify exact field names by reading the file.
- **19 existing thiserror error enums** (project convention) — sample:
  `DraftRegistryError`, `EngineError`, `LoadError`. All follow:
  `#[derive(Debug, thiserror::Error)]` + per-variant `#[error("...")]` + `#[source]`
  for wrapped errors.

### Established Patterns

- **Error enums** — AGENTS.md "Error Type Conventions":
  - `#[derive(thiserror::Error)]` — never hand-written Display/Error
  - `#[error("...")]` on every variant
  - `#[source]` for wrapped errors (preserves chain)
  - `From<E>` impls for cross-crate conversion in `error/mod.rs`
  - No `Box<dyn std::error::Error>` in public APIs
- **Test patterns** — unit tests in `#[cfg(test)] mod tests {}` blocks; integration
  tests in `crates/*/tests/*.rs`. Use `FakeModel`/`StubModel` from `vllm-testing`.
- **Verification gates** — `just clippy`, `just fmt-check`, `just nextest` (skips
  `#[ignore]`), `just nextest-all`.

### Integration Points

- **`TensorParallelError` definition site:** `crates/traits/src/types.rs:86-112`
  (verified via Read).
- **`Engine::step()` source-chain loss:** `crates/core/src/engine.rs:677` —
  `Err(crate::error::EngineError::ModelError(e.to_string()))` (verified).
- **`Box<dyn Error>` site:** `crates/dist/src/grpc.rs:129` —
  `Result<(), Box<dyn std::error::Error>>` (verified).
- **Stub architectures:** `crates/model/src/{gemma3,llama4,phi4,mistral_small}/`
  directories + `register.rs` files; collapse policy lives in
  `crates/model/src/arch/registry.rs::register_all_archs()`.
- **Prefix cache hit rate:** `crates/core/src/scheduler/engine.rs:555-559` —
  `pub fn prefix_cache_hit_rate(&self) -> f64 { 0.0 }` (verified).

</code_context>

<specifics>

## Specific Ideas

- **CODE-01 variant preservation:** All 6 variants (`InvalidWorldSize`,
  `InvalidRank`, `DeviceMismatch`, `InputSizeMismatch`, `AllReduceFailed(String)`,
  `CudaError(String)`) must remain. Change is mechanical: `impl Display` and
  `impl Error` → `#[derive(thiserror::Error)]` with `#[error("...")]` per variant.
- **CODE-02 From impl:** Add `impl From<vllm_traits::ModelError> for EngineError`
  if not present; add a `ModelErrorSource(#[source] vllm_traits::ModelError)`
  variant OR extend existing `ModelError` variant to accept a generic. Audit
  intent: "preserve the underlying `vllm_traits::ModelError`" — implies a new
  `Model(#[source] vllm_traits::ModelError)` variant. Decision: add new variant,
  deprecate `ModelError(String)` with `#[deprecated(since = "0.23.0", note = "...")]`.
- **CODE-03 GrpcError variants:** Minimum variants: `Bind(String)`,
  `Transport(#[source] tonic::transport::Error)`. May add `Io(#[source] std::io::Error)`
  if `TcpListener::bind` is the only error site.
- **CODE-04 stub policy:** Since Phase 43 will collapse the 4 stubs into one
  `StubArchitecture`, the cleanest approach is to add the policy in the
  `StubArchitecture::create_model()` (after Phase 43 lands). For Phase 40, add
  the policy check in each of the 4 stub architectures' `create_model()` so
  the policy is enforced even pre-collapse. Phase 43 will inherit the policy
  from the unified struct.
- **CODE-05 preference:** Implement against metrics if both counters exist;
  remove the API otherwise. Do NOT keep the placeholder returning `0.0`.

</specifics>

<deferred>

None — phase scope is strictly the 5 CODE-* findings. Phase 43 ARCH-05
consolidates stub architectures; Phase 43 ARCH-09/10 unify greedy_sample and
Architecture types.

</deferred>
