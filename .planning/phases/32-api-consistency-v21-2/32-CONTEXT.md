# Phase 32: API Consistency (v21.2) - Context

**Gathered:** 2026-06-27
**Status:** Ready for planning
**Mode:** Auto-generated (infrastructure phase — smart discuss skipped)

<domain>
## Phase Boundary

Make API surface uniform — typed errors throughout, ergonomic builders, structured error context, sync/async trait splits where the runtime requires it. Document conventions so future API additions don't regress.

Concrete deliverables:
- Document builder/struct-literal convention in AGENTS.md (API-01, API-07)
- Add `#[source]` for error chains; add `From<candle_core::Error>` for `EngineError` (API-02, API-09)
- Replace 2 `Box<dyn Error>` in model lib with typed errors (API-03)
- Split `FallbackStrategy` into sync + async traits (API-08)
- Add request_id/seq_id fields to error variants (API-11)
- Add `Default` for object-safe traits + compile-only `dyn Trait` tests (API-06, API-10)
- Introduce 22 new builders where only `Default` exists (API-05)

Note: Several v19.0 audit items (API-04 mutex unwraps, EngineError variants) were already addressed in v20 phases; this phase focuses on remaining items.

</domain>

<decisions>
## Implementation Decisions

### the agent's Discretion

All implementation choices are at the agent's discretion — API consistency phase.

Specific implementation choices to make during planning:
- Whether to break `DraftRegistryError::LoadFailed` signature for `#[source]` or add a new variant
- Which structs get builders (limit to public API types)
- How to split `FallbackStrategy` without breaking existing callers

</decisions>

<code_context inherited="false">
## Existing Code Insights

### Reusable Assets
- `SpeculationConfigBuilder` — existing builder pattern
- `RequestBuilder` / `BatchBuilder` in vllm-testing — existing builders
- `thiserror` available for new error types
- `tokio::task::JoinError` available for task join errors

### Established Patterns
- All error enums use `#[derive(thiserror::Error)]` (Phase 30 invariant)
- Cross-crate `From` impls in `error/mod.rs`
- `Arc<Mutex<...>>` patterns use `lock_mutex` helper

### Integration Points
- `EngineError::from` impls in `crates/core/src/error/mod.rs`
- `DraftRegistryError` in `crates/core/src/speculative/registry/errors.rs`
- `FallbackStrategy` trait + impls in `crates/core/src/circuit_breaker/strategy.rs`

</code_context>

<specifics>
## Specific Ideas

No specific requirements — API consistency phase. Key constraints:

- **Backward compatibility:** All breaking API changes need `#[deprecated]` markers + migration path
- **vllm-dist feature-gate:** Must remain feature-gated
- **Test invariants:** 1144+ tests must remain green; clippy/fmt must remain clean
- **Doc coverage:** Maintain ≥60% workspace doc coverage (currently 97.8%)

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>
