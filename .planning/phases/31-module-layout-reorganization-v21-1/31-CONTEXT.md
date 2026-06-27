# Phase 31: Module Layout Reorganization (v21.1) - Context

**Gathered:** 2026-06-27
**Status:** Ready for planning
**Mode:** Auto-generated (infrastructure phase — smart discuss skipped)

<domain>
## Phase Boundary

Reorganize oversized God modules into focused sub-trees so contributors can navigate and modify the codebase without cross-cutting concerns; error types live at their semantic boundaries (not in convenient-but-wrong locations). Concrete deliverables:

- Decompose `draft_registry.rs` (929 LOC) into `registry/{loader,lifecycle}.rs` + thin re-export shim
- Unify `engine.rs` + `engine/speculative.rs` into a single `engine/speculative/` sub-tree
- Move `qwen3_config.rs` to `qwen3/config.rs`; extract `attention/mod.rs` utilities to `attention/util.rs`
- Decide vllm-testing lemon pair (split or document infeasibility); move `test_fixtures.rs` into vllm-testing
- Move `TensorParallelError` to `vllm-dist::error` with re-export from `vllm-traits`; migrate callers
- Verify/remove unused exports in vllm-testing

All changes preserve 1144+ tests, clippy/fmt cleanliness, and vllm-dist feature-gating.

</domain>

<decisions>
## Implementation Decisions

### the agent's Discretion

All implementation choices are at the agent's discretion — pure infrastructure/refactor phase. Use ROADMAP phase goal, success criteria, and codebase conventions to guide decisions.

Specific implementation choices to make during planning:
- Exact file split boundaries (where each sub-module boundary lives)
- Re-export strategy (shim vs flat re-export)
- Whether to split vllm-testing or document infeasibility (the phase allows either)
- Migration order (which file splits first to minimize churn)

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- vllm-core has 9 engine modules + speculative sub-tree at `crates/core/src/engine/`
- vllm-model has draft_registry at `crates/core/src/draft_registry.rs` (929 LOC)
- vllm-testing at `crates/testing/` exposes shared test fixtures
- vllm-dist feature-gated behind `--features multi-node`
- TensorParallelError currently defined in vllm-traits (canonical home should be vllm-dist)

### Established Patterns
- Sub-tree pattern: `crates/core/src/scheduler/{queue,preemption,eviction,batch}/`
- Module re-export pattern: `pub use submodule::*;` in `mod.rs`
- Feature-gate pattern: `#[cfg(feature = "multi-node")]` + `Cargo.toml` feature definition
- Backward-compat via `pub type` aliases + `#[deprecated]` markers (Phase 25/30 precedent)

### Integration Points
- All engine code reaches draft_registry via `use crate::draft_registry::*`
- Test fixtures consumed by server integration tests via `use vllm_testing::fixtures::*`
- vllm-dist error reachable via `vllm_traits::TensorParallelError` (will be re-export)

</code_context>

<specifics>
## Specific Ideas

No specific requirements — infrastructure/refactor phase. Key constraints:

- **Backward compatibility:** All public API removals need `#[deprecated]` markers + migration path (per DEP-01/02 from Phase 30)
- **vllm-dist feature-gate:** Must remain feature-gated; never compiled by default
- **Test invariants:** 1144+ tests must remain green; clippy/fmt must remain clean
- **LOC budget:** Each leaf file <300 LOC where possible

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>
