# Phase 25: P0 Critical Fixes (v20.1) - Context

**Gathered:** 2026-06-27
**Status:** Ready for planning
**Mode:** Autonomous (infrastructure phase — discuss skipped)

<domain>
## Phase Boundary

Phase 25 (v20.1) eliminates 5 P0 architecture / API / error violations identified by the v19.0 audit (BACKLOG.md):

- **P0-01 (ARCH-F-11)**: Eliminate `vllm-model → vllm-dist` dependency edge — feature-gate vllm-dist usage in vllm-model behind `--features multi-node`
- **P0-02 (ARCH-F-12)**: Make `vllm-core → vllm-model` dependency feature-gated — split the optional `cuda-graph` feature so default build does not require vllm-model
- **P0-03 (API-F-01)**: Convert `ModelError` from struct → enum so it can be matched exhaustively; migrate all `match ModelError` sites to enum variants
- **P0-04 (API-F-02)**: Make 8 non-object-safe traits object-safe (or split + associated types) so `dyn Trait` works at all 14+ existing call sites
- **P0-05 (API-F-03)**: Convert `CudaGraphError` to `#[derive(thiserror::Error)]`; remove hand-written `Display`/`Error` impls (~14 LOC); preserve `From<OldStrError>` for semver compat

Phase 25 is high-risk architectural change — rollback criteria and pre-Phase-25 baseline capture are mandatory before any code changes.

</domain>

<decisions>
## Implementation Decisions

### the agent's Discretion

All implementation choices for individual code changes are at the agent's discretion. The phase decisions are already locked by:

1. **ROADMAP.md Phase 25 success criteria** — 6 numbered criteria define what must be TRUE (cargo tree output, cargo build with/without features, cargo doc enum verification, dyn compile-only tests, thiserror derive, 287+ tests remain green, clippy clean)
2. **v20.0 constraint preservation** — All 287+ tests must remain green; public API removals require `#[deprecated]` markers; vllm-dist feature-gated (not removed)
3. **Rollback criteria** — pre-Phase-25 baseline (commit SHA + cargo tree snapshot + green test baseline) must be captured; rollback via `git revert` chain if any trigger fires (integration test failure, default-feature build failure, >2 P0 regressions, clippy error)
4. **Backward-compat preservation** — any removed public API MUST retain ≥1 minor-version `#[deprecated]` marker or type alias; new public types in first stable release must be marked deprecated-with-migration-note

The planner should generate 5 plans (25-01..25-05) matching the ROADMAP.md plan list:

- **25-01**: Eliminate `vllm-model → vllm-dist` edge (feature-gate) — ARCH-F-11
- **25-02**: Feature-gate `vllm-core → vllm-model` (cuda-graph) — ARCH-F-12
- **25-03**: Convert `ModelError` struct → enum + migrate match sites — API-F-01
- **25-04**: Convert `CudaGraphError` to thiserror + From shim — API-F-03
- **25-05**: Make 8 traits object-safe (or split + associated types) + add `dyn_safety.rs` — API-F-02

Plans should be ordered so that structural changes (25-01, 25-02) precede type changes (25-03, 25-05), since feature-gating may expose or hide code paths that affect error type and trait usage.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets (from v19.0 audit)

- `.planning/audit/architecture/REPORT.md` — full ARCH-F-11 / ARCH-F-12 findings with code citations and remediation guidance
- `.planning/audit/api/REPORT.md` — full API-F-01 / API-F-02 / API-F-03 findings with current trait/error type definitions and call sites
- `.planning/audit/BACKLOG.md` — P0 finding list with priority rationale
- `.planning/audit/MIGRATION-ROADMAP.md` — proposed v20.1 sub-phase plan (maps 1:1 to Phase 25 plans)
- `.planning/PROJECT.md` — project decisions: "vllm-dist feature-gate (v20.0 decision)" + "Object-safety co-fix (v20.0 decision)"
- `.planning/STATE.md` — decision log: "Combined into single Phase 25 to avoid regression risk from sequential changes"

### Established Patterns

- `thiserror::Error` derive is the project standard for error enums (used in `EngineError`, `SchedulerError`, `KvCacheError`, etc.) — reference these for `ModelError` / `CudaGraphError` conversion
- Cargo feature gates follow pattern `default = []` + `optional = true` + `#[cfg(feature = "multi-node")]` in code; `vllm-dist` is currently a workspace dependency
- Trait object-safety rules: no generic methods, no `Self` in return types (except `Self`), no `Self: Sized` methods — when fixing, prefer splitting trait or using associated types

### Integration Points

- `crates/model/Cargo.toml` — declare `vllm-dist` as `optional = true` + `multi-node` feature; update `default` features
- `crates/core/Cargo.toml` — declare `vllm-model` as `optional = true` + `cuda-graph` feature
- `crates/testing/tests/dyn_safety.rs` — NEW file for compile-only object-safety tests (≥8 tests, one per affected trait)
- `crates/model/src/error.rs` — `ModelError` definition lives here
- `crates/core/src/cuda_graph.rs` — `CudaGraphError` definition lives here
- `crates/server/src/main.rs` — entry point (consumes `ModelError` indirectly)

</code_context>

<specifics>
## Specific Ideas

- **Pre-Phase-25 baseline capture is mandatory** — capture last passing commit SHA + `cargo tree -p vllm-model` snapshot + full green test baseline before any change
- **Per-crate rollback hooks** — `pub type ModelError = OldModelError` type alias restoration if enum conversion causes downstream breakage
- **Compile-only dyn_safety tests** — should use `fn _assert_obj_safe<T: ?Sized + Trait>() {}` pattern; no `#[test]` needed since compile-failure is the test
- **Feature flag symmetry** — `--features multi-node` enables vllm-dist everywhere; `--features cuda-graph` enables vllm-model in vllm-core

</specifics>

<deferred>
## Deferred Ideas

- **`Engine::step()` speculative-mode hang** — pre-existing bug, not introduced by Phase 25; deferred to v20.7+
- **`vllm-dist` resurrection (multi-node work)** — code stays feature-gated; actual multi-node implementation deferred to v20.7+
- **44 P2 issues + 13 P3 informational findings** — out of v20.0 scope, deferred to v20.7+

</deferred>
