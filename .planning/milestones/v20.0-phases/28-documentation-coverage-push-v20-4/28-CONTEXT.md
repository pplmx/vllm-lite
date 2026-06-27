# Phase 28: Documentation Coverage Push (v20.4) - Context

**Gathered:** 2026-06-27
**Status:** Ready for planning
**Mode:** Autonomous (infrastructure phase — discuss skipped)

<domain>
## Phase Boundary

Phase 28 (v20.4) raises workspace doc coverage from baseline (7.6% per audit, current post-Phase-25/26/27 may differ) toward ≥60% workspace-wide with per-crate targets of ≥80%. The phase:

- **DOC-01**: Adds a coverage measurement script (`scripts/doc_coverage.sh`) producing a per-crate table
- **DOC-02**: Backfills `///` doc comments on undoc'd `pub` items across 6 crates (traits/dist/server/model/core/testing)
- **DOC-03**: Adds `//!` module-level docs to source files lacking them
- **DOC-04**: Fixes the broken `SchedulerEngine::new(config, 1024)` example in README.md
- **DOC-05**: Updates README.md "Supported Architectures" table to list all 10 registered architectures
- **DOC-06**: Updates README.md crate count from "7 crates" to "6 crates" (matches Cargo.toml [workspace] members)
- **DOC-07**: Reconciles AGENTS.md "Architecture" section with current crate structure; verifies file:line refs exist

</domain>

<decisions>
## Implementation Decisions

### the agent's Discretion

All implementation choices are at the agent's discretion. Locked constraints:

1. **Coverage target**: workspace ≥60%; per-crate ≥80% (per REQUIREMENTS.md DOC-01/02)
2. **Doc style**: 1-line `///` summary is acceptable; full examples not required for trivial items
3. **Module-level `//!`**: short purpose statement (1-3 lines) is acceptable; full architecture rationale not required
4. **Phase 28 is non-breaking**: no public API signatures change; only doc comments and README/AGENTS.md prose updates
5. **All 287+ tests must remain green** (FINAL-01 invariant); no test modifications
6. **Order**: DOC-01 (script + baseline) must precede DOC-02 (so we can measure before/after)

Plans generated for Phase 28:
- **28-01**: Add coverage measurement script + baseline report (DOC-01)
- **28-02**: Backfill `///` on `traits` crate (0% → 80%)
- **28-03**: Backfill `///` on `dist` crate (2.7% → 80%)
- **28-04**: Backfill `///` on `server` crate (4.9% → 80%)
- **28-05**: Backfill `///` on `model` crate (8.5% → 80%)
- **28-06**: Backfill `///` on `core` crate (9.0% → 80%)
- **28-07**: Backfill `///` on `testing` crate (12.9% → 80%)
- **28-08**: Add `//!` to source files lacking module-level docs (DOC-03)
- **28-09**: Fix README code example + architecture list + crate count (DOC-04, DOC-05, DOC-06)
- **28-10**: Reconcile AGENTS.md Architecture section (DOC-07)

</decisions>

