# Requirements: vllm-lite

**Defined:** 2026-06-27
**Milestone:** v19.0 Codebase Health Audit (analysis-only)
**Core Value:** Fast, memory-efficient LLM inference with continuous batching, paged KV cache, and tensor parallelism — deployed with production-grade ops tooling and security.

## v19.0 Requirements

This milestone is **analysis-only**. No code changes will be made. Each requirement produces an audit artifact (report, table, list) consumed by future milestones (v20.0+) for actual remediation.

### Architecture audit

- [ ] **ARCH-01**: Crate dependency graph audited (verify directional flow: traits ← core ← model/server/dist; no upward deps)
- [ ] **ARCH-02**: Module boundary audit complete (each module has single, clear responsibility; no God modules)
- [ ] **ARCH-03**: Circular dependency scan complete (cargo metadata + manual review)
- [ ] **ARCH-04**: Layering consistency audit (e.g., scheduler doesn't import from server; model doesn't import from core)
- [ ] **ARCH-05**: Test architecture audit (unit/integration/bench separation; shared testing crate hygiene)

### Naming audit

- [ ] **NAME-01**: File naming audit complete (identify casually-named files like `17_*.rs` stage-info-named files; document each finding)
- [ ] **NAME-02**: Type/struct/enum naming consistency audit (PascalCase, descriptive, no redundant suffixes)
- [ ] **NAME-03**: Function/method naming audit (snake_case, action verbs, consistent prefix patterns)
- [ ] **NAME-04**: Variable naming audit (descriptive, no single-letter except indices; consistent naming for similar concepts)
- [ ] **NAME-05**: Module name audit (matches file name; consistent depth)

### Comments + documentation audit

- [ ] **DOCS-01**: Doc-comment coverage measured for public API (target: ≥80% on `pub` items)
- [ ] **DOCS-02**: Module-level documentation audit (each module has `//!` or top-of-file context)
- [ ] **DOCS-03**: Stale comment / TODO audit (identify comments referencing old code, dead TODOs, misleading docstrings)
- [ ] **DOCS-04**: External documentation audit (root README, AGENTS.md, .planning docs accuracy against current codebase)
- [ ] **DOCS-05**: Architecture-decision records (ADRs) — identify documented rationale vs. tribal knowledge

### API + error handling audit

- [ ] **API-01**: Public API surface consistency (function signatures, return types, builder patterns)
- [ ] **API-02**: Error type audit (thiserror usage, error variants coverage, error message quality)
- [ ] **API-03**: Error ergonomics audit (Result types, error context propagation, `From` conversions)
- [ ] **API-04**: Trait design audit (object safety, async/sync consistency, default method usage)
- [ ] **API-05**: Deprecation hygiene (deprecated items properly marked, migration paths documented)

### Synthesis

- [ ] **SYNTH-01**: Cross-dimensional synthesis report complete (correlates findings across ARCH/NAME/DOCS/API)
- [ ] **SYNTH-02**: Prioritized remediation backlog produced (P0/P1/P2 with impact, cost, suggested phase)
- [ ] **SYNTH-03**: Suggested v20.0+ migration roadmap (which findings group into which future phase)

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Excluded                                  | Reason                                                                                          |
| ----------------------------------------- | ----------------------------------------------------------------------------------------------- |
| Any code modification                     | v19.0 is analysis-only; renaming/refactoring deferred to v20.0+                                 |
| New features                              | Not the goal of an audit milestone                                                              |
| Performance optimization                  | Separate audit/optimization cycle                                                               |
| Test re-runs / CI changes                 | No code touched, existing CI still validates                                                    |
| Migration execution (even simple rename)  | Even trivial changes deferred to dedicated future milestone with its own audit + execution cycle |
| Security audit                            | Out of scope — covered by earlier security hardening (v13.0)                                    |
| Performance benchmarking                  | Out of scope — covered by earlier benchmarking suite (v14.0)                                    |

## Traceability

| Requirement | Phase    | Status   |
| ----------- | -------- | -------- |
| ARCH-01     | Phase 20 | Pending  |
| ARCH-02     | Phase 20 | Pending  |
| ARCH-03     | Phase 20 | Pending  |
| ARCH-04     | Phase 20 | Pending  |
| ARCH-05     | Phase 20 | Pending  |
| NAME-01     | Phase 21 | Pending  |
| NAME-02     | Phase 21 | Pending  |
| NAME-03     | Phase 21 | Pending  |
| NAME-04     | Phase 21 | Pending  |
| NAME-05     | Phase 21 | Pending  |
| DOCS-01     | Phase 22 | Pending  |
| DOCS-02     | Phase 22 | Pending  |
| DOCS-03     | Phase 22 | Pending  |
| DOCS-04     | Phase 22 | Pending  |
| DOCS-05     | Phase 22 | Pending  |
| API-01      | Phase 23 | Pending  |
| API-02      | Phase 23 | Pending  |
| API-03      | Phase 23 | Pending  |
| API-04      | Phase 23 | Pending  |
| API-05      | Phase 23 | Pending  |
| SYNTH-01    | Phase 24 | Pending  |
| SYNTH-02    | Phase 24 | Pending  |
| SYNTH-03    | Phase 24 | Pending  |

**Coverage:**

- v19.0 requirements: 23 total
- Mapped to phases: 23
- Unmapped: 0 ✓

**Audit execution model:**

Each audit phase produces:

1. A report at `.planning/audit/{dimension}/REPORT.md` with detailed findings
2. A summary table at `.planning/audit/{dimension}/SUMMARY.md` (P0/P1/P2 prioritized)
3. Raw inventory data (file lists, naming tables, doc-coverage stats) where applicable

Synthesis phase (Phase 24) reads all four dimension reports and produces:

- `.planning/audit/SYNTHESIS.md` — cross-cutting findings
- `.planning/audit/BACKLOG.md` — P0/P1/P2 remediation backlog with impact/cost/suggested-phase columns
- `.planning/audit/MIGRATION-ROADMAP.md` — proposed v20.0+ phase breakdown (advisory only)

---

*Requirements defined: 2026-06-27*
*Last updated: 2026-06-27 after initial definition*
