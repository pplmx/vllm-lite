---
gsd_state_version: 1.0
milestone: v19.0
milestone_name: Codebase Health Audit
status: complete
last_updated: "2026-06-27T13:30:00.000Z"
last_activity: 2026-06-27
progress:
  total_phases: 5
  completed_phases: 5
  total_plans: 5
  completed_plans: 5
  percent: 100
---
# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-27)

**Core value:** Fast, memory-efficient LLM inference with continuous batching, paged KV cache, and tensor parallelism — deployed with production-grade ops tooling and security.
**Current focus:** v19.0 Codebase Health Audit — SHIPPED 2026-06-27 (5 phases, 23/23 requirements, audit passed, 0 source code modified)

---

## Current Position

Phase: All v19.0 phases complete (20, 21, 22, 23, 24)
Plan: All 5 plans shipped
Status: Milestone complete — audit passed, archived to .planning/milestones/v19.0-*
Last activity: 2026-06-27 — v19.0 archived; next milestone (v20.0+) should consume `.planning/audit/BACKLOG.md`

## Performance Metrics

**Velocity (v19.0 actual):**

- Total plans completed: 5 (v19.0)
- Total commits across v19.0: ~18 (context+plan × 5, audit × 5, summary × 5, archive × 1)
- Audit artifacts produced: 11 markdown files / 3,578 lines
- Findings consolidated: 100 (5 P0 + 38 P1 + 44 P2 + 13 P3)
- Source code modified: **0 lines** (analysis-only constraint honored)

**Velocity (v18.0 historical):**

- Total plans completed: 5 (v18.0)
- Total commits across v18.0: 5 (one per phase) + 9 review fixes + 1 fmt
- Total test count change: 209 → 287+ tests (lib + integration)
- Total lib test count change: 209 → 263 (+54 tests, unchanged by Phase 19)
- Total integration test count: 14 → 23 (+9 in engine_v18_wiring.rs)
- Total bench count: 3 → 4 (+1 bench)

**By Phase (v19.0):**

| Phase | Plans | Findings | Artifacts |
| ----- | ----- | --------:| --------- |
| 20    | 1     | 17       | REPORT.md (481L) + SUMMARY.md (66L) |
| 21    | 1     | 26       | REPORT.md (431L) + SUMMARY.md (100L) |
| 22    | 1     | 24       | REPORT.md (566L) + SUMMARY.md (179L) |
| 23    | 1     | 33       | REPORT.md (653L) + SUMMARY.md (113L) |
| 24    | 1     | 100 consolidated | SYNTHESIS.md (391L) + BACKLOG.md (280L) + MIGRATION-ROADMAP.md (318L) |

**By Phase (v18.0 historical):**

| Phase | Plans | Tests Added | Cumulative Lib Total |
| ----- | ----- | ----------- | -------------------- |
| 18.1  | 1     | 15          | 224                  |
| 18.2  | 1     | 30          | 254                  |
| 18.3  | 1     | 9           | 263                  |
| 18.4  | 1     | 14 (integration) | 263              |
| 19    | 1     | 9 (integration) | 263              |

**Recent Trend:**

- Last 5 commits: docs(audit) phase 20 → phase 21 → phase 22 → phase 23 → phase 24 (synthesis)
- v19.0 shipped cleanly: all 5 phases → audit passed → archived
- No source code modifications across the entire milestone (analysis-only constraint enforced)
- Trend: Audit-before-refactor pattern validated; produces 100 actionable findings for v20.0+

---

## Accumulated Context

### Decisions

- v18.0 uses 4 phases (18.1-18.4) plus Phase 19 gap-closure, continuing numbering from v17.0 which ended at phase 17.4
- v18.0 phase ordering: Draft Registry + External Loading → Lifecycle + Memory Budget → Request Routing + Fallback → Integration Tests + Benchmarks → Gap Closure
- Phase 18.1 collocates MMLT-01..03 + LIFE-01 (registry foundation before lifecycle/routing depend on it)
- Phase 18.4 is a validation phase with 0 direct requirements — verifies Phase 18.1-18.3 work end-to-end
- Phase 19 is the v18.0 audit gap closure — wires resolver into Engine step loop, FALL-02, exporter, server config, production DraftLoader
- **v19.0 roadmap decision**: 5 phases (20-24), one per audit dimension + synthesis; v18.0 ended at Phase 19, so v19.0 starts at Phase 20
- **v19.0 execution model**: Each audit phase (20-23) is a single read-only subagent dispatch producing `REPORT.md` + `SUMMARY.md` in `.planning/audit/{dimension}/`; Phase 24 reads all four reports and produces synthesis + backlog + migration roadmap
- **v19.0 parallelism**: Phases 20-23 each read disjoint slices of the codebase, so they could run in parallel — but reports are sequenced 20→21→22→23 to make the audit dimension narrative coherent for Phase 24's synthesis reader
- **v19.0 out-of-scope**: No code changes, no new features, no performance optimization, no test re-runs, no CI changes — strictly audit artifacts only
- **v19.0 deliverable shape**: `.planning/audit/` directory with 4 dimension subdirs (each with REPORT.md + SUMMARY.md) + 3 root files (SYNTHESIS.md / BACKLOG.md / MIGRATION-ROADMAP.md)

### Architecture Patterns Established

1. **Registry is loader-agnostic** — `DraftModelRegistry` does not depend on `vllm-model`. The actual `ModelLoader` invocation lives at the caller (server or test harness), which then hands a `Box<dyn ModelBackend>` to `attach_loaded`. This keeps the registry usable from contexts where `vllm-model` is unavailable.

2. **Lazy loading via state machine** — `DraftState::{Unloaded, Loaded}` makes lazy semantics explicit. `register` is metadata-only; `attach_loaded` is the only path to `Loaded`; `unload` returns to `Unloaded`.

3. **MemoryBudget is shared via Arc** — Engine and registry both hold `Arc<MemoryBudget>` so they share a single source of truth. Reservations are atomic via `try_reserve_draft`.

4. **Per-request resolution, not per-batch** — `DraftResolver::resolve` is called per-request. Mixed-draft routing (RTE-03) emerges naturally from per-request resolution.

5. **FALL-01 is silent, FALL-02 is sticky** — Load failures log + metric + silently fall back to SelfSpec (user unaware). Runtime errors set `Sequence.degraded_draft = true` (sticky for lifetime of sequence).

6. **Resolver is wired in Engine** — `Engine.draft_resolver: Option<Arc<DraftResolver>>`. When `Some`, `step_speculative_inner` dispatches via `generate_per_seq_drafts`. When `None`, falls back to legacy single-draft path. Backward-compatible with `new_boxed`.

7. **Forward error capture via catch_unwind** — `generate_per_seq_drafts` wraps each per-seq forward in `catch_unwind(AssertUnwindSafe(...))` to handle both `Result::Err` and panic. Degraded seqs are skipped on subsequent steps.

8. **Audit-before-refactor (v19.0)** — Non-functional refactoring work (file renaming, module boundary tightening, error-type unification, doc coverage catch-up) is gated on a structured audit. Audit findings become input to v20.0+ remediation phases, each with measurable acceptance criteria.

### Pending Todos

None — v19.0 milestone complete. All audit phases executed; synthesis + backlog + migration roadmap delivered.

### Blockers/Concerns

- `Engine::step()` in speculative mode hangs — pre-existing bug (not introduced by v18.0 or v19.0). 2 Phase 19 integration tests are `#[ignore]`d awaiting the fix. Not blocking audit; will be addressed in v20.0+ remediation.
- Real-model benchmark missing — current bench uses stub backends. Deferred until a small real checkpoint is available. Out of scope for v19.0 audit.
- v20.0+ decisions needed:
  - Phase ordering for v20.1 (P0 fixes) vs v20.2 (module tree restoration) — can be parallel?
  - `vllm-dist` fate: continue / feature-gate / remove (~1,600 LOC unused)
  - Doc coverage target: accept 60% intermediate or push to 80%?

---

## Deferred Items

| Category    | Item                                              | Status            | Deferred At |
| ----------- | ------------------------------------------------- | ----------------- | ----------- |
| Multi-Model | External draft model support (MULTI-01, MULTI-02) | Shipped in v18.0  | 2026-05-13   |
| Multi-Model | Draft hot-swap mid-request (MULTI-04)             | Deferred to v19+ → v20.0+ | 2026-06-27 |
| Multi-Model | Draft fine-tuning hooks (MULTI-05)                | Out of scope      | 2026-05-13   |
| Multi-Model | Cross-GPU draft placement (MULTI-06)              | Deferred to v19+ → v20.0+ | 2026-06-27 |
| Engine      | Wire DraftResolver into step_speculative loop     | **Closed by Phase 19** | 2026-06-27 |
| Engine      | HTTP exporter integration of v18.0 metrics        | **Closed by Phase 19** | 2026-06-27 |
| Engine      | Real-model benchmark (vs stub backend)            | Deferred to v20.0+ | 2026-06-27 |
| Engine      | Fix Engine::step() speculative-mode hang         | Deferred to v20.0+ | 2026-06-27 |
| Refactor    | **5 P0 architecture/API violations**              | Deferred to v20.1 (see BACKLOG.md) | 2026-06-27 |
| Refactor    | **38 P1 issues** (module tree, docs, errors, naming) | Deferred to v20.2-v20.6 (see MIGRATION-ROADMAP.md) | 2026-06-27 |
| Refactor    | **44 P2 issues** + **13 P3 informational**        | Deferred to v20.x (optional) | 2026-06-27 |
| Architecture| `vllm-dist` fate decision (continue/gate/remove)  | Deferred to v20.1 (open question) | 2026-06-27 |
| Documentation | README broken code example fix                   | Deferred to v20.4/20.5 | 2026-06-27 |
| Documentation | 12+ tribal-knowledge decisions need ADRs          | Deferred to v20.5 | 2026-06-27 |

## Session Continuity

Last session: 2026-06-27 v19.0 milestone complete (5/5 phases, audit passed, archived)
Stopped at: v19.0 archived to `.planning/milestones/v19.0-{ROADMAP,REQUIREMENTS}.md`; backlog ready in `.planning/audit/BACKLOG.md` for v20.0+ consumption
Resume file: None
Next command: `/gsd-new-milestone v20.0 Layering Restoration & P0 Fixes` (or similar — see `.planning/audit/MIGRATION-ROADMAP.md` for full proposed v20.x phase breakdown)

---

*State updated: 2026-06-27 — v19.0 milestone complete (5/5 phases, audit passed, 100 findings consolidated, archived)*
