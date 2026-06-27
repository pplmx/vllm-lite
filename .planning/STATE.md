---
gsd_state_version: 1.0
milestone: v19.0
milestone_name: Codebase Health Audit
status: planning
last_updated: "2026-06-27T05:00:00.000Z"
last_activity: 2026-06-27
progress:
  total_phases: 5
  completed_phases: 0
  total_plans: 5
  completed_plans: 0
  percent: 0
---
# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-26)

**Core value:** Fast, memory-efficient LLM inference with continuous batching, paged KV cache, and tensor parallelism — deployed with production-grade ops tooling and security.
**Current focus:** v19.0 Codebase Health Audit (analysis-only, no code changes) — roadmap defined (5 phases), execution not started.

---

## Current Position

Phase: Phase 20 (Architecture Audit) — planning complete, execution not started
Plan: —
Status: Defining roadmap
Last activity: 2026-06-27 — Milestone v19.0 roadmap created (Phases 20-24)

## Performance Metrics

**Velocity (v18.0 historical):**

- Total plans completed: 5 (v18.0)
- Total commits across v18.0: 5 (one per phase) + 9 review fixes + 1 fmt
- Total test count change: 209 → 287+ tests (lib + integration)
- Total lib test count change: 209 → 263 (+54 tests, unchanged by Phase 19)
- Total integration test count: 14 → 23 (+9 in engine_v18_wiring.rs)
- Total bench count: 3 → 4 (+1 bench)

**v19.0 Projection:**

- 5 phases, all analysis-only (no code modifications)
- Each phase produces concrete artifacts in `.planning/audit/`
- Phase 24 (synthesis) gates on Phase 20-23 outputs
- No test changes expected (existing 287+ tests still validate code as-is)

**By Phase (v18.0 historical):**

| Phase | Plans | Tests Added | Cumulative Lib Total |
| ----- | ----- | ----------- | -------------------- |
| 18.1  | 1     | 15          | 224                  |
| 18.2  | 1     | 30          | 254                  |
| 18.3  | 1     | 9           | 263                  |
| 18.4  | 1     | 14 (integration) | 263              |
| 19    | 1     | 9 (integration) | 263              |

**Recent Trend:**

- Last 5 commits: feat(18.1) → feat(18.2) → feat(18.3) → test+bench(18.4) → gap-closure(19)
- Phase 19 closed all 3 audit gaps + 2 cross-phase issues; 9 code-review fixes applied
- Trend: All phases shipped cleanly, no rollbacks, all clippy/fmt checks passed

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

- Phase 20: Architecture Audit — execute subagent to produce `.planning/audit/architecture/{REPORT.md,SUMMARY.md}`
- Phase 21: Naming Audit — execute subagent after Phase 20
- Phase 22: Comments + Documentation Audit — execute subagent after Phase 21
- Phase 23: API + Error Handling Audit — execute subagent after Phase 22
- Phase 24: Synthesis — execute after all four dimension reports exist; produces SYNTHESIS.md / BACKLOG.md / MIGRATION-ROADMAP.md

### Blockers/Concerns

- `Engine::step()` in speculative mode hangs — pre-existing bug (not introduced by v18.0). 2 Phase 19 integration tests are `#[ignore]`d awaiting the fix. **Audit does not require code changes**, so this does not block v19.0.
- Real-model benchmark missing — current bench uses stub backends. Deferred until a small real checkpoint is available. **Out of scope for v19.0.**

---

## Deferred Items

| Category    | Item                                              | Status            | Deferred At |
| ----------- | ------------------------------------------------- | ----------------- | ----------- |
| Multi-Model | External draft model support (MULTI-01, MULTI-02) | Shipped in v18.0  | 2026-05-13   |
| Multi-Model | Draft hot-swap mid-request (MULTI-04)             | Deferred to v19+  | 2026-06-27   |
| Multi-Model | Draft fine-tuning hooks (MULTI-05)                | Out of scope      | 2026-05-13   |
| Multi-Model | Cross-GPU draft placement (MULTI-06)              | Deferred to v19+  | 2026-06-27   |
| Engine      | Wire DraftResolver into step_speculative loop     | **Closed by Phase 19** | 2026-06-27 |
| Engine      | HTTP exporter integration of v18.0 metrics        | **Closed by Phase 19** | 2026-06-27 |
| Engine      | Real-model benchmark (vs stub backend)            | Deferred to v19+  | 2026-06-27   |
| Engine      | Fix Engine::step() speculative-mode hang         | Deferred to v19+  | 2026-06-27   |
| Refactor    | File renaming (`17_*.rs` stage-info cleanup)      | Deferred to v20.0+ (after v19.0 audit) | 2026-06-27 |
| Refactor    | Module boundary tightening                        | Deferred to v20.0+ (after v19.0 audit) | 2026-06-27 |
| Refactor    | Doc-comment coverage catch-up to ≥80%              | Deferred to v20.0+ (after v19.0 audit) | 2026-06-27 |
| Refactor    | Error type unification across crates              | Deferred to v20.0+ (after v19.0 audit) | 2026-06-27 |

## Session Continuity

Last session: 2026-06-27 v18.0 + Phase 19 gap closure complete; v19.0 milestone started
Stopped at: v19.0 roadmap defined (5 phases, 23/23 requirements mapped)
Resume file: None

---

*State updated: 2026-06-27 — v19.0 roadmap created (5 audit phases, analysis-only); v18.0 milestone complete (5/5 phases, audit passed, 287+ tests passing)*
