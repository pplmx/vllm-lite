---
gsd_state_version: 1.0
milestone: v18.0
milestone_name: Multi-Model Speculative Decoding
status: planning
last_updated: "2026-06-27T02:00:00.000Z"
last_activity: 2026-06-27
progress:
  total_phases: 4
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
  percent: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-26)

**Core value:** Fast, memory-efficient LLM inference with continuous batching, paged KV cache, and tensor parallelism — deployed with production-grade ops tooling and security.
**Current focus:** v18.0 Multi-Model Speculative Decoding — roadmap created (4 phases, 14/14 requirements mapped)

---

## Current Position

Phase: Not started (roadmap defined)
Plan: —
Status: Roadmap complete; awaiting /gsd-plan-phase 18.1
Last activity: 2026-06-27 — v18.0 roadmap created (Phases 18.1-18.4)

## Performance Metrics

**Velocity:**

- Total plans completed: 0
- Average duration: —
- Total execution time: —

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
| ----- | ----- | ----- | -------- |
| —     | —     | —     | —        |

**Recent Trend:**

- Last 5 plans: —
- Trend: —

---

## Accumulated Context

### Decisions

- v17.0 uses 4 phases (17.1-17.4), continuing numbering from v16.0 which ended at phase 16.4
- Multi-model speculation (MULTI-01, MULTI-02) deferred to v18.0 per research recommendation
- Phase order dictated by dependency chain: engine integration → self-spec → adaptive + bench → warmup + metrics
- v18.0 uses 4 phases (18.1-18.4), continuing numbering from v17.0 which ended at phase 17.4
- v18.0 phase ordering: Draft Registry + External Loading → Lifecycle + Memory Budget → Request Routing + Fallback → Integration Tests + Benchmarks
- Phase 18.1 collocates MMLT-01..03 + LIFE-01 (registry foundation before lifecycle/routing depend on it)
- Phase 18.4 is a validation phase with 0 direct requirements — verifies Phase 18.1-18.3 work end-to-end

### Pending Todos

None yet.

### Blockers/Concerns

- Phase 18.1 needs careful design for DraftModelRegistry integration with existing ArchitectureRegistry — potential overlap with `crates/model/src/arch/` patterns
- Phase 18.2's VRAM budget enforcement requires runtime KV cache growth tracking — needs to coordinate with existing MemoryManager APIs from v17
- Phase 18.3's mixed-routing batch composition adds scheduler complexity beyond current batch_composer logic

---

## Deferred Items

| Category    | Item                                              | Status            | Deferred At |
| ----------- | ------------------------------------------------- | ----------------- | ----------- |
| Multi-Model | External draft model support (MULTI-01, MULTI-02) | Active in v18.0   | 2026-05-13   |

## Session Continuity

Last session: —
Stopped at: Milestone v18.0 roadmap created
Resume file: None

---

*State updated: 2026-06-27 — v18.0 roadmap created (4 phases, 14/14 reqs mapped)*
*Prior state: 2026-05-13 — Milestone v17.0 roadmap created*
