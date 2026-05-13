---
gsd_state_version: 1.0
milestone: v17.0
milestone_name: Production Speculative Decoding
status: planning
last_updated: "2026-05-13T02:56:30.671Z"
last_activity: 2026-05-13
progress:
  total_phases: 4
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
  percent: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-13)

**Core value:** Fast, memory-efficient LLM inference with continuous batching, paged KV cache, and tensor parallelism — deployed with production-grade ops tooling and security.
**Current focus:** Milestone v17.0 Production Speculative Decoding — Phase 17.1 Engine Integration

---

## Current Position

Phase: 1 of 4 (Phase 17.1: Engine Integration)
Plan: — (not yet planned)
Status: Roadmap created, ready to plan
Last activity: 2026-05-13 — Milestone v17.0 roadmap created

Progress: [░░░░░░░░░░] 0%

## Performance Metrics

**Velocity:**
- Total plans completed: 0
- Average duration: —
- Total execution time: —

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| — | — | — | — |

**Recent Trend:**
- Last 5 plans: —
- Trend: —

---

## Accumulated Context

### Decisions

- v17.0 uses 4 phases (17.1-17.4), continuing numbering from v16.0 which ended at phase 16.4
- Multi-model speculation (MULTI-01, MULTI-02) deferred to v18.0 per research recommendation
- Phase order dictated by dependency chain: engine integration → self-spec → adaptive + bench → warmup + metrics

### Pending Todos

None yet.

### Blockers/Concerns

- Phase 17.2 (Self-Spec Forward Pass) needs `/gsd-research-phase` during planning — the layer-truncated forward pass across the architecture registry boundary needs careful design
- Phase 17.1 research flags this as well-documented (no deeper research needed), but `ModelBackend` may need `forward_with_logits()` extension

---

## Deferred Items

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| Multi-Model | External draft model support (MULTI-01, MULTI-02) | Deferred to v18.0 | 2026-05-13 |

## Session Continuity

Last session: —
Stopped at: Milestone v17.0 roadmap created
Resume file: None

---

*State updated: 2026-05-13 — Milestone v17.0 roadmap created*
