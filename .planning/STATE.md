---
gsd_state_version: 1.0
milestone: v18.0
milestone_name: Multi-Model Speculative Decoding
status: complete
last_updated: "2026-06-27T03:00:00.000Z"
last_activity: 2026-06-27
progress:
  total_phases: 4
  completed_phases: 4
  total_plans: 4
  completed_plans: 4
  percent: 100
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-26)

**Core value:** Fast, memory-efficient LLM inference with continuous batching, paged KV cache, and tensor parallelism — deployed with production-grade ops tooling and security.
**Current focus:** v18.0 Multi-Model Speculative Decoding — SHIPPED 2026-06-27 (4 phases, 14/14 requirements)

---

## Current Position

Phase: All v18.0 phases complete (18.1, 18.2, 18.3, 18.4)
Plan: All 4 plans shipped
Status: Milestone complete — ready for audit
Last activity: 2026-06-27 — v18.0 phase 18.4 integration tests + benchmarks shipped

## Performance Metrics

**Velocity:**

- Total plans completed: 4 (v18.0)
- Total commits across v18.0: 4 (one per phase)
- Total test count change: 209 → 277 (+68 tests)
- Total lib test count change: 209 → 263 (+54 tests)
- Total integration test count: 0 → 14 (new file)
- Total bench count: 3 → 4 (+1 bench)

**By Phase:**

| Phase | Plans | Tests Added | Cumulative Lib Total |
| ----- | ----- | ----------- | -------------------- |
| 18.1  | 1     | 15          | 224                  |
| 18.2  | 1     | 30          | 254                  |
| 18.3  | 1     | 9           | 263                  |
| 18.4  | 1     | 14 (integration) | 277 (263 + 14)   |

**Recent Trend:**

- Last 4 commits: feat(18.1) → feat(18.2) → feat(18.3) → test+bench(18.4)
- Trend: All phases shipped cleanly, no rollbacks, all clippy/fmt checks passed

---

## Accumulated Context

### Decisions

- v18.0 uses 4 phases (18.1-18.4), continuing numbering from v17.0 which ended at phase 17.4
- v18.0 phase ordering: Draft Registry + External Loading → Lifecycle + Memory Budget → Request Routing + Fallback → Integration Tests + Benchmarks
- Phase 18.1 collocates MMLT-01..03 + LIFE-01 (registry foundation before lifecycle/routing depend on it)
- Phase 18.4 is a validation phase with 0 direct requirements — verifies Phase 18.1-18.3 work end-to-end

### Architecture Patterns Established

1. **Registry is loader-agnostic** — `DraftModelRegistry` does not depend on `vllm-model`. The actual `ModelLoader` invocation lives at the caller (server or test harness), which then hands a `Box<dyn ModelBackend>` to `attach_loaded`. This keeps the registry usable from contexts where `vllm-model` is unavailable.

2. **Lazy loading via state machine** — `DraftState::{Unloaded, Loaded}` makes lazy semantics explicit. `register` is metadata-only; `attach_loaded` is the only path to `Loaded`; `unload` returns to `Unloaded`.

3. **MemoryBudget is shared via Arc** — Engine and registry both hold `Arc<MemoryBudget>` so they share a single source of truth. Reservations are atomic via `try_reserve_draft`.

4. **Per-request resolution, not per-batch** — `DraftResolver::resolve` is called per-request. Mixed-draft routing (RTE-03) emerges naturally from per-request resolution.

5. **FALL-01 is silent, FALL-02 is sticky** — Load failures log + metric + silently fall back to SelfSpec (user unaware). Runtime errors set `Sequence.degraded_draft = true` (sticky for lifetime of sequence).

### Pending Todos

None — milestone complete.

### Blockers/Concerns

None.

---

## Deferred Items

| Category    | Item                                              | Status            | Deferred At |
| ----------- | ------------------------------------------------- | ----------------- | ----------- |
| Multi-Model | External draft model support (MULTI-01, MULTI-02) | Shipped in v18.0  | 2026-05-13   |
| Multi-Model | Draft hot-swap mid-request (MULTI-04)             | Deferred to v19+  | 2026-06-27   |
| Multi-Model | Draft fine-tuning hooks (MULTI-05)                | Out of scope      | 2026-05-13   |
| Multi-Model | Cross-GPU draft placement (MULTI-06)              | Deferred to v19+  | 2026-05-13   |
| Engine      | Wire DraftResolver into step_speculative loop     | Deferred to v19+  | 2026-06-27   |
| Engine      | HTTP exporter integration of v18.0 metrics        | Deferred to v19+  | 2026-06-27   |
| Engine      | Real-model benchmark (vs stub backend)            | Deferred to v19+  | 2026-06-27   |

## Session Continuity

Last session: 2026-06-27 v18.0 complete
Stopped at: All v18.0 phases shipped, awaiting milestone audit
Resume file: None

---

*State updated: 2026-06-27 — v18.0 milestone complete (4/4 phases, 277 tests passing)*
