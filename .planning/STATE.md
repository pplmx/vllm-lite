---
gsd_state_version: 1.0
milestone: v20.0
milestone_name: Codebase Remediation (BACKLOG.md-driven)
status: in_progress
last_updated: "2026-06-27T14:05:00.000Z"
last_activity: 2026-06-27
progress:
  total_phases: 6
  completed_phases: 3
  total_plans: 39
  completed_plans: 17
  percent: 50
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-27)

**Core value:** Fast, memory-efficient LLM inference with continuous batching, paged KV cache, and tensor parallelism — deployed with production-grade ops tooling and security.
**Current focus:** v20.0 Codebase Remediation — Phases 25/26/27 shipped (commit 2ffd4af, f940639, 6fdf94e); executing Phase 28 (Documentation Coverage Push)

---

## Current Position

Phase: 28 of 30 (Phase 28 — Documentation Coverage Push)
Plan: —
Status: Phases 25/26/27 complete in git history; autonomous workflow resumed for 28/29/30
Last activity: 2026-06-27 — detected Phases 25-27 already shipped via git log; ROADMAP.md heading format repaired; STATE.md updated

Progress: [█████░░░░░] 50%

---

## Performance Metrics

**Velocity (v20.0 baseline — from v18.0/v19.0 historical):**

- Total test count at v19.0 end: 287+ tests (lib + integration; +78 vs v17.0)
- Test count target post-v20.0: 297+ tests (Phase 25: +8 dyn_safety, Phase 27: +15 error variant/context)
- Audit artifacts: 11 markdown files / 3,578 lines in `.planning/audit/`
- Findings to remediate: 100 (5 P0 + 38 P1 + 44 P2 + 13 P3)
- P0 items being addressed in Phase 25: 5 (ARCH-F-11, ARCH-F-12, API-F-01, API-F-02, API-F-03)
- Estimated effort: ~190h total (per MIGRATION-ROADMAP.md)

**Velocity (v19.0 actual — for context):**

- Total plans completed: 5 (v19.0)
- Audit-only milestone (0 source code modified; analysis-only constraint honored)
- Findings consolidated: 100 across 4 audit dimensions

**Velocity (v18.0 historical):**

- Total plans completed: 5 (v18.0) + 1 (Phase 19 gap closure)
- Total test count change: 209 → 287+ tests
- Total bench count: 3 → 4

**By Phase (planned v20.0):**

| Phase | Plans | Requirements | Theme |
| ----- | ----- | ------------:| ----- |
| 25    | 5     | 5            | P0 critical fixes + object-safety + vllm-dist feature-gate |
| 26    | 4     | 7            | Module tree restoration + vllm-dist feature-gate |
| 27    | 7     | 7            | Error handling standardization |
| 28    | 10    | 7            | Doc coverage 7.6% → ≥60% |
| 29    | 4     | 12           | External docs reconciliation + ADRs |
| 30    | 9     | 10           | Naming + deprecation hygiene + final verification |

**Recent Trend:**

- Last 5 commits: docs(audit) phase 20 → 21 → 22 → 23 → 24, then archive v19.0, then v20.0 milestone + requirements definition, then v20.0 roadmap
- v19.0 shipped cleanly: all 5 audit phases delivered, 100 findings consolidated, archived
- v20.0 starting position: 6-phase roadmap defined; 48/48 v1 requirements mapped; BACKLOG.md-driven execution

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
- **v20.0 roadmap decision**: 6 phases (25-30) mapped from REQUIREMENTS.md traceability; phase numbers 25-30 continue from v19.0's Phase 24 endpoint. Each phase corresponds to a v20.X sub-phase from MIGRATION-ROADMAP.md (25=v20.1, 26=v20.2, 27=v20.3, 28=v20.4, 29=v20.5, 30=v20.6)
- **v20.0 phase dependency graph**: 25 → (26 ‖ 28) → (27 + 29) → 30; 27 depends on 25 (ModelError must be enum first); 29 depends on 28 (/// docs must exist before ADR references); 30 runs last by convention (independent)
- **v20.0 risk profile**: Phase 25 high-risk architectural changes (layering, ModelError enum, object-safety) — requires explicit rollback criteria and pre-Phase-25 baseline capture
- **v20.0 constraint preservation**: All 287+ existing tests must remain green throughout (FINAL-01 invariant); public API removals require `#[deprecated]` markers (DEP-01/02); `vllm-dist` is feature-gated (not removed) — MT-07 + EXT-07 outcome
- **v20.0 counts note**: REQUIREMENTS.md traceability maps 48 requirements (5 P0 + 7 MT + 7 ERR + 7 DOC + 12 EXT + 10 NAM/DEP/CMT/FINAL), though REQUIREMENTS.md preamble says "46 total" — minor source inconsistency (count derived from traceability table is 48, which is used in ROADMAP.md)

### Architecture Patterns Established (v18.0 — preserved in v20.0)

1. **Registry is loader-agnostic** — `DraftModelRegistry` does not depend on `vllm-model`. The actual `ModelLoader` invocation lives at the caller (server or test harness), which then hands a `Box<dyn ModelBackend>` to `attach_loaded`.
2. **Lazy loading via state machine** — `DraftState::{Unloaded, Loaded}` makes lazy semantics explicit. `register` is metadata-only; `attach_loaded` is the only path to `Loaded`; `unload` returns to `Unloaded`.
3. **MemoryBudget is shared via Arc** — Engine and registry both hold `Arc<MemoryBudget>` so they share a single source of truth. Reservations are atomic via `try_reserve_draft`.
4. **Per-request resolution, not per-batch** — `DraftResolver::resolve` is called per-request. Mixed-draft routing (RTE-03) emerges naturally from per-request resolution.
5. **FALL-01 is silent, FALL-02 is sticky** — Load failures log + metric + silently fall back to SelfSpec. Runtime errors set `Sequence.degraded_draft = true` (sticky for lifetime of sequence).
6. **Resolver is wired in Engine** — `Engine.draft_resolver: Option<Arc<DraftResolver>>`. When `Some`, `step_speculative_inner` dispatches via `generate_per_seq_drafts`. When `None`, falls back to legacy single-draft path. Backward-compatible with `new_boxed`.
7. **Forward error capture via catch_unwind** — `generate_per_seq_drafts` wraps each per-seq forward in `catch_unwind(AssertUnwindSafe(...))` to handle both `Result::Err` and panic.
8. **Audit-before-refactor** — Non-functional refactoring work (file renaming, module boundary tightening, error-type unification, doc coverage catch-up) is gated on a structured audit. Audit findings become input to v20.0+ remediation phases, each with measurable acceptance criteria.
9. **vllm-dist feature-gate (v20.0 decision)** — Keep code, exclude from default build; enable for multi-node work via `--features multi-node`. `TensorParallelConfig` remains public in `vllm-dist`.
10. **Object-safety co-fix (v20.0 decision)** — Fix `ModelError` + non-object-safe traits together in v20.1 (both P0). Combined into single Phase 25 to avoid regression risk from sequential changes.

### Pending Todos

None — v20.0 roadmap defined; awaiting first phase plan (Phase 25).

### Blockers/Concerns

- `Engine::step()` in speculative mode hangs — pre-existing bug (not introduced by v18.0 or v19.0). 2 Phase 19 integration tests are `#[ignore]`d awaiting the fix. NOT blocking v20.0; will be addressed in v20.7+ or later remediation cycle.
- Real-model benchmark missing — current bench uses stub backends. Out of scope for v20.0 (purely remediation); deferred indefinitely.
- v20.0 Phase 25 risk — P0 architectural changes (layering, ModelError enum, object-safety) have non-trivial blast radius; rollback criteria defined in ROADMAP.md but real-world test impact unknown until execution begins.
- v20.0 doc coverage target — REQUIREMENTS.md DOC-01 target is ≥60% workspace (per-crate target ≥80%); MIGRATION-ROADMAP.md proposes 60% as achievable in ~46h, 80% in ~60h. Phase 28 plan accommodates both targets via per-crate thresholds.

---

## Deferred Items

| Category    | Item                                              | Status            | Deferred At |
| ----------- | ------------------------------------------------- | ----------------- | ----------- |
| Multi-Model | External draft model support (MULTI-01, MULTI-02) | Shipped in v18.0  | 2026-05-13   |
| Multi-Model | Draft hot-swap mid-request (MULTI-04)             | Deferred to v20.7+ | 2026-06-27 |
| Multi-Model | Draft fine-tuning hooks (MULTI-05)                | Out of scope      | 2026-05-13   |
| Multi-Model | Cross-GPU draft placement (MULTI-06)              | Deferred to v20.7+ | 2026-06-27 |
| Engine      | Real-model benchmark (vs stub backend)            | Deferred to v20.7+ | 2026-06-27 |
| Engine      | Fix Engine::step() speculative-mode hang         | Deferred to v20.7+ | 2026-06-27 |
| Refactor    | **5 P0 architecture/API violations**              | **Phase 25 (v20.1)** | 2026-06-27 |
| Refactor    | **Module tree restoration** (orphan files + test migration) | **Phase 26 (v20.2)** | 2026-06-27 |
| Refactor    | **Error handling standardization** (13 enums + 25+ mutex + 4 variants) | **Phase 27 (v20.3)** | 2026-06-27 |
| Refactor    | **Doc coverage push 7.6%→≥60%**                   | **Phase 28 (v20.4)** | 2026-06-27 |
| Refactor    | **External docs + 12 ADRs**                       | **Phase 29 (v20.5)** | 2026-06-27 |
| Refactor    | **Naming + deprecation hygiene + final verify**   | **Phase 30 (v20.6)** | 2026-06-27 |
| Refactor    | 44 P2 issues + 13 P3 informational                | Deferred to v20.7+ (not in v20.0 scope) | 2026-06-27 |
| Architecture| `vllm-dist` resurrection (multi-node work)        | Deferred to v20.7+ (feature-gated in v20.0) | 2026-06-27 |
| Documentation | RFU-01 doc coverage 60% → 80% (if 60% achieved) | Deferred to v20.7+ (depends on v20.4 outcome) | 2026-06-27 |
| Documentation | RFU-03 Migrate to parking_lot::Mutex             | Deferred to v20.7+ (optional follow-up) | 2026-06-27 |
| Architecture| ARF-02 Replace `kv_cache_fp8` with unified FP8    | Deferred to v20.7+ (depends on v20.2 outcome) | 2026-06-27 |
| Architecture| ARF-03 Split `engine.rs` God module (1,038 LOC)   | Deferred to v20.7+ (only if Phase 27 reveals deeper issues) | 2026-06-27 |

---

## Session Continuity

Last session: 2026-06-27 v20.0 roadmap defined
Stopped at: v20.0 roadmap written (6 phases, 48/48 requirements mapped from REQUIREMENTS.md traceability); STATE.md and ROADMAP.md committed; ready for `/gsd-plan-phase 25`
Resume file: None
Next command: `/gsd-plan-phase 25` (Phase 25 — P0 Critical Fixes; high-risk architectural changes with rollback criteria)

---

*State updated: 2026-06-27 — v20.0 roadmap defined (6 phases, 48/48 requirements mapped, phase dependency graph captured, Phase 25 includes rollback criteria, Phase 30 includes FINAL-01..04 verification gates)*
