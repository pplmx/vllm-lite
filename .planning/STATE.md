---
gsd_state_version: 1.0
milestone: v21.0
milestone_name: P2/P3 Backlog Cleanup
status: planning
last_updated: "2026-06-27T12:00:00.000Z"
last_activity: 2026-06-27
progress:
  total_phases: 5
  completed_phases: 0
  total_plans: 30
  completed_plans: 0
  percent: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-27)

**Core value:** Fast, memory-efficient LLM inference with continuous batching, paged KV cache, and tensor parallelism — deployed with production-grade ops tooling and security.
**Current focus:** v21.0 P2/P3 Backlog Cleanup — ROADMAP CREATED (5 phases scoped: 31-35; 42 requirements mapped 100%; awaiting `/gsd-discuss-phase 31`)

---

## Current Position

Phase: Phase 31 (next, not started)
Plan: —
Status: Roadmap defined; pending phase discussion / planning
Last activity: 2026-06-27 — v21.0 roadmap created (ROADMAP.md updated with Phase 31-35; REQUIREMENTS.md traceability complete; STATE.md updated)

## Performance Metrics

**Velocity (v20.0 actual — MILESTONE COMPLETE):**

- Total test count post-v20.0: **1144 tests** (+857 vs v19.0's 287; final count with --all-features)
- Test count change: 287 (v19.0) → 1139 (default) → 1144 (--all-features)
- Doc coverage: 19.5% → **97.8%** (workspace pub items); 99.6% module docs (Phase 28 outcome)
- Clippy: clean (0 warnings, 0 errors, including 3 pre-existing kv_cache_fp8 errors fixed in Phase 30)
- cargo fmt: clean (133 files auto-fixed for Phase 28 doc-backfill indent issue)
- ADRs created: **12** (Phase 29)
- Phases shipped: 6/6 (Phase 25/26/27/28/29/30)

**By Phase (actual v20.0):**

| Phase | Plans | Requirements | Theme | Status |
| ----- | ----- | ------------:| ----- | :----: |
| 25    | 5     | 5            | P0 critical fixes + object-safety + vllm-dist feature-gate | ✓ |
| 26    | 4     | 7            | Module tree restoration + vllm-dist feature-gate | ✓ |
| 27    | 7     | 7            | Error handling standardization | ✓ |
| 28    | 10    | 7            | Doc coverage 7.6% → 97.8% | ✓ |
| 29    | 4     | 12           | External docs reconciliation + 12 ADRs | ✓ |
| 30    | 9     | 10           | Naming + deprecation hygiene + final verification | ✓ |

**v21.0 Planned (scope defined 2026-06-27):**

| Phase | Plans | Requirements | Theme | Status |
| ----- | ----- | ------------:| ----- | :----: |
| 31    | 6     | 9            | Module Layout Reorganization (ML-01..09) | Pending |
| 32    | 8     | 11           | API Consistency (API-01..11) | Pending |
| 33    | 4     | 8            | Naming Consistency (NAM-01..08) | Pending |
| 34    | 4     | 4            | External Doc Fixes (DOC-01..04) | Pending |
| 35    | 8     | 10           | P3 Actionable + FINAL gates (P3-01..06 + FINAL-01..04) | Pending |
| **Total** | **30** | **42** |  |  |

**Recent Trend:**

- Last 5 commits (v20.0): refactor(model) Phase 30 NAM/DEP/CMT, chore(fmt) Phase 28 indent fix, docs(adr) Phase 29, docs(core) Phase 28 DOC-02/03 backfill, docs(AGENTS) Phase 28 DOC-07
- v20.0 shipped cleanly: all 6 phases delivered, FINAL-01..04 gates green, milestone archived
- v21.0 roadmap created: 5 phases (31-35) covering 44 P2 + 13 P3 backlog items, ~71h estimated

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
- **v21.0 roadmap decision**: 5 phases (31-35), one per theme (Module Layout / API / Naming / External Docs / P3+FINAL); v20.0 ended at Phase 30, so v21.0 starts at Phase 31
- **v21.0 phase dependency graph**: 31 → 32 → 33 → 34 → 35 (linear chain); 31 unblocks 32 (engine splits relocate `FallbackStrategy`); 32 unblocks 33 (API surface stable before doc updates); 33 unblocks 34 (AGENTS.md must exist before PROJECT.md cross-links); 35 depends on all (FINAL gates run last)
- **v21.0 scope decision**: All 44 P2 + selected actionable P3 (vs only high-impact subset); 2 of 44 P2 are zero-effort "no action" findings (NAME-F-13, NAME-F-20) per BACKLOG.md, so effective P2 work = 42 items across 5 phases
- **v21.0 risk profile**: Phase 31 highest-risk (draft_registry split touches Phase 18 code; engine splits affect speculative step loop); Phase 32 medium-risk (FallbackStrategy split touches trait ecosystem); Phases 33-35 low-risk (documentation + small fixes)
- **v21.0 constraint preservation**: All 1144+ existing tests must remain green (FINAL-01 invariant); public API removals require `#[deprecated]` markers (DEP-01/02 from v20.6 still applies); `vllm-dist` remains feature-gated (not removed); doc coverage ≥60% baseline must not regress (achieved 97.8%)
- **v21.0 sub-phase mapping**: 31=v21.1, 32=v21.2, 33=v21.3, 34=v21.4, 35=v21.5 (mirrors v20.0's 25=v20.1, ..., 30=v20.6 convention)
- **v21.0 P3 selection**: 6 of 13 P3 items actionable (ARCH-F-15, API-F-25, API-F-27, API-F-31, API-F-32, API-F-33); 7 remaining are informational negative findings (NAME-F-12..F-26) deferred per REQUIREMENTS.md RFU-04
- **v21.0 carry-over from v20.0**: Engine::step() speculative-mode hang + 5 cargo doc warnings + 1 gguf TODO — explicitly out of scope (OPS-02/03 deferred to bug-fix milestone)

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
11. **AGENTS.md as living style guide (v20.0 + v21.0 pattern)** — Naming conventions (verb policy, suffix conventions, tensor-math exemption, test-file location) are documented in AGENTS.md rather than enforced mechanically. v21.0 Phases 33-34 extend this with remaining ambiguities (`*Manager` suffix, `create_*` vs `build_*`, async/sync split rationale).

### Pending Todos

- [ ] **Phase 31**: Module Layout Reorganization (ML-01..09) — `/gsd-discuss-phase 31` then `/gsd-plan-phase 31`
- [ ] **Phase 32**: API Consistency (API-01..11)
- [ ] **Phase 33**: Naming Consistency (NAM-01..08)
- [ ] **Phase 34**: External Doc Fixes (DOC-01..04)
- [ ] **Phase 35**: P3 Actionable + FINAL gates (P3-01..06 + FINAL-01..04)

### Blockers/Concerns

- `Engine::step()` in speculative mode hangs — pre-existing bug (not introduced by v18.0 or v19.0). 2 Phase 19 integration tests are `#[ignore]`d awaiting the fix. NOT blocking v21.0; explicitly out of scope per PROJECT.md Out of Scope (v21.0).
- Real-model benchmark missing — current bench uses stub backends. Out of scope for v21.0 (purely backlog closure); deferred indefinitely per OPS-01.
- 5 pre-existing cargo doc broken-link warnings — deferred per OPS-03; out of scope for v21.0.
- vllm-testing lemon pair split (ML-05) may be infeasible if cross-crate dependency graph becomes too tangled — fallback is to document why kept (per success criterion).

---

## Deferred Items

| Category    | Item                                              | Status            | Deferred At |
| ----------- | ------------------------------------------------- | ----------------- | ----------- |
| Multi-Model | External draft model support (MULTI-01, MULTI-02) | Shipped in v18.0  | 2026-05-13   |
| Multi-Model | Draft hot-swap mid-request (MULTI-04)             | Out of scope (v21.0) | 2026-06-27 |
| Multi-Model | Draft fine-tuning hooks (MULTI-05)                | Out of scope      | 2026-05-13   |
| Multi-Model | Cross-GPU draft placement (MULTI-06)              | Out of scope (v21.0) | 2026-06-27 |
| Engine      | Real-model benchmark (vs stub backend)            | Deferred (OPS-01) | 2026-06-27 |
| Engine      | Fix Engine::step() speculative-mode hang         | Out of scope (v21.0) | 2026-06-27 |
| Refactor    | **5 P0 architecture/API violations**              | **Shipped in Phase 25 (v20.1)** | 2026-06-27 |
| Refactor    | **Module tree restoration** (orphan files + test migration) | **Shipped in Phase 26 (v20.2)** | 2026-06-27 |
| Refactor    | **Error handling standardization** (13 enums + 25+ mutex + 4 variants) | **Shipped in Phase 27 (v20.3)** | 2026-06-27 |
| Refactor    | **Doc coverage push 7.6%→97.8%**                  | **Shipped in Phase 28 (v20.4)** | 2026-06-27 |
| Refactor    | **External docs + 12 ADRs**                       | **Shipped in Phase 29 (v20.5)** | 2026-06-27 |
| Refactor    | **Naming + deprecation hygiene + final verify**   | **Shipped in Phase 30 (v20.6)** | 2026-06-27 |
| Refactor    | **Module layout reorganization** (draft_registry split, engine/speculative subtree, qwen3_config, attention/util) | **Phase 31 (v21.1)** | 2026-06-27 |
| Refactor    | **API consistency** (builders, error sources, FallbackStrategy split, error context) | **Phase 32 (v21.2)** | 2026-06-27 |
| Refactor    | **Naming consistency** (flash_v3 rename, AGENTS.md expansions, non-tensor single-letter rename) | **Phase 33 (v21.3)** | 2026-06-27 |
| Refactor    | **External doc fixes** (DeepSeek, vllm-dist ADR, Phase 5 ref, PROJECT.md cross-links) | **Phase 34 (v21.4)** | 2026-06-27 |
| Refactor    | **P3 actionable + FINAL gates** (gemma4 unwrap, MIGRATING.md, CircuitBreakerError variant, CudaGraphError Clone) | **Phase 35 (v21.5)** | 2026-06-27 |
| Architecture| `vllm-dist` resurrection (multi-node work)        | Deferred to v21.x+ (feature-gated in v20.0; ML-06 in v21.1 relocates the error type but doesn't resurrect multi-node) | 2026-06-27 |
| Documentation | RFU-03 Migrate to parking_lot::Mutex             | Partially addressed in v21.2 (API-04 covers predictive_batching.rs); broader migration deferred | 2026-06-27 |
| Architecture| ARF-02 Replace `kv_cache_fp8` with unified FP8    | Deferred to v21.x+ (depends on v21.1 integration revealing design issues) | 2026-06-27 |
| Architecture| ARF-03 Split `engine.rs` God module (1,038 LOC)   | Partially addressed in v21.1 (ML-02 collapses engine.rs + engine/speculative.rs into sub-tree); deeper split deferred | 2026-06-27 |
| Refactor    | 7 P3 negative findings (NAME-F-12..F-26)          | Informational only (RFU-04); no action in v21.0 | 2026-06-27 |

---

## Session Continuity

Last session: 2026-06-27 v21.0 roadmap created
Stopped at: v21.0 roadmap complete (5 phases 31-35, 42/42 requirements mapped, ROADMAP.md updated, REQUIREMENTS.md traceability complete, STATE.md refreshed); ready for `/gsd-discuss-phase 31`
Resume file: None
Next command: `/gsd-discuss-phase 31` (gather adaptive context before planning Module Layout Reorganization)

---

*State updated: 2026-06-27 — v21.0 roadmap created (5 phases: 31-35; 42 requirements mapped 100%; linear dependency chain 31→32→33→34→35; ~71h estimated; preserves v20.0 invariants: 1144 tests pass, clippy/fmt clean, doc coverage 97.8%, #[deprecated] for public API changes, vllm-dist feature-gated)*
