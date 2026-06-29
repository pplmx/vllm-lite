# scheduler Module: The v24.0 Split (1 file → 8)

**Date:** 2026-06-29
**Status:** Historical retrospective
**Context version:** v24.0 (Phase D)

## Context

Prior to v24.0, the scheduler module was a single ~3000 LOC monolith
(`crates/core/src/scheduler.rs`). It contained:

- `SchedulerEngine` (main entry, ~600 LOC)
- Batch composition logic (~800 LOC)
- Memory/block management (~900 LOC)
- Scheduling policies (FCFS, priority, SJF; ~300 LOC)
- Radix cache for prefix matching (~400 LOC)
- CUDA Graph integration (~250 LOC)
- Packing logic (~250 LOC)

By v23.0 close, this file had become:

- **Hard to navigate**: 3000 LOC in one file meant scrolling/searching
  to find anything. New contributors gave up.
- **Long build times**: changing ANY scheduler component forced full
  recompilation of the whole file, even if your edit was orthogonal.
- **Merge conflicts**: 5+ concurrent PRs touching different parts of
  scheduler constantly conflicted at the file level.
- **Unclear ownership**: each sub-domain (batch, memory, policy) had
  its own invariants but they were interleaved with general scheduler
  code, making invariants hard to state and test.

## v24.0 Refactor (Phase D-1 to D-3c)

v24.0 split `scheduler.rs` into a directory of focused modules:

```text
crates/core/src/scheduler/
├── batch.rs                    (99 LOC)  — Batch type + helpers
├── batch_composer/             (1020 LOC) — prefill + decode composition
│   ├── compose.rs
│   ├── mod.rs
│   └── validate.rs
├── cuda_graph.rs               (163 LOC) — CUDA Graph integration
├── engine/                     (910 LOC)  — SchedulerEngine split
│   ├── graph.rs
│   ├── memory.rs
│   ├── mod.rs
│   ├── state.rs
│   └── update.rs
├── memory/                     (934 LOC) — block allocation/eviction
│   ├── allocator.rs
│   ├── eviction.rs
│   └── mod.rs
├── mod.rs                      (129 LOC) — module root
├── observer.rs                 (189 LOC) — metrics emission
├── packing.rs                  (131 LOC) — sequence packing
├── packing/                    — (separate directory, split in D-3)
├── phase_scheduler.rs          (216 LOC) — prefill/decode transitions
├── policy/                     (199 LOC) — scheduling policies
│   ├── fcfs.rs
│   ├── mod.rs
│   ├── priority.rs
│   ├── sjf.rs
│   └── trait_def.rs
├── preemption.rs               (215 LOC) — preemption logic
├── radix_cache/                (262 LOC) — prefix caching
│   ├── mod.rs
│   ├── node.rs
│   └── tree.rs
├── request_queue.rs            (438 LOC) — wait queue
└── stats.rs                    (63 LOC)  — metrics aggregation
```

**Total**: 1 file (3000 LOC) → 19 files (avg ~270 LOC each, largest 858).

## Sub-phase Breakdown

The split was done in 5 sub-phases to minimize risk:

- **D-1 (engine split)**: `SchedulerEngine` extracted into `engine/`
  subdirectory. Public API unchanged.
- **D-2 (scheduler sub-files)**: top-level scheduler functions moved
  into focused files (`request_queue.rs`, `phase_scheduler.rs`, etc.)
- **D-3a (types + ssm splits)**: large type definitions broken into
  dedicated files.
- **D-3b (soft-target splits)**: file size targets (≤500 LOC) applied
  to remaining large files via soft splits (no logic changes).
- **D-3c (visibility tightening)**: `pub(crate)` applied to items
  that shouldn't be visible outside the scheduler module.

Each sub-phase was independently reviewable and mergeable.

## Test Coverage During Refactor

To ensure no regression, each sub-phase was accompanied by:

- All existing tests (unit + integration) passing
- Mutation score on changed modules (cargo-mutants, after v30.0 K)
- Property-based test properties on extracted components (v28.0)

The refactor produced **zero behavioral changes** — it was purely
structural.

## Lessons

1. **Sub-phase decomposition beats big-bang refactors**. 5 small PRs
   are far less risky than 1 large PR, especially when each is
   independently testable.
2. **Module size limits guide decomposition**. Targeting ≤500 LOC per
   file (v23.0's soft limit) gives a tangible criterion; without it,
   "split when it's too big" is subjective and gets deferred.
3. **Visibility tightening (`pub(crate)`) follows naturally**. Once
   items are in focused sub-modules, the boundary between
   "module-internal" and "module-public" becomes obvious — you can see
   which items are only used within the sub-module.
4. **Property-based tests make refactors safer**. v28.0's proptest
   properties on invariants (e.g., `batch.seq_ids.len() <=
   max_batch_size`) catch accidental invariant violations during
   extraction.

## See also

- v24.0 plan: `docs/superpowers/plans/2026-06-28-v24-phase-d*.md`
- v24.0 ROADMAP: `.planning/milestones/v24.0-ROADMAP.md`
- v23.0 audit that motivated the refactor
- v30.0 Phase K mutation scan of all scheduler sub-modules (100% score)
