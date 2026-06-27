# Phase 41: Stale Documentation (v23.2) - Context

**Gathered:** 2026-06-28
**Status:** Ready for planning
**Mode:** Smart discuss auto-accepted (infrastructure phase)

<domain>

## Phase Boundary

Bring all user-facing documentation in sync with v23.0 reality. The v22.0 post-ship
audit surfaced 8 specific stale-documentation findings (DOC-02..09). Each is
discrete and bounded:

1. **`CLAUDE.md` rewrite (DOC-02)** — Currently describes "4 crates", references
   `qwen3/attention.rs` (renamed to `components/attention/gqa.rs`), Rust 1.75
   (actual: 1.85), and a generic `Engine<M: ModelBackend>` (actual: non-generic
   `Box<dyn ModelBackend>`).
2. **`README.md` Scheduling policy example (DOC-03)** — Lines 459-473 reference
   non-existent `vllm_core::scheduler::FcfsPolicy`; actual paths are
   `vllm_core::scheduler::policy::{FcfsPolicy, SjfPolicy, PriorityPolicy}`.
3. **`CHANGELOG.md` backfill (DOC-04)** — Missing v19.0, v20.0, v21.0, v22.0
   entries. Synthesize from archived milestone files.
4. **`MIGRATING.md` v22.0 entry (DOC-05)** — Missing v22.0 entry covering
   security middleware wiring, parking_lot migration, LazyLock upgrade.
5. **`docs/architecture.md` creation (DOC-06)** — File does not exist; create
   unified v23.0 architecture overview.
6. **README badge update (DOC-07)** — Test count badge shows `1100+`, actual is
   `1179`; add version pin note.
7. **`docs/optimization_guide.md:50` API example (DOC-08)** — `Engine::with_config`
   example uses old signature; current signature is `Option<M>`.
8. **`docs/optimization_guide.md` perf numbers date tag (DOC-09)** — Performance
   numbers need date tagging; reconcile with v22.0 bench results.

FINAL-01 invariant: 1179 tests remain green post-update (doc changes don't
affect logic).

</domain>

<decisions>

## Implementation Decisions

### the agent's Discretion

All implementation choices are at the agent's discretion — pure documentation
phase. Specific guidance per requirement:

- **DOC-02 CLAUDE.md:** Rewrite to current reality. Reference actual file paths
  (`crates/model/src/components/attention/gqa.rs` not `qwen3/attention.rs`).
  Crate count: 6 (traits, core, model, server, testing, dist). Rust 1.85. Engine
  signature: `Engine { /* Box<dyn ModelBackend> fields */ }`.
- **DOC-03 README imports:** Update import paths to actual scheduler policy
  module location. The example should be a complete compilable snippet.
- **DOC-04 CHANGELOG:** Add 4 entries (v19.0, v20.0, v21.0, v22.0) with:
  - Date
  - Phase count and names
  - Key accomplishments (3-5 bullets per milestone)
  - Test count at end
  - Tech debt roll-forward
  Synthesize from `.planning/milestones/v{19,20,21,22}.0-*.md` files.
- **DOC-05 MIGRATING:** Add v22.0 entry covering the 3 changes (security
  middleware, parking_lot, LazyLock) with example diffs.
- **DOC-06 architecture.md:** Create new file. Cover:
  - Engine orchestration (split into engine/, engine/spec_dispatch/)
  - Scheduler split (queue, preemption, eviction, batch)
  - Paged tensor split (logical KV cache in core/, physical in model/paged_tensor/)
  - Architecture registry pattern
  - Multi-model spec flow
  - Cross-reference relevant ADRs in `docs/adr/`
- **DOC-07 README badge:** Update from `1100+` to `1179`; add version pin note
  (e.g., "as of v23.0").
- **DOC-08 optimization_guide.md:** Fix `Engine::with_config` example at line 50
  to match current `Option<M>` signature. If `Engine::with_config` no longer
  exists, replace with the actual current API surface.
- **DOC-09 optimization_guide.md perf numbers:** Add `(date: YYYY-MM-DD)` tag to
  each perf number; reconcile with v22.0 bench results in
  `docs/benchmark-results/`.

</decisions>

