# Phase 43: Architecture Cleanup + Final Verification (v23.4) - Context

**Gathered:** 2026-06-28
**Status:** Ready for planning
**Mode:** Smart discuss auto-accepted (infrastructure phase)

<domain>

## Phase Boundary

Remove ~2000 LOC of dead code, consolidate 4 stub architectures into one,
fix the `core → model` upward dependency, clean unused deps, and unify
duplicate implementations. Final verification gates close v23.0.

10 architecture requirements (ARCH-01..10) + 5 final gates (FINAL-01..05):

1. **ARCH-01**: Delete `scheduler/batch_planner.rs` (363 LOC; no prod callers)
2. **ARCH-02**: Delete `scheduler/predictive_batching.rs` (481 LOC; no prod callers)
3. **ARCH-03**: Delete `core/src/kv_cache/mod.rs` (7 LOC shim)
4. **ARCH-04**: Consolidate unused internal modules (~1057 LOC):
   - `core/src/sync.rs` (10 LOC) — used internally; scope to `pub(crate)`
   - `routing/HashRouter` (180 LOC) — re-exported only; no prod usage; scope to `pub(crate)`
   - `ha/{FailoverManager,LeaderElection}` (307 LOC) — re-exported only; scope to `pub(crate)`
   - `circuit_breaker/*` (555 LOC) — used by error/recovery; scope to `pub(crate)`
5. **ARCH-05**: Collapse 4 stub architectures (`gemma3`, `llama4`, `phi4`,
   `mistral_small`) into one parameterized `StubArchitecture` struct
6. **ARCH-06**: Fix `core → model` upward dep via cuda-graph feature
7. **ARCH-07**: Remove unused `reqwest` from `crates/server/Cargo.toml`
8. **ARCH-08**: Move `rayon` from `[dependencies]` to `[dev-dependencies]`
   in `crates/model/Cargo.toml`
9. **ARCH-09**: Unify 3 `greedy_sample`/`argmax` implementations
10. **ARCH-10**: Unify 2 `Architecture` types (`arch::Architecture` trait + `config::Architecture` enum)

FINAL gates:
- FINAL-01: 1179 tests remain green
- FINAL-02: clippy clean across workspace + all targets + all features
- FINAL-03: fmt clean
- FINAL-04: test count ≥ 1179 (no regression)
- FINAL-05: Update `.planning/PROJECT.md` + `.planning/STATE.md` with v23.0 outcomes

</domain>

<decisions>

## Implementation Decisions

### the agent's Discretion

All implementation choices are at the agent's discretion — pure cleanup phase.
Specific guidance per requirement:

- **ARCH-01..03 (delete dead modules):** Straight `rm`. Verify no external
  callers before deletion via grep. Update `scheduler/mod.rs` to remove
  re-exports.
- **ARCH-04 (consolidate unused):** Change `pub` → `pub(crate)` for items
  that have internal users but no public consumers. Re-exports in `lib.rs`
  get removed.
- **ARCH-05 (stub collapse):** Phase 40 already added the typed `LoadError`
  policy. The collapse itself: create `crates/model/src/stub/architecture.rs`
  with parameterized `StubArchitecture { name: String }` impl. Move the 4
  stubs' `Architecture` impl into a single generic that returns the same
  shape based on name.
- **ARCH-06 (cuda-graph dep):** Verify `crates/core/Cargo.toml` doesn't
  depend on `vllm-model` for non-cuda-graph functionality. If it does,
  move cuda graph types to `vllm-traits` or new `vllm-kernels`.
- **ARCH-07 (reqwest):** `grep -rn "reqwest" crates/server/src/` — if zero
  matches in source code, remove from `Cargo.toml`.
- **ARCH-08 (rayon):** `grep -rn "rayon" crates/model/src/` — check if
  only used in tests/benches; if so, move from `[dependencies]` to
  `[dev-dependencies]`.
- **ARCH-09 (greedy_sample):** Three impls at `core/sampling.rs`,
  `model/causal_lm/mod.rs`, `engine/spec_dispatch/drafts.rs`. Pick canonical
  impl; replace other call sites with it.
- **ARCH-10 (Architecture types):** `arch::Architecture` trait and
  `config::Architecture` enum — pick one or merge.

</decisions>

