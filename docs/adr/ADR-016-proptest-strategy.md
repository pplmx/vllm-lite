# ADR-016: Property-Based Testing Strategy (proptest)

**Date:** 2026-06-29
**Status:** Accepted
**Context version:** v28.0 / v30.0

## Context

vllm-lite's core scheduling, KV cache management, and sampling logic are
stateful, branchy code with many invariants that are easy to express but
hard to enumerate manually:

- BatchComposer: `seq_ids.len() <= max_batch_size`, parallel vectors stay
  consistent across rebalances, prefill total_tokens == sum of prompt_lens
- BlockAllocator: every allocated block ID is unique within an allocator's
  lifetime; LIFO reuse on free; capacity bounding
- RequestQueue: FIFO order within priority class; phase index consistency
  across enqueue/dequeue
- RadixTree: insert+lookup round-trip; longest-prefix bound; clear() drops
  all references

Example-based unit tests miss these invariants when they fail on edge
cases the developer didn't think to enumerate (e.g., empty token sequences
in BatchComposer — v28.0's `tokens_len - 1` underflow bug).

## Decision

Use `proptest` 1.11 as a workspace dev-dependency, with each property
test as a `#[cfg(test)] mod tests {}` block embedded in the source file
under test. Default 100 cases per run (`PROPTEST_CASES=100`).

Coverage scope:
- `crates/core/src/scheduler/radix_cache/tree.rs` (RadixTree)
- `crates/core/src/scheduler/memory/allocator.rs` (BlockAllocator)
- `crates/core/src/scheduler/request_queue.rs` (RequestQueue)
- `crates/core/src/scheduler/batch_composer/compose.rs` (BatchComposer)

Generator conventions:
- Custom `Arbitrary` impls on internal types (`Sequence`, `BatchCompositionConfig`)
  produce realistic-but-extreme inputs (empty lists, max sizes, off-by-one).
- Shrinking is enabled by default; failing cases shrink to a minimal
  reproducer before being saved.
- Failing seeds are saved to
  `<CARGO_MANIFEST_DIR>/proptest-regressions/<source-relative-path>.txt`
  and **tracked in git** (per `.gitignore` comment, v30.0 fix).

## Consequences

Easier:
- Edge-case bugs (v28.0 found one) surface automatically without exhaustive
  hand-written tests.
- New components can be covered cheaply by adding 5-10 lines of `proptest!`
  per invariant.
- Regression seeds form a permanent, deterministic bug-replay record.

Harder / new risks:
- Generator maintenance: `Arbitrary` impls must stay in sync with the
  type they generate. When a field is added, generators break loudly
  (good) but require updates (cost).
- `PROPTEST_CASES=100` keeps CI runtime manageable (~5-10s per module)
  but is lower than the 1000-10000 recommended for thorough coverage.
  v31+ may revisit.
- `proptest-regressions/` is a new tracked artifact. `.gitignore`
  comments prevent future contributors from mis-ignoring it (v30.0).

## Alternatives considered

- **quickcheck** — older, less actively maintained, weaker shrinking,
  smaller Rust ecosystem integration. Rejected.
- **arbitrary-framework** (the standalone `arbitrary` crate) — works
  for fuzz-derived inputs but lacks `proptest`'s ergonomics for
  property-only tests. Rejected; we use it transitively via fuzz targets.
- **hand-written exhaustive tests** — would multiply LOC by ~10x for
  equivalent coverage. Rejected.

## See also

- v28.0 plan: `docs/superpowers/plans/2026-06-28-v28-property-testing.md`
- v28.0 CHANGELOG entry: empty-tokens underflow bug fix
- ADR-017 (fuzz) — complementary layer
- ADR-018 (mutation) — validates test sensitivity to logic changes
