# radix_cache Evolution: From `kv_cache/` to `scheduler/radix_cache/`

**Date:** 2026-06-29
**Status:** Historical retrospective
**Context version:** v23.0 (Phase 43 ARCH-01..04)

## Context

vllm-lite's prefix caching (RadixAttention-style) was originally
implemented in a top-level `crates/core/src/kv_cache/` module. By v22.0,
this module had become a maintenance burden:

- **Dead code accumulation**: ~150 LOC of unused helpers (block
  validation utilities, alternative eviction strategies) accumulated
  without consumers.
- **Conceptual mismatch**: KV cache block allocation is fundamentally
  a scheduler concern (it gates how many concurrent requests can run),
  but lived at the same level as `scheduler/`. New contributors didn't
  know whether to add block-allocation logic in `kv_cache/` or
  `scheduler/memory/`.
- **Tight coupling**: `kv_cache/` reached into `scheduler/` for
  `Sequence` types and into `engine/` for model hooks, creating
  circular import candidates.

## v23.0 Decision (Phase 43 ARCH-01..04)

The v23.0 audit classified `kv_cache/mod.rs` as **dead code** (the
`BlockAllocator` and `PrefixCache` had been **moved** to
`scheduler/memory/allocator.rs` and `scheduler/radix_cache/`
respectively during v22.0 cleanup). The remaining 150 LOC in
`kv_cache/mod.rs` was scaffolding with no production callers.

Resolution: **delete the top-level `kv_cache/` module entirely**.
Production prefix-caching logic had already been living in
`scheduler/radix_cache/` for ~6 months (since v22.0); the duplicate
module was pure liability.

## Current State (post-v23.0)

Prefix caching lives in:

- `crates/core/src/scheduler/radix_cache/tree.rs` — radix tree data
  structure for prefix matching (213 LOC)
- `crates/core/src/scheduler/radix_cache/node.rs` — tree node (45 LOC)
- `crates/core/src/scheduler/radix_cache/mod.rs` — public API (4 LOC)

Adjacent (not prefix-cache, but related):

- `crates/core/src/scheduler/memory/allocator.rs` — block allocator
  (363 LOC)
- `crates/core/src/scheduler/memory/eviction.rs` — eviction policy
  (313 LOC)

## Lessons

1. **Module placement reflects ownership, not implementation**. KV
   block allocation is a scheduling concern because it gates
   concurrency — it belongs with the scheduler.
2. **Dead code accumulates silently**. The 150 LOC scaffolding had no
   production callers for ~6 months before the audit caught it. v23.0
   established a precedent for explicit deletion rather than leaving
   "in case it's useful" code.
3. **Migration is a 2-step process**. v22.0 moved the live code to
   `scheduler/radix_cache/`. v23.0 deleted the old module. Doing it
   in one step would have been a big-bang refactor; doing it in two
   steps allowed each to be verified independently.

## See also

- v23.0 audit: `.planning/STATE.md`
- v23.0 Phase 43: ARCH-01..04 (dead module deletion)
- v22.0 cleanup: scheduler module split
- v30.0 Phase K mutation scan of `scheduler/radix_cache/` (100% score)
