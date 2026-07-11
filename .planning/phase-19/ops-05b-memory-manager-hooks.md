# Phase 19 OPS-05b — MemoryManager distributed-KV hooks

**Date:** 2026-07-11
**Scope:** `vllm-core` (MemoryManager, SchedulerEngine, EngineBuilder)
**Status:** Shipped
**Phase goal (long-term):** Wire vllm-dist into the Engine end-to-end so multi-node inference is real, not just available as a library.

---

## 1. Why this phase

OPS-05a (commit `db18f16`) added the *seam* — Engine owns a `DistributedKVCache`, but no call site writes through it. That made the cache unreachable from production paths.

OPS-05b threads the cache down to `MemoryManager` so that every `allocate(n)` and `free(ranges)` round-trips through the cache. The cache becomes a real participant in the block lifecycle, not a passive observer.

This unlocks two things:
1. **Cache stats are now meaningful** — `updates` / `invalidations` counters track real engine activity, not direct manual puts.
2. **OPS-05c (gRPC peer sync) has something to sync** — peer nodes can observe block transitions via the cache API.

---

## 2. What changed

### `MemoryManager` field + builder

`crates/core/src/scheduler/memory/mod.rs` adds a feature-gated field and a `with_distributed_kv` builder method:

```rust
#[derive(Debug)]
pub struct MemoryManager {
    allocator: BlockAllocator,
    eviction_policy: EvictionPolicy,
    preemption_manager: PreemptionManager,
    /// Optional distributed KV-cache; when `Some`, every `allocate`
    /// registers new blocks and every `free` invalidates them so peer
    /// nodes can observe activity. Phase 19 OPS-05b.
    #[cfg(feature = "multi-node")]
    distributed_kv: Option<Arc<DistributedKVCache>>,
}

#[cfg(feature = "multi-node")]
#[must_use]
pub fn with_distributed_kv(mut self, cache: Arc<DistributedKVCache>) -> Self {
    self.distributed_kv = Some(cache);
    self
}

#[cfg(feature = "multi-node")]
pub fn set_distributed_kv(&mut self, cache: Arc<DistributedKVCache>) {
    self.distributed_kv = Some(cache);
}
```

Both `with_distributed_kv` (chained builder style) and `set_distributed_kv` (post-construction setter) are exposed — the chainable form is the ergonomic default; the setter is what the Engine's existing `Engine::set_distributed_kv` setter uses internally.

### `MemoryManager::allocate` / `free` hooks

The two methods that already wrap `BlockAllocator` now also write through the cache:

```rust
pub fn allocate(&mut self, num_blocks: usize) -> Option<Vec<BlockId>> {
    let blocks = self.allocator.allocate(num_blocks);
    #[cfg(feature = "multi-node")]
    if let (Some(cache), Some(blocks)) = (self.distributed_kv.as_ref(), blocks.as_ref()) {
        for &block_id in blocks {
            cache.put(u64::try_from(block_id).unwrap_or(u64::MAX), 0);
        }
    }
    blocks
}

pub fn free(&mut self, blocks: &[BlockId]) {
    #[cfg(feature = "multi-node")]
    if let Some(cache) = self.distributed_kv.as_ref() {
        for &block_id in blocks {
            cache.invalidate(u64::try_from(block_id).unwrap_or(u64::MAX));
        }
    }
    self.allocator.free(blocks);
}
```

Key design points:

- **Block id → u64** for the cache key. `BlockId` is a `usize` newtype; the `unwrap_or(u64::MAX)` defends against the (impossible-in-practice) case of a block id too large to fit in `u64` — on every platform the workspace targets this is a no-op.
- **Value = 0** for now. This is the placeholder for the content hash that OPS-05b2 will compute. The *key* (block id) is enough to track existence across nodes today.
- **Free hooks `release_blocks` indirectly**: `release_blocks` calls `free` internally, so we get coverage for the eviction-policy path for free.
- **Preemption hooks too**: `MemoryManager::execute_preemption` (called by `SchedulerEngine::execute_preemption`) calls `self.free(seq.kv_blocks.as_ref())`, so preemption-driven frees also bump `invalidations`.

### `SchedulerEngine::set_distributed_kv` + `memory_mut`

`crates/core/src/scheduler/engine/memory.rs` adds the propagator + a test-friendly accessor:

```rust
pub const fn memory_mut(&mut self) -> &mut MemoryManager {
    &mut self.memory
}

#[cfg(feature = "multi-node")]
pub fn set_distributed_kv(&mut self, cache: Arc<vllm_dist::DistributedKVCache>) {
    self.memory.set_distributed_kv(cache);
}
```

`memory_mut` is a test seam — production code drives block allocation via the request lifecycle (`add_request` / `build_batch`), but tests need direct access to assert that `allocate(n)` bumps the cache counter.

### Engine setter propagates to scheduler

`crates/core/src/engine/distributed_kv.rs::set_distributed_kv` now does two things instead of one:

```rust
#[cfg(feature = "multi-node")]
pub(crate) fn set_distributed_kv(&mut self, cache: Arc<vllm_dist::DistributedKVCache>) {
    self.scheduler.set_distributed_kv(Arc::clone(&cache));
    self.distributed_kv = Some(cache);
}
```

The clone keeps a strong reference inside the scheduler's `MemoryManager` (necessary for the cache to outlive the engine's own `distributed_kv` field) while also storing a reference in the engine for status accessors.

The `Arc::clone` is intentional — the cache is meant to be shared with peer processes / threads, so the cost of one extra strong-ref is irrelevant.

---

## 3. Tests

### Unit tests (`crates/core/src/scheduler/memory/tests.rs`)

Three new tests, all feature-gated behind `multi-node`:

| Test | Asserts |
|------|---------|
| `test_memory_manager_allocate_bumps_cache_updates` | `allocate(3)` → `cache.stats().updates == 3`; cumulative across two calls. |
| `test_memory_manager_free_bumps_cache_invalidations` | `allocate(3)` + `free(blocks)` → `updates == 3`, `invalidations == 3`. |
| `test_memory_manager_without_cache_is_a_no_op` | Default construction (no cache) → allocate / free work, no panic. |

### Integration test (`crates/core/tests/distributed_kv_integration.rs`)

One new test:

| Test | Asserts |
|------|---------|
| `engine_propagates_distributed_kv_to_scheduler_memory_manager` | `EngineBuilder::with_distributed_kv(...)` → `scheduler.memory_mut().allocate(2)` → `engine.distributed_kv_stats().updates == 2`. Verifies the end-to-end wiring from builder → engine → scheduler → memory manager → cache. |

The existing OPS-05a tests (`engine_without_distributed_kv_reports_disabled`, etc.) still pass — they're unaffected because they don't drive block allocation.

---

## 4. Verification

| Check | Result |
|-------|--------|
| `cargo build -p vllm-core` (default) | ✅ |
| `cargo build -p vllm-core --features cuda-graph` | ✅ |
| `cargo build -p vllm-core --features multi-node` | ✅ |
| `cargo build -p vllm-core --features cuda-graph,multi-node` | ✅ |
| `cargo build --tests -p vllm-core --features multi-node` | ✅ |
| `cargo test -p vllm-core --features multi-node --lib scheduler::memory` | ✅ 31 passed (3 new + 28 existing) |
| `cargo test -p vllm-core --features multi-node --test distributed_kv_integration` | ✅ 5 passed (1 new + 4 existing) |
| `cargo test --all-features --workspace` | ✅ **1269 passed, 0 failed** |
| `cargo fmt --all --check` | ✅ clean |
| `cargo clippy -p vllm-core --all-targets --features multi-node -- -D correctness -D suspicious -D perf` | ✅ clean |

---

## 5. Test count delta

| Bucket | Before (OPS-05a) | After (OPS-05b) | Δ |
|--------|------------------:|----------------:|---|
| `vllm-core` lib tests (multi-node feature on) | 306 | **309** | +3 (memory unit tests) |
| `vllm-core` integration tests (multi-node feature on) | 4 | **5** | +1 (`engine_propagates_distributed_kv_to_scheduler_memory_manager`) |
| Total workspace tests (`--all-features`) | 1265 | **1269** | +4 |

All 4 new tests pass; no existing tests regress.

---

## 6. Files changed

```
crates/core/src/scheduler/memory/mod.rs              | +62 -4
crates/core/src/scheduler/memory/tests.rs            | +84 -1
crates/core/src/scheduler/engine/memory.rs           | +19
crates/core/src/engine/distributed_kv.rs             | +8 -3
crates/core/tests/distributed_kv_integration.rs      | +36
```

Net: 5 files modified, ~200 LOC.

---

## 7. What is **not** wired up (explicit non-goals)

OPS-05b is intentionally narrow. The cache observes block lifecycle; it does **not**:

1. **Compute content hashes** — `value` is `0` today. OPS-05b2 will replace this with a hash of (sequence_id, position, content) so peer nodes can answer "do you have the KV for prefix X?"
2. **Use the prefix cache** — `RadixTree::longest_prefix_match` does not consult the distributed cache. Same seam as #1.
3. **Sync across nodes** — `DistributedKVCache::put` only updates the local in-process map. OPS-05c wires the `CacheMessage` / `NodeService` gRPC plumbing so peers observe each other's puts.
4. **Handle migrations / splits** — re-installing the cache via `set_distributed_kv` does not migrate existing tracked blocks. Live migration is OPS-05b3 (if needed).
5. **`PipelineParallel` / `PipelineStage` integration** — tensor-level pipeline is out of scope for any "seam only" phase.

---

## 8. Migration toward OPS-05b2 / OPS-05c

OPS-05b2 (content hashing):
1. Add a `BlockHasher` trait in `vllm-traits` so the hash function is pluggable.
2. `MemoryManager::allocate` calls `cache.put(block_id, hasher.hash(block_id, parent_hash))`.
3. Prefix-cache lookup consults `cache.get(prefix_hash)` before computing fresh blocks.

OPS-05c (gRPC peer sync):
1. `GrpcState` becomes a peer-to-peer replication transport for `CacheMessage::Put` / `Invalidate`.
2. `CacheConfig` grows a `peer_urls` field.
3. The dead `NodeServiceClient` / `NodeServiceServer` (currently unused tonic-generated code) becomes live.
