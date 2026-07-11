# Phase 19 OPS-05b3 — Distributed prefix-cache lookup

**Date:** 2026-07-11
**Scope:** `vllm-dist` (`DistributedKVCache::lookup_prefix`), `vllm-core` (MemoryManager, SchedulerEngine, ObserverEvent)
**Status:** Shipped
**Phase goal (long-term):** Wire vllm-dist into the Engine end-to-end so multi-node inference is real, not just available as a library.

---

## 1. Why this phase

OPS-05b2 (commit `ec453b2`) established content-derived chain hashes
in `DistributedKVCache::put`. The cache can now store entries keyed
by content hash, but there was no API to *look up* a prefix by its
chain — just per-key `get()`.

OPS-05b3 closes that loop: when a new request arrives with a token
prefix, the scheduler computes the chain hash for each block and
asks the cache. The cache returns the longest consecutive prefix
that's been seen (locally today; once OPS-05c lands, on peer nodes
too).

What's *not* in scope: actually transferring KV blocks from a peer.
OPS-05b3 establishes the lookup; block transfer requires the gRPC
plumbing in OPS-05c. Until then the result is informational —
observers (metrics / tracing) can report "X% of incoming prompts
had a remote prefix hit" so operators can decide when to enable
cross-node sync.

---

## 2. What changed

### `DistributedKVCache::lookup_prefix` (`crates/dist/src/distributed_kv/cache.rs`)

```rust
/// Walk `keys` in order; return the count of consecutive hits
/// from the start.
///
/// The first miss stops the walk and the prefix length up to
/// (but not including) that key is returned. If every key hits,
/// the full `keys.len()` is returned.
pub fn lookup_prefix(&self, keys: &[u64]) -> usize;
```

Single write-lock acquisition on the local map (more efficient
than calling `get` once per key). Bumps `hits` / `misses` per key
the same way `get` does — prefix-lookup telemetry is
indistinguishable from individual gets at the counter level.

Stats semantics: on a partial match `[k1=k, k2=k, k3=miss, k4, k5]`,
the call bumps `hits += 2`, `misses += 3` (the first miss plus the
remaining unseen keys). The remaining keys aren't individually
queried, so we count them as misses to keep the hit/miss ratio
informative.

### `DistributedPrefixMatch` struct (`crates/core/src/scheduler/memory/mod.rs`)

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DistributedPrefixMatch {
    /// Number of consecutive blocks whose chain hash was present
    /// in the distributed cache.
    pub matched_blocks: usize,
    /// Number of tokens covered by the matched prefix (always
    /// `matched_blocks * BLOCK_SIZE`, capped at `prompt.len()`).
    pub matched_tokens: usize,
    /// Name of the `BlockHasher` used to compute the chain —
    /// recorded so observers can verify which hash space the
    /// match is in.
    pub hasher_name: &'static str,
}
```

Why no block IDs? The distributed cache stores *content hashes*,
not local block ids. A peer's KV blocks live in the peer's
allocator; we can't reuse them as-is without a block transfer
protocol (OPS-05c). Until then, "how many blocks of this prefix
are cached *somewhere*" is the only useful datum.

### `MemoryManager::lookup_distributed_prefix` (`crates/core/src/scheduler/memory/mod.rs`)

```rust
#[cfg(feature = "multi-node")]
#[must_use]
pub fn lookup_distributed_prefix(
    &self,
    prompt_tokens: &[TokenId],
) -> Option<DistributedPrefixMatch>;
```

Computes the chain hash for each `BLOCK_SIZE`-token chunk and asks
the cache. Returns the longest matched prefix length, or `None` if
no chain hash is present. Returns `None` when no
`DistributedKVCache` is wired in (no-cache manager has nothing
to look up).

Also added `MemoryManager::hasher() -> &dyn BlockHasher` accessor
— useful for diagnostics (logging which hash space is in use) and
for tests that need to compute chain hashes the same way the
manager would.

### `record_block_tokens` re-orientation (`crates/core/src/scheduler/memory/mod.rs`)

The cache now stores `(content_hash, block_id)` from
`record_block_tokens` — *content hash as key*, block_id as value.
This is the opposite direction from `allocate`, which puts
`(block_id, placeholder_hash)` for block-id-keyed existence
tracking. Both entries coexist in the cache; the lookup API
queries by content hash.

The orientation flip was a design call: the prior "block-id as
key" made sense for block-existence tracking (OPS-05b), but
content-addressable lookup (the goal of OPS-05b3) needs the
content hash to be the key.

### `SchedulerEngine::lookup_distributed_prefix` (`crates/core/src/scheduler/engine/memory.rs`)

Thin wrapper around `MemoryManager::lookup_distributed_prefix` so
callers don't need to reach into the manager directly.

### `SchedulerEngine::add_request` hook (`crates/core/src/scheduler/engine/state/request.rs`)

After the local `RadixTree::longest_prefix_match` check (still
first — local is faster than remote), `add_request` now calls
`lookup_distributed_prefix(&req.prompt)` and dispatches an
observer event with the matched token count. The result is
informational: the engine doesn't reuse remote blocks (OPS-05c
plumbing required), but the metric flows through.

### `ObserverEvent::DistributedPrefixMatched` (`crates/core/src/scheduler/observer.rs`)

New variant:

```rust
DistributedPrefixMatched {
    seq_id: SeqId,
    matched_tokens: usize,
},
```

Plus a matching `on_distributed_prefix_matched` trait method on
`SchedulerObserver` (with no-op impl in `NoopSchedulerObserver`
and test impls in `crates/core/tests/observer.rs`). `matched_tokens
== 0` indicates a full miss — implementations that only care about
hits can filter on `matched_tokens > 0`.

---

## 3. Tests

### `crates/dist/src/distributed_kv/cache.rs` (6 new unit tests)

| Test | Asserts |
|------|---------|
| `test_lookup_prefix_empty_input_returns_zero` | Empty `keys` is a no-op (no lock, no counter bumps). |
| `test_lookup_prefix_all_hits` | All 4 keys hit → returns 4, `hits == 4`, `misses == 0`. |
| `test_lookup_prefix_partial_match_stops_at_first_miss` | Hits on 10, 20; miss on 30 → returns 2; misses counts 30 + the rest. |
| `test_lookup_prefix_first_key_misses` | First key miss → returns 0; all 3 keys count as misses. |
| `test_lookup_prefix_invalid_entries_count_as_miss` | `cache.invalidate(key)` → subsequent `lookup_prefix` treats it as a miss. |
| `test_lookup_prefix_distinguishes_different_caches` | Two independent `DistributedKVCache` instances don't see each other's entries (lookup is purely local — OPS-05c is what makes it peer-aware). |

### `crates/core/src/scheduler/memory/tests.rs` (6 new unit tests)

| Test | Asserts |
|------|---------|
| `test_memory_manager_lookup_distributed_prefix_full_hit` | All 3 chain hashes pre-published → 3 blocks matched, `hasher_name == "xorshift"`. |
| `test_memory_manager_lookup_distributed_prefix_partial_match` | Only first of 2 chain hashes present → `matched_blocks == 1`, `matched_tokens == BLOCK_SIZE`. |
| `test_memory_manager_lookup_distributed_prefix_no_match` | Empty cache → returns `None`. |
| `test_memory_manager_lookup_distributed_prefix_empty_prompt` | Empty `prompt_tokens` → returns `None` (no work to do). |
| `test_memory_manager_lookup_distributed_prefix_no_cache_returns_none` | No cache wired in → returns `None` (no-op). |
| `test_memory_manager_lookup_distributed_prefix_round_trip_with_record` | Allocate → record (with chain cursor) → lookup same prompt → full hit. Mirrors production scheduler flow. |

### `crates/core/tests/distributed_kv_integration.rs` (2 new integration tests)

| Test | Asserts |
|------|---------|
| `engine_scheduler_lookup_distributed_prefix_round_trip` | End-to-end through Engine + EngineBuilder + scheduler: allocate, record, lookup → `DistributedPrefixMatch { matched_blocks: 2, matched_tokens: 32, hasher_name: "xorshift" }`. |
| `engine_scheduler_lookup_distributed_prefix_partial_match` | Record 1 of 2 blocks → lookup → `matched_blocks == 1`. |

### Updated tests

- `test_memory_manager_record_block_tokens_advances_chain` (OPS-05b2) — updated cache-key assertions to match the new `(hash, block_id)` orientation. The actual chain-property assertions are unchanged.

### Updated test impls

- `crates/core/tests/observer.rs` — `TrackingObserver` and `PanickingObserver` both gained `on_distributed_prefix_matched` impls to satisfy the extended trait.

---

## 4. Verification

| Check | Result |
|-------|--------|
| `cargo build -p vllm-dist` | ✅ |
| `cargo build -p vllm-core` (default) | ✅ |
| `cargo build -p vllm-core --features multi-node` | ✅ |
| `cargo build -p vllm-core --features cuda-graph,multi-node` | ✅ |
| `cargo test -p vllm-dist --lib distributed_kv::cache` | ✅ 12 passed (6 new) |
| `cargo test -p vllm-core --features multi-node --lib scheduler::memory` | ✅ 41 passed (6 new) |
| `cargo test -p vllm-core --features multi-node --lib` | ✅ 319 passed |
| `cargo test -p vllm-core --features multi-node --test observer` | ✅ 9 passed |
| `cargo test -p vllm-core --features multi-node --test distributed_kv_integration` | ✅ 7 passed (2 new) |
| `cargo test --all-features --workspace` | ✅ **1298 passed, 0 failed** |
| `cargo fmt --all --check` | ✅ clean |
| `cargo clippy -p vllm-core --all-targets --features multi-node -- -D correctness -D suspicious -D perf` | ✅ clean |
| `cargo clippy -p vllm-dist --all-targets -- -D correctness -D suspicious -D perf` | ✅ clean |

---

## 5. Test count delta

| Bucket | Before (OPS-05b2) | After (OPS-05b3) | Δ |
|--------|------------------:|----------------:|---|
| `vllm-dist` lib tests (`distributed_kv::cache`) | 6 | 12 | +6 (`lookup_prefix` unit tests) |
| `vllm-core` lib tests (multi-node feature on, `scheduler::memory`) | 35 | 41 | +6 (`MemoryManager::lookup_distributed_prefix` tests) |
| `vllm-core` integration tests (`distributed_kv_integration`, multi-node feature on) | 5 | 7 | +2 (scheduler-level lookup integration) |
| Other tests updated, no count change | — | — | 0 |
| **Total workspace tests (`--all-features`)** | **1284** | **1298** | **+14** |

All 14 new tests pass; no existing tests regress.

---

## 6. Files changed

```
crates/dist/src/distributed_kv/cache.rs            | +102 -0
crates/core/src/scheduler/memory/mod.rs            | +95 -7
crates/core/src/scheduler/memory/tests.rs          | +141 -3
crates/core/src/scheduler/observer.rs              | +17 -0
crates/core/src/scheduler/engine/memory.rs         | +12 -0
crates/core/src/scheduler/engine/state/request.rs  | +22 -0
crates/core/tests/observer.rs                      | +15 -0
crates/core/tests/distributed_kv_integration.rs    | +91 -0
```

Net: 8 files modified, ~410 LOC delta.

---

## 7. What is **not** wired up (explicit non-goals)

OPS-05b3 is intentionally narrow. It establishes the lookup API but does **not**:

1. **gRPC peer sync** — `DistributedKVCache::put` still updates only the local map. OPS-05c wires `tonic` so peers observe each other's puts. Until then, a "distributed hit" really means "local hit" — useful for measuring local cache effectiveness, not cross-node coherence.
2. **Block transfer on hit** — even when `lookup_distributed_prefix` returns a hit, the engine doesn't fetch the KV blocks. That's a separate transfer protocol; requires OPS-05c plumbing plus a new `CacheMessage::BlockTransfer` variant.
3. **Memory accounting on hit** — the engine currently still allocates fresh blocks for the matched prefix. Reducing local allocations when a remote hit covers the prefix is a follow-up (depends on #2).
4. **Metrics counters for hit/miss rate** — observers receive the events; metrics wiring is left to the implementation (the `on_distributed_prefix_matched` callback is the integration point).
5. **Cross-engine hit aggregation** — each engine has its own distributed cache. Aggregating hit rates across engines is the server's job.

---

## 8. Migration toward OPS-05c

OPS-05c: gRPC peer sync for the distributed cache.

1. `GrpcState` becomes a peer-to-peer replication transport for `CacheMessage::Put` / `Invalidate` / `Update` / `Write`.
2. `CacheConfig` grows a `peer_urls: Vec<String>` field.
3. The dead `NodeServiceClient` / `NodeServiceServer` (currently generated but unused `tonic` code in `vllm-dist/generated/`) becomes live.

When OPS-05c ships, every entry put on one node is broadcast to all peer nodes. The same `lookup_distributed_prefix` API then transparently returns cross-node prefix hits — no scheduler-side changes needed.

OPS-05c + block-transfer protocol are the two missing pieces to make the OPS-05 phase goal real: end-to-end cross-node inference with prefix-cache reuse.

---

## 9. Summary

| What | Before | After |
|------|--------|-------|
| Cache lookup | Per-key `get` only | New `lookup_prefix(&[u64]) -> usize` for prefix walks |
| Scheduler prefix lookup | Local `RadixTree` only | Local + distributed (post-OPS-05c: cross-node) |
| Observer visibility | `RequestArrived` only | + `DistributedPrefixMatched { seq_id, matched_tokens }` |
| `MemoryManager` chain-cursor surface | `record_block_tokens` (writer) | + `lookup_distributed_prefix` (reader) + `hasher()` accessor |
| Cache entry orientation from `record_block_tokens` | `(block_id, hash)` | `(content_hash, block_id)` — content-addressable |
| Workspace tests | 1284 | **1298** (+14) |
