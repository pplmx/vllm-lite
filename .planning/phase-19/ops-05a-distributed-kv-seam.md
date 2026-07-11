# Phase 19 OPS-05a — DistributedKVCache seam inside Engine

**Date:** 2026-07-11
**Scope:** `vllm-core`, `vllm-dist`
**Status:** Shipped
**Phase goal (long-term):** Wire vllm-dist into the Engine end-to-end so multi-node inference is real, not just available as a library.

---

## 1. Why this phase, and why this small scope

OPS-05 was deferred from v23.0 and earlier because vllm-dist's four
subsystems (tensor parallel, pipeline parallel, distributed KV cache,
gRPC transport) sit below the Engine with **zero external users** outside
the crate. The 3233 LOC of dist code is dormant — `qwen3/tp.rs` is the
only consumer, and only for `TensorParallelConfig`.

Three of those subsystems sit at the **tensor-level** API (`PipelineStage::forward(StageInput)`) and would require a deep Engine refactor to integrate cleanly (the Engine today only sees `ModelBackend` tokens, never hidden states). That work is real but ~15-25 hours and a separate architectural decision.

The fourth subsystem, `DistributedKVCache`, is **engine-natural**: it operates on `(u64 key, u64 value_hash)` metadata pairs that slot into the Engine's existing KV-block bookkeeping without touching model internals.

**OPS-05a ships the smallest honest subset of that work** — the field, the builder hook, and the status accessors — so:

1. The cache is reachable from Engine callers (not just from `vllm_dist` users).
2. The accessor pattern is locked in (consistent with `cuda_graph_enabled` / `cuda_graph_stats`).
3. The allocator-level integration is unblocked for OPS-05b.

We explicitly do **not** claim end-to-end cross-node inference. The cache is owned but not yet wired into `BlockAllocator`. That's OPS-05b.

---

## 2. What changed

### Cargo

`crates/core/Cargo.toml` gains:

```toml
[features]
cuda-graph = ["dep:vllm-model"]
multi-node  = ["dep:vllm-dist"]        # NEW

[dependencies.vllm-dist]                # NEW
path = "../dist"
optional = true
```

The `multi-node` feature is gated on `dep:vllm-dist`, mirroring how
`cuda-graph` gates on `dep:vllm-model`. Both features stay independent —
single-node binaries don't pull in vllm-dist.

### Engine field

`crates/core/src/engine/mod.rs` adds one field:

```rust
/// Optional distributed KV-cache for cross-node cache coherence.
/// Phase 19 OPS-05a surfaces the [`vllm_dist::DistributedKVCache`]
/// seam to engine callers; the allocator-level hooks that wire block
/// allocate/free into the cache are OPS-05b. The field exists today
/// so callers can construct a multi-node engine, query its status,
/// and own the cache for the engine's lifetime.
#[cfg(feature = "multi-node")]
distributed_kv: Option<Arc<DistributedKVCache>>,
```

`Arc` because the cache is the same instance the server (or a peer
process) will share — the type is `Send + Sync` and intended for
cross-thread / cross-process observation.

### Accessors

`crates/core/src/engine/distributed_kv.rs` (new file):

```rust
impl Engine {
    // Multi-node build
    #[cfg(feature = "multi-node")]
    pub(crate) fn set_distributed_kv(&mut self, cache: Arc<DistributedKVCache>);

    // Multi-node build
    #[cfg(feature = "multi-node")]
    pub const fn distributed_kv_enabled(&self) -> bool;

    // Non-multi-node build
    #[cfg(not(feature = "multi-node"))]
    pub const fn distributed_kv_enabled(&self) -> bool { false }

    // Multi-node build
    #[cfg(feature = "multi-node")]
    pub fn distributed_kv_stats(&self) -> Option<CacheStats>;
}
```

Three feature combinations × two accessors × one setter. The
`#[cfg(not(feature = "multi-node"))] const fn distributed_kv_enabled` stub
returns `false` so call sites compile unchanged on single-node builds —
same pattern as `cuda_graph_enabled`.

### Builder

`crates/core/src/engine/ctor/builder.rs` adds the parallel
`with_distributed_kv` method:

```rust
#[cfg(feature = "multi-node")]
pub fn with_distributed_kv(mut self, cache: Arc<DistributedKVCache>) -> Self;
```

In `build()`, if the cache was set, it's installed via the new
`set_distributed_kv` setter — same pattern as `with_cuda_graph_executor`.

### Constructor

`crates/core/src/engine/ctor/mod.rs::with_config_boxed` initializes the
new field to `None` (under the feature gate). All four constructors
(`new_boxed`, `with_config`, `with_config_boxed`, `with_drafts_boxed`,
`with_budget_boxed`) go through `with_config_boxed` so they all get the
field automatically.

### Debug

`Engine::Debug` now reports the strong-ref count of the cache (mirrors
how `draft_resolver` is rendered) so logs / debug dumps tell operators
whether the cache is shared or per-engine.

### Tests

`crates/core/tests/distributed_kv_integration.rs` (new, 4 tests,
feature-gated behind `multi-node`):

| Test | Asserts |
|------|---------|
| `engine_without_distributed_kv_reports_disabled` | Default `EngineBuilder` → `distributed_kv_enabled() == false`, `distributed_kv_stats().is_none()` |
| `engine_with_distributed_kv_reports_enabled` | `with_distributed_kv(...)` flips the enabled flag |
| `engine_distributed_kv_stats_reflect_cache_state` | Two `put()` + one `get()` miss → engine reports `updates == 2`, `misses == 1` |
| `multiple_engines_can_share_a_cache_via_arc` | Two engines wrapped around the same `Arc<DistributedKVCache>` see consistent stats |

---

## 3. Verification

| Check | Result |
|-------|--------|
| `cargo build -p vllm-core` (default features) | ✅ |
| `cargo build -p vllm-core --features cuda-graph` | ✅ |
| `cargo build -p vllm-core --features multi-node` | ✅ |
| `cargo build -p vllm-core --features cuda-graph,multi-node` | ✅ |
| `cargo test -p vllm-core --all-features` | ✅ (4 new distributed_kv tests + all existing) |
| `cargo test --all-features --workspace` | ✅ **1265 passed, 0 failed, 48 ignored** |
| `cargo fmt --all --check` | ✅ clean |
| `cargo clippy -p vllm-core --all-targets --features multi-node -- -D correctness -D suspicious -D perf` | ✅ clean (pedantic warnings only, all pre-existing) |

---

## 4. Test count delta

| Bucket | Before | After | Δ |
|--------|-------:|------:|---|
| `vllm-core` integration tests (multi-node feature on) | n/a | **+4** | +4 (`distributed_kv_integration.rs`) |
| Total workspace tests (`--all-features`) | 1261 | **1265** | +4 |

All 4 new tests pass; no existing tests regress.

---

## 5. Files changed

```
crates/core/Cargo.toml                              | +6
crates/core/src/engine/mod.rs                       | +14
crates/core/src/engine/distributed_kv.rs            | +63 (new)
crates/core/src/engine/ctor/mod.rs                  | +2
crates/core/src/engine/ctor/builder.rs              | +21
crates/core/tests/distributed_kv_integration.rs     | +101 (new)
```

Net: 5 files modified, 2 files added, ~200 LOC.

---

## 6. What is **not** wired up (explicit non-goals)

The phase deliberately leaves these for later:

1. **No allocator hooks**: `BlockAllocator::allocate(n)` and `BlockAllocator::free(ranges)` do not call into the cache. OPS-05b will thread the cache through `SchedulerEngine` so that every block transition bumps the cache's `updates` / `invalidations` counters.
2. **No prefix-cache integration**: The scheduler's prefix-cache lookup (in `crates/core/src/scheduler/engine/state/batch.rs:100-105`) does not yet consult the distributed cache. Same seam as #1 — once the allocator hooks are in, the prefix-cache call sites become obvious extension points.
3. **No gRPC peer sync**: `DistributedKVCache::new` builds the local-only path. The `CacheMessage` / `NodeService` gRPC plumbing in `vllm-dist` is generated but no caller invokes it. OPS-05c.
4. **No `PipelineParallel` integration**: `PipelineStage::forward(StageInput)` is a tensor-level API; wiring it through Engine needs an embedding lookup outside the model and a hidden-state extraction between stages. Out of scope for any "seam only" phase — that's a real Engine refactor with its own ADR.
5. **No `TensorParallelManager` integration**: TP is already wired into Qwen3 model construction; the Engine doesn't see it. No change needed today.

---

## 7. Migration toward OPS-05b

OPS-05b should add:

1. `SchedulerEngine` gains `distributed_kv: Option<Arc<DistributedKVCache>>` field.
2. `BlockAllocator` (or a wrapper around it in `scheduler/memory/mod.rs`) calls `cache.put(block_id, ...)` on allocate and `cache.invalidate(block_id)` on free.
3. The prefix-cache lookup consults `cache.get(prefix_hash)` before computing a fresh KV block.
4. The metrics collector picks up `cache.stats()` on every snapshot.

When that lands, the cache stops being a passive observer and becomes the cross-node coherence layer that OPS-05 has always promised.

---

## 8. Migration toward OPS-05c

OPS-05c is the gRPC wiring:

1. `GrpcState` (in `crates/dist/src/grpc.rs`) becomes a peer-to-peer replication transport for `CacheMessage::Put` / `CacheMessage::Invalidate` / `CacheMessage::Get`.
2. The Engine's `DistributedKVCache` is constructed with peer URLs from `AppConfig`.
3. `CacheConfig` already has `node_id` and `num_nodes`; peer URLs are the missing piece.

This is also where the existing `tonic`-generated `NodeServiceClient`/`NodeServiceServer` (currently dead code) becomes live.
