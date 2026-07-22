# Phase 41 — Engine-layer Multi-Node Plumbing (OPS-32a second half)

**Date:** 2026-07-22
**Scope:** `vllm-core` (MemoryManager, SchedulerEngine, EngineBuilder, Engine), `vllm-model` (PagedKvCache write API), `vllm-server` (`bootstrap/engine.rs` + `main.rs` gRPC wiring)
**Status:** Design
**Phase goal (long-term):** Wire the P40 `PagedKvCacheWrapper` into the Engine so multi-node KV block replication works end-to-end without manual `ServerState` plumbing. Closes the second half of OPS-32a (ADR-020 §5 deferred engine wiring).

---

## 1. Why this phase

P40 (commit chain `dfa20fa0` … `c591dcae`, 2026-07-22) shipped the load-bearing half of OPS-32a: a concrete `PagedKvCacheWrapper` in `crates/model/src/paged_tensor/paged_kv_cache_wrapper.rs` (gated by `#[cfg(feature = "multi-node")]`) that wraps `Arc<PagedKvCache>` and implements `vllm_dist::BlockDataSource`. The 2-node gRPC round-trip integration test (`real_paged_kv_cache_bytes_round_trip_via_wrapper` in `crates/model/tests/paged_kv_cache_wrapper_e2e.rs`) proves the wrapper can read K/V bytes via `fetch_block` and return them over `TransferKVBlock` to a peer — but the test wires the wrapper into the receiver's `DistributedKVCache::with_block_data_source` manually, not via the engine.

In production today, the server constructs an Engine, hands it a model, and runs inference. There is no path from a server config flag ("enable multi-node KV replication on this node") to (1) constructing an `Arc<PagedKvCache>` shared between the model and the engine, (2) attaching the wrapper to `MemoryManager`, and (3) handing the wrapper to `start_grpc_server_with_listener` as the `BlockDataSource`. The result: the gRPC server answers `Status::unavailable("TransferKVBlock called but no BlockDataSource wired in")` for every block transfer, exactly the gap that the P40 spec §8 explicitly deferred.

P41 closes that gap with bounded scope: 4 reviewable tasks that mirror the pattern set by P19's `with_distributed_kv` builder + the OPS-05b memory-manager hooks, plus a thin server-main wiring step. The receiver-side `write_kv_batch` integration (turning received bytes back into installed KV) is documented but explicitly deferred to P42+ — it requires a new `pub(crate) write_layer_block` helper on `PagedKvCache` that P41 introduces as a stub (skeleton + test-only callers), then P42 wires into the engine's receive path.

### 1.1 What this batch explicitly does NOT close

- **Receiver-side `write_kv_batch` end-to-end** — P41 ships the `pub(crate) write_layer_block` helper on `PagedKvCache` (mirroring the P40 T1 `read_layer_block` helper), but does NOT wire it into the gRPC server's receive handler. The receiver side of a real multi-node KV pull still requires:
  - A `write_kv_batch` helper that takes the wire-shape byte slice + block_id and writes per-layer K/V (mirrors what `forward` writes during a forward pass, but with explicit per-block scope).
  - The gRPC server's `TransferKVBlock` handler in `crates/dist/src/grpc.rs` to call `write_kv_batch` after a successful `BlockDataSource::fetch_block`.
  - An integration test that drives a 2-node gRPC pull end-to-end: receiver requests block X → sender's wrapper reads bytes → receiver writes them via `write_kv_batch` → receiver reads back the same block via its local `read_layer_block` and asserts equality.

  These three items are mechanically similar to the sender-side code P40 shipped, but they require their own design + 1-2 days of implementation, plus end-to-end test fixtures that don't exist today (a real 2-node setup that also exercises the receiver's KV cache, not just the sender's). Tracked as **P42 candidate** — see §6.

- **OPS-32b / OPS-32c / OPS-32d / OPS-32e** (streaming RPCs, wire compression, owner-based routing, failure recovery) — all remain v32+ candidates per the P40 spec §8 deferred list. None are addressable until the receiver-side path is closed (P42).

---

## 2. Scope split (P41 vs. P42)

| Work | P41 (this design) | P42 (deferred) |
|------|-------------------|----------------|
| `MemoryManager::block_data_source` field + setter | ✅ ships | n/a |
| `MemoryManager::set_block_data_source` propagates the wrapper to gRPC-side callers | ✅ ships | n/a |
| `EngineBuilder::with_paged_kv_cache(Arc<PagedKvCache>)` | ✅ ships | n/a |
| `Engine::set_paged_kv_cache` (mirror of `set_distributed_kv`) | ✅ ships | n/a |
| `crates/server/src/bootstrap/engine.rs` constructs `Arc<PagedKvCache>` from the model loader + attaches to engine | ✅ ships | n/a |
| `crates/server/src/main.rs` (or `bootstrap/`) wires `start_grpc_server_with_listener(node_id, listener, None, Some(wrapper))` | ✅ ships | n/a |
| `PagedKvCache::write_layer_block` helper (mirrors `read_layer_block`) | ✅ ships (helper + 4 unit tests) | n/a |
| Hash-verification helper on wrapper side (`verify_chain_hash`) | ✅ ships | n/a |
| End-to-end 2-node gRPC pull (sender reads → receiver writes → both agree on bytes) | n/a (deferred) | ✅ P42 candidate |
| `crates/dist/src/grpc.rs` `TransferKVBlock` handler calls `write_kv_batch` after `fetch_block` | n/a (deferred) | ✅ P42 candidate |
| Receiver-side `write_kv_batch` public method (composes `write_layer_block` × num_layers) | n/a (deferred) | ✅ P42 candidate |

**Why split here:**

- The memory-manager + engine + server-main wiring (P41's load) is mechanically similar to the OPS-05a/b pattern and can be reviewed/tested in isolation against the existing P40 unit + integration tests (which exercise the wrapper directly without going through the engine).
- The receiver-side `write_kv_batch` integration requires new public APIs on `PagedKvCache` AND a 2-node gRPC fixture that exercises the receiver's local cache (today's P40 fixture exercises the sender's cache only). That's a separate PR with its own design choices (write semantics: append vs overwrite? per-token vs per-block? `BlockHasher` recompute or skip?).
- The project style (P21-P40) prefers reviewable batches at the natural architectural seam — same as P19 split OPS-05a/b and P40 split OPS-32a.

---

## 3. Goals

1. **G1 — `MemoryManager` holds an optional `BlockDataSource`.** New `block_data_source: Option<Arc<dyn BlockDataSource + Send + Sync>>` field on `MemoryManager` (feature-gated `#[cfg(feature = "multi-node")]`). New `with_block_data_source(...)` builder method + `set_block_data_source(...)` post-construction setter, both gated under `multi-node`. New `block_data_source(&self) -> Option<Arc<dyn BlockDataSource + Send + Sync>>` getter for the gRPC server to query (so `start_grpc_server_with_listener` can be called with `Some(block_data_source.clone())` even when the engine constructor took `None`).

2. **G2 — Engine wires `Arc<PagedKvCache>` through to `MemoryManager`.** New `EngineBuilder::with_paged_kv_cache(Arc<PagedKvCache>) -> Self` builder method (chained, mirrors `with_distributed_kv`). The builder constructs the `PagedKvCacheWrapper` internally (`Arc::new(PagedKvCacheWrapper::new(cache.clone()))`) and stores both:
   - The wrapper as the `MemoryManager::block_data_source` (via `MemoryManager::set_block_data_source`).
   - The raw `Arc<PagedKvCache>` somewhere readable (the Engine's `Engine::paged_kv_cache` field, gated `#[cfg(feature = "multi-node")]`) so the server can fetch it for the gRPC server wiring.
   New `Engine::set_paged_kv_cache(Arc<PagedKvCache>)` post-construction setter, mirroring `set_distributed_kv`. Crate-internal because the wrapper type isn't exported (gated behind `multi-node` per ADR-008).

3. **G3 — Server bootstrap constructs the wrapper from the loaded model.** `bootstrap/engine.rs::build_engine` gains a new section after model loading: if `app_config.server.multi_node.enabled` is true (new config flag, default false), construct `Arc<PagedKvCache>` via the existing model-loader API (`loader.paged_kv_cache()` or similar — TBD in T1, may require a new `ModelLoader::paged_kv_cache() -> Arc<PagedKvCache>` getter that returns the Arc already held inside the loader's `Model` trait object). Pass it through `EngineBuilder::with_paged_kv_cache(...)` (requires the engine construction in `build_engine` to use the builder path instead of the direct `Engine::new_boxed` / `with_budget_boxed` / `with_drafts_boxed` paths — see §5.5 for the migration plan).

4. **G4 — Server main wires the wrapper to the gRPC server.** `crates/server/src/main.rs` (or the appropriate bootstrap module) gains a gRPC server bootstrap that calls `start_grpc_server_with_listener(node_id, listener, None, Some(wrapper))` where `wrapper = engine.paged_kv_cache_wrapper()` (a new `Engine::paged_kv_cache_wrapper() -> Option<Arc<dyn BlockDataSource + Send + Sync>>` getter, returns the `MemoryManager::block_data_source()` clone). The CLI gains a `--multi-node <node_id>` flag (or reads from `app_config.server.multi_node`) that drives the bootstrap; without the flag the gRPC server is not started (single-node default).

5. **G5 — `PagedKvCache::write_layer_block` helper + 4 unit tests.** New `pub(crate) fn write_layer_block(&mut self, layer_idx: usize, block_id: usize, k: &[f32], v: &[f32]) -> Result<(), PagedKvCacheError>` on `PagedKvCache` (mirrors `read_layer_block` from P40 T1). Validates `layer_idx < num_layers`, `block_id < num_blocks_count_per_layer`, `k.len() == block_size * num_heads * head_dim`, `v.len() == block_size * num_heads * head_dim`. Writes K and V to the layer/block position. The K/V storage layout matches what `read_layer_block` returns (so the round-trip is bit-exact). Returns a typed error for each failure mode. **4 new unit tests** in `crates/model/src/paged_tensor/tensor_store/tests.rs` (mirroring the 4 `read_layer_block` tests): zero-init / written round-trip / oob layer / oob block.

6. **G6 — Hash-verification helper on the wrapper.** New `pub fn verify_chain_hash(&self, block_id: u64, chain_hash: u64) -> bool` on `PagedKvCacheWrapper` that checks `self.inner.block_hashes[0][block_id] == Some(chain_hash)`. Pure read, no allocation, no I/O. **1 new unit test** (`verify_chain_hash_returns_true_for_written_block`) + 1 negative (`verify_chain_hash_returns_false_for_mismatch`).

7. **G7 — Honoring is end-to-end (sans receiver write).** After P41 lands, a sender process that has both an Engine with `Arc<PagedKvCache>` wired AND a running gRPC server with the wrapper as `BlockDataSource` will answer `TransferKVBlock` calls with real K/V bytes from the local cache (not `Status::unavailable`). The peer can verify the chain_hash via `wrapper.verify_chain_hash(...)`. **The peer cannot yet write the received bytes into its own local cache** — that's P42.

---

## 4. Non-goals (P41 explicitly defers)

- **Receiver-side `write_kv_batch` end-to-end** — see §2 and §6. P42 candidate.
- **OPS-32b streaming RPCs** — P40 §8 deferred; remains v32+.
- **OPS-32c wire compression** — P40 §8 deferred; remains v32+.
- **OPS-32d owner-based routing** — P40 §8 deferred; remains v32+.
- **OPS-32e failure recovery** — P40 §8 deferred; remains v32+.
- **Server CLI plumbing for `multi-node` flag** — T4 of P41 introduces the bootstrap + gRPC-server wiring, but the CLI flag itself (`--multi-node <node_id>` or similar) is a thin shim. The full `peer_urls` / cluster-config / `CacheConfig` plumbing that OPS-05c added at the library level remains library-only (matches the P12 OPERATIONS.md note: "`peer_urls` is library-level only — no CLI / `VLLM_*` env var exists yet").

---

## 5. Architecture

### 5.1 `MemoryManager::block_data_source` field + setters

`crates/core/src/scheduler/memory/mod.rs` gains (all `#[cfg(feature = "multi-node")]`-gated):

```rust
/// Optional `BlockDataSource` that produces the actual K/V tensor bytes
/// when a peer requests them via `TransferKVBlock`. Set via
/// `EngineBuilder::with_paged_kv_cache(...)` (which constructs a
/// `PagedKvCacheWrapper` from the loader's `Arc<PagedKvCache>` and
/// passes it through here). Phase 41 OPS-32a second-half.
#[cfg(feature = "multi-node")]
block_data_source: Option<Arc<dyn vllm_dist::distributed_kv::BlockDataSource + Send + Sync>>,
```

The new builder + setter + getter:

```rust
#[cfg(feature = "multi-node")]
impl MemoryManager {
    #[must_use]
    pub fn with_block_data_source(
        mut self,
        source: Arc<dyn vllm_dist::distributed_kv::BlockDataSource + Send + Sync>,
    ) -> Self {
        self.block_data_source = Some(source);
        self
    }

    pub fn set_block_data_source(
        &mut self,
        source: Arc<dyn vllm_dist::distributed_kv::BlockDataSource + Send + Sync>,
    ) {
        self.block_data_source = Some(source);
    }

    pub fn block_data_source(
        &self,
    ) -> Option<Arc<dyn vllm_dist::distributed_kv::BlockDataSource + Send + Sync>> {
        self.block_data_source.as_ref().map(Arc::clone)
    }
}
```

The `Send + Sync` bound on the trait object matches what `start_grpc_server_with_listener` requires (it's a `Arc<dyn BlockDataSource>` parameter).

### 5.2 `PagedKvCache::write_layer_block` helper

`crates/model/src/paged_tensor/tensor_store/mod.rs` gains (alongside the existing `read_layer_block`):

```rust
/// Write per-layer K and V tensors for a single block (P41, the
/// receiver-side counterpart of `read_layer_block` from P40 T1).
///
/// The slice layouts match what `read_layer_block` returns: each is a
/// flat `[f32; block_size * num_heads * head_dim]` row-major slice.
/// Returns `PagedKvCacheError::LayerOutOfRange` /
/// `BlockOutOfRange` / `KLengthMismatch` / `VLengthMismatch` on
/// invalid input. No-op for `quantized=true` caches — the cache
/// stores K/V as f32 internally; quantization is applied at write
/// time but the public API takes f32 (matches the existing
/// `write_kv` path's contract).
pub(crate) fn write_layer_block(
    &mut self,
    layer_idx: usize,
    block_id: usize,
    k: &[f32],
    v: &[f32],
) -> Result<(), PagedKvCacheError> {
    if layer_idx >= self.num_layers {
        return Err(PagedKvCacheError::LayerOutOfRange {
            layer: layer_idx,
            num_layers: self.num_layers,
        });
    }
    let per_layer_blocks = self.num_blocks_count_per_layer();
    if block_id >= per_layer_blocks {
        return Err(PagedKvCacheError::BlockOutOfRange {
            block: block_id,
            num_blocks: per_layer_blocks,
        });
    }
    let expected_len = BLOCK_SIZE * self.num_heads * self.head_dim;
    if k.len() != expected_len {
        return Err(PagedKvCacheError::KLengthMismatch {
            actual: k.len(),
            expected: expected_len,
        });
    }
    if v.len() != expected_len {
        return Err(PagedKvCacheError::VLengthMismatch {
            actual: v.len(),
            expected: expected_len,
        });
    }
    self.storage.write_layer_block(layer_idx, block_id, k, v);
    self.block_hashes[layer_idx].insert(/* chain-hash recompute */ 0, block_id);
    Ok(())
}
```

The `block_hashes` insert uses `0` as a placeholder for now (matches OPS-05b's deterministic-placeholder pattern); P42 will thread the real `BlockHasher` through.

The 4 new unit tests in `tests.rs` (placed adjacent to the existing 4 `read_layer_block` tests):

```rust
#[test]
fn write_layer_block_returns_ok_for_valid_layer_and_block() {
    let mut cache = small_cache();
    let k = vec![1.0; BLOCK_SIZE * cache.num_heads * cache.head_dim];
    let v = vec![2.0; BLOCK_SIZE * cache.num_heads * cache.head_dim];
    cache.write_layer_block(0, 0, &k, &v).unwrap();
    let (k_out, v_out) = cache.read_layer_block(0, 0).unwrap();
    assert_eq!(k_out, k);
    assert_eq!(v_out, v);
}

#[test]
fn write_layer_block_returns_err_for_oob_layer() { /* ... */ }

#[test]
fn write_layer_block_returns_err_for_oob_block() { /* ... */ }

#[test]
fn write_layer_block_returns_err_for_length_mismatch() { /* ... */ }
```

### 5.3 `EngineBuilder::with_paged_kv_cache` + `Engine::set_paged_kv_cache`

`crates/core/src/engine/ctor/builder.rs` gains:

```rust
/// Wire a `PagedKvCache` into the engine for multi-node KV block
/// replication. Constructs a `PagedKvCacheWrapper` internally and
/// threads it through to `MemoryManager::block_data_source` so the
/// gRPC server can answer `TransferKVBlock` calls with real bytes.
///
/// Mirrors `with_distributed_kv(...)` (which wires the metadata
/// cache); this method wires the byte-producer side. Both can be
/// installed independently — `with_distributed_kv` is for the
/// metadata replication (`DistributedKVCache`); `with_paged_kv_cache`
/// is for the byte transfer (`BlockDataSource`). Phase 41 OPS-32a
/// second-half.
#[cfg(feature = "multi-node")]
pub fn with_paged_kv_cache(mut self, cache: Arc<vllm_model::paged_tensor::PagedKvCache>) -> Self {
    self.paged_kv_cache = Some(cache);
    self
}
```

The `EngineBuilder` gains a new `paged_kv_cache: Option<Arc<PagedKvCache>>` field. The `.build()` method calls `engine.set_paged_kv_cache(cache)` if `Some`.

`crates/core/src/engine/ctor/builder.rs::Engine::set_paged_kv_cache` (in a new `crates/core/src/engine/paged_kv_cache.rs` to mirror `cuda_graph.rs` / `distributed_kv.rs`):

```rust
impl crate::engine::Engine {
    /// Install a `PagedKvCache` after construction (Phase 41
    /// OPS-32a second-half). Constructs a `PagedKvCacheWrapper`
    /// internally and propagates it to `MemoryManager::block_data_source`.
    /// Also stores the raw `Arc<PagedKvCache>` so the server can
    /// pass it to the model forward pass.
    #[cfg(feature = "multi-node")]
    pub(crate) fn set_paged_kv_cache(
        &mut self,
        cache: Arc<vllm_model::paged_tensor::PagedKvCache>,
    ) {
        use vllm_model::paged_tensor::PagedKvCacheWrapper;
        use std::sync::Arc as StdArc;
        let wrapper: StdArc<dyn vllm_dist::distributed_kv::BlockDataSource + Send + Sync> =
            StdArc::new(PagedKvCacheWrapper::new(Arc::clone(&cache)));
        self.scheduler.set_block_data_source(StdArc::clone(&wrapper));
        self.paged_kv_cache = Some(cache);
        self.paged_kv_cache_wrapper = Some(wrapper);
    }

    /// Returns the `BlockDataSource` wrapper if a `PagedKvCache` is wired in.
    #[cfg(feature = "multi-node")]
    pub(crate) fn paged_kv_cache_wrapper(
        &self,
    ) -> Option<Arc<dyn vllm_dist::distributed_kv::BlockDataSource + Send + Sync>> {
        self.paged_kv_cache_wrapper.as_ref().map(Arc::clone)
    }
}
```

The Engine gains two new fields (`#[cfg(feature = "multi-node")]`-gated):

```rust
#[cfg(feature = "multi-node")]
paged_kv_cache: Option<Arc<vllm_model::paged_tensor::PagedKvCache>>,
#[cfg(feature = "multi-node")]
paged_kv_cache_wrapper: Option<Arc<dyn vllm_dist::distributed_kv::BlockDataSource + Send + Sync>>,
```

### 5.4 `SchedulerEngine::set_block_data_source`

`crates/core/src/scheduler/engine/memory.rs` (the propagator module per OPS-05b §2) gains:

```rust
#[cfg(feature = "multi-node")]
impl SchedulerEngine {
    pub fn set_block_data_source(
        &self,
        source: Arc<dyn vllm_dist::distributed_kv::BlockDataSource + Send + Sync>,
    ) {
        self.memory.set_block_data_source(source);
    }
}
```

(Using `&self` + interior mutability of `MemoryManager` because `SchedulerEngine::memory` is behind a Mutex — matches the existing `set_distributed_kv` propagator's pattern.)

### 5.5 Server bootstrap wiring

`crates/server/src/bootstrap/engine.rs` gains a new section after model loading:

```rust
// Construct the engine via EngineBuilder when the model_loader
// exposes an Arc<PagedKvCache>. Today the engine is built via
// Engine::new_boxed / with_budget_boxed / with_drafts_boxed —
// P41 T3 introduces a builder path that also threads the cache
// through. The legacy paths are preserved for backward compat
// (single-node default).

#[cfg(feature = "multi-node")]
let paged_kv_cache: Option<Arc<vllm_model::paged_tensor::PagedKvCache>> = {
    // Requires ModelLoader to expose the cache — see T1 in the
    // plan for the getter API. When the loader doesn't expose it
    // (legacy paths, stub mode, etc.) the engine is built without
    // multi-node wiring and the gRPC server is not started.
    loader.paged_kv_cache_clone()
};

// Build the engine. If paged_kv_cache is Some, use the builder
// path; otherwise use the legacy direct constructors.
#[cfg(feature = "multi-node")]
let engine = if let Some(cache) = paged_kv_cache {
    EngineBuilder::new(model)
        .with_draft_model(draft_model.unwrap_or_default_box())
        .with_config(SchedulerConfig::default())
        .with_num_kv_blocks(app_config.engine.num_kv_blocks)
        .with_max_draft_tokens(app_config.engine.max_draft_tokens)
        .with_paged_kv_cache(cache)
        .build()
} else {
    // legacy direct constructors (unchanged from P40)
    Engine::new_boxed(model, draft_model)
    // ...
};
```

`crates/server/src/main.rs` (or a new `bootstrap/grpc.rs`) gains:

```rust
#[cfg(feature = "multi-node")]
if let Some(node_id) = &app_config.server.multi_node.node_id {
    let wrapper = engine.paged_kv_cache_wrapper()
        .context("multi-node enabled but engine has no wrapper")?;
    let listener = tokio::net::TcpListener::bind(
        app_config.server.multi_node.bind_addr.as_str()
    ).await
    .context("failed to bind multi-node gRPC listener")?;
    tokio::spawn(async move {
        vllm_dist::start_grpc_server_with_listener(
            node_id.clone(),
            listener,
            None,  // receiver-side cache is None in P41; P42 wires it
            Some(wrapper),
        ).await
        .context("multi-node gRPC server failed")
    });
    tracing::info!(node_id, "Multi-node gRPC server started");
}
```

### 5.6 Server config additions

`crates/server/src/config/*.yaml` (the `app_config` schema) gains a new section:

```yaml
server:
  # ... existing fields ...
  multi_node:
    # When set, the engine constructs a PagedKvCacheWrapper and
    # starts a gRPC server that answers TransferKVBlock calls with
    # real K/V bytes from the local cache. The receiver side is
    # wired in P42; in P41 the server is sender-only.
    enabled: false
    node_id: null         # optional explicit node id (defaults to random uuid)
    bind_addr: "0.0.0.0:50051"
```

The default `enabled: false` preserves single-node behavior bit-for-bit.

### 5.7 Module declarations + re-exports

- `crates/model/src/paged_tensor/mod.rs` — `PagedKvCache` is already `pub` (P40 T4); no change.
- `crates/core/src/engine/mod.rs` — declare new `paged_kv_cache` submodule (mirror of `cuda_graph` / `distributed_kv`).
- `crates/server/src/bootstrap/mod.rs` — declare new `grpc` submodule (if we put the gRPC bootstrap there).

---

## 6. P42 candidate — receiver-side write path

Tracked here so the work isn't lost; **not** part of P41.

### 6.1 `PagedKvCache::write_kv_batch`

New `pub(crate) fn write_kv_batch(&mut self, block_id: usize, per_layer_kv: &[(Vec<f32>, Vec<f32>)]) -> Result<(), PagedKvCacheError>` on `PagedKvCache`. Validates `per_layer_kv.len() == self.num_layers` + each `(k, v)` length matches `BLOCK_SIZE * num_heads * head_dim`. Delegates to `write_layer_block` × num_layers. Returns early on first error.

### 6.2 `crates/dist/src/grpc.rs::TransferKVBlock` handler

Update the handler to:
1. Call `block_data_source.fetch_block(req.block_id)` (already wired).
2. **NEW**: When the local receiver has a `PagedKvCache` installed, call `cache.write_kv_batch(req.block_id, deserialize(per_layer_bytes))` to install the bytes into the receiver's cache.
3. **NEW**: Compute the chain hash from the received K/V and verify it matches `req.expected_hash` (defense-in-depth — the wrapper's `verify_chain_hash` does the same check on the sender side; doing it again on the receiver catches corruption in transit).
4. Return `(block_id, chain_hash, bytes, num_tokens)` as today.

The handler needs a way to know whether the local node has a `PagedKvCache` installed. New `Option<Arc<dyn WriteableBlockSink>>` field on the gRPC server state, set from `start_grpc_server_with_listener`'s 5th parameter (new). When `None`, the handler behaves as today (just returns the bytes, no local install). When `Some`, the handler writes before returning.

### 6.3 End-to-end 2-node test

New `crates/dist/tests/two_node_e2e_pull.rs`:
1. Spawn two `DistributedKVCache` instances on two threads (matches the existing `kv_block_transfer.rs` pattern).
2. Sender writes K/V for block 0 via `write_layer_block`.
3. Receiver has a `PagedKvCache` installed (with the same shape as the sender's).
4. Receiver calls `cache.fetch_block(0)` → server answers via sender's wrapper.
5. Receiver's `write_kv_batch` installs the bytes.
6. Receiver reads block 0 via its local `read_layer_block` → asserts equality with what the sender wrote.

This test is the proof-of-end-to-end for the multi-node story. Without it, P41 leaves the receiver side untested.

### 6.4 Estimated scope

P42 is roughly equivalent to P40 in scope: 1 new public method (`write_kv_batch`) + 1 gRPC handler change + 1 end-to-end integration test. Estimated 0.5-1 day.

---

## 7. Risks & mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| `EngineBuilder` migration in `bootstrap/engine.rs` regresses the existing single-node path | Medium | High | The legacy `Engine::new_boxed` / `with_budget_boxed` / `with_drafts_boxed` constructors stay unchanged. The new builder path is selected via `if let Some(cache) = paged_kv_cache` — when `None` (single-node default) the legacy path runs. The default `multi_node.enabled: false` keeps the existing 1763 tests passing bit-for-bit. |
| `MemoryManager::block_data_source` setter requires interior mutability through `SchedulerEngine::set_block_data_source` but `MemoryManager` is currently accessed via `&mut self.memory` only | Low | Medium | Use the existing `SchedulerEngine::set_distributed_kv` pattern: `&self` + a `Mutex<MemoryManager>` or `parking_lot::Mutex<MemoryManager>` (the project standard per CLAUDE.md). Verify by checking how `set_distributed_kv` is currently invoked — it's called from `Engine::set_distributed_kv(&mut self)` which holds `&mut Engine`, so `&mut SchedulerEngine` is available, which means `&mut MemoryManager` is too. The `set_block_data_source` propagator can take `&mut self` instead of `&self`. |
| `ModelLoader` doesn't currently expose `Arc<PagedKvCache>` to callers | Medium | Medium | New `ModelLoader::paged_kv_cache() -> Option<Arc<PagedKvCache>>` getter. If the loader doesn't hold an Arc (legacy / stub / external weights), the getter returns `None` and the builder path is skipped (legacy constructor runs). |
| `write_layer_block` storage layout doesn't match `read_layer_block` exactly | Low | High | Both helpers share the same `self.storage.read_layer_block(layer_idx, block_id, k_out, v_out)` / `self.storage.write_layer_block(layer_idx, block_id, k, v)` pair, with identical indexing. The 4 unit tests pin the round-trip. |
| Server-side `multi_node.bind_addr` conflict on a host that already binds port 50051 | Low | Low | T4 logs the bind error and returns it to the operator (no auto-restart / port-iteration logic — that's v32+ work per OPS-32d). The error message names the address so operators can update `app_config.server.multi_node.bind_addr`. |
| `Arc<dyn BlockDataSource + Send + Sync>` requires `BlockDataSource: Send + Sync` | Negligible | Low | Verified — `PagedKvCacheWrapper` is a `#[derive(Clone, Debug)]` newtype around `Arc<PagedKvCache>`, and `Arc<T>: Send + Sync` when `T: Send + Sync`. `PagedKvCache` derives `Debug` and contains `Vec<Tensor>` + `Vec<HashMap<...>>` which are `Send + Sync`. The wrapper is `Send + Sync` end-to-end. |
| Multi-node + single-node feature interaction — operator enables `multi_node.enabled: true` but doesn't set `node_id` | Low | Low | The bootstrap falls back to `uuid::Uuid::new_v4().to_string()` when `node_id: null`. The bootstrap logs the auto-generated id so operators can reference it. |
| T3 introduces a builder-based engine path but `Engine::new_boxed` / `with_budget_boxed` / `with_drafts_boxed` are still used by tests | Low | Medium | Verify by grepping `Engine::new_boxed` usage after T3 lands. If the existing tests still pass via the legacy path (because `paged_kv_cache` is None), no test breakage. The builder path is opt-in. |

---

## 8. Success criteria

- [ ] `cargo build -p vllm-core --features multi-node` is green.
- [ ] `cargo build --workspace --all-features` is green (covers the multi-node + cuda-graph + default combo).
- [ ] `cargo build -p vllm-core` (default) is green — the new fields are gated behind `multi-node`.
- [ ] `cargo clippy -p vllm-core --all-targets --features multi-node -- -D clippy::correctness -D clippy::suspicious -D clippy::perf` is green.
- [ ] `cargo fmt --all --check` passes.
- [ ] `cargo nextest run -p vllm-core --all-features --no-fail-fast` passes (existing tests + new memory-manager / scheduler tests + new wrapper tests).
- [ ] `cargo nextest run --workspace --all-features --no-fail-fast` passes (no regression in the existing 1763 tests).
- [ ] `cargo nextest run --workspace --no-fail-fast` passes (default-features regression guard — the multi-node wiring is invisible in default builds).
- [ ] `bash .planning/phase-12e/check-public-api.sh` exits 0 (public-API delta is 0 for default features; +1 new `EngineBuilder::with_paged_kv_cache` method + +1 new `EngineBuilder::paged_kv_cache` field + +3 new `MemoryManager` methods under `--features multi-node`).
- [ ] `OPERATIONS.md` §"Multi-Node (Experimental)" §"What works" grows from 4 → 6 bullets (engine plumbing + hash-verification helper). §"What is not" shrinks by one item (engine plumbing line moved to "What works").
- [ ] `CHANGELOG.md` [Unreleased] gains a `public-api: vllm-core` bullet (the `MemoryManager` setters / `EngineBuilder::with_paged_kv_cache` additions) per the public-api-check gate.
- [ ] `.planning/v31.0-MASTER-PLAN.md` Phase 31-G row's deliverable is updated: "engine plumbing (MemoryManager + EngineBuilder + server main) deferred to P41+" → "engine plumbing (MemoryManager + EngineBuilder + server main) shipped in P41; receiver-side write deferred to P42+".
- [ ] ADR-020 §"Status" line updated: "engine plumbing deferred to P41+" → "engine plumbing shipped in P41; receiver-side write_kv_batch integration deferred to P42+".

## 9. Test count delta

| Bucket | Before (P40 T8) | After (P41) | Δ |
|--------|------------------:|------------:|---|
| `vllm-model` lib tests (multi-node feature on) | 6 (wrapper) + 4 (read_layer_block) | + 4 (write_layer_block) = **14** | +4 |
| `vllm-core` lib tests (multi-node feature on) | 3 (OPS-05b memory) | + 4 (block_data_source setter / getter / builder / clone) = **7** | +4 |
| `vllm-core` integration tests (multi-node feature on) | 1 (OPS-05b end-to-end) | unchanged | 0 |
| Total workspace tests | 1763 | **1771** | **+8** |

(P41 ships 8 new tests, no existing tests regress. P42 candidate is ~6 more tests for the receiver-side write path.)

## 10. Decision log

| Decision | Rationale | Date |
|----------|-----------|------|
| `EngineBuilder::with_paged_kv_cache` constructs the wrapper internally | Builders that take `Arc<dyn BlockDataSource>` would force callers to also import `vllm_dist`, violating ADR-008's `core → dist` boundary. The builder's responsibility is composition; the wrapper is a composition detail. | 2026-07-22 |
| `block_data_source: Option<Arc<dyn BlockDataSource + Send + Sync>>` rather than `Option<Arc<PagedKvCacheWrapper>>` | Keeps `MemoryManager` decoupled from `vllm_model`'s concrete wrapper type. Mirrors how `DistributedKVCache` takes the trait object directly. | 2026-07-22 |
| Receiver-side `write_kv_batch` deferred to P42 | P41's load is "wire the sender end end-to-end through the engine". The receiver end requires new public APIs on `PagedKvCache` + gRPC handler changes + an end-to-end test that doesn't exist today. Splitting here keeps each PR reviewable. | 2026-07-22 |
| `PagedKvCache::write_layer_block` ships in P41 with `block_hashes` set to 0 placeholder | The write helper itself is mechanical (mirrors `read_layer_block`) and the test surface is small. The placeholder hash matches OPS-05b's deterministic-placeholder pattern — the real `BlockHasher` recompute is P42 work. | 2026-07-22 |
| `Hash-verification helper` is a read-only `verify_chain_hash` on the wrapper | The OPS-31d protocol layer already verifies `chain_hash` on the wire; the wrapper-side helper is defense-in-depth + useful for tests. Read-only, no allocation, no I/O. | 2026-07-22 |
| Server config uses `multi_node.enabled: false` default | Single-node is the dominant deployment; opt-in is safer than opt-out. Mirrors how `cuda-graph` and `multi-node` are already gated behind Cargo features. | 2026-07-22 |
| Builder-based engine path in `bootstrap/engine.rs` is selected when `paged_kv_cache.is_some()` | The legacy `Engine::new_boxed` / `with_budget_boxed` / `with_drafts_boxed` paths stay unchanged. The new builder path is opt-in. The single-node default (`multi_node.enabled: false` → `paged_kv_cache = None`) keeps the existing 1763 tests passing bit-for-bit. | 2026-07-22 |
| `multi_node.bind_addr` defaults to `0.0.0.0:50051` (matches the OPERATIONS.md quickstart) | No magic, no port auto-iteration. Operators can override via config. | 2026-07-22 |
| `node_id` defaults to `uuid::Uuid::new_v4().to_string()` | No global counter, no risk of collision across restarts of the same process. The bootstrap logs the id so operators can reference it. | 2026-07-22 |

## 11. See also

- P40 spec: `docs/superpowers/specs/2026-07-22-p40-paged-kv-cache-wrapper-design.md` (defines the wrapper + the P41 deferred work this spec closes the first half of)
- P40 plan: `docs/superpowers/plans/2026-07-22-p40-paged-kv-cache-wrapper.md`
- OPS-05a/b memory-manager hooks: `.planning/phase-19/ops-05a-distributed-kv-seam.md`, `.planning/phase-19/ops-05b-memory-manager-hooks.md` (the patterns this design mirrors)
- OPS-31d KV block transfer protocol: `.planning/phase-19/ops-31d-kv-block-transfer.md` (the protocol layer the wrapper's `fetch_block` output feeds)
- ADR-008 (crate layering): the `core → dist` boundary that motivates the builder-internal wrapper construction
- ADR-020 (multi-node KV block transfer architecture): the §5 deferred-engine-wiring decision this design closes the first half of
- v31.0 master plan: `.planning/v31.0-MASTER-PLAN.md` Phase 31-G row
- OPERATIONS.md §"Multi-Node (Experimental)" §"What works": the operator-facing doc this batch updates
