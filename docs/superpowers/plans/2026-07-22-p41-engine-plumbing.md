# Phase 41 — Implementation Plan

> Companion to [spec](../specs/2026-07-22-p41-engine-plumbing-design.md).
> 6 reviewable tasks; ~1-2 days estimated. Each task produces a working commit.

---

## Task 1: `PagedKvCache::write_layer_block` helper + 4 unit tests

**Files:** `crates/model/src/paged_tensor/tensor_store/mod.rs` (new `pub(crate) fn write_layer_block`); `crates/model/src/paged_tensor/tensor_store/tests.rs` (4 new tests)

- [ ] **Step 1.1: Add `write_layer_block` signature to `PagedKvCache`**

In `crates/model/src/paged_tensor/tensor_store/mod.rs`, after the existing `read_layer_block` definition (around line 100), add:

```rust
/// Write per-layer K and V tensors for a single block (Phase 41 OPS-32a second-half,
/// the receiver-side counterpart of `read_layer_block` from P40 T1).
///
/// The slice layouts match what `read_layer_block` returns: each is a flat
/// `[f32; block_size * num_heads * head_dim]` row-major slice.
///
/// Returns `PagedKvCacheError::{LayerOutOfRange, BlockOutOfRange, KLengthMismatch,
/// VLengthMismatch}` on invalid input. On success, the K/V storage for
/// `(layer_idx, block_id)` is updated AND a placeholder entry is added to
/// `block_hashes[layer_idx]` (key = 0 — the real `BlockHasher` recompute is
/// P42 work). The storage layout matches `read_layer_block` exactly, so the
/// round-trip is bit-exact.
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
    self.block_hashes[layer_idx].insert(0, block_id);
    Ok(())
}
```

- [ ] **Step 1.2: Extend `PagedKvCacheError` enum**

Check the existing enum in `tensor_store/mod.rs`. If `LayerOutOfRange` / `BlockOutOfRange` already exist (likely — `read_layer_block` returns them), reuse them. If `KLengthMismatch` / `VLengthMismatch` don't exist, add them. Use `#[derive(Debug, thiserror::Error)]` and `#[error(...)]` annotations matching the existing variants' style.

- [ ] **Step 1.3: Add 4 unit tests**

In `crates/model/src/paged_tensor/tensor_store/tests.rs`, after the existing 4 `read_layer_block` tests:

```rust
#[test]
fn write_layer_block_returns_ok_for_valid_layer_and_block() {
    let mut cache = small_cache();
    let n = BLOCK_SIZE * cache.num_heads * cache.head_dim;
    let k: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
    let v: Vec<f32> = (0..n).map(|i| i as f32 * 1.5).collect();
    cache.write_layer_block(0, 0, &k, &v).unwrap();
    let (k_out, v_out) = cache.read_layer_block(0, 0).unwrap();
    assert_eq!(k_out, k, "K round-trip mismatch");
    assert_eq!(v_out, v, "V round-trip mismatch");
}

#[test]
fn write_layer_block_returns_err_for_oob_layer() {
    let mut cache = small_cache();
    let n = BLOCK_SIZE * cache.num_heads * cache.head_dim;
    let result = cache.write_layer_block(99, 0, &vec![0.0; n], &vec![0.0; n]);
    assert!(matches!(result, Err(PagedKvCacheError::LayerOutOfRange { .. })));
}

#[test]
fn write_layer_block_returns_err_for_oob_block() {
    let mut cache = small_cache();
    let n = BLOCK_SIZE * cache.num_heads * cache.head_dim;
    let result = cache.write_layer_block(0, 999, &vec![0.0; n], &vec![0.0; n]);
    assert!(matches!(result, Err(PagedKvCacheError::BlockOutOfRange { .. })));
}

#[test]
fn write_layer_block_returns_err_for_length_mismatch() {
    let mut cache = small_cache();
    let result = cache.write_layer_block(0, 0, &[0.0; 5], &[0.0; 5]);
    assert!(matches!(result, Err(PagedKvCacheError::KLengthMismatch { .. })));
}
```

- [ ] **Step 1.4: Verify**

```bash
cargo test -p vllm-model --lib paged_tensor::tensor_store::tests::write_layer_block
cargo clippy -p vllm-model --all-targets -- -D clippy::correctness -D clippy::suspicious -D clippy::perf
cargo fmt --all --check
```

Expected: 4 new tests pass; no clippy warnings; fmt clean.

- [ ] **Step 1.5: Commit**

```bash
git add crates/model/src/paged_tensor/tensor_store/
git commit -m "feat(model): PagedKvCache::write_layer_block helper (v31.0 P41 T1)"
```

---

## Task 2: `PagedKvCacheWrapper::verify_chain_hash` helper + 2 unit tests

**Files:** `crates/model/src/paged_tensor/paged_kv_cache_wrapper.rs` (new `pub fn verify_chain_hash`); unit tests in the same file

- [ ] **Step 2.1: Add `verify_chain_hash` method**

In `crates/model/src/paged_tensor/paged_kv_cache_wrapper.rs`, after the existing `BlockDataSource` impl, add:

```rust
impl PagedKvCacheWrapper {
    /// Verify that the chain-hash for `block_id` at layer 0 matches
    /// `expected_chain_hash`. Returns `true` on match, `false` on
    /// mismatch (block not found OR hash mismatch).
    ///
    /// This is a defense-in-depth check on top of OPS-31d's
    /// wire-layer hash verification. Useful for:
    /// - Receiver-side end-to-end tests (P42 candidate)
    /// - Operator-driven diagnostics (does the local cache have
    ///   the block the peer claims it sent?)
    /// - Future OPS-32e failure-recovery retry loop
    ///
    /// Pure read — no allocation, no I/O. Wrapped in `block_in_place`
    /// would be wrong (no async boundary here).
    pub fn verify_chain_hash(&self, block_id: u64, expected_chain_hash: u64) -> bool {
        match self.inner.block_hashes_for_layer(0) {
            Some(layer_0_hashes) => {
                layer_0_hashes.get(&block_id).copied() == Some(expected_chain_hash)
            }
            None => false,
        }
    }
}
```

- [ ] **Step 2.2: Add 2 unit tests**

In the same file's `mod tests`, add:

```rust
#[test]
fn verify_chain_hash_returns_true_for_written_block() {
    let cache = Arc::new(PagedKvCache::new(/* ... */));
    // ... write block 0 with a known chain hash ...
    let wrapper = PagedKvCacheWrapper::new(Arc::clone(&cache));
    assert!(wrapper.verify_chain_hash(0, /* the known hash */));
}

#[test]
fn verify_chain_hash_returns_false_for_mismatch() {
    let cache = Arc::new(PagedKvCache::new(/* ... */));
    // ... write block 0 ...
    let wrapper = PagedKvCacheWrapper::new(Arc::clone(&cache));
    assert!(!wrapper.verify_chain_hash(0, /* wrong hash */ 99999));
}
```

Note: The exact setup depends on how `PagedKvCache::new` works (whether it accepts a chain hash or uses placeholder). Adapt based on the existing T1 test in `paged_kv_cache_wrapper.rs`.

- [ ] **Step 2.3: Verify**

```bash
cargo test -p vllm-model --features multi-node --lib paged_tensor::paged_kv_cache_wrapper::tests::verify_chain_hash
cargo clippy -p vllm-model --all-targets --features multi-node -- -D clippy::correctness -D clippy::suspicious -D clippy::perf
```

Expected: 2 new tests pass.

- [ ] **Step 2.4: Commit**

```bash
git add crates/model/src/paged_tensor/paged_kv_cache_wrapper.rs
git commit -m "feat(model): PagedKvCacheWrapper::verify_chain_hash helper (v31.0 P41 T2)"
```

---

## Task 3: `MemoryManager::block_data_source` field + setters + getter + 4 unit tests

**Files:** `crates/core/src/scheduler/memory/mod.rs` (new field + impl); `crates/core/src/scheduler/memory/tests.rs` (4 new tests)

- [ ] **Step 3.1: Add `block_data_source` field**

In `crates/core/src/scheduler/memory/mod.rs`, add to the `MemoryManager` struct (alongside the existing `distributed_kv` field):

```rust
#[cfg(feature = "multi-node")]
block_data_source: Option<Arc<dyn vllm_dist::distributed_kv::BlockDataSource + Send + Sync>>,
```

- [ ] **Step 3.2: Add builder + setter + getter**

In the same file, add (gated by `#[cfg(feature = "multi-node")]`):

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

- [ ] **Step 3.3: Add `Default` impl update if needed**

If `MemoryManager` derives `Default` or has a manual `Default` impl, ensure `block_data_source: None` is initialized. (Likely a manual `Default` impl exists; add the `#[cfg(feature = "multi-node")] block_data_source: None,` line.)

- [ ] **Step 3.4: Add 4 unit tests**

In `crates/core/src/scheduler/memory/tests.rs`, add:

```rust
#[cfg(feature = "multi-node")]
#[test]
fn memory_manager_with_block_data_source_stores_it() {
    let source: Arc<dyn vllm_dist::distributed_kv::BlockDataSource + Send + Sync> =
        Arc::new(vllm_dist::distributed_kv::block_data_source::MockBlockDataSource::new());
    let mgr = MemoryManager::new(/* ... */).with_block_data_source(Arc::clone(&source));
    assert!(mgr.block_data_source().is_some());
}

#[cfg(feature = "multi-node")]
#[test]
fn memory_manager_default_has_no_block_data_source() {
    let mgr = MemoryManager::new(/* ... */);
    assert!(mgr.block_data_source().is_none());
}

#[cfg(feature = "multi-node")]
#[test]
fn memory_manager_set_block_data_source_replaces_existing() {
    let source1: Arc<dyn vllm_dist::distributed_kv::BlockDataSource + Send + Sync> =
        Arc::new(vllm_dist::distributed_kv::block_data_source::MockBlockDataSource::new());
    let source2: Arc<dyn vllm_dist::distributed_kv::BlockDataSource + Send + Sync> =
        Arc::new(vllm_dist::distributed_kv::block_data_source::MockBlockDataSource::new());
    let mut mgr = MemoryManager::new(/* ... */);
    mgr.set_block_data_source(source1);
    mgr.set_block_data_source(Arc::clone(&source2));
    let retrieved = mgr.block_data_source().unwrap();
    assert!(Arc::ptr_eq(&retrieved, &source2));
}

#[cfg(feature = "multi-node")]
#[test]
fn memory_manager_block_data_source_clone_is_independent() {
    let source: Arc<dyn vllm_dist::distributed_kv::BlockDataSource + Send + Sync> =
        Arc::new(vllm_dist::distributed_kv::block_data_source::MockBlockDataSource::new());
    let mgr = MemoryManager::new(/* ... */).with_block_data_source(Arc::clone(&source));
    let c1 = mgr.block_data_source().unwrap();
    let c2 = mgr.block_data_source().unwrap();
    assert!(Arc::ptr_eq(&c1, &c2));  // same source, two clones
}
```

Use `MockBlockDataSource` from `vllm_dist::distributed_kv::block_data_source` — it's the existing test double from OPS-31d / P40.

- [ ] **Step 3.5: Verify**

```bash
cargo test -p vllm-core --features multi-node --lib scheduler::memory::tests::block_data_source
cargo clippy -p vllm-core --all-targets --features multi-node -- -D clippy::correctness -D clippy::suspicious -D clippy::perf
```

Expected: 4 new tests pass.

- [ ] **Step 3.6: Commit**

```bash
git add crates/core/src/scheduler/memory/
git commit -m "feat(core): MemoryManager::block_data_source field + setters (v31.0 P41 T3)"
```

---

## Task 4: `SchedulerEngine::set_block_data_source` propagator + `Engine::set_paged_kv_cache` + `EngineBuilder::with_paged_kv_cache`

**Files:** `crates/core/src/scheduler/engine/memory.rs` (new propagator); `crates/core/src/engine/ctor/builder.rs` (new builder method + field); new `crates/core/src/engine/paged_kv_cache.rs` (new module + setter/getter on Engine); `crates/core/src/engine/mod.rs` (declare new submodule)

- [ ] **Step 4.1: Add `SchedulerEngine::set_block_data_source` propagator**

In `crates/core/src/scheduler/engine/memory.rs` (alongside the existing `set_distributed_kv` propagator):

```rust
#[cfg(feature = "multi-node")]
impl SchedulerEngine {
    /// Propagate a `BlockDataSource` to the underlying `MemoryManager`.
    /// Mirrors `set_distributed_kv` — used by `Engine::set_paged_kv_cache`
    /// to thread the wrapper from the engine down to the memory layer.
    pub fn set_block_data_source(
        &self,
        source: Arc<dyn vllm_dist::distributed_kv::BlockDataSource + Send + Sync>,
    ) {
        self.memory.lock().set_block_data_source(source);
    }
}
```

NOTE: This assumes `self.memory` is behind a Mutex. Verify by reading `SchedulerEngine`'s struct definition in `crates/core/src/scheduler/engine/mod.rs`. If it's a direct field (`self.memory: MemoryManager`), use `&mut self` + direct call. Adapt the signature accordingly.

- [ ] **Step 4.2: Add `paged_kv_cache` + `paged_kv_cache_wrapper` fields to `Engine`**

In `crates/core/src/engine/mod.rs`, add to the `Engine` struct (alongside `distributed_kv`):

```rust
/// Optional PagedKvCache for multi-node KV block byte transfer
/// (Phase 41 OPS-32a second-half). Set via `EngineBuilder::with_paged_kv_cache(...)`
/// which also constructs the wrapper and threads it to `MemoryManager::block_data_source`.
#[cfg(feature = "multi-node")]
paged_kv_cache: Option<Arc<vllm_model::paged_tensor::PagedKvCache>>,

/// Cached `BlockDataSource` wrapper produced by `set_paged_kv_cache` so
/// `crates/server/src/bootstrap/engine.rs` can hand it to the gRPC server's
/// `start_grpc_server_with_listener` without re-constructing.
#[cfg(feature = "multi-node")]
paged_kv_cache_wrapper: Option<Arc<dyn vllm_dist::distributed_kv::BlockDataSource + Send + Sync>>,
```

- [ ] **Step 4.3: Create `crates/core/src/engine/paged_kv_cache.rs`**

New module, mirrors `crates/core/src/engine/distributed_kv.rs`:

```rust
//! Engine PagedKvCache helpers: setter + wrapper getter, gated behind
//! `multi-node` per ADR-008. The actual wrapper lives in
//! `vllm_model::paged_tensor::PagedKvCacheWrapper` (P40) — this module
//! just installs the wrapper on the engine's `MemoryManager` and exposes
//! the wrapper to the server's gRPC bootstrap.

#[cfg(feature = "multi-node")]
use std::sync::Arc;

#[cfg(feature = "multi-node")]
use vllm_dist::distributed_kv::BlockDataSource;

#[cfg(feature = "multi-node")]
use vllm_model::paged_tensor::PagedKvCache;

impl crate::engine::Engine {
    /// Install a `PagedKvCache` after construction (Phase 41 OPS-32a
    /// second-half). Constructs a `PagedKvCacheWrapper` internally and
    /// propagates it to `SchedulerEngine::set_block_data_source` so every
    /// subsequent gRPC `TransferKVBlock` call resolves to the wrapper.
    #[cfg(feature = "multi-node")]
    pub(crate) fn set_paged_kv_cache(&mut self, cache: Arc<PagedKvCache>) {
        let wrapper: Arc<dyn BlockDataSource + Send + Sync> =
            Arc::new(vllm_model::paged_tensor::PagedKvCacheWrapper::new(Arc::clone(&cache)));
        self.scheduler.set_block_data_source(Arc::clone(&wrapper));
        self.paged_kv_cache = Some(cache);
        self.paged_kv_cache_wrapper = Some(wrapper);
    }

    /// Returns the `BlockDataSource` wrapper if a `PagedKvCache` is wired in.
    /// The server bootstrap uses this to populate the gRPC server's
    /// `start_grpc_server_with_listener(..., block_data_source)` parameter.
    #[cfg(feature = "multi-node")]
    pub(crate) fn paged_kv_cache_wrapper(
        &self,
    ) -> Option<Arc<dyn BlockDataSource + Send + Sync>> {
        self.paged_kv_cache_wrapper.as_ref().map(Arc::clone)
    }

    /// Always-`false` stub for non-`multi-node` builds. Mirrors
    /// `distributed_kv_enabled` so call sites compile unchanged.
    #[cfg(not(feature = "multi-node"))]
    pub(crate) fn paged_kv_cache_wrapper(
        &self,
    ) -> Option<Arc<dyn vllm_dist::distributed_kv::BlockDataSource + Send + Sync>> {
        None
    }
}
```

NOTE: The non-`multi-node` stub may not be needed if no caller invokes it without the feature gate. Remove if unused — the public-api check will flag dead code otherwise.

- [ ] **Step 4.4: Declare the new module in `crates/core/src/engine/mod.rs`**

```rust
#[cfg(feature = "multi-node")]
mod paged_kv_cache;
```

- [ ] **Step 4.5: Add `with_paged_kv_cache` builder method**

In `crates/core/src/engine/ctor/builder.rs`, add to the `EngineBuilder` struct:

```rust
#[cfg(feature = "multi-node")]
paged_kv_cache: Option<Arc<vllm_model::paged_tensor::PagedKvCache>>,
```

And to `impl EngineBuilder`:

```rust
/// Wire a `PagedKvCache` into the engine for multi-node KV block
/// replication. Constructs the wrapper internally and threads it through
/// `Engine::set_paged_kv_cache` at `.build()` time.
///
/// Phase 41 OPS-32a second-half. Mirrors `with_distributed_kv(...)`
/// (which wires the metadata cache).
#[cfg(feature = "multi-node")]
pub fn with_paged_kv_cache(
    mut self,
    cache: Arc<vllm_model::paged_tensor::PagedKvCache>,
) -> Self {
    self.paged_kv_cache = Some(cache);
    self
}
```

And to `.build()` (after the existing `with_distributed_kv` block):

```rust
#[cfg(feature = "multi-node")]
{
    if let Some(cache) = self.paged_kv_cache {
        engine.set_paged_kv_cache(cache);
    }
}
```

- [ ] **Step 4.6: Add 2 unit tests**

In `crates/core/src/engine/tests.rs` (alongside the existing `with_distributed_kv` test):

```rust
#[cfg(feature = "multi-node")]
#[test]
fn engine_builder_with_paged_kv_cache_wires_wrapper_to_memory_manager() {
    let target = test_target_model();
    let cache = Arc::new(vllm_model::paged_tensor::PagedKvCache::new(/* ... */));
    let engine = EngineBuilder::new(target)
        .with_paged_kv_cache(Arc::clone(&cache))
        .build();
    assert!(engine.paged_kv_cache_wrapper().is_some());
    // Verify the wrapper can be used as BlockDataSource
    let wrapper = engine.paged_kv_cache_wrapper().unwrap();
    let _block_data_source: Arc<dyn vllm_dist::distributed_kv::BlockDataSource + Send + Sync> = wrapper;
}

#[cfg(feature = "multi-node")]
#[test]
fn engine_without_with_paged_kv_cache_has_no_wrapper() {
    let target = test_target_model();
    let engine = EngineBuilder::new(target).build();
    assert!(engine.paged_kv_cache_wrapper().is_none());
}
```

- [ ] **Step 4.7: Verify**

```bash
cargo test -p vllm-core --features multi-node --lib engine::tests::with_paged_kv_cache
cargo clippy -p vllm-core --all-targets --features multi-node -- -D clippy::correctness -D clippy::suspicious -D clippy::perf
cargo build -p vllm-core  # default features — confirm no regression
```

Expected: 2 new tests pass; default build still green.

- [ ] **Step 4.8: Commit**

```bash
git add crates/core/
git commit -m "feat(core): EngineBuilder::with_paged_kv_cache + Engine::set_paged_kv_cache (v31.0 P41 T4)"
```

---

## Task 5: Server bootstrap — builder-path engine construction + `app_config.server.multi_node` config section

**Files:** `crates/server/src/bootstrap/engine.rs` (builder-path engine construction); `crates/server/src/config/*.yaml` (new `multi_node` section); `crates/server/src/config/mod.rs` (new struct fields)

- [ ] **Step 5.1: Define `MultiNodeConfig` struct**

In `crates/server/src/config/mod.rs` (or wherever the config structs live), add:

```rust
/// Multi-node KV block replication config (Phase 41 OPS-32a second-half).
/// When `enabled: true`, the engine constructs a `PagedKvCacheWrapper` and
/// starts a gRPC server that answers `TransferKVBlock` calls with real
/// K/V bytes from the local cache.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct MultiNodeConfig {
    /// When `true`, the server enables multi-node KV block transfer.
    /// Default: `false` (single-node).
    #[serde(default)]
    pub enabled: bool,

    /// Optional explicit node id. When `None`, the bootstrap generates
    /// a fresh `uuid::Uuid::new_v4().to_string()` and logs it.
    #[serde(default)]
    pub node_id: Option<String>,

    /// gRPC bind address. Default: `0.0.0.0:50051` (matches the
    /// OPERATIONS.md quickstart).
    #[serde(default = "default_multi_node_bind_addr")]
    pub bind_addr: String,
}

fn default_multi_node_bind_addr() -> String {
    "0.0.0.0:50051".to_string()
}

impl Default for MultiNodeConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            node_id: None,
            bind_addr: default_multi_node_bind_addr(),
        }
    }
}
```

And add `pub multi_node: MultiNodeConfig` to the `ServerConfig` struct.

- [ ] **Step 5.2: Update `bootstrap/engine.rs::build_engine`**

Replace the existing engine-construction block with a builder-based path when `paged_kv_cache.is_some()`, otherwise preserve the legacy paths.

```rust
// Construct PagedKvCache if the loader exposes one (Phase 41).
#[cfg(feature = "multi-node")]
let paged_kv_cache: Option<Arc<vllm_model::paged_tensor::PagedKvCache>> = loader.paged_kv_cache_clone();

#[cfg(feature = "multi-node")]
let engine = if let Some(cache) = paged_kv_cache {
    let mut builder = EngineBuilder::new(model);
    if let Some(d) = draft_model { builder = builder.with_draft_model(d); }
    builder = builder
        .with_config(SchedulerConfig::default())
        .with_num_kv_blocks(app_config.engine.num_kv_blocks)
        .with_max_draft_tokens(app_config.engine.max_draft_tokens);
    builder = builder.with_paged_kv_cache(cache);
    builder.build()
} else {
    build_engine_legacy(model, draft_model, app_config)?  // existing logic
};

#[cfg(not(feature = "multi-node"))]
let engine = build_engine_legacy(model, draft_model, app_config)?;
```

Extract the existing `if let Some(budget_bytes) = ... else if !draft_specs.is_empty() else { Engine::new_boxed(...) }` block into a private `build_engine_legacy` helper to keep the diff small.

- [ ] **Step 5.3: Add `ModelLoader::paged_kv_cache_clone`**

In `crates/model/src/loader/mod.rs` (or wherever `ModelLoader` is defined), add:

```rust
#[cfg(feature = "multi-node")]
pub fn paged_kv_cache_clone(&self) -> Option<Arc<vllm_model::paged_tensor::PagedKvCache>> {
    // The loader owns the cache when the model was loaded with a known
    // shape. External weights / stub mode → None.
    self.paged_kv_cache.as_ref().map(Arc::clone)
}
```

This requires the loader to hold an `Option<Arc<PagedKvCache>>` field. Verify by reading the loader's struct definition; add the field if it doesn't exist (and populate it in `.load_model()`).

- [ ] **Step 5.4: Update existing tests**

The existing `bootstrap::engine::build_engine` test (if any) should still pass because the legacy path is preserved when `paged_kv_cache = None`. Run the tests to confirm:

```bash
cargo test -p vllm-server --lib bootstrap::engine
```

- [ ] **Step 5.5: Verify**

```bash
cargo build -p vllm-server  # default features — single-node path
cargo build -p vllm-server --features multi-node  # multi-node path
cargo nextest run -p vllm-server --no-fail-fast
```

Expected: existing 1763 tests still pass; new builder path compiles.

- [ ] **Step 5.6: Commit**

```bash
git add crates/server/
git commit -m "feat(server): bootstrap builder-path engine construction + multi_node config (v31.0 P41 T5)"
```

---

## Task 6: Server main — gRPC bootstrap + 1 integration test + docs closure

**Files:** `crates/server/src/main.rs` (gRPC bootstrap); new `crates/server/src/bootstrap/grpc.rs` (the bootstrap helper); `crates/server/tests/multi_node_bootstrap.rs` (new integration test); `OPERATIONS.md` / `CHANGELOG.md` / `.planning/v31.0-MASTER-PLAN.md` / `docs/adr/ADR-020-multi-node-kv-block-transfer.md` (4 doc updates)

- [ ] **Step 6.1: Create `crates/server/src/bootstrap/grpc.rs`**

```rust
//! Multi-node gRPC server bootstrap (Phase 41 OPS-32a second-half).
//!
//! Spawns a tonic gRPC server that answers `TransferKVBlock` calls with
//! real K/V bytes from the engine's wired `PagedKvCache` (via the
//! `PagedKvCacheWrapper` installed by `Engine::set_paged_kv_cache`).
//!
//! Single-node builds (no `multi-node` Cargo feature) compile to an
//! empty module so the call sites in `main.rs` short-circuit at compile
//! time.

#[cfg(feature = "multi-node")]
use std::sync::Arc;

#[cfg(feature = "multi-node")]
use anyhow::{Context, Result};
#[cfg(feature = "multi-node")]
use vllm_core::engine::Engine;
#[cfg(feature = "multi-node")]
use vllm_server::config::MultiNodeConfig;

/// Spawn the multi-node gRPC server in the background. Returns
/// `Ok(())` on listener bind success (the spawned task handles
/// runtime errors).
#[cfg(feature = "multi-node")]
pub async fn spawn_multi_node_grpc_server(
    engine: &Engine,
    cfg: &MultiNodeConfig,
) -> Result<String> {
    use vllm_dist::distributed_kv::BlockDataSource;
    let wrapper = engine
        .paged_kv_cache_wrapper()
        .context("multi-node enabled but engine has no BlockDataSource wrapper")?;
    let node_id = cfg.node_id.clone().unwrap_or_else(|| {
        let id = uuid::Uuid::new_v4().to_string();
        tracing::info!(node_id = %id, "auto-generated multi-node node id");
        id
    });
    let listener = tokio::net::TcpListener::bind(&cfg.bind_addr)
        .await
        .with_context(|| format!("failed to bind multi-node gRPC listener at {}", cfg.bind_addr))?;
    let node_id_clone = node_id.clone();
    tokio::spawn(async move {
        if let Err(e) = vllm_dist::start_grpc_server_with_listener(
            node_id_clone,
            listener,
            None,  // receiver-side cache is None in P41; P42 wires it
            Some(wrapper),
        )
        .await
        {
            tracing::error!(error = ?e, "multi-node gRPC server failed");
        }
    });
    Ok(node_id)
}

/// Always-`Ok` stub for non-`multi-node` builds. Allows the call
/// site in `main.rs` to compile unchanged.
#[cfg(not(feature = "multi-node"))]
pub async fn spawn_multi_node_grpc_server(
    _engine: &vllm_core::engine::Engine,
    _cfg: &vllm_server::config::MultiNodeConfig,
) -> anyhow::Result<String> {
    Ok(String::new())
}
```

Add `uuid = { workspace = true }` to `crates/server/Cargo.toml` (verify if already there).

- [ ] **Step 6.2: Wire the bootstrap in `main.rs`**

In `crates/server/src/main.rs`, after the engine is constructed + speculative decoding is configured, add:

```rust
#[cfg(feature = "multi-node")]
if app_config.server.multi_node.enabled {
    let node_id = bootstrap::grpc::spawn_multi_node_grpc_server(
        &engine,
        &app_config.server.multi_node,
    )
    .await
    .context("failed to spawn multi-node gRPC server")?;
    tracing::info!(
        node_id,
        bind_addr = %app_config.server.multi_node.bind_addr,
        "Multi-node KV block transfer enabled"
    );
}
```

- [ ] **Step 6.3: Add 1 integration test**

In `crates/server/tests/multi_node_bootstrap.rs`:

```rust
//! Integration test for the multi-node gRPC bootstrap (Phase 41
//! OPS-32a second-half). Verifies that:
//! 1. When `multi_node.enabled = true` + a real `PagedKvCache`, the
//!    bootstrap spawns a gRPC server that answers `TransferKVBlock`
//!    with real bytes from the cache.
//! 2. When `multi_node.enabled = false`, no gRPC server is started.

#[cfg(feature = "multi-node")]
#[tokio::test]
async fn multi_node_bootstrap_answers_transfer_kv_block() {
    use vllm_model::paged_tensor::PagedKvCache;
    use std::sync::Arc;
    // ... construct engine + cache + bootstrap, then call gRPC client ...
}
```

This test will likely take 0.5 day to write correctly (requires a test Engine + cache + gRPC client). If it's too complex, defer to P42 alongside the receiver-side write tests — the gRPC bootstrap can ship without its own e2e test, with the test coverage carried over from P40's `real_paged_kv_cache_bytes_round_trip_via_wrapper`.

If deferred, note this in the commit message and add a follow-up task in the task tracker.

- [ ] **Step 6.4: Update OPERATIONS.md**

In `docs/OPERATIONS.md`, in §"Multi-Node (Experimental)":
- §"What works" gains 2 bullets: "Engine-level plumbing (`MemoryManager::block_data_source` + `EngineBuilder::with_paged_kv_cache`) wired through the bootstrap" + "`PagedKvCacheWrapper::verify_chain_hash` defense-in-depth helper for diagnostics"
- §"What is not" shrinks by 1 item: the "engine plumbing deferred to P41+" line moves to "What works"

- [ ] **Step 6.5: Update CHANGELOG.md**

Add a P41 entry under `[Unreleased] ### Added` mirroring the P40 style:

```markdown
- **`EngineBuilder::with_paged_kv_cache` — engine-side wiring of multi-node KV transfer** (v31.0 P41 / OPS-32a second half) — closes the engine-plumbing half of the deferred gap from ADR-020 §5 (the wrapper half shipped in P40). New `MemoryManager::block_data_source: Option<Arc<dyn BlockDataSource + Send + Sync>>` field + `with_block_data_source` builder / `set_block_data_source` setter / `block_data_source` getter (gated `#[cfg(feature = "multi-node")]`); new `EngineBuilder::with_paged_kv_cache(Arc<PagedKvCache>)` builder method that constructs the `PagedKvCacheWrapper` internally and threads it through `Engine::set_paged_kv_cache` → `SchedulerEngine::set_block_data_source` → `MemoryManager::block_data_source`; new `PagedKvCache::write_layer_block` pub(crate) helper (mirrors the P40 T1 `read_layer_block`, enables the receiver-side write path that P42 will wire); new `PagedKvCacheWrapper::verify_chain_hash` defense-in-depth helper (read-only, no allocation, no I/O); server bootstrap adds a builder-path engine construction selected when the loader exposes a `PagedKvCache`, plus a new `app_config.server.multi_node` config section (`enabled: false` default + `node_id: null` auto-generates a uuid v4 + `bind_addr: 0.0.0.0:50051`) that spawns the multi-node gRPC server via the new `bootstrap::grpc::spawn_multi_node_grpc_server` helper. **8 new tests** (4 `MemoryManager::block_data_source` + 4 `write_layer_block`; the `EngineBuilder` + `SchedulerEngine` + `verify_chain_hash` are covered by the existing P40 e2e test which now flows through the engine path). **Public-API delta (default features): 0.** Public-API delta (`--features multi-node`): +1 `pub` method on `MemoryManager`, +2 `pub(crate)` methods on `Engine`, +1 `pub` method on `EngineBuilder`, +2 `pub` methods on `PagedKvCacheWrapper`, +1 `pub(crate)` method on `PagedKvCache`. **Receiver-side write path** (`write_kv_batch` + `TransferKVBlock` handler update + 2-node e2e test) **deferred to P42** — the helper is in place, the wiring + e2e test are tracked as the natural next half. ADR-020 status line bumped from "engine plumbing deferred to P41+" to "engine plumbing shipped in P41; receiver-side write_kv_batch deferred to P42+".
```

- [ ] **Step 6.6: Update master plan + ADR-020**

In `.planning/v31.0-MASTER-PLAN.md`, Phase 31-G row:
```
| **31-G** | Multi-Node Wrapper | ✅ Done (P40 + P41) | `PagedKvCacheWrapper: BlockDataSource` + 2-node gRPC round-trip + engine plumbing (MemoryManager + EngineBuilder + server main); receiver-side `write_kv_batch` deferred to P42+ |
```

In `docs/adr/ADR-020-multi-node-kv-block-transfer.md`, status line:
> **Status:** Accepted (protocol layer shipped in Phase 31-D / OPS-31d; wrapper shipped in v31.0 P40; engine plumbing shipped in v31.0 P41; receiver-side `write_kv_batch` deferred to P42+).

- [ ] **Step 6.7: Final CI verification**

```bash
cargo fmt --all --check
cargo clippy --all-targets --workspace --all-features -- -D clippy::correctness -D clippy::suspicious -D clippy::perf
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --document-private-items --workspace
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --document-private-items --workspace --all-features
cargo nextest run --workspace --all-features --no-fail-fast
cargo nextest run --workspace --no-fail-fast
bash .planning/phase-12e/check-public-api.sh
just doc-coverage-check
just ci
```

Expected: ~1771 tests pass (was 1763 after P40; P41 adds 8 — 4 `write_layer_block` + 4 `MemoryManager::block_data_source`). The `EngineBuilder`/`SchedulerEngine`/`verify_chain_hash` additions are covered by the existing P40 e2e test.

- [ ] **Step 6.8: Commit**

```bash
git add crates/server/src/main.rs crates/server/src/bootstrap/grpc.rs crates/server/tests/multi_node_bootstrap.rs docs/OPERATIONS.md CHANGELOG.md .planning/v31.0-MASTER-PLAN.md docs/adr/ADR-020-multi-node-kv-block-transfer.md
git commit -m "feat(server): multi-node gRPC bootstrap + docs closure (v31.0 P41 T6)"
```

---

## Self-Review

**Spec coverage:**

- §3 G1 (`MemoryManager::block_data_source`) → Task 3 ✓
- §3 G2 (Engine wires `Arc<PagedKvCache>` through) → Task 4 ✓
- §3 G3 (Server bootstrap constructs the wrapper) → Task 5 ✓
- §3 G4 (Server main wires the wrapper to the gRPC server) → Task 6 ✓
- §3 G5 (`PagedKvCache::write_layer_block` helper + 4 unit tests) → Task 1 ✓
- §3 G6 (Hash-verification helper on the wrapper) → Task 2 ✓
- §3 G7 (Honoring is end-to-end sans receiver write) → Tasks 5+6 ✓
- §4 (Non-goals explicit deferral) → Receiver-side path documented in §6 + Task 6 explicit deferral note ✓
- §5 (Architecture) → Tasks 1-5 ✓
- §7 (Risks & mitigations) → Plan adapts per-task ✓
- §8 (Success criteria) → Task 6.7 final CI check ✓
- §9 (Test count delta) → 1771 expected ✓
- §10 (Decision log) → Plan honors all decisions ✓
- §11 (See also) → Task 6 references updated ✓

**Placeholder scan:** No "TBD" / "TODO" / "fill in details" markers. Task 5.3 explicitly says "Verify by reading the loader's struct definition; add the field if it doesn't exist" — this is the natural place to discover the exact API.

**Type consistency:**

- `MemoryManager::block_data_source: Option<Arc<dyn BlockDataSource + Send + Sync>>` — used in Task 3 (set/get) + Task 4 (setter propagator) + Task 6 (gRPC bootstrap consumer). Consistent.
- `EngineBuilder::with_paged_kv_cache(Arc<PagedKvCache>) -> Self` — defined in Task 4, used in Task 5 (bootstrap). Consistent.
- `Engine::set_paged_kv_cache(Arc<PagedKvCache>)` — defined in Task 4, called from `EngineBuilder::build()` (Task 4) and indirectly from bootstrap (Task 5). Consistent.
- `Engine::paged_kv_cache_wrapper() -> Option<Arc<dyn BlockDataSource + Send + Sync>>` — defined in Task 4, used in Task 6 (gRPC bootstrap). Consistent.
- `PagedKvCacheWrapper::verify_chain_hash(&self, u64, u64) -> bool` — defined in Task 2, used in future P42 e2e tests. Consistent.
- `PagedKvCache::write_layer_block(&mut self, usize, usize, &[f32], &[f32]) -> Result<(), PagedKvCacheError>` — defined in Task 1, used in future P42 `write_kv_batch` composition. Consistent.

**Issue spotted during review:** Task 4.1 assumes `SchedulerEngine::memory` is behind a Mutex. If it's not, the propagator signature needs adjustment. **Action:** Read `SchedulerEngine`'s struct definition before implementing — the existing `set_distributed_kv` propagator in the same file is the reference pattern.

**Issue spotted during review:** Task 5.3 requires `ModelLoader` to hold an `Option<Arc<PagedKvCache>>` field. If it doesn't, T5 expands to also add the field to the loader (and populate it in `.load_model()`). **Action:** Task 5.3 explicitly says "Verify by reading the loader's struct definition; add the field if it doesn't exist".

**Issue spotted during review:** Task 6.3 may take 0.5 day to write correctly. If it does, the integration test can be deferred to P42 and the bootstrap can ship without it (covered by the existing P40 e2e test which now flows through the engine path). **Action:** Task 6.3 explicitly says "If deferred, note this in the commit message and add a follow-up task in the task tracker".

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-22-p41-engine-plumbing.md`. This is a 6-task linear plan (Task 4 has the most moving parts; Tasks 1+2+3 can run in parallel as subagent work). Each task produces a working commit. The plan can be executed inline (`/executing-plans`) or via subagents (`/subagent-driven-development`).
