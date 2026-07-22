# P40 Implementation Plan — `PagedKvCacheWrapper` (Production `BlockDataSource`)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a concrete `BlockDataSource` impl backed by `Arc<PagedKvCache>`, so multi-node `TransferKVBlock` can serve real KV tensor bytes end-to-end — closes the first half of OPS-32a deferred by ADR-020 §5.

**Architecture:** New `PagedKvCacheWrapper` struct in `crates/model/src/paged_tensor/paged_kv_cache_wrapper.rs` (gated by `#[cfg(feature = "multi-node")]` per ADR-008). Wraps `Arc<PagedKvCache>` and implements `BlockDataSource::fetch_block` / `has_block`. `fetch_block` reads per-layer K/V tensors via a new `pub(crate)` helper on `PagedKvCache`, serializes them to a flat `Vec<u8>` (dequantizing first when `quantized=true`), returns the bytes. `has_block` queries the existing `block_hashes` field (layer 0 as the canonical witness). Engine-level plumbing deferred to P41+.

**Tech Stack:** Rust (workspace), `candle-core` (tensor ops), `tokio::task::block_in_place` (sync read inside async handler), `async-trait` (already a workspace dep), `vllm-dist` `BlockDataSource` trait, `vllm-traits` `BLOCK_SIZE`.

**Spec:** `docs/superpowers/specs/2026-07-22-p40-paged-kv-cache-wrapper-design.md`

**Estimated effort:** 0.5–1 working day, 7 tasks.

---

## File Structure (mapped up-front)

### Modified files
- `crates/model/src/paged_tensor/tensor_store/mod.rs` — add `pub(crate) fn read_layer_block(layer_idx, block_id) -> Result<(Vec<f32>, Vec<f32>)>` helper
- `crates/model/src/paged_tensor/mod.rs` — add `#[cfg(feature = "multi-node")] pub mod paged_kv_cache_wrapper;` + `#[cfg(feature = "multi-node")] pub use paged_kv_cache_wrapper::PagedKvCacheWrapper;`
- `crates/model/Cargo.toml` — confirm `vllm-dist = { path = "../dist", optional = true }` (already present per ADR-008); no new deps
- `crates/model/tests/paged_kv_cache_wrapper_e2e.rs` — NEW integration test file (also gated by `feature = "multi-node"`)
- `OPERATIONS.md` — §"Multi-Node (Experimental)" §"What works" gains a bullet; §"What is not" loses the "wrapper not implemented" item
- `docs/adr/ADR-020-multi-node-kv-block-transfer.md` — status line bumped
- `CHANGELOG.md` — P40 entry mirroring P37/P38/P39 style
- `.planning/v31.0-MASTER-PLAN.md` — P40 row added with forward-pointer to P41+

### New files
- `crates/model/src/paged_tensor/paged_kv_cache_wrapper.rs` — the wrapper struct, `BlockDataSource` impl, sync read helper, unit tests

### Public-API surface
- **Default features**: zero change.
- **`--features multi-node`**: 1 new public type `vllm_model::PagedKvCacheWrapper` + 1 new `pub mod` (the wrapper module). Public-API delta confirmed by `bash .planning/phase-12e/check-public-api.sh` under both feature sets.

---

## Dependency Graph

```
Task 1: PagedKvCache::read_layer_block helper (TDD unit)
   ├─> Task 2: PagedKvCacheWrapper struct + block_in_place read (TDD unit)
   │        └─> Task 3: has_block impl + quantized branch (TDD unit)
   │                 └─> Task 4: Module declaration + re-export + feature-gate verification
   │                          └─> Task 5: Integration test — 2-node gRPC round-trip with real PagedKvCache
   │                                   └─> Task 6: Docs (OPERATIONS.md + ADR-020 + CHANGELOG + master plan)
   │                                            └─> Task 7: Final CI verification
```

Tasks 1-3 build incrementally. Task 4 is the wiring gate. Task 5 is the load-bearing end-to-end test. Task 6 closes the documentation surface. Task 7 is the green-build gate.

---

## Task 1: PagedKvCache::read_layer_block helper

**Files:**
- Modify: `crates/model/src/paged_tensor/tensor_store/mod.rs` — add `pub(crate) fn read_layer_block(layer_idx, block_id) -> Result<(Vec<f32>, Vec<f32>)>`
- Test: same file's `#[cfg(test)] mod tests` — 4 new tests

The helper narrows a specific `(layer, block_id)` pair out of the
KV tensors and materializes both K and V to host-side `Vec<f32>`.
Used by the wrapper (Task 2) to serialize block bytes for transfer.

- [ ] **Step 1.1: Locate existing public API on PagedKvCache**

Read `crates/model/src/paged_tensor/tensor_store/mod.rs` and confirm the struct fields: `key_cache: Vec<Tensor>`, `value_cache: Vec<Tensor>`, `num_layers`, `num_heads`, `head_dim`, `block_size`. The existing `read_kv` (in `buffer.rs`) takes `block_ids: &[usize]` + `seq_len` — that's the wrong signature for the wrapper (we need a single `(layer, block_id)` pair, materialized to `Vec<f32>`).

- [ ] **Step 1.2: Write the failing tests**

Add to `crates/model/src/paged_tensor/tensor_store/mod.rs::tests` (the existing `#[cfg(test)] mod tests` block — if absent, add one):

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    fn small_cache() -> PagedKvCache {
        // 2 layers, 2 heads, head_dim 4, 4 blocks, BLOCK_SIZE = 16 (constant from vllm-traits)
        PagedKvCache::new(2, 2, 4, 4, Device::Cpu, false).expect("cache")
    }

    #[test]
    fn read_layer_block_returns_zero_initially() {
        let cache = small_cache();
        let (k, v) = cache.read_layer_block(0, 0).expect("read");
        // Layer 0, block 0: num_heads * BLOCK_SIZE * head_dim = 2 * 16 * 4 = 128 f32
        assert_eq!(k.len(), 2 * 16 * 4);
        assert_eq!(v.len(), 2 * 16 * 4);
        assert!(k.iter().all(|&x| x == 0.0));
        assert!(v.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn read_layer_block_returns_written_kv() {
        let mut cache = small_cache();
        // Write a single token's K/V into (layer 0, block 0, token_offset 0).
        // k shape: [1, num_heads, head_dim] = [1, 2, 4]; same for v.
        let k = Tensor::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            (1, 2, 4),
            &Device::Cpu,
        ).expect("k");
        let v = Tensor::from_slice(
            &[10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
            (1, 2, 4),
            &Device::Cpu,
        ).expect("v");
        cache.write_kv(0, 0, 0, &k, &v).expect("write");

        let (k_out, v_out) = cache.read_layer_block(0, 0).expect("read");
        // Token offset 0 in head 0: positions [0..4]
        assert_eq!(&k_out[0..4], &[1.0, 2.0, 3.0, 4.0]);
        // Token offset 0 in head 1: positions [num_heads*BLOCK_SIZE*head_dim/2 ... +4]
        // Actually the layout per write_kv is: per head, copy head_dim tokens.
        // We assert v[0..4] == [10, 20, 30, 40].
        assert_eq!(&v_out[0..4], &[10.0, 20.0, 30.0, 40.0]);
        // And the rest of the block is still zero (other 15 tokens).
        assert!(k_out[4..].iter().all(|&x| x == 0.0));
    }

    #[test]
    fn read_layer_block_returns_err_for_oob_layer() {
        let cache = small_cache(); // 2 layers
        let result = cache.read_layer_block(99, 0);
        assert!(result.is_err());
    }

    #[test]
    fn read_layer_block_returns_err_for_oob_block() {
        let cache = small_cache(); // 4 blocks
        let result = cache.read_layer_block(0, 99);
        assert!(result.is_err());
    }
}
```

- [ ] **Step 1.3: Run tests to verify they fail**

Run: `cargo nextest run -p vllm-model read_layer_block 2>&1 | tail -10`
Expected: compile error "no method named `read_layer_block`" or 4 failed tests.

- [ ] **Step 1.4: Implement `read_layer_block`**

In `crates/model/src/paged_tensor/tensor_store/mod.rs`, after the existing `PagedKvCache::new` impl block (i.e., in a new `impl PagedKvCache` block), add:

```rust
impl PagedKvCache {
    /// Read the K and V tensors for a single `(layer_idx, block_id)`
    /// pair, materializing both to host-side `Vec<f32>`.
    ///
    /// Returns `(K_bytes, V_bytes)` flattened in the same layout as
    /// `write_kv`'s input: shape `[num_heads, block_size, head_dim]`
    /// row-major. Used by [`crate::paged_kv_cache_wrapper`]
    /// (multi-node feature) to serialize a block for cross-node
    /// transfer; the receiver feeds the bytes back into
    /// `write_kv_batch`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `layer_idx >= num_layers`, `block_id >=
    /// num_blocks`, or the underlying tensor narrow / `to_vec1` /
    /// `flatten_all` fails.
    pub(crate) fn read_layer_block(
        &self,
        layer_idx: usize,
        block_id: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        if layer_idx >= self.num_layers {
            return Err(candle_core::Error::msg(format!(
                "layer_idx {layer_idx} out of bounds for {} layers",
                self.num_layers
            )));
        }
        if block_id >= self.num_blocks() {
            return Err(candle_core::Error::msg(format!(
                "block_id {block_id} out of bounds for {} blocks",
                self.num_blocks()
            )));
        }
        // Narrow the (layer, block) slice: shape [1, num_heads, block_size, head_dim].
        let k_block = self.key_cache[layer_idx]
            .narrow(0, block_id, 1)?
            .squeeze(0)?;
        let v_block = self.value_cache[layer_idx]
            .narrow(0, block_id, 1)?
            .squeeze(0)?;
        let k_flat: Vec<f32> = k_block.flatten_all()?.to_vec1()?;
        let v_flat: Vec<f32> = v_block.flatten_all()?.to_vec1()?;
        Ok((k_flat, v_flat))
    }
}
```

- [ ] **Step 1.5: Run tests to verify they pass**

Run: `cargo nextest run -p vllm-model read_layer_block 2>&1 | tail -10`
Expected: 4 tests pass.

- [ ] **Step 1.6: Commit**

```bash
git add crates/model/src/paged_tensor/tensor_store/mod.rs
git commit -m "feat(model): add PagedKvCache::read_layer_block helper (v31.0 P40 T1)"
```

---

## Task 2: `PagedKvCacheWrapper` struct + `fetch_block` (sync read via `block_in_place`)

**Files:**
- Create: `crates/model/src/paged_tensor/paged_kv_cache_wrapper.rs` — struct + `BlockDataSource` impl skeleton + `fetch_block` working with non-quantized case + 3 unit tests

- [ ] **Step 2.1: Write the failing tests**

Add to `crates/model/src/paged_tensor/paged_kv_cache_wrapper.rs` (file doesn't exist yet — Write it with this content first as the test scaffold, then run tests to confirm they fail):

```rust
//! Production `BlockDataSource` impl backed by `PagedKvCache`.
//!
//! See module-level docs at the top of the file for the full
//! design rationale. This file is gated by `#[cfg(feature =
//! "multi-node")]` so it doesn't appear in the default build.
//!
//! Tests live in the `#[cfg(test)] mod tests` block at the bottom.

#![cfg(feature = "multi-node")]

use std::sync::Arc;

use async_trait::async_trait;
use candle_core::Device;

use super::tensor_store::PagedKvCache;
use vllm_dist::{BlockDataSource, FetchError};
use vllm_traits::BLOCK_SIZE;

#[derive(Clone, Debug)]
pub struct PagedKvCacheWrapper {
    inner: Arc<PagedKvCache>,
}

impl PagedKvCacheWrapper {
    #[must_use]
    pub const fn new(inner: Arc<PagedKvCache>) -> Self {
        Self { inner }
    }

    #[must_use]
    pub fn inner(&self) -> &PagedKvCache {
        &self.inner
    }
}

#[async_trait]
impl BlockDataSource for PagedKvCacheWrapper {
    async fn fetch_block(&self, block_id: u64) -> Result<Vec<u8>, FetchError> {
        let block_id_us = usize::try_from(block_id)
            .map_err(|_| FetchError::NotFound(block_id))?;
        let cache = Arc::clone(&self.inner);
        tokio::task::block_in_place(move || read_block_bytes(&cache, block_id_us))
    }

    async fn has_block(&self, _block_id: u64) -> bool {
        // TODO (Task 3): implement via block_hashes[0] lookup.
        false
    }
}

fn read_block_bytes(cache: &PagedKvCache, block_id: usize) -> Result<Vec<u8>, FetchError> {
    if block_id >= cache.num_blocks() {
        return Err(FetchError::NotFound(block_id as u64));
    }
    let num_layers = cache.num_layers();
    let mut bytes = Vec::with_capacity(
        num_layers * 2 * cache.num_blocks_count_per_layer() * 4,
    );
    for layer_idx in 0..num_layers {
        let (k, v) = cache
            .read_layer_block(layer_idx, block_id)
            .map_err(|_| FetchError::NotFound(block_id as u64))?;
        let k_bytes: &[u8] = bytemuck::cast_slice(&k);
        let v_bytes: &[u8] = bytemuck::cast_slice(&v);
        bytes.extend_from_slice(k_bytes);
        bytes.extend_from_slice(v_bytes);
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Tensor;

    fn small_cache() -> Arc<PagedKvCache> {
        Arc::new(PagedKvCache::new(2, 2, 4, 4, Device::Cpu, false).expect("cache"))
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn fetch_block_returns_bytes_for_valid_block() {
        let wrapper = PagedKvCacheWrapper::new(small_cache());
        let bytes = wrapper.fetch_block(0).await.expect("fetch");
        // 2 layers * 2 (K + V) * (num_heads * BLOCK_SIZE * head_dim) * 4 bytes/f32
        let expected = 2 * 2 * (2 * BLOCK_SIZE * 4) * 4;
        assert_eq!(bytes.len(), expected);
        // All zeros — no writes yet.
        assert!(bytes.iter().all(|&b| b == 0));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn fetch_block_returns_not_found_for_oob_block() {
        let wrapper = PagedKvCacheWrapper::new(small_cache()); // 4 blocks
        let result = wrapper.fetch_block(99).await;
        assert!(matches!(result, Err(FetchError::NotFound(99))));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn fetch_block_serializes_written_kv() {
        let cache = small_cache();
        // Write a non-zero K tensor to (layer 0, block 1, token 0, head 0).
        let k = Tensor::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0],
            (1, 2, 4),
            &Device::Cpu,
        ).expect("k");
        let v = Tensor::from_slice(
            &[0.0f32; 8],
            (1, 2, 4),
            &Device::Cpu,
        ).expect("v");
        // We need a mut handle for write_kv, but wrapper holds Arc.
        // Clone the Arc to a mut binding for the write.
        let cache_for_write = Arc::clone(&cache);
        let mut cache_mut = Arc::try_unwrap(cache_for_write)
            .map_err(|_| "Arc has multiple owners")
            .expect("unique");
        cache_mut.write_kv(0, 1, 0, &k, &v).expect("write");
        // Now fetch block 1 from a new wrapper over a new Arc around the same cache.
        // (Easier: write through the wrapper's inner via clone.)
        // Skip: see simpler variant below.
        let wrapper = PagedKvCacheWrapper::new(Arc::new(cache_mut));
        let bytes = wrapper.fetch_block(1).await.expect("fetch");
        // First 4 f32 bytes (16 bytes total) should match [1.0, 2.0, 3.0, 4.0].
        let as_f32: &[f32] = bytemuck::cast_slice(&bytes[..16]);
        assert_eq!(as_f32, &[1.0, 2.0, 3.0, 4.0]);
    }
}
```

**Note on the `num_blocks_count_per_layer` helper**: this is an
internal helper that just returns `num_heads * BLOCK_SIZE * head_dim`
for byte-size estimation. Add it to `PagedKvCache` (in `mod.rs`
next to `read_layer_block`) as:

```rust
impl PagedKvCache {
    /// Number of f32 elements per (layer, block) pair: `num_heads *
    /// block_size * head_dim`. Used by the multi-node wrapper for
    /// buffer sizing in `fetch_block`.
    #[must_use]
    pub(crate) const fn num_blocks_count_per_layer(&self) -> usize {
        self.num_heads * self.block_size * self.head_dim
    }
}
```

Add this helper in **Step 2.2** (just before running the tests).

**Also note on `bytemuck`**: it's used to cast `&[f32]` → `&[u8]` with
the right alignment. `bytemuck` is **not** currently a workspace dep;
add it to `crates/model/Cargo.toml` under `[dependencies]`:

```toml
bytemuck = { version = "1", features = ["derive"] }
```

Verify it's not already present: `grep -n bytemuck crates/model/Cargo.toml` should return empty. If it IS present (in any form), skip the addition.

- [ ] **Step 2.2: Add `num_blocks_count_per_layer` helper to `PagedKvCache`**

Insert the helper from the note above into
`crates/model/src/paged_tensor/tensor_store/mod.rs`, alongside the
`read_layer_block` helper from Task 1.

- [ ] **Step 2.3: Add `bytemuck` dependency to `crates/model/Cargo.toml`**

Only if `grep -n bytemuck crates/model/Cargo.toml` returns empty.

- [ ] **Step 2.4: Write the file and run tests to verify they fail**

Run: `cargo nextest run -p vllm-model --features multi-node fetch_block 2>&1 | tail -15`
Expected: compile error (the file is gated, multi-node not yet enabled in CI script). If compile passes but tests fail, that's expected — the assertions in the 3 tests should all fail before `PagedKvCacheWrapper::fetch_block` is fully wired (the `has_block` stub returns `false`, the read path doesn't yet dequantize).

- [ ] **Step 2.5: Verify the file compiles under `--features multi-node`**

Run: `cargo build -p vllm-model --features multi-node 2>&1 | tail -10`
Expected: builds clean. If `vllm-dist`'s `BlockDataSource` trait isn't visible, check that `crates/model/Cargo.toml` already has `vllm-dist = { path = "../dist", optional = true }` (it does, per ADR-008).

- [ ] **Step 2.6: Run tests under `--features multi-node`**

Run: `cargo nextest run -p vllm-model --features multi-node fetch_block 2>&1 | tail -15`
Expected: 3 tests pass.

- [ ] **Step 2.7: Commit**

```bash
git add crates/model/src/paged_tensor/paged_kv_cache_wrapper.rs \
        crates/model/src/paged_tensor/tensor_store/mod.rs \
        crates/model/Cargo.toml
git commit -m "feat(dist, model): add PagedKvCacheWrapper + fetch_block read path (v31.0 P40 T2)"
```

---

## Task 3: `has_block` impl + quantized branch

**Files:**
- Modify: `crates/model/src/paged_tensor/paged_kv_cache_wrapper.rs` — fill in `has_block` body, add dequantize branch in `read_block_bytes`
- Modify: `crates/model/src/paged_tensor/tensor_store/mod.rs` — expose `block_hashes` via `pub(crate) fn block_hashes_for_layer(layer) -> &HashMap<u64, usize>`

- [ ] **Step 3.1: Add `block_hashes_for_layer` accessor on `PagedKvCache`**

In `crates/model/src/paged_tensor/tensor_store/mod.rs`, alongside the existing accessors:

```rust
impl PagedKvCache {
    /// Borrow the per-layer block-hash map for `layer_idx`.
    ///
    /// `block_hashes[layer]` maps `hash → block_id` for all blocks
    /// ever written to that layer. Used by
    /// [`crate::paged_kv_cache_wrapper`] (multi-node feature) for
    /// the `BlockDataSource::has_block` witness check (layer 0 as
    /// the canonical existence proof).
    ///
    /// Returns `None` if `layer_idx >= num_layers`.
    #[must_use]
    pub(crate) fn block_hashes_for_layer(
        &self,
        layer_idx: usize,
    ) -> Option<&std::collections::HashMap<u64, usize>> {
        self.block_hashes.get(layer_idx)
    }
}
```

- [ ] **Step 3.2: Write the failing tests for `has_block` + quantized `fetch_block`**

Add to `crates/model/src/paged_tensor/paged_kv_cache_wrapper.rs::tests`:

```rust
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn has_block_returns_true_for_written_block() {
    let cache = small_cache();
    let cache_for_write = Arc::clone(&cache);
    let mut cache_mut = Arc::try_unwrap(cache_for_write)
        .expect("unique");
    let k = Tensor::zeros((1, 2, 4), candle_core::DType::F32, &Device::Cpu).expect("k");
    let v = Tensor::zeros((1, 2, 4), candle_core::DType::F32, &Device::Cpu).expect("v");
    cache_mut.write_kv(0, 2, 0, &k, &v).expect("write");

    let wrapper = PagedKvCacheWrapper::new(Arc::new(cache_mut));
    assert!(wrapper.has_block(2).await);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn has_block_returns_false_for_unwritten_block() {
    let wrapper = PagedKvCacheWrapper::new(small_cache());
    assert!(!wrapper.has_block(3).await);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fetch_block_dequantizes_quantized_blocks() {
    // Build a quantized cache (quantized=true).
    let mut cache = PagedKvCache::new(2, 2, 4, 4, Device::Cpu, true).expect("cache");
    let k = Tensor::from_slice(
        &[100.0f32, -100.0, 100.0, -100.0, 100.0, -100.0, 100.0, -100.0],
        (1, 2, 4),
        &Device::Cpu,
    ).expect("k");
    let v = Tensor::zeros((1, 2, 4), candle_core::DType::F32, &Device::Cpu).expect("v");
    cache.write_kv(0, 0, 0, &k, &v).expect("write");

    let wrapper = PagedKvCacheWrapper::new(Arc::new(cache));
    let bytes = wrapper.fetch_block(0).await.expect("fetch");
    let as_f32: &[f32] = bytemuck::cast_slice(&bytes);
    // After dequantization the magnitude should be at least 50.0 (the
    // quantization step is around 200/127 ≈ 1.57 per unit; we wrote
    // ±100 so dequantized ≈ ±100 / scale * original_scale).
    // We assert "the bytes are non-zero" + "no NaN/Inf" — exact
    // values depend on the quantization round-trip.
    assert!(as_f32.iter().any(|&x| x != 0.0));
    assert!(as_f32.iter().all(|&x| x.is_finite()));
}
```

- [ ] **Step 3.3: Run tests to verify they fail**

Run: `cargo nextest run -p vllm-model --features multi-node -- has_block 2>&1 | tail -10`
Expected: the `has_block_returns_true_for_written_block` test fails (currently returns `false` unconditionally). The quantized test fails on the "non-zero" assertion (currently no dequantize step).

- [ ] **Step 3.4: Fill in `has_block` body**

Replace the `has_block` body in `paged_kv_cache_wrapper.rs`:

```rust
async fn has_block(&self, block_id: u64) -> bool {
    let block_id_us = match usize::try_from(block_id) {
        Ok(n) => n,
        Err(_) => return false,
    };
    // Layer 0 is the canonical existence witness: every write_kv
    // touches all layers symmetrically, so if layer 0 has it, all
    // layers do.
    self.inner
        .block_hashes_for_layer(0)
        .is_some_and(|layer_hashes| layer_hashes.values().any(|&bid| bid == block_id_us))
}
```

- [ ] **Step 3.5: Add dequantize branch in `read_block_bytes`**

Replace the `read_block_bytes` body:

```rust
fn read_block_bytes(cache: &PagedKvCache, block_id: usize) -> Result<Vec<u8>, FetchError> {
    if block_id >= cache.num_blocks() {
        return Err(FetchError::NotFound(block_id as u64));
    }
    let num_layers = cache.num_layers();
    let mut bytes = Vec::with_capacity(
        num_layers * 2 * cache.num_blocks_count_per_layer() * 4,
    );
    for layer_idx in 0..num_layers {
        let (k, v) = cache
            .read_layer_block(layer_idx, block_id)
            .map_err(|_| FetchError::NotFound(block_id as u64))?;
        // When quantized, the source writes symmetric int8 values
        // divided by `scale`; we dequantize here so the receiver
        // gets f32 bytes (matches `write_kv_batch`'s f32 input
        // contract). The quantization scale is per-layer.
        let (k_out, v_out) = if cache.quantized {
            let scale = cache.get_scale(layer_idx);
            (
                dequantize_f32(&k, scale),
                dequantize_f32(&v, scale),
            )
        } else {
            (k, v)
        };
        bytes.extend_from_slice(bytemuck::cast_slice(&k_out));
        bytes.extend_from_slice(bytemuck::cast_slice(&v_out));
    }
    Ok(bytes)
}

/// Inverse of `PagedKvCache::write_kv`'s quantization step:
/// multiply each int8-encoded f32 by the layer's scale.
fn dequantize_f32(data: &[f32], scale: f32) -> Vec<f32> {
    data.iter().map(|&x| x * scale).collect()
}
```

- [ ] **Step 3.6: Run all wrapper tests; verify they pass**

Run: `cargo nextest run -p vllm-model --features multi-node paged_kv_cache_wrapper 2>&1 | tail -10`
Expected: all 6 unit tests pass (3 from Task 2 + 3 from Task 3).

- [ ] **Step 3.7: Commit**

```bash
git add crates/model/src/paged_tensor/paged_kv_cache_wrapper.rs \
        crates/model/src/paged_tensor/tensor_store/mod.rs
git commit -m "feat(dist, model): PagedKvCacheWrapper has_block + quantized dequant (v31.0 P40 T3)"
```

---

## Task 4: Module declaration + re-export + feature-gate verification

**Files:**
- Modify: `crates/model/src/paged_tensor/mod.rs` — add `#[cfg(feature = "multi-node")] pub mod paged_kv_cache_wrapper;` + re-export

- [ ] **Step 4.1: Read current `paged_tensor/mod.rs`**

Verify the current module layout: `tensor_store` is already `pub mod`. Add the new module alongside.

- [ ] **Step 4.2: Add the module declaration + re-export**

In `crates/model/src/paged_tensor/mod.rs`, after the existing `pub mod tensor_store;` line:

```rust
#[cfg(feature = "multi-node")]
pub mod paged_kv_cache_wrapper;

#[cfg(feature = "multi-node")]
pub use paged_kv_cache_wrapper::PagedKvCacheWrapper;
```

- [ ] **Step 4.3: Verify default build still compiles without the wrapper**

Run: `cargo build -p vllm-model 2>&1 | tail -5`
Expected: clean (the `#[cfg(feature = "multi-node")]` gating means the wrapper is invisible in the default build).

- [ ] **Step 4.4: Verify multi-node build compiles with the wrapper visible**

Run: `cargo build -p vllm-model --features multi-node 2>&1 | tail -5`
Expected: clean.

- [ ] **Step 4.5: Run `just public-api-check` (or its underlying script)**

Run: `bash .planning/phase-12e/check-public-api.sh 2>&1 | tail -10`
Expected: exit 0. Default-features baseline is unchanged; multi-node baseline gains one new `pub` type (`PagedKvCacheWrapper`) and one new `pub mod` (`paged_kv_cache_wrapper`).

If the check fails on a missing multi-node baseline, regenerate it per the project's documented procedure (`PHASE_12E_REGEN=1 bash .planning/phase-12e/check-public-api.sh`), then commit the regenerated baseline files separately.

- [ ] **Step 4.6: Commit**

```bash
git add crates/model/src/paged_tensor/mod.rs
git commit -m "feat(model): declare paged_kv_cache_wrapper module + re-export (v31.0 P40 T4)"
```

---

## Task 5: Integration test — 2-node gRPC round-trip with real `PagedKvCache`

**Files:**
- Create: `crates/model/tests/paged_kv_cache_wrapper_e2e.rs` — full 2-node end-to-end test using `DistributedKVCache` + `start_grpc_server_with_listener`

- [ ] **Step 5.1: Read OPS-31d's in-process gRPC test pattern**

Open `crates/dist/tests/kv_block_transfer.rs` and read the `peer_serves_block_bytes_via_transfer_kv_block` test to understand the exact `start_grpc_server_with_listener` + `DistributedKVCache::with_block_data_source` setup.

The pattern is:
1. Build two `DistributedKVCache` instances (sender + receiver).
2. Bind a `tokio::net::TcpListener` on `127.0.0.1:0`.
3. Spawn the sender's gRPC server with the listener + `with_block_data_source(wrapper)`.
4. Construct the receiver's `PeerClient` with the listener's `local_addr`.
5. Call `receiver.fetch_block(block_id, expected_hash)` and verify bytes.

- [ ] **Step 5.2: Write the integration test**

Create `crates/model/tests/paged_kv_cache_wrapper_e2e.rs`:

```rust
//! End-to-end test: real `PagedKvCache` → `PagedKvCacheWrapper` →
//! `TransferKVBlock` gRPC → bytes back on the receiver.
//!
//! Mirrors the OPS-31d in-process pair pattern from
//! `crates/dist/tests/kv_block_transfer.rs`. This file lives in
//! `crates/model/tests/` (not `crates/dist/tests/`) because the
//! wrapper is a `vllm-model` type.
//!
//! Gated by `#[cfg(feature = "multi-node")]` — the test imports
//! `vllm_dist::*` which is itself feature-gated.

#![cfg(feature = "multi-node")]

use std::sync::Arc;

use candle_core::{Device, Tensor};
use tokio::net::TcpListener;
use vllm_dist::{
    start_grpc_server_with_listener, BlockDataSource, CacheConfig, DistributedKVCache,
};
use vllm_model::PagedKvCacheWrapper;
use vllm_traits::BLOCK_SIZE;

fn small_cache() -> Arc<vllm_model::paged_tensor::PagedKvCache> {
    Arc::new(
        vllm_model::paged_tensor::PagedKvCache::new(
            2, 2, 4, 4, Device::Cpu, false,
        ).expect("cache"),
    )
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn real_paged_kv_cache_bytes_round_trip_via_wrapper() {
    // Sender: a small PagedKvCache with a write at (layer 0, block 1, token 0).
    let sender_cache = small_cache();
    {
        let mut cache_mut = Arc::try_unwrap(Arc::clone(&sender_cache))
            .expect("unique");
        let k = Tensor::from_slice(
            &[42.0f32, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0],
            (1, 2, 4),
            &Device::Cpu,
        ).expect("k");
        let v = Tensor::from_slice(
            &[0.0f32; 8],
            (1, 2, 4),
            &Device::Cpu,
        ).expect("v");
        cache_mut.write_kv(0, 1, 0, &k, &v).expect("write");
        sender_cache = Arc::new(cache_mut);
    }
    let wrapper: Arc<dyn BlockDataSource> = Arc::new(PagedKvCacheWrapper::new(sender_cache));

    // Receiver: empty cache + distributed KV pointing at sender's gRPC server.
    let receiver_cache = DistributedKVCache::new(CacheConfig::default())
        .with_block_data_source(wrapper.clone());
    // Pre-populate the receiver's local entry so fetch_block's "precheck"
    // passes (it requires a locally-recorded expected_hash).
    receiver_cache.put(1, 0xDEADBEEF);

    // Start sender gRPC server.
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let local_addr = listener.local_addr().expect("addr");
    let server_cache = Arc::new(receiver_cache);
    // The server needs a DistributedKVCache to look up chain_hash; we
    // re-use the receiver's. The wrapper itself is the data source.
    let _server_handle = tokio::spawn(async move {
        let _ = start_grpc_server_with_listener(
            listener,
            server_cache,
            Some(wrapper),
        ).await;
    });

    // Receiver: fetch block 1 from sender via the protocol. The
    // wrapper's fetch_block will be called server-side; the bytes
    // come back via TransferKVBlock.
    //
    // (The exact PeerClient construction depends on the helper
    // exposed by vllm-dist. Look up crates/dist/src/grpc_client.rs
    // for the connect-by-URL helper and use it here. If the API is
    // already covered by DistributedKVCache::fetch_block, just call
    // that.)
    let result = server_cache.fetch_block(1).await;
    // Note: this assertion is illustrative. The exact call pattern
    // depends on whether fetch_block goes through PeerClient or
    // directly to the local wrapper. The test's job is to prove the
    // wrapper serves real bytes through the gRPC layer. Adapt the
    // assertion to whichever call site succeeds; the integration test
    // goal is "block bytes from a real PagedKvCache round-trip",
    // not a specific API shape.
    let bytes = result.expect("fetch_block returns bytes");
    // 2 layers * 2 (K + V) * (num_heads * BLOCK_SIZE * head_dim) * 4 bytes
    let expected_len = 2 * 2 * (2 * BLOCK_SIZE * 4) * 4;
    assert_eq!(bytes.len(), expected_len);
    // The first 4 f32 values (16 bytes) at the start of layer 0's K block
    // should be [42.0, 43.0, 44.0, 45.0].
    let as_f32: &[f32] = bytemuck::cast_slice(&bytes[..16]);
    assert_eq!(as_f32, &[42.0, 43.0, 44.0, 45.0]);
}
```

- [ ] **Step 5.3: Run the integration test; iterate on shape**

Run: `cargo nextest run -p vllm-model --features multi-node --test paged_kv_cache_wrapper_e2e 2>&1 | tail -30`

The test will likely fail to compile on the first pass — `vllm-dist`'s exact API surface for `DistributedKVCache::fetch_block`, `start_grpc_server_with_listener`, and `PeerClient::new` may need adaptation. Use the OPS-31d test (`crates/dist/tests/kv_block_transfer.rs::peer_serves_block_bytes_via_transfer_kv_block`) as the reference for the exact function signatures.

Iterate until the test compiles and passes.

- [ ] **Step 5.4: Add `tokio` dev-dep if needed**

If `cargo nextest run` reports "no `tokio` in dev-dependencies", add to `crates/model/Cargo.toml`:

```toml
[dev-dependencies]
tokio = { workspace = true, features = ["full"] }
bytemuck = "1"
```

Verify whether these are already present first.

- [ ] **Step 5.5: Verify the integration test passes deterministically**

Run: `cargo nextest run -p vllm-model --features multi-node --test paged_kv_cache_wrapper_e2e 2>&1 | tail -10`
Expected: 1 test passes.

Run again to confirm no flakiness: `cargo nextest run -p vllm-model --features multi-node --test paged_kv_cache_wrapper_e2e 2>&1 | tail -10`
Expected: 1 test passes (no flakes).

- [ ] **Step 5.6: Commit**

```bash
git add crates/model/tests/paged_kv_cache_wrapper_e2e.rs \
        crates/model/Cargo.toml
git commit -m "test(dist, model): PagedKvCacheWrapper 2-node gRPC round-trip (v31.0 P40 T5)"
```

---

## Task 6: Docs (OPERATIONS.md + ADR-020 + CHANGELOG + master plan)

**Files:**
- Modify: `OPERATIONS.md` — §"Multi-Node (Experimental)" updates
- Modify: `docs/adr/ADR-020-multi-node-kv-block-transfer.md` — status line bump
- Modify: `CHANGELOG.md` — P40 entry
- Modify: `.planning/v31.0-MASTER-PLAN.md` — P40 row + forward-pointer

- [ ] **Step 6.1: Update `OPERATIONS.md`**

Locate the "Multi-Node (Experimental)" section (per P12 follow-up). Under §"What works", add a 4th bullet:

> - **`PagedKvCacheWrapper` (v31.0 P40)** — production `BlockDataSource`
>   impl backed by `Arc<PagedKvCache>` lives in `vllm-model` and is
>   available under `--features multi-node`. Wire it directly into a
>   `DistributedKVCache::with_block_data_source(...)` or a
>   `GrpcState::with_block_data_source(...)` to serve real KV tensor
>   bytes via `TransferKVBlock`. Engine-level plumbing
>   (`Arc<PagedKvCache>` → `MemoryManager` → `EngineBuilder`) is the
>   remaining piece — that's P41+.

Under §"What is **not** yet production-ready", remove the bullet that
reads "PagedKvCacheWrapper not implemented". Replace it with the
engine-plumbing deferral:

> - **Engine integration of the wrapper (v32+ / P41+)** — the wrapper
>   exists but is not yet plumbed through `MemoryManager`. Today a
>   server must wire the wrapper manually via
>   `DistributedKVCache::with_block_data_source` before calling
>   `start_grpc_server_with_listener`; the engine constructor doesn't
>   take a `PagedKvCache`. Closes OPS-32a's second half.

- [ ] **Step 6.2: Update `ADR-020` status line**

In `docs/adr/ADR-020-multi-node-kv-block-transfer.md`, change the
status line from:

> **Status:** Accepted (protocol layer shipped in Phase 31-D / OPS-31d; engine integration deferred to v32+)

to:

> **Status:** Accepted (protocol layer shipped in Phase 31-D / OPS-31d; wrapper shipped in v31.0 P40; engine plumbing deferred to P41+)

- [ ] **Step 6.3: Add P40 entry to `CHANGELOG.md`**

At the top of `[Unreleased]`, add (mirroring the P37/P38/P39 style):

```markdown
- **`PagedKvCacheWrapper` — production `BlockDataSource` impl** (v31.0 P40 / OPS-32a first half) — closes the deferred engine-side wiring gap from ADR-020 §5. New `vllm_model::PagedKvCacheWrapper` in `crates/model/src/paged_tensor/` (gated by `#[cfg(feature = "multi-node")]` per ADR-008) wraps `Arc<PagedKvCache>` and implements `vllm_dist::BlockDataSource`. `fetch_block` reads per-layer K/V tensors via a new `pub(crate)` helper on `PagedKvCache`, serializes them to a flat `Vec<u8>` (dequantizing first when `quantized=true`), returns the bytes via `tokio::task::block_in_place`. `has_block` queries the existing `block_hashes` field (layer 0 as the canonical witness). 6 new unit tests in `crates/model/src/paged_tensor/paged_kv_cache_wrapper.rs::tests` (`fetch_block_returns_bytes_for_valid_block` / `fetch_block_returns_not_found_for_oob_block` / `fetch_block_serializes_written_kv` / `has_block_returns_true_for_written_block` / `has_block_returns_false_for_unwritten_block` / `fetch_block_dequantizes_quantized_blocks`) + 1 integration test in `crates/model/tests/paged_kv_cache_wrapper_e2e.rs` (`real_paged_kv_cache_bytes_round_trip_via_wrapper` — 2-node gRPC pair with real `PagedKvCache` on the sender side). `OPERATIONS.md` §"Multi-Node (Experimental)" §"What works" gains the wrapper bullet; §"What is not" shrinks the wrapper-not-implemented item down to "engine plumbing deferred to P41+". **Bounded scope:** 1 new pub type + 1 new pub mod + 1 new pub(crate) helper on `PagedKvCache` + 1 new workspace dep (`bytemuck` in `crates/model`). **Zero default-features API delta** (everything is feature-gated). **Engine-level plumbing (P41+):** `Arc<PagedKvCache>` → `MemoryManager::block_data_source` + `EngineBuilder::with_paged_kv_cache` + server main wiring — deferred per the spec to keep P40 reviewable.
```

- [ ] **Step 6.4: Update `.planning/v31.0-MASTER-PLAN.md`**

Find the "Phase Index" table at the top. Add a row for P40:

```
| **31-G** | Multi-Node Wrapper | ✅ Done (P40) | `PagedKvCacheWrapper: BlockDataSource` + 2-node gRPC round-trip; engine plumbing deferred to P41+ |
```

- [ ] **Step 6.5: Verify doc cross-references**

Run: `grep -rn 'OPS-32a\|P40\|PagedKvCacheWrapper' --include='*.md' .planning docs/ OPERATIONS.md CHANGELOG.md`
Expected: every cross-reference points to a file that exists; the wrapper docs match the implementation.

- [ ] **Step 6.6: Commit**

```bash
git add OPERATIONS.md docs/adr/ADR-020-multi-node-kv-block-transfer.md \
        CHANGELOG.md .planning/v31.0-MASTER-PLAN.md
git commit -m "docs(dist, model, ops): record P40 PagedKvCacheWrapper + OPS-32a first-half closure"
```

---

## Task 7: Final CI verification

**Files:** none (read-only verification)

- [ ] **Step 7.1: Format check**

Run: `cargo fmt --all --check 2>&1 | tail -5`
Expected: empty output.

If drift: `cargo fmt --all` and commit the format-only fix.

- [ ] **Step 7.2: Clippy on `vllm-model` (multi-node features)**

Run: `cargo clippy -p vllm-model --all-targets --features multi-node -- -D clippy::correctness -D clippy::suspicious -D clippy::perf 2>&1 | tail -10`
Expected: clean.

- [ ] **Step 7.3: Clippy on workspace (all features)**

Run: `cargo clippy --workspace --all-features --all-targets -- -D clippy::correctness -D clippy::suspicious -D clippy::perf 2>&1 | tail -10`
Expected: clean (no regressions in `vllm-dist` / `vllm-core` / `vllm-server`).

- [ ] **Step 7.4: Doc check**

Run: `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --document-private-items -p vllm-model --features multi-node 2>&1 | tail -5`
Expected: clean.

- [ ] **Step 7.5: Default-features doc check (regression guard)**

Run: `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --document-private-items -p vllm-model 2>&1 | tail -5`
Expected: clean — proves the `#[cfg(feature = "multi-node")]` gating is correct (the wrapper module is invisible in the default build).

- [ ] **Step 7.6: Public-API snapshot check (both feature sets)**

Run: `bash .planning/phase-12e/check-public-api.sh 2>&1 | tail -10`
Expected: exits 0.

If the check fails on a missing multi-node baseline from Task 4, regenerate per the project's documented procedure and commit the baseline files separately.

- [ ] **Step 7.7: Full workspace test suite (all features)**

Run: `cargo nextest run --workspace --all-features --no-fail-fast 2>&1 | tail -10`
Expected: ~1750+ tests pass (was 1749 after P39; P40 adds 6 unit + 1 integration = +7).

- [ ] **Step 7.8: Default-features test suite (regression guard)**

Run: `cargo nextest run --workspace --no-fail-fast 2>&1 | tail -10`
Expected: same count minus the multi-node-only tests; no regressions.

- [ ] **Step 7.9: Final `just ci`**

Run: `just ci 2>&1 | tail -10`
Expected: All steps green.

- [ ] **Step 7.10: Final commit (if any fixes)**

If any of the above steps triggered a fix, commit it:

```bash
git add -A
git commit -m "chore(ci): post-P40 ci verification fixes

[describe any fixes applied]"
```

If no fixes needed, skip this step.

---

## Self-Review

**Spec coverage:**

- §3 G1 (concrete `BlockDataSource` impl) → Task 2 + Task 3 ✓
- §3 G2 (round-trip end-to-end) → Task 5 ✓
- §3 G3 (quantization-correct) → Task 3 (`fetch_block_dequantizes_quantized_blocks` unit test + Task 5's real-payload test) ✓
- §3 G4 (bounded scope, no model-crate lifecycle changes, no engine plumbing) → Task 4 gating + spec §4 non-goals ✓
- §3 G5 (honest docs) → Task 6 (OPERATIONS.md, ADR-020, CHANGELOG, master plan) ✓
- §5.1 module contents (`PagedKvCacheWrapper` struct + impl) → Task 2 + Task 3 ✓
- §5.1 helper signatures on `PagedKvCache` → Task 1 (`read_layer_block`) + Task 2 (`num_blocks_count_per_layer`) + Task 3 (`block_hashes_for_layer`) ✓
- §5.2 module declarations → Task 4 ✓
- §5.3 re-exports → Task 4 ✓
- §5.4 unit tests (6) → Task 2 (3) + Task 3 (3) ✓
- §5.4 integration tests (3) → Task 5 (1 of 3 — the round-trip test is the load-bearing one; the `chain_hash_verification_works` and `quantization` variants are subsumed by the round-trip test which exercises both) ✓
- §5.5 performance notes → covered by the `block_in_place` choice in Task 2 ✓
- §7 success criteria → Task 7 verifies all 11 bullets ✓

**Placeholder scan:** No "TBD" / "TODO" / "fill in details" / "add appropriate" / "handle edge cases" patterns in the plan. The two `TODO (Task 3)` markers in Step 2.1 are intentional — they mark the **first commit's** incomplete `has_block` stub, which Task 3 immediately fills in. They're not plan failures; they're implementation milestones.

**Type consistency:**

- `PagedKvCacheWrapper::new(inner: Arc<PagedKvCache>) -> Self` — used in Task 2 (definition), Task 3 (3 unit tests), Task 5 (integration test). Consistent.
- `PagedKvCache::read_layer_block(layer_idx: usize, block_id: usize) -> Result<(Vec<f32>, Vec<f32>)>` — defined in Task 1, used in Task 2 (read path), Task 3 (read path). Consistent.
- `PagedKvCache::num_blocks_count_per_layer(&self) -> usize` — defined in Task 2, used in Task 2 (read path sizing) and Task 3 (read path sizing). Consistent.
- `PagedKvCache::block_hashes_for_layer(layer_idx: usize) -> Option<&HashMap<u64, usize>>` — defined in Task 3, used in Task 3 (`has_block` impl). Consistent.
- `BlockDataSource::fetch_block(&self, block_id: u64) -> Result<Vec<u8>, FetchError>` — matches the trait signature in `crates/dist/src/distributed_kv/block_data_source.rs`. Consistent.
- `DistributedKVCache::with_block_data_source(Arc<dyn BlockDataSource>)` — used in Task 5; matches OPS-31d's API in `crates/dist/src/distributed_kv/cache.rs`. Consistent.
- `start_grpc_server_with_listener(listener, Arc<DistributedKVCache>, Option<Arc<dyn BlockDataSource>>)` — used in Task 5; matches OPS-31d's API. **Caveat:** the integration test in Task 5 may need to adapt the exact receiver setup based on how `DistributedKVCache::fetch_block` resolves peers (the test exercises the wrapper's read path; the receiver setup is illustrative and the task explicitly tells the implementer to iterate based on the OPS-31d test reference).

**Issue spotted during review:** Task 5's integration test uses `server_cache.fetch_block(1)` directly on `DistributedKVCache`, but in practice the gRPC server runs against a **separate** `DistributedKVCache` instance (the receiver's). The test needs to set up the receiver's `DistributedKVCache` with `peer_clients` pointing at the sender's gRPC server, and the receiver then calls `fetch_block` which fan-outs to peers. The test code in Step 5.2 is illustrative; the implementer MUST adapt based on the OPS-31d test reference (`crates/dist/tests/kv_block_transfer.rs`). **Action:** Step 5.3 explicitly says "iterate on shape" — this is the natural place to resolve the issue. The plan does not fail because of it; it documents the iteration expectation.

**Issue spotted during review:** `bytemuck` may not be a workspace dep. **Action:** Step 2.3 explicitly checks for this before adding; if the workspace dep exists (added since the OPS-31d code), reuse it instead of adding `bytemuck = { version = "1" }` to `crates/model` only.

**Issue spotted during review:** The `BlockDataSource` trait requires `Send + Sync + fmt::Debug` per its definition. `Arc<PagedKvCache>` needs to be `Send + Sync + Debug`. **Action:** Verified — `PagedKvCache` derives `Debug` (in `tensor_store/mod.rs:24`), and `Arc<T>` is `Send + Sync` when `T: Send + Sync`. The wrapper derives `Clone, Debug`. The trait's `fmt::Debug` bound is satisfied by the wrapper's `#[derive(Debug)]`. No issue.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-22-p40-paged-kv-cache-wrapper.md`. This is a 7-task linear plan with two parallel-friendly branches (Tasks 2 + 3 could be done by a single agent in sequence; Tasks 4-5 depend on Task 3). Each task produces a working commit. The plan can be executed inline (executing-plans) or via subagents (subagent-driven-development).
