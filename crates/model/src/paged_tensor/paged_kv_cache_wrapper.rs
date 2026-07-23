//! Production `BlockDataSource` impl backed by `PagedKvCache`.
//!
//! This module is gated by `#[cfg(feature = "multi-node")]` at the
//! module declaration in `paged_tensor/mod.rs` so it doesn't appear
//! in the default build.
//!
//! Tests live in the `#[cfg(test)] mod tests` block at the bottom.

use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::Mutex;

use super::tensor_store::PagedKvCache;
use vllm_dist::{BlockDataSource, BlockSink, FetchError, WriteError};

/// Production `BlockDataSource` (sender) + `BlockSink` (receiver)
/// impl backed by a `PagedKvCache`.
///
/// The cache is held inside a `parking_lot::Mutex` so both the
/// sender path (`fetch_block` / `has_block` / `verify_chain_hash`)
/// and the receiver path (`write_block`) can share a single
/// underlying `PagedKvCache` without aliasing `&mut self` issues.
/// `parking_lot::Mutex` is preferred over `tokio::sync::Mutex` here
/// because the hot path is sync (candle tensor work) wrapped in
/// `block_in_place`; the async-only mutex would add overhead without
/// any contention benefit since writes are short-lived.
///
/// **Construction contract**: `PagedKvCacheWrapper::new` requires
/// unique ownership of the input `Arc<PagedKvCache>`. The
/// `PagedKvCache` itself isn't `Clone`, so we can't share the
/// underlying data across multiple wrappers — the engine bootstrap
/// passes its sole `Arc<PagedKvCache>` to the wrapper and goes
/// through the wrapper for all subsequent access. This invariant is
/// enforced at construction time via `Arc::try_unwrap`.
#[derive(Clone, Debug)]
pub struct PagedKvCacheWrapper {
    inner: Arc<Mutex<PagedKvCache>>,
}

impl PagedKvCacheWrapper {
    /// Wrap a `PagedKvCache` for both sender (`BlockDataSource`) and
    /// receiver (`BlockSink`) use. Takes unique ownership of the
    /// input `Arc<PagedKvCache>` and stores it inside a
    /// `parking_lot::Mutex` for concurrent access.
    ///
    /// # Panics
    ///
    /// Panics if `inner` is not the sole strong reference. Tests
    /// construct a fresh `Arc` for the wrapper, so this is fine.
    /// The production path uses [`Self::from_arc_mutex`] (which
    /// takes a pre-wrapped `Arc<Mutex<PagedKvCache>>` so the engine
    /// can hold its own `Arc<Mutex<PagedKvCache>>` for diagnostics).
    #[must_use]
    pub fn new(inner: Arc<PagedKvCache>) -> Self {
        let cache = Arc::try_unwrap(inner)
            .expect("PagedKvCacheWrapper::new requires unique Arc<PagedKvCache> ownership");
        Self {
            inner: Arc::new(Mutex::new(cache)),
        }
    }

    /// Construct a wrapper from a pre-wrapped
    /// `Arc<parking_lot::Mutex<PagedKvCache>>`. The production path —
    /// the engine bootstrap calls this so the engine itself can hold
    /// its own `Arc<Mutex<PagedKvCache>>` for diagnostics / future
    /// read paths.
    #[must_use]
    pub const fn from_arc_mutex(inner: Arc<Mutex<PagedKvCache>>) -> Self {
        Self { inner }
    }

    /// Accessor for the underlying `Arc<Mutex<PagedKvCache>>`. Used
    /// by diagnostics and the engine bootstrap to share the same
    /// Mutex (and therefore the same data) with another owner.
    #[must_use]
    pub const fn inner(&self) -> &Arc<Mutex<PagedKvCache>> {
        &self.inner
    }

    /// Verify that the chain-hash for `block_id` at layer 0 matches
    /// `expected_chain_hash`. Returns `true` on match, `false` on
    /// mismatch (block not found at layer 0 OR hash mismatch).
    ///
    /// Layer 0 is the canonical existence witness: every
    /// `PagedKvCache::write_kv` writes to all layers symmetrically,
    /// so layer 0 is a sound existence check (the same invariant
    /// `BlockDataSource::has_block` relies on).
    ///
    /// This is a defense-in-depth check on top of OPS-31d's
    /// wire-layer hash verification. Useful for:
    /// - Receiver-side end-to-end tests (P42)
    /// - Operator-driven diagnostics (does the local cache have the
    ///   block the peer claims it sent?)
    /// - Future OPS-32e failure-recovery retry loop
    ///
    /// Pure read — no allocation, no I/O.
    #[must_use]
    pub fn verify_chain_hash(&self, block_id: u64, expected_chain_hash: u64) -> bool {
        let Ok(block_id_us) = usize::try_from(block_id) else {
            return false;
        };
        let cache = self.inner.lock();
        cache
            .block_hashes_for_layer(0)
            .map_or(false, |layer_hashes| {
                layer_hashes
                    .iter()
                    .any(|(hash, bid)| *bid == block_id_us && *hash == expected_chain_hash)
            })
    }
}

#[async_trait]
impl BlockDataSource for PagedKvCacheWrapper {
    async fn fetch_block(&self, block_id: u64) -> Result<Vec<u8>, FetchError> {
        let block_id_us = usize::try_from(block_id).map_err(|_| FetchError::NotFound(block_id))?;
        let cache_lock = Arc::clone(&self.inner);
        tokio::task::block_in_place(move || {
            let cache = cache_lock.lock();
            read_block_bytes(&cache, block_id_us)
        })
    }

    async fn has_block(&self, block_id: u64) -> bool {
        let Ok(block_id_us) = usize::try_from(block_id) else {
            return false;
        };
        // Layer 0 is the canonical existence witness: every write_kv
        // touches all layers symmetrically, so if layer 0 has it, all
        // layers do.
        let cache = self.inner.lock();
        cache
            .block_hashes_for_layer(0)
            .is_some_and(|layer_hashes| layer_hashes.values().any(|&bid| bid == block_id_us))
    }
}

#[async_trait]
impl BlockSink for PagedKvCacheWrapper {
    /// P42 receiver-side install. Symmetric to `fetch_block`:
    /// - converts `block_id: u64` → `usize` (returning `OutOfRange`
    ///   if it can't fit — the sink can't tell what `usize::MAX`
    ///   means, but the caller asked for it)
    /// - delegates the candle-touching work to `block_in_place`
    ///   so the inner `Mutex` lock is held only for the duration of
    ///   the synchronous tensor write.
    ///
    /// `PagedKvCache::write_block_bytes` (T1) does all the per-layer
    /// dispatch + `block_hashes` updates, so `write_block` is a thin
    /// shim that converts the cache's `candle_core::Error` to a typed
    /// `WriteError`.
    async fn write_block(&self, block_id: u64, bytes: &[u8]) -> Result<(), WriteError> {
        let block_id_us =
            usize::try_from(block_id).map_err(|_| WriteError::OutOfRange { block_id })?;
        let cache_lock = Arc::clone(&self.inner);
        tokio::task::block_in_place(move || {
            let mut cache = cache_lock.lock();
            // Pre-check the OOB case so the error variant is correct.
            // The cache's own bounds check returns a candle error
            // we'd otherwise map to InvalidBytes; the wrapper layer
            // distinguishes "block can't fit" from "bytes are wrong
            // length" because they're different client-side mistakes.
            if block_id_us >= cache.num_blocks() {
                return Err(WriteError::OutOfRange {
                    block_id: block_id_us as u64,
                });
            }
            write_block_bytes_inner(&mut cache, block_id_us, bytes)
        })
    }
}

fn read_block_bytes(cache: &PagedKvCache, block_id: usize) -> Result<Vec<u8>, FetchError> {
    if block_id >= cache.num_blocks() {
        return Err(FetchError::NotFound(block_id as u64));
    }
    let num_layers = cache.num_layers();
    let mut bytes = Vec::with_capacity(num_layers * 2 * cache.num_blocks_count_per_layer() * 4);
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
            (dequantize_f32(&k, scale), dequantize_f32(&v, scale))
        } else {
            (k, v)
        };
        bytes.extend_from_slice(bytemuck::cast_slice(&k_out));
        bytes.extend_from_slice(bytemuck::cast_slice(&v_out));
    }
    Ok(bytes)
}

/// P42 receiver-side counterpart of `read_block_bytes`. Returns a
/// `WriteError::InvalidBytes` if `bytes.len()` doesn't match the
/// cache's expected shape, with `expected` filled in from the
/// cache's metadata.
fn write_block_bytes_inner(
    cache: &mut PagedKvCache,
    block_id: usize,
    bytes: &[u8],
) -> Result<(), WriteError> {
    let expected = cache.num_layers() * 2 * cache.num_blocks_count_per_layer() * 4;
    cache
        .write_block_bytes(block_id, bytes)
        .map_err(|_| WriteError::InvalidBytes {
            block_id: block_id as u64,
            expected,
            actual: bytes.len(),
        })
}

/// Inverse of `PagedKvCache::write_kv`'s quantization step:
/// multiply each int8-encoded f32 by the layer's scale.
fn dequantize_f32(data: &[f32], scale: f32) -> Vec<f32> {
    data.iter().map(|&x| x * scale).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};
    use vllm_traits::BLOCK_SIZE;

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
        // Write a non-zero K tensor to (layer 0, block 1, token 0).
        // Shape [1, num_heads=2, head_dim=4] needs 8 f32 values total:
        // head 0 gets [1,2,3,4], head 1 gets [5,6,7,8].
        let k = Tensor::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            (1, 2, 4),
            &Device::Cpu,
        )
        .expect("k");
        let v = Tensor::from_slice(&[0.0f32; 8], (1, 2, 4), &Device::Cpu).expect("v");
        // We need a mut handle for write_kv, but wrapper holds Arc.
        // Unwrap the Arc (we still hold the only owner) to get a `&mut`
        // handle, then re-wrap after writing.
        let mut cache_mut = Arc::try_unwrap(cache).expect("unique Arc owner");
        cache_mut.write_kv(0, 1, 0, &k, &v).expect("write");
        let wrapper = PagedKvCacheWrapper::new(Arc::new(cache_mut));
        let bytes = wrapper.fetch_block(1).await.expect("fetch");
        // First 4 f32 bytes (16 bytes total) should match [1.0, 2.0, 3.0, 4.0].
        let as_f32: &[f32] = bytemuck::cast_slice(&bytes[..16]);
        assert_eq!(as_f32, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn has_block_returns_true_for_written_block() {
        let cache = small_cache();
        let mut cache_mut = Arc::try_unwrap(cache).expect("unique Arc owner");
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
            &[
                100.0f32, -100.0, 100.0, -100.0, 100.0, -100.0, 100.0, -100.0,
            ],
            (1, 2, 4),
            &Device::Cpu,
        )
        .expect("k");
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

    // ──────────────────────────────────────────────────────────────────────
    // verify_chain_hash tests (P41 T2). The helper itself is in the
    // impl block above. Uses `write_layer_block` (P41 T1) to fully
    // populate the block, so the chain hash is deterministic and
    // computable from the input K data.
    // ──────────────────────────────────────────────────────────────────────

    #[test]
    fn verify_chain_hash_returns_true_for_written_block() {
        let n = 2 * BLOCK_SIZE * 4; // num_heads=2, BLOCK_SIZE=16, head_dim=4
        let k: Vec<f32> = (0..n).map(|i| i as f32 * 0.25 + 0.5).collect();
        let v: Vec<f32> = (0..n).map(|i| i as f32 * 1.5).collect();

        let mut cache_mut = Arc::try_unwrap(small_cache()).expect("unique Arc owner");
        cache_mut
            .write_layer_block(0, 2, &k, &v)
            .expect("write_layer_block");
        let expected_hash = PagedKvCache::compute_block_hash_from_slice(&k);

        let wrapper = PagedKvCacheWrapper::new(Arc::new(cache_mut));
        assert!(wrapper.verify_chain_hash(2, expected_hash));
    }

    #[test]
    fn verify_chain_hash_returns_false_for_mismatch() {
        let n = 2 * BLOCK_SIZE * 4;
        let k: Vec<f32> = (0..n).map(|i| i as f32 * 0.25 + 0.5).collect();
        let v: Vec<f32> = (0..n).map(|i| i as f32 * 1.5).collect();

        let mut cache_mut = Arc::try_unwrap(small_cache()).expect("unique Arc owner");
        cache_mut
            .write_layer_block(0, 2, &k, &v)
            .expect("write_layer_block");
        let actual_hash = PagedKvCache::compute_block_hash_from_slice(&k);

        let wrapper = PagedKvCacheWrapper::new(Arc::new(cache_mut));
        // Pass a wrong hash → must return false even though block 2 exists.
        assert!(!wrapper.verify_chain_hash(2, actual_hash.wrapping_add(1)));
        // Pass an unknown block → must return false.
        assert!(!wrapper.verify_chain_hash(99, actual_hash));
    }

    // ──────────────────────────────────────────────────────────────────────
    // BlockSink impl tests (P42 T3). Round-trip via `write_block` →
    // `read_layer_block` (which is the same wire-shape path that
    // `fetch_block` produces in production) and verify the hash
    // recomputation in `write_layer_block` lights up
    // `verify_chain_hash`.
    // ──────────────────────────────────────────────────────────────────────

    fn make_sink_test_bytes(num_layers: usize, per_layer_f32s: usize, seed: u32) -> Vec<u8> {
        // Same LCG as the tensor_store tests so we can assert
        // round-trip equality across the crate.
        let mut state = seed.wrapping_add(1);
        let total = num_layers * 2 * per_layer_f32s;
        let mut bytes = Vec::with_capacity(total * 4);
        for _ in 0..total {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            #[allow(clippy::cast_possible_truncation)]
            let f = (state as f32) * 1e-9;
            bytes.extend_from_slice(&f.to_le_bytes());
        }
        bytes
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn write_block_round_trip_via_read_layer_block_is_bit_exact() {
        let cache = small_cache(); // 2 layers, 2 heads, head_dim 4, 4 blocks
        let num_layers = cache.num_layers();
        let per_layer_f32s = cache.num_blocks_count_per_layer();

        // Build the wire bytes the same way `fetch_block` would:
        // [K_layer_0][V_layer_0][K_layer_1][V_layer_1].
        let bytes = make_sink_test_bytes(num_layers, per_layer_f32s, 0xCAFE_F00D);

        // Move the cache into the wrapper (sole owner). Subsequent
        // readback goes through `wrapper.inner().lock()`.
        let wrapper = PagedKvCacheWrapper::new(cache);
        wrapper.write_block(1, &bytes).await.expect("write_block");

        // Read back via the same per-layer read API the sender uses.
        // Scope the lock so it drops before verify_chain_hash below
        // (parking_lot::Mutex isn't reentrant).
        {
            let cache_lock = wrapper.inner().lock();
            for layer_idx in 0..num_layers {
                let (k_out, v_out) = cache_lock
                    .read_layer_block(layer_idx, 1)
                    .expect("read_layer_block");
                let offset = layer_idx * 2 * per_layer_f32s;
                let k_in: &[f32] =
                    bytemuck::cast_slice(&bytes[offset * 4..(offset + per_layer_f32s) * 4]);
                let v_in: &[f32] = bytemuck::cast_slice(
                    &bytes[(offset + per_layer_f32s) * 4..(offset + 2 * per_layer_f32s) * 4],
                );
                assert_eq!(k_out, k_in, "K round-trip bit-exact (layer {layer_idx})");
                assert_eq!(v_out, v_in, "V round-trip bit-exact (layer {layer_idx})");
            }
        }
        // The K slice we serialized via the LCG matches what
        // `write_layer_block` recomputed the hash over — that's the
        // exact hash stored in `block_hashes` and the one
        // `verify_chain_hash` reads back.
        let layer_0_k: &[f32] = bytemuck::cast_slice(&bytes[..per_layer_f32s * 4]);
        let lcg_hash = PagedKvCache::compute_block_hash_from_slice(layer_0_k);
        assert!(
            wrapper.verify_chain_hash(1, lcg_hash),
            "verify_chain_hash must reflect the bytes written via BlockSink"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn write_block_returns_invalid_bytes_for_wrong_length() {
        let wrapper = PagedKvCacheWrapper::new(small_cache());
        let wrong = vec![0u8; 100]; // not a multiple of (num_layers * 2 * 128 * 4)
        let result = wrapper.write_block(0, &wrong).await;
        match result {
            Err(WriteError::InvalidBytes {
                block_id,
                expected: _,
                actual,
            }) => {
                assert_eq!(block_id, 0);
                assert_eq!(actual, 100);
            }
            other => panic!("expected InvalidBytes, got {other:?}"),
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn write_block_returns_out_of_range_for_oversize_block_id() {
        let wrapper = PagedKvCacheWrapper::new(small_cache()); // 4 blocks
        let valid_len = 2 * 2 * 2 * BLOCK_SIZE * 4 * 4; // num_layers * 2 * per_layer_f32s * 4
        let valid = vec![0u8; valid_len];
        let result = wrapper.write_block(99, &valid).await;
        match result {
            Err(WriteError::OutOfRange { block_id }) => assert_eq!(block_id, 99),
            other => panic!("expected OutOfRange, got {other:?}"),
        }
    }
}
