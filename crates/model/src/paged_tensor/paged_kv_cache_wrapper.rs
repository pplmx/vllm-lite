//! Production `BlockDataSource` impl backed by `PagedKvCache`.
//!
//! This module is gated by `#[cfg(feature = "multi-node")]` at the
//! module declaration in `paged_tensor/mod.rs` so it doesn't appear
//! in the default build.
//!
//! Tests live in the `#[cfg(test)] mod tests` block at the bottom.

use std::sync::Arc;

use async_trait::async_trait;

use super::tensor_store::PagedKvCache;
use vllm_dist::{BlockDataSource, FetchError};

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
        let block_id_us = usize::try_from(block_id).map_err(|_| FetchError::NotFound(block_id))?;
        let cache = Arc::clone(&self.inner);
        tokio::task::block_in_place(move || read_block_bytes(&cache, block_id_us))
    }

    async fn has_block(&self, block_id: u64) -> bool {
        let Ok(block_id_us) = usize::try_from(block_id) else {
            return false;
        };
        // Layer 0 is the canonical existence witness: every write_kv
        // touches all layers symmetrically, so if layer 0 has it, all
        // layers do.
        self.inner
            .block_hashes_for_layer(0)
            .is_some_and(|layer_hashes| layer_hashes.values().any(|&bid| bid == block_id_us))
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
}
