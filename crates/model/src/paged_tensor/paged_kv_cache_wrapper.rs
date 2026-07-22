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
    let mut bytes = Vec::with_capacity(num_layers * 2 * cache.num_blocks_count_per_layer() * 4);
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
        let mut cache_mut = Arc::try_unwrap(cache)
            .map_err(|_| "Arc has multiple owners")
            .expect("unique");
        cache_mut.write_kv(0, 1, 0, &k, &v).expect("write");
        let wrapper = PagedKvCacheWrapper::new(Arc::new(cache_mut));
        let bytes = wrapper.fetch_block(1).await.expect("fetch");
        // First 4 f32 bytes (16 bytes total) should match [1.0, 2.0, 3.0, 4.0].
        let as_f32: &[f32] = bytemuck::cast_slice(&bytes[..16]);
        assert_eq!(as_f32, &[1.0, 2.0, 3.0, 4.0]);
    }
}
