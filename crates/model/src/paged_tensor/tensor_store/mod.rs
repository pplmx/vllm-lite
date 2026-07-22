//! Paged tensor store: physical KV cache backing store, block pool, and layout helpers.
//!
//! `PagedKvCache` is the public facade re-exported from this module.
//! Sub-modules: `buffer` (KV read/write operations), `layout` (block
//! hashes + scale layout), `pool` (block free-list).

// crates/model/src/paged_tensor/tensor_store/mod.rs
//
// Facade for the paged tensor store subsystem. Sub-modules:
// - `buffer` — PagedKvCache write/read operations on the K/V buffer.
// - `layout` — PagedKvCache hash, scale, and block-size layout helpers.
// - `pool`   — `CacheBlock` and `KvCachePool` block-allocator types.

pub use vllm_traits::BLOCK_SIZE;

mod buffer;
mod layout;
mod pool;

use std::collections::HashMap;

use candle_core::{DType, Device, Result, Tensor};

#[derive(Debug)]
/// Physical paged KV cache backing store for attention layers.
///
/// Stores per-layer K/V tensors in fixed-size blocks (`BLOCK_SIZE` tokens per
/// block). Attention paths call [`Self::write_kv`] / [`Self::read_kv`] during
/// prefill and decode; the scheduler's logical block allocator tracks which
/// physical blocks are assigned to each sequence.
pub struct PagedKvCache {
    key_cache: Vec<Tensor>,
    value_cache: Vec<Tensor>,
    num_layers: usize,
    num_heads: usize,
    head_dim: usize,
    block_size: usize,
    device: Device,
    pub quantized: bool,
    pub scales: Vec<f32>,
    pub block_hashes: Vec<HashMap<u64, usize>>,
}

impl PagedKvCache {
    /// Construct a new instance from the given configuration.
    /// # Errors
    ///
    /// Returns `Err` if any required tensor allocation or weight loading fails.
    pub fn new(
        num_layers: usize,
        num_heads: usize,
        head_dim: usize,
        num_blocks: usize,
        device: Device,
        quantized: bool,
    ) -> Result<Self> {
        let mut key_cache = Vec::with_capacity(num_layers);
        let mut value_cache = Vec::with_capacity(num_layers);

        for _ in 0..num_layers {
            let shape = (num_blocks, num_heads, BLOCK_SIZE, head_dim);
            let key = Tensor::zeros(shape, DType::F32, &device)?;
            let value = Tensor::zeros(shape, DType::F32, &device)?;
            key_cache.push(key);
            value_cache.push(value);
        }

        let scales = vec![1.0f32; num_layers];
        let block_hashes = vec![HashMap::new(); num_layers];

        Ok(Self {
            key_cache,
            value_cache,
            num_layers,
            num_heads,
            head_dim,
            block_size: BLOCK_SIZE,
            device,
            quantized,
            scales,
            block_hashes,
        })
    }

    /// Read the K and V tensors for a single `(layer_idx, block_id)`
    /// pair, materializing both to host-side `Vec<f32>`.
    ///
    /// Returns `(K_bytes, V_bytes)` flattened in the same row-major
    /// layout as the on-disk block written by `write_kv`: shape
    /// `[num_heads, block_size, head_dim]`. Used by the multi-node
    /// `PagedKvCacheWrapper` to serialize a block for cross-node
    /// transfer; the receiver feeds the bytes back into
    /// `write_kv_batch`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `layer_idx >= num_layers`, `block_id >=
    /// num_blocks`, or the underlying tensor narrow / `flatten_all`
    /// / `to_vec1` fails.
    #[allow(dead_code)] // reachable under --features multi-node via PagedKvCacheWrapper (P40 T2)
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

    /// Number of f32 elements per (layer, block) pair: `num_heads *
    /// block_size * head_dim`. Used by the multi-node wrapper for
    /// buffer sizing in `fetch_block`.
    #[must_use]
    #[allow(dead_code)] // reachable under --features multi-node via PagedKvCacheWrapper (P40 T2)
    pub(crate) const fn num_blocks_count_per_layer(&self) -> usize {
        self.num_heads * self.block_size * self.head_dim
    }

    /// Borrow the per-layer block-hash map for `layer_idx`.
    ///
    /// `block_hashes[layer]` maps `hash → block_id` for all blocks
    /// ever written to that layer. Used by the multi-node
    /// `paged_kv_cache_wrapper` module (gated by `#[cfg(feature =
    /// "multi-node")]`) for the `BlockDataSource::has_block`
    /// witness check (layer 0 as the canonical existence proof).
    ///
    /// Returns `None` if `layer_idx >= num_layers`.
    #[must_use]
    #[allow(dead_code)] // reachable under --features multi-node via PagedKvCacheWrapper (P40 T3)
    pub(crate) fn block_hashes_for_layer(
        &self,
        layer_idx: usize,
    ) -> Option<&std::collections::HashMap<u64, usize>> {
        self.block_hashes.get(layer_idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

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
        )
        .expect("k");
        let v = Tensor::from_slice(
            &[10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
            (1, 2, 4),
            &Device::Cpu,
        )
        .expect("v");
        cache.write_kv(0, 0, 0, &k, &v).expect("write");

        let (k_out, v_out) = cache.read_layer_block(0, 0).expect("read");
        // write_kv flattens as [head][token][head_dim] row-major.
        // The k input has shape [1, num_heads=2, head_dim=4] so it
        // fills both heads at token_offset=0:
        //   head 0, token 0: positions [0..4]            = [1, 2, 3, 4]
        //   head 0, tokens 1..16: positions [4..64]      = zeros
        //   head 1, token 0: positions [64..68]          = [5, 6, 7, 8]
        //   head 1, tokens 1..16: positions [68..128]    = zeros
        assert_eq!(&k_out[0..4], &[1.0, 2.0, 3.0, 4.0]);
        assert!(k_out[4..64].iter().all(|&x| x == 0.0));
        assert_eq!(&k_out[64..68], &[5.0, 6.0, 7.0, 8.0]);
        assert!(k_out[68..].iter().all(|&x| x == 0.0));

        assert_eq!(&v_out[0..4], &[10.0, 20.0, 30.0, 40.0]);
        assert!(v_out[4..64].iter().all(|&x| x == 0.0));
        assert_eq!(&v_out[64..68], &[50.0, 60.0, 70.0, 80.0]);
        assert!(v_out[68..].iter().all(|&x| x == 0.0));
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
