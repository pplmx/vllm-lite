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
}
