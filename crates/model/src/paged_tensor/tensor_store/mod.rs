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

/// `PagedKvCache`: paged kv cache.
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
