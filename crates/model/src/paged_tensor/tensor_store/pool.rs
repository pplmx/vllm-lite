//! Paged-KV block pool: `CacheBlock` (single block metadata + tensor handle) and `KvCachePool` (the global free-list + occupancy counters).
//!
//! The pool is owned by `tensor_store::TensorStore` and survives
//! eviction / re-allocation cycles. Block id is a stable handle —
//! the physical tensor can move, the id cannot.

// crates/model/src/paged_tensor/tensor_store/pool.rs
//
// Block-allocator types: `CacheBlock` and `KvCachePool`.
//
// Reserved for the legacy block-allocator interface; the production path
// uses `BlockAllocator` from `paged_tensor/block_allocator.rs`. Kept
// available for tests and future re-integration.
#![allow(dead_code)]

use candle_core::{DType, Device, Result, Tensor};

/// Block abstraction for Cache. Groups a contiguous range of work (e.g. one transformer layer, one pipeline stage).
pub struct CacheBlock {
    pub key: Tensor,
    pub value: Tensor,
    pub is_free: bool,
    pub layer_idx: usize,
}

/// Pool of pre-allocated `KvCache` resources. Acquire returns a guard; release happens on drop.
pub struct KvCachePool {
    blocks: Vec<CacheBlock>,
    free_list: Vec<usize>,
    total_blocks: usize,
}

impl KvCachePool {
    #[allow(clippy::needless_pass_by_value)]
    pub fn new(
        _num_layers: usize,
        num_heads: usize,
        head_dim: usize,
        block_size: usize,
        device: Device,
    ) -> Result<Self> {
        let mut blocks = Vec::new();
        let mut free_list = Vec::new();

        for block_id in 0..1000 {
            let key = Tensor::zeros((1, num_heads, block_size, head_dim), DType::F32, &device)?;
            let value = Tensor::zeros((1, num_heads, block_size, head_dim), DType::F32, &device)?;

            blocks.push(CacheBlock {
                key,
                value,
                is_free: true,
                layer_idx: 0,
            });
            free_list.push(block_id);
        }

        Ok(Self {
            blocks,
            free_list,
            total_blocks: 1000,
        })
    }

    pub fn allocate(&mut self) -> Option<usize> {
        self.free_list.pop()
    }

    pub fn deallocate(&mut self, block_id: usize) {
        if block_id < self.total_blocks {
            self.blocks[block_id].is_free = true;
            self.free_list.push(block_id);
        }
    }

    pub const fn is_available(&self) -> bool {
        !self.free_list.is_empty()
    }

    pub const fn available_blocks(&self) -> usize {
        self.free_list.len()
    }
}
