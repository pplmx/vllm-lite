// crates/model/src/paged_tensor/tensor_store/pool.rs
//
// Block-allocator types: `CacheBlock` and `KvCachePool`.

use candle_core::{DType, Device, Result, Tensor};

/// CacheBlock: cache block.
pub struct CacheBlock {
    pub key: Tensor,
    pub value: Tensor,
    pub is_free: bool,
    pub layer_idx: usize,
}

/// KvCachePool: kv cache pool.
pub struct KvCachePool {
    blocks: Vec<CacheBlock>,
    free_list: Vec<usize>,
    total_blocks: usize,
}

impl KvCachePool {
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

    pub fn is_available(&self) -> bool {
        !self.free_list.is_empty()
    }

    pub fn available_blocks(&self) -> usize {
        self.free_list.len()
    }
}
