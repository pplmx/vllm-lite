// crates/model/src/paged_tensor/tensor_store/layout.rs
//
// PagedKvCache layout helpers: hash computation, block-hash lookup,
// quantization scale tracking, and block-size accessor.

use super::PagedKvCache;
use candle_core::Tensor;

impl PagedKvCache {
    #[must_use]
    pub fn num_blocks(&self) -> usize {
        self.key_cache.first().map_or(0, |t| t.shape().dims()[0])
    }

    #[must_use]
    pub const fn num_layers(&self) -> usize {
        self.num_layers
    }

    #[must_use]
    pub fn get_scale(&self, layer_idx: usize) -> f32 {
        self.scales.get(layer_idx).copied().unwrap_or(1.0)
    }

    pub(super) fn update_scale(&mut self, layer_idx: usize, new_scale: f32) {
        if layer_idx < self.scales.len() {
            self.scales[layer_idx] = new_scale;
        }
    }

    #[must_use]
    pub const fn block_size(&self) -> usize {
        self.block_size
    }

    #[must_use]
    pub fn compute_block_hash(block: &Tensor) -> u64 {
        if let Ok(data) = block.to_vec1::<f32>() {
            let hash: u64 = data
                .iter()
                .map(|&x| (x.abs() * 1000.0) as u64)
                .fold(0u64, |acc, x| acc.wrapping_mul(31).wrapping_add(x));
            hash
        } else {
            0
        }
    }

    #[must_use]
    pub fn find_matching_blocks(&self, prompt_hash: u64, layer_idx: usize) -> Vec<usize> {
        let mut matches = Vec::new();
        if let Some(hash_map) = self.block_hashes.get(layer_idx) {
            for (&hash, &block_id) in hash_map {
                if prompt_hash == hash {
                    matches.push(block_id);
                }
            }
        }
        matches
    }
}
