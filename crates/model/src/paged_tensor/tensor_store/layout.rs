//! `PagedKV` cache layout helpers: block-hash computation (xxHash of token prefix + parent hash) and the lookup table used during prefix-cache hits.
//!
//! Hash stability across re-starts is what makes prefix caching
//! survive process restarts; the implementation lives here.

// crates/model/src/paged_tensor/tensor_store/layout.rs
//
// PagedKvCache layout helpers: hash computation, block-hash lookup,
// quantization scale tracking, and block-size accessor.

use super::PagedKvCache;

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

    /// Compute block hash directly from a host-side `&[f32]` buffer.
    ///
    /// H-13 (PERF-02): avoids the redundant device round-trip in
    /// `write_kv` when the host-side data is already available.
    #[must_use]
    pub fn compute_block_hash_from_slice(data: &[f32]) -> u64 {
        // invariant: block hashes only need rough quantization of |x|; the
        // truncation to u64 is part of the hash-mixing algorithm.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        data.iter()
            .map(|&x| (x.abs() * 1000.0) as u64)
            .fold(0u64, |acc, x| acc.wrapping_mul(31).wrapping_add(x))
    }
}
