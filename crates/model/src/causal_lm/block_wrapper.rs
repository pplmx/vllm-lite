//! Generic wrapper exposing a concrete decoder block through [`PagedDecoderBlock`] + [`TransformerBlock`].

use crate::components::TransformerBlock;
use crate::components::decoder_block::PagedDecoderBlock;
use crate::config::ModelConfig;
use crate::paged_tensor::PagedKvCache;
use candle_core::{Result, Tensor};

/// BlockWrapper: block wrapper.
pub struct BlockWrapper<B> {
    inner: B,
    inner_dim: usize,
    num_kv_heads: usize,
}

impl<B> BlockWrapper<B> {
    /// new: new.
    pub fn new(inner: B, config: &ModelConfig) -> Self {
        Self {
            inner_dim: config.head_dim,
            num_kv_heads: config.num_kv_heads,
            inner,
        }
    }

    /// inner: inner.
    pub fn inner(&self) -> &B {
        &self.inner
    }
}

impl<B: PagedDecoderBlock> PagedDecoderBlock for BlockWrapper<B> {
    fn forward_prefill(
        &self,
        x: &Tensor,
        kv_cache: &mut PagedKvCache,
        layer_idx: usize,
        block_ids: &[usize],
        positions: &[usize],
    ) -> Result<Tensor> {
        self.inner
            .forward_prefill(x, kv_cache, layer_idx, block_ids, positions)
    }

    fn forward_decode(
        &self,
        x: &Tensor,
        kv_cache: &mut PagedKvCache,
        layer_idx: usize,
        block_ids: &[usize],
        num_computed_tokens: usize,
        positions: &[usize],
    ) -> Result<Tensor> {
        self.inner.forward_decode(
            x,
            kv_cache,
            layer_idx,
            block_ids,
            num_computed_tokens,
            positions,
        )
    }
}

impl<B: PagedDecoderBlock + Send + Sync> TransformerBlock for BlockWrapper<B> {
    fn inner_dim(&self) -> usize {
        self.inner_dim
    }

    fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }
}
