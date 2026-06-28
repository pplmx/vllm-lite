//! Architecture registry block trait.
//!
//! [`TransformerBlock`] extends [`PagedDecoderBlock`] with layer metadata
//! for [`crate::arch::Architecture::create_block`] dynamic dispatch.
//!
//! [`PagedDecoderBlock`]: crate::components::decoder_block::PagedDecoderBlock

use crate::components::decoder_block::PagedDecoderBlock;
use crate::paged_tensor::PagedKvCache;
use candle_core::{Result, Tensor};

/// `TransformerBlock`: transformer block trait.
pub trait TransformerBlock: PagedDecoderBlock + Send + Sync {
    fn inner_dim(&self) -> usize;
    fn num_kv_heads(&self) -> usize;
}

/// No-op paged prefill for registry-only block stubs (does not write KV).
pub(crate) fn passthrough_paged_prefill(
    x: &Tensor,
    _kv_cache: &mut PagedKvCache,
    _layer_idx: usize,
    _block_ids: &[usize],
    _positions: &[usize],
) -> Result<Tensor> {
    Ok(x.clone())
}

/// No-op paged decode for registry-only block stubs (does not write KV).
pub(crate) fn passthrough_paged_decode(
    x: &Tensor,
    _kv_cache: &mut PagedKvCache,
    _layer_idx: usize,
    _block_ids: &[usize],
    _num_computed_tokens: usize,
    _positions: &[usize],
) -> Result<Tensor> {
    Ok(x.clone())
}
