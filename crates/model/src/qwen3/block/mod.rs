//! Qwen3 transformer block: one decoder layer with grouped-query attention + `SwiGLU` MLP + QK-norm.
//!
//! Supports both the standard GQA variant and the MLA variant
//! (`qwen3/mla_attention.rs`). Each block reads from / writes to the
//! paged KV cache through `AttentionConfig`.
//!
//! Module layout:
//!
//! - [`self`] (`mod.rs`) — `TransformerBlock` struct + `Deref` + `PagedDecoderBlock` impl
//! - [`construct`] — `new`, `new_with_tp` (feature-gated), `new_with_weights`
//! - [`weights`] — `from_weights` (HuggingFace weight map loader)
//! - [`factory`] — free functions `new_block` + `block_from_weights`
//!
//! Tests live in `tests.rs` (sibling file) to keep this module under the
//! 800-line soft cap.
#![allow(clippy::type_complexity, clippy::module_name_repetitions)]

mod construct;
pub(crate) mod factory;
mod weights;

#[cfg(test)]
mod tests;

use crate::components::RopeGqaDecoderBlock;
use crate::components::decoder_block::PagedDecoderBlock;
use candle_core::{Result, Tensor};
use std::ops::Deref;

#[derive(Debug)]
/// Qwen3 decoder layer wrapping the shared RoPE-GQA block.
pub struct TransformerBlock(RopeGqaDecoderBlock);

impl Deref for TransformerBlock {
    type Target = RopeGqaDecoderBlock;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl PagedDecoderBlock for TransformerBlock {
    fn forward_prefill(
        &self,
        x: &Tensor,
        kv_cache: &mut crate::paged_tensor::PagedKvCache,
        layer_idx: usize,
        block_ids: &[usize],
        positions: &[usize],
    ) -> Result<Tensor> {
        self.0
            .forward_prefill(x, kv_cache, layer_idx, block_ids, positions)
    }

    fn forward_prefill_continue(
        &self,
        x: &Tensor,
        kv_cache: &mut crate::paged_tensor::PagedKvCache,
        layer_idx: usize,
        block_ids: &[usize],
        positions: &[usize],
        num_computed_tokens: usize,
    ) -> Result<Tensor> {
        self.0.forward_prefill_continue(
            x,
            kv_cache,
            layer_idx,
            block_ids,
            positions,
            num_computed_tokens,
        )
    }

    fn forward_decode(
        &self,
        x: &Tensor,
        kv_cache: &mut crate::paged_tensor::PagedKvCache,
        layer_idx: usize,
        block_ids: &[usize],
        num_computed_tokens: usize,
        positions: &[usize],
    ) -> Result<Tensor> {
        self.0.forward_decode(
            x,
            kv_cache,
            layer_idx,
            block_ids,
            num_computed_tokens,
            positions,
        )
    }
}

// Re-export the free factory functions so `super::block::{new_block,
// block_from_weights}` continues to work from sibling modules.
pub(crate) use factory::{block_from_weights, new_block};
