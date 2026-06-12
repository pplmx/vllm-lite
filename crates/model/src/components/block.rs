//! Architecture registry block trait.
//!
//! Production models use [`super::decoder_block::RopeGqaDecoderBlock`] (or architecture-specific
//! wrappers). This trait exists for [`crate::arch::Architecture::create_block`] dynamic dispatch.

use candle_core::{Result, Tensor};

pub trait TransformerBlock: Send + Sync {
    fn forward(
        &mut self,
        hidden_states: &Tensor,
        positions: &[usize],
        kv_block_ids: &[usize],
        num_computed: usize,
        is_prefill: bool,
    ) -> Result<Tensor>;

    fn inner_dim(&self) -> usize;
    fn num_kv_heads(&self) -> usize;
}
