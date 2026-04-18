//! Transformer block trait for layer abstraction.

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
