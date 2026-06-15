use crate::types::{BatchOutput, SeqId, TokenId};

#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct ModelError {
    message: String,
}

impl ModelError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

#[cfg(feature = "candle")]
impl From<candle_core::Error> for ModelError {
    fn from(e: candle_core::Error) -> Self {
        ModelError::new(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, ModelError>;

#[cfg(feature = "candle")]
use candle_core::Tensor;

pub trait ModelBackend: Send + Sync {
    fn forward(
        &mut self,
        seq_ids: &[SeqId],
        input_tokens: &[Vec<TokenId>],
        positions: &[Vec<usize>],
        kv_block_ids: &[Vec<usize>],
        num_computed_tokens: &[usize],
        is_prefill: &[bool],
    ) -> Result<BatchOutput>;

    fn forward_logits(
        &mut self,
        seq_ids: &[SeqId],
        input_tokens: &[Vec<TokenId>],
        positions: &[Vec<usize>],
        kv_block_ids: &[Vec<usize>],
        num_computed_tokens: &[usize],
        is_prefill: &[bool],
    ) -> Result<Vec<Vec<f32>>>;

    fn embed(
        &mut self,
        input_tokens: &[Vec<TokenId>],
        positions: &[Vec<usize>],
    ) -> Result<Vec<Vec<f32>>>;

    #[cfg(feature = "candle")]
    fn forward_with_cache(
        &mut self,
        _input_tokens: &[TokenId],
        _num_computed: usize,
        _kv_block_ids: &[usize],
        _positions: &[usize],
        _is_prefill: bool,
    ) -> Result<(Tensor, usize)> {
        Err(ModelError::new(
            "forward_with_cache not implemented for this model backend",
        ))
    }

    #[cfg(feature = "candle")]
    fn vocab_size(&self) -> usize;

    fn num_layers(&self) -> usize;

    fn num_heads(&self) -> usize;

    /// Forward pass stopped after `upto_layer` layers.
    /// Default impl ignores `upto_layer` and calls `self.forward()`.
    #[allow(clippy::too_many_arguments)]
    fn forward_to_layer(
        &mut self,
        seq_ids: &[SeqId],
        input_tokens: &[Vec<TokenId>],
        positions: &[Vec<usize>],
        kv_block_ids: &[Vec<usize>],
        num_computed_tokens: &[usize],
        is_prefill: &[bool],
        upto_layer: usize,
    ) -> Result<BatchOutput> {
        let _ = upto_layer;
        self.forward(
            seq_ids,
            input_tokens,
            positions,
            kv_block_ids,
            num_computed_tokens,
            is_prefill,
        )
    }
}
