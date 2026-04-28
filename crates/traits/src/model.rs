use crate::types::{BatchOutput, SeqId, TokenId};

#[derive(Debug)]
pub struct ModelError {
    msg: String,
}

impl ModelError {
    pub fn new(msg: impl Into<String>) -> Self {
        Self { msg: msg.into() }
    }
}

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Model error: {}", self.msg)
    }
}

impl std::error::Error for ModelError {}

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
        input_tokens: &[TokenId],
        num_computed: usize,
        kv_block_ids: &[usize],
        positions: &[usize],
        is_prefill: bool,
    ) -> Result<(Tensor, usize)> {
        let seq_id: SeqId = 1;
        let tokens = vec![input_tokens.to_vec()];
        let pos = vec![positions.to_vec()];
        let blocks = vec![kv_block_ids.to_vec()];
        let num_computed = vec![num_computed];
        let is_prefill = vec![is_prefill];

        let output = self.forward(
            &[seq_id],
            &tokens,
            &pos,
            &blocks,
            &num_computed,
            &is_prefill,
        )?;

        if let Some(&next_token) = output.next_tokens.first() {
            let logits = Tensor::zeros(
                (1, 1, self.vocab_size()),
                candle_core::DType::F32,
                &candle_core::Device::Cpu,
            )?;
            Ok((logits, next_token as usize))
        } else {
            Err(ModelError::new("No output token"))
        }
    }

    #[cfg(feature = "candle")]
    fn vocab_size(&self) -> usize;

    fn num_layers(&self) -> usize;

    fn num_heads(&self) -> usize;
}
