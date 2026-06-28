use std::sync::Arc;

use crate::types::{BatchOutput, SeqId, TokenId};

/// ModelError: model error.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ModelError {
    #[error("{0}")]
    Message(String),
    #[error("backend error: {0}")]
    Backend(String),
    #[error("configuration error: {0}")]
    Config(String),
    #[cfg(feature = "candle")]
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

impl ModelError {
    pub fn new(message: impl Into<String>) -> Self {
        Self::Message(message.into())
    }
}

/// Result: result.
pub type Result<T> = std::result::Result<T, ModelError>;

#[cfg(feature = "candle")]
use candle_core::Tensor;

/// ModelBackend: model backend trait.
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

/// Default stub `ModelBackend`: returns token 0 for every sequence and
/// zero-valued embeddings/logits with vocab-shaped layout.
///
/// Lives in `vllm-traits` next to the trait definition so the inherent
/// `dyn ModelBackend::default_arc()` method can construct it.
/// Testing/utility stubs with richer behavior live in `vllm-testing`.
#[derive(Debug, Default, Clone, Copy)]
pub struct StubModelBackend;

impl ModelBackend for StubModelBackend {
    fn forward(
        &mut self,
        seq_ids: &[SeqId],
        _input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
        _kv_block_ids: &[Vec<usize>],
        _num_computed_tokens: &[usize],
        _is_prefill: &[bool],
    ) -> Result<BatchOutput> {
        Ok(BatchOutput {
            seq_ids: seq_ids.to_vec(),
            next_tokens: vec![0; seq_ids.len()],
        })
    }

    fn forward_logits(
        &mut self,
        _seq_ids: &[SeqId],
        input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
        _kv_block_ids: &[Vec<usize>],
        _num_computed_tokens: &[usize],
        _is_prefill: &[bool],
    ) -> Result<Vec<Vec<f32>>> {
        let vocab = {
            #[cfg(feature = "candle")]
            {
                self.vocab_size()
            }
            #[cfg(not(feature = "candle"))]
            {
                1
            }
        };
        Ok(input_tokens
            .iter()
            .map(|tokens| vec![0.0; tokens.len() * vocab])
            .collect())
    }

    fn embed(
        &mut self,
        input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
    ) -> Result<Vec<Vec<f32>>> {
        Ok(input_tokens
            .iter()
            .map(|tokens| vec![0.0; tokens.len()])
            .collect())
    }

    #[cfg(feature = "candle")]
    fn vocab_size(&self) -> usize {
        1
    }

    fn num_layers(&self) -> usize {
        0
    }

    fn num_heads(&self) -> usize {
        0
    }
}

impl dyn ModelBackend {
    /// Returns an `Arc<Self>` containing the no-op [`StubModelBackend`] stub.
    ///
    /// This is the closest equivalent to `Arc::<dyn ModelBackend>::default()`;
    /// Rust's orphan rule prevents a direct `impl Default for Arc<dyn ...>`
    /// because `Arc` is foreign and there is no local type appearing before
    /// the uncovered trait-object parameter.
    pub fn default_arc() -> Arc<Self> {
        Arc::new(StubModelBackend)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stub_model_backend_returns_zero_tokens() {
        let mut backend = StubModelBackend;
        let output = backend
            .forward(
                &[1, 2, 3],
                &[vec![0], vec![0], vec![0]],
                &[vec![0], vec![0], vec![0]],
                &[vec![0], vec![0], vec![0]],
                &[0, 0, 0],
                &[true, true, true],
            )
            .expect("stub forward should succeed");
        assert_eq!(output.seq_ids, vec![1, 2, 3]);
        assert_eq!(output.next_tokens, vec![0, 0, 0]);
    }

    #[test]
    fn model_backend_default_arc_works() {
        let mut backend: Arc<dyn ModelBackend> = <dyn ModelBackend>::default_arc();
        let output = Arc::get_mut(&mut backend)
            .expect("default Arc should be uniquely owned")
            .forward(
                &[1, 2, 3],
                &[vec![0], vec![0], vec![0]],
                &[vec![0], vec![0], vec![0]],
                &[vec![0], vec![0], vec![0]],
                &[0, 0, 0],
                &[true, true, true],
            )
            .expect("stub forward should succeed");
        assert_eq!(output.seq_ids, vec![1, 2, 3]);
        assert_eq!(output.next_tokens, vec![0, 0, 0]);
    }
}
