//! `ModelBackend` trait and the wire types that flow through it.
//!
//! The trait is the single seam between `vllm-core` (scheduler, KV cache,
//! engine) and `vllm-model` (architecture implementations). Adding a new
//! architecture means implementing `ModelBackend` and registering the
//! impl in `vllm_model::arch::ArchitectureRegistry`.
#![allow(clippy::module_name_repetitions)]

use std::sync::Arc;

use crate::types::{BatchOutput, SeqId, TokenId};

/// Error type for Model. Returned from every fallible public API; covers I/O, validation, and resource-limit failures. Use [`Result<T>`] alias in the same module.
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

/// Convenience alias used by every public API in `vllm-traits`.
pub type Result<T> = std::result::Result<T, ModelError>;

#[cfg(feature = "candle")]
use candle_core::Tensor;

/// `ModelBackend`. See the type definition for fields and behavior.
pub trait ModelBackend: Send + Sync {
    /// Run a forward pass producing next-token logits for each sequence.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the underlying backend fails (e.g. tensor allocation,
    /// shape mismatch, I/O, or backend-specific runtime errors).
    fn forward(
        &mut self,
        seq_ids: &[SeqId],
        input_tokens: &[Vec<TokenId>],
        positions: &[Vec<usize>],
        kv_block_ids: &[Vec<usize>],
        num_computed_tokens: &[usize],
        is_prefill: &[bool],
    ) -> Result<BatchOutput>;

    /// Run a forward pass returning raw per-token logits for each sequence.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the underlying backend fails (e.g. tensor allocation,
    /// shape mismatch, I/O, or backend-specific runtime errors).
    fn forward_logits(
        &mut self,
        seq_ids: &[SeqId],
        input_tokens: &[Vec<TokenId>],
        positions: &[Vec<usize>],
        kv_block_ids: &[Vec<usize>],
        num_computed_tokens: &[usize],
        is_prefill: &[bool],
    ) -> Result<Vec<Vec<f32>>>;

    /// Compute embeddings for the given input tokens without sampling.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the underlying backend fails to produce embeddings.
    fn embed(
        &mut self,
        input_tokens: &[Vec<TokenId>],
        positions: &[Vec<usize>],
    ) -> Result<Vec<Vec<f32>>>;

    #[cfg(feature = "candle")]
    /// Run a candle-only forward pass using the model's paged KV cache.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the candle backend fails to allocate tensors or
    /// perform the forward pass.
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
    ///
    /// # Errors
    ///
    /// Returns `Err` if the underlying forward pass fails.
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
    #[must_use]
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
            // invariant: pre-conditions make this infallible at this call site.
            .expect("default Arc should be uniquely owned")
            .forward(
                &[1, 2, 3],
                &[vec![0], vec![0], vec![0]],
                &[vec![0], vec![0], vec![0]],
                &[vec![0], vec![0], vec![0]],
                &[0, 0, 0],
                &[true, true, true],
            )
            // invariant: pre-conditions make this infallible at this call site.
            .expect("stub forward should succeed");
        assert_eq!(output.seq_ids, vec![1, 2, 3]);
        assert_eq!(output.next_tokens, vec![0, 0, 0]);
    }
}
