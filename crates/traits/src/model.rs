//! `ModelBackend` trait and the wire types that flow through it.
//!
//! The trait is the single seam between `vllm-core` (scheduler, KV cache,
//! engine) and `vllm-model` (architecture implementations). Adding a new
//! architecture means implementing `ModelBackend` and registering the
//! impl in `vllm_model::arch::ArchitectureRegistry`.
#![allow(clippy::module_name_repetitions)]

use std::sync::Arc;

use crate::types::{BatchOutput, SampledToken, SeqId, TokenId};

/// Error type for Model. Returned from every fallible public API; covers I/O, validation, and
/// resource-limit failures. Use [`Result<T>`] alias in the same module.
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
    /// Construct a generic message-bearing error variant.
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
    /// Vocabulary size of the loaded model weights.
    fn vocab_size(&self) -> usize;

    /// Number of transformer layers in the model stack.
    fn num_layers(&self) -> usize;

    /// Number of query attention heads.
    fn num_heads(&self) -> usize;

    /// Whether a prefix-cache hit can safely skip prefill for a NEW
    /// sequence sharing a cached prompt prefix.
    ///
    /// Default `true` (the paged KV cache fully captures a plain
    /// transformer's sequence state). Stateful backends whose hidden state
    /// is NOT stored in the KV cache — e.g. the Qwen3.5 hybrid GDN recurrent
    /// layers — MUST return `false`: a full prefix hit would admit the
    /// sequence straight into Decode with fresh `None` recurrent states,
    /// erroring on the first decode.
    fn supports_prefix_caching(&self) -> bool {
        true
    }

    /// Called by the engine when a sequence finishes (`max_tokens`, stop
    /// sequence, or cancel) so the backend can release per-sequence state.
    ///
    /// Default no-op. Stateful backends (e.g. the Qwen3.5 hybrid GDN
    /// recurrent-state map) MUST drop the sequence's entry here — without
    /// it, every finished request leaks its per-sequence state for the
    /// engine's lifetime.
    fn on_sequence_finished(&mut self, _seq_id: SeqId) {}

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
            next_tokens: (0..seq_ids.len())
                .map(|_| SampledToken {
                    token: TokenId::default(),
                    logprob: 0.0,
                    top_logprobs: Vec::new(),
                })
                .collect(),
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
        // P36: BatchOutput::next_tokens is `Vec<SampledToken>`.
        let expected: Vec<SampledToken> = (0..3)
            .map(|_| SampledToken {
                token: TokenId::default(),
                logprob: 0.0,
                top_logprobs: Vec::new(),
            })
            .collect();
        assert_eq!(output.next_tokens, expected);
    }

    /// `supports_prefix_caching` must default to `true`.
    ///
    /// The trait method's default body is `true`, and `cargo mutants` replaced
    /// it with `false`. The engine reads it to decide whether a full prefix
    /// hit may admit a sequence straight into decode
    /// (`engine/lifecycle.rs`); `false` turns every prefix hit into a full
    /// re-prefill, so the prefix cache stops paying off while still costing
    /// the memory it reserves. `StubModelBackend` does not override it, which
    /// is exactly the default a plain transformer inherits.
    #[test]
    fn supports_prefix_caching_defaults_to_true() {
        assert!(
            StubModelBackend.supports_prefix_caching(),
            "a stateless backend whose KV fully captures the sequence must \
             allow skipping prefill on a prefix hit"
        );
    }

    /// `StubModelBackend`'s `forward_logits` must be shape-consistent.
    ///
    /// `cargo mutants` replaced the body with `Ok(vec![])`,
    /// `Ok(vec![vec![]])`, `Ok(vec![vec![0.0]])` and `Ok(vec![vec![1.0]])` —
    /// four survivors. Callers zip the per-sequence logits against the batch,
    /// so an empty outer vector silently drops every sequence, and a single
    /// inner vector attributes every sequence's logits to the first one. The
    /// stub documents a `tokens.len() * vocab` layout, and that layout is what
    /// the speculative verification path expects.
    #[test]
    fn stub_forward_logits_is_size_consistent() {
        let mut backend = StubModelBackend;
        let input = vec![vec![5, 6, 7], vec![8]];
        let positions = vec![vec![0, 1, 2], vec![0]];
        let logits = backend
            .forward_logits(
                &[1, 2],
                &input,
                &positions,
                &[vec![0], vec![0]],
                &[0, 0],
                &[true, true],
            )
            .expect("the stub never fails");

        assert_eq!(logits.len(), 2, "one entry per sequence, not one total");
        // `vocab_size` is `#[cfg(feature = "candle")]` on the trait; the stub's
        // `forward_logits` falls back to a vocab of 1 without it, and the
        // assertion must follow the same layout the code produces.
        #[cfg(feature = "candle")]
        let vocab = backend.vocab_size();
        #[cfg(not(feature = "candle"))]
        let vocab = 1;
        for (i, row) in logits.iter().enumerate() {
            assert_eq!(
                row.len(),
                input[i].len() * vocab,
                "sequence {i} must report tokens.len() * vocab logits"
            );
            assert!(
                row.iter().all(|&v| v == 0.0),
                "the stub emits zero logits, not fabricated values"
            );
        }

        // An empty batch yields no rows, not one fabricated row.
        assert!(
            backend
                .forward_logits(&[], &[], &[], &[], &[], &[])
                .expect("the stub never fails")
                .is_empty()
        );
    }

    /// `StubModelBackend`'s `embed` must be shape-consistent too.
    ///
    /// Five more survivors here (`Ok(vec![])` and the single-row variants).
    /// The stub's contract is one row per sequence, `tokens.len()` wide.
    #[test]
    fn stub_embed_is_size_consistent() {
        let mut backend = StubModelBackend;
        let input = vec![vec![1, 2, 3], vec![4, 5]];
        let positions = vec![vec![0, 1, 2], vec![0, 1]];
        let embeddings = backend
            .embed(&input, &positions)
            .expect("the stub never fails");

        assert_eq!(embeddings.len(), 2, "one row per sequence");
        for (i, row) in embeddings.iter().enumerate() {
            assert_eq!(
                row.len(),
                input[i].len(),
                "sequence {i} must report one embedding per token"
            );
            assert!(row.iter().all(|&v| v == 0.0), "the stub emits zeros");
        }

        assert!(
            backend
                .embed(&[], &[])
                .expect("the stub never fails")
                .is_empty()
        );
    }

    /// Every `StubModelBackend` accessor must stay the placeholder it documents.
    ///
    /// The trait defaults are `num_layers() == 0` and `num_heads() == 0`, with
    /// `vocab_size() == 1` under the `candle` feature, and `cargo mutants`
    /// replaced them with `1`, `1` and `0`. These values are not decorative:
    /// `vocab_size` sizes the logits row in `forward_logits`, so a `0` there
    /// makes every row empty, and the engine uses `num_layers` to decide
    /// whether a model can be sharded.
    #[test]
    fn stub_model_backend_accessors_are_the_documented_placeholders() {
        let backend = StubModelBackend;
        assert_eq!(
            backend.num_layers(),
            0,
            "the stub has no layers; a constant 1 would look shardable"
        );
        assert_eq!(backend.num_heads(), 0, "the stub has no attention heads");
        #[cfg(feature = "candle")]
        assert_eq!(
            backend.vocab_size(),
            1,
            "the stub's vocab is 1, and it sizes the logits row"
        );
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
        let expected: Vec<SampledToken> = (0..3)
            .map(|_| SampledToken {
                token: TokenId::default(),
                logprob: 0.0,
                top_logprobs: Vec::new(),
            })
            .collect();
        assert_eq!(output.next_tokens, expected);
    }
}
