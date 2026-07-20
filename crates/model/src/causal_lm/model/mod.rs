//! Causal language model wrapper: ties embeddings → transformer layers → logits + sampler into the `ModelBackend` interface.
//!
//! The model itself is a thin facade; the per-architecture work lives
//! in `llama/`, `qwen3/`, `qwen3_5/`, etc. `forward` dispatches to the
//! architecture-specific forward impl selected by the registry.
//!
//! Construction helpers (`new_with_block_fn`, `from_hf_weights_ln`,
//! `new_rms`, `from_hf_weights_rms`) live in the sibling [`construct`]
//! module so this file stays focused on the facade and the
//! `ModelBackend` trait impl.

mod construct;

use super::{
    LayerCtx, embed_sequence, embed_with_paged_layers, forward_batch, forward_with_paged_kv,
    greedy_sample_token, logits_to_vector, map_candle, mean_pool_embeddings, run_layers_upto,
};
use crate::components::decoder_block::PagedDecoderBlock;
use crate::config::ModelConfig;
use crate::paged_tensor::PagedKvCache;
use candle_core::{Device, Tensor};
use candle_nn::{Embedding, Module};
use vllm_traits::{BatchOutput, BlockId, ModelBackend, Result, SampledToken, SeqId, TokenId};

#[derive(Debug)]
/// Generic decoder-only causal language model shell.
pub struct CausalLm<B, Norm, Head> {
    config: ModelConfig,
    embed_tokens: Embedding,
    layers: Vec<B>,
    norm: Norm,
    lm_head: Head,
    kv_cache: PagedKvCache,
    device: Device,
    embed_through_layers: bool,
}

impl<B, Norm, Head> CausalLm<B, Norm, Head>
where
    B: PagedDecoderBlock,
    Norm: Module,
    Head: Module,
{
    #[must_use]
    pub const fn with_embed_through_layers(mut self, enabled: bool) -> Self {
        self.embed_through_layers = enabled;
        self
    }

    /// Run the forward pass with the paged KV cache enabled.
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    pub fn forward_with_cache(
        &mut self,
        tokens: &[TokenId],
        num_computed_tokens: usize,
        block_ids: &[BlockId],
        positions: &[usize],
        is_prefill: bool,
    ) -> Result<(Tensor, usize)> {
        forward_with_paged_kv(
            &self.embed_tokens,
            &self.layers,
            &self.norm,
            &self.lm_head,
            &self.device,
            self.config.vocab_size,
            tokens,
            num_computed_tokens,
            block_ids,
            positions,
            is_prefill,
            &mut self.kv_cache,
        )
    }
}

impl<B, Norm, Head> ModelBackend for CausalLm<B, Norm, Head>
where
    B: PagedDecoderBlock + Send + Sync,
    Norm: Module + Send + Sync,
    Head: Module + Send + Sync,
{
    fn forward_with_cache(
        &mut self,
        input_tokens: &[TokenId],
        num_computed: usize,
        kv_block_ids: &[usize],
        positions: &[usize],
        is_prefill: bool,
    ) -> Result<(Tensor, usize)> {
        self.forward_with_cache(
            input_tokens,
            num_computed,
            kv_block_ids,
            positions,
            is_prefill,
        )
    }

    fn forward(
        &mut self,
        seq_ids: &[SeqId],
        input_tokens: &[Vec<TokenId>],
        positions: &[Vec<usize>],
        kv_block_ids: &[Vec<usize>],
        num_computed_tokens: &[usize],
        is_prefill: &[bool],
    ) -> Result<BatchOutput> {
        forward_batch(seq_ids, is_prefill, |i, prefill| {
            let (logits, _) = self.forward_with_cache(
                &input_tokens[i],
                num_computed_tokens[i],
                &kv_block_ids[i],
                &positions[i],
                prefill,
            )?;
            let token = greedy_sample_token(&logits, prefill)?;
            // P36: model-layer `forward` is the legacy greedy path; it
            // populates the sampled token but leaves logprob/top_logprobs
            // empty (the engine samples via sample_one_with_params
            // anyway). Returning a placeholder SampledToken is safe
            // because `forward` is no longer on the engine hot path —
            // `forward_logits` + `sample_batch_with_params` is.
            Ok(SampledToken {
                token,
                logprob: 0.0,
                top_logprobs: Vec::new(),
            })
        })
    }

    fn forward_logits(
        &mut self,
        seq_ids: &[SeqId],
        input_tokens: &[Vec<TokenId>],
        positions: &[Vec<usize>],
        kv_block_ids: &[Vec<usize>],
        num_computed_tokens: &[usize],
        is_prefill: &[bool],
    ) -> Result<Vec<Vec<f32>>> {
        let mut results = Vec::with_capacity(seq_ids.len());
        for i in 0..seq_ids.len() {
            let (logits, _) = self.forward_with_cache(
                &input_tokens[i],
                num_computed_tokens[i],
                &kv_block_ids[i],
                &positions[i],
                is_prefill[i],
            )?;
            results.push(logits_to_vector(&logits, is_prefill[i])?);
        }
        Ok(results)
    }

    fn embed(
        &mut self,
        input_tokens: &[Vec<TokenId>],
        positions: &[Vec<usize>],
    ) -> Result<Vec<Vec<f32>>> {
        if self.embed_through_layers {
            embed_with_paged_layers(
                &self.embed_tokens,
                &self.layers,
                &self.norm,
                &self.device,
                self.config.hidden_size,
                &mut self.kv_cache,
                input_tokens,
                positions,
            )
        } else {
            input_tokens
                .iter()
                .map(|tokens| {
                    mean_pool_embeddings(
                        &self.embed_tokens,
                        tokens,
                        &self.device,
                        self.config.hidden_size,
                    )
                })
                .collect()
        }
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn num_layers(&self) -> usize {
        self.config.num_layers
    }

    fn num_heads(&self) -> usize {
        self.config.num_heads
    }

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
        forward_batch(seq_ids, is_prefill, |i, prefill| {
            let tokens = &input_tokens[i];
            if tokens.is_empty() {
                return Ok(SampledToken {
                    token: TokenId::default(),
                    logprob: 0.0,
                    top_logprobs: Vec::new(),
                });
            }

            let hidden = embed_sequence(&self.embed_tokens, tokens, &self.device, prefill)?;
            let mut ctx = LayerCtx {
                kv_cache: &mut self.kv_cache,
                block_ids: &kv_block_ids[i],
                positions: &positions[i],
                num_computed_tokens: num_computed_tokens[i],
                is_prefill: prefill,
                aux: None,
            };
            let hidden = run_layers_upto(&self.layers, hidden, &mut ctx, upto_layer)?;
            let hidden = map_candle(self.norm.forward(&hidden))?;
            let logits = map_candle(self.lm_head.forward(&hidden))?;
            let token = greedy_sample_token(&logits, prefill)?;
            // See comment in the `forward` impl: model-layer
            // `forward_to_layer` is a legacy path; populating token
            // with placeholder logprob is safe because the engine
            // uses `forward_logits` + `sample_batch_with_params`.
            Ok(SampledToken {
                token,
                logprob: 0.0,
                top_logprobs: Vec::new(),
            })
        })
    }
}
