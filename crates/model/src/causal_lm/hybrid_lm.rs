//! Generic hybrid causal LM shell (GDN linear layers + paged full attention).
//!
//! Unlike [`super::CausalLm`], tracks per-sequence GDN auxiliary state and supports
//! an optional LM head with tied-embedding fallback.

use std::collections::HashMap;
use std::sync::Arc;

use super::{
    LayerAuxMut, LayerCtx, embed_sequence, forward_batch, greedy_sample_token, logits_to_vector,
    map_candle, run_layers, run_layers_upto,
};
use crate::components::gated_delta::GatedDeltaState;
use crate::paged_tensor::PagedKvCache;
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Embedding, Linear};
use parking_lot::Mutex;
use vllm_traits::{BatchOutput, BlockId, ModelBackend, Result, SampledToken, SeqId, TokenId};

/// Config surface required by [`HybridLm`] for inference and `ModelBackend` metadata.
pub trait HybridLmConfig: Clone {
    fn vocab_size(&self) -> usize;
    fn hidden_size(&self) -> usize;
    fn num_layers(&self) -> usize;
    fn num_kv_heads(&self) -> usize;
}

#[derive(Debug)]
/// `HybridLm`. See the type definition for fields and behavior.
pub struct HybridLm<B, Norm, C> {
    config: C,
    pub(crate) embed_tokens: Embedding,
    pub(crate) layers: Vec<B>,
    pub(crate) norm: Norm,
    pub(crate) lm_head: Option<Linear>,
    pub(crate) kv_cache: Arc<Mutex<PagedKvCache>>,
    gdn_states: HashMap<SeqId, Vec<Option<GatedDeltaState>>>,
    device: Device,
}

impl<B, Norm, C> HybridLm<B, Norm, C>
where
    B: super::DecoderLayer + Send + Sync,
    Norm: Module + Send + Sync,
    C: HybridLmConfig,
{
    pub fn from_parts(
        config: C,
        embed_tokens: Embedding,
        layers: Vec<B>,
        norm: Norm,
        lm_head: Option<Linear>,
        kv_cache: Arc<Mutex<PagedKvCache>>,
        device: Device,
    ) -> Self {
        Self {
            config,
            embed_tokens,
            layers,
            norm,
            lm_head,
            kv_cache,
            gdn_states: HashMap::new(),
            device,
        }
    }

    /// Run the forward pass with the paged `KV` cache enabled.
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    pub fn forward_with_cache(
        &mut self,
        seq_id: SeqId,
        tokens: &[TokenId],
        num_computed_tokens: usize,
        block_ids: &[BlockId],
        positions: &[usize],
        is_prefill: bool,
    ) -> Result<(Tensor, usize)> {
        let vocab_size = self.config.vocab_size();
        if tokens.is_empty() {
            let logits = map_candle(Tensor::zeros((1, 1, vocab_size), DType::F32, &self.device))?;
            return Ok((logits, 0));
        }

        let hidden = embed_sequence(&self.embed_tokens, tokens, &self.device, is_prefill)?;
        let num_layers = self.layers.len();
        let gdn_states = self
            .gdn_states
            .entry(seq_id)
            .or_insert_with(|| vec![None; num_layers]);

        let hidden = {
            let mut kv_cache = self.kv_cache.lock();
            let mut ctx = LayerCtx {
                kv_cache: &mut kv_cache,
                block_ids,
                positions,
                num_computed_tokens,
                is_prefill,
                aux: Some(LayerAuxMut::Gdn(gdn_states)),
            };
            run_layers(&self.layers, hidden, &mut ctx)?
        };
        let hidden = map_candle(self.norm.forward(&hidden))?;
        let logits = forward_lm_head(&self.embed_tokens, self.lm_head.as_ref(), &hidden)?;
        Ok((logits, 0))
    }

    /// Returns a clone of the shared `Arc<Mutex<PagedKvCache>>` for
    /// multi-node `KV` block transfer wiring (Phase 41 OPS-32a second-half).
    #[must_use]
    pub fn paged_kv_cache(&self) -> Arc<Mutex<PagedKvCache>> {
        Arc::clone(&self.kv_cache)
    }
}

impl<B, Norm, C> ModelBackend for HybridLm<B, Norm, C>
where
    B: super::DecoderLayer + Send + Sync,
    Norm: Module + Send + Sync,
    C: HybridLmConfig + Send + Sync,
{
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
                seq_ids[i],
                &input_tokens[i],
                num_computed_tokens[i],
                &kv_block_ids[i],
                &positions[i],
                prefill,
            )?;
            let token = greedy_sample_token(&logits, prefill)?;
            // See comment in causal_lm::model::forward: model-layer
            // legacy `forward` returns placeholder SampledToken
            // (the engine uses `forward_logits` + sample_batch_with_params).
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
                seq_ids[i],
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
        const EMBED_SEQ_ID: SeqId = 0;
        let mut embeddings = Vec::with_capacity(input_tokens.len());
        let hidden_size = self.config.hidden_size();
        let num_layers = self.layers.len();

        for (i, tokens) in input_tokens.iter().enumerate() {
            if tokens.is_empty() {
                embeddings.push(vec![0.0; hidden_size]);
                continue;
            }

            let positions = if i < positions.len() && !positions[i].is_empty() {
                positions[i].clone()
            } else {
                (0..tokens.len()).collect()
            };
            let block_ids = [0usize];

            let hidden = embed_sequence(&self.embed_tokens, tokens, &self.device, true)?;
            self.gdn_states.insert(EMBED_SEQ_ID, vec![None; num_layers]);
            let gdn_states = self
                .gdn_states
                // invariant: `EMBED_SEQ_ID` was just inserted above; cannot be missing.
                .get_mut(&EMBED_SEQ_ID)
                .expect("embed gdn states");

            let hidden = {
                let mut kv_cache = self.kv_cache.lock();
                let mut ctx = LayerCtx {
                    kv_cache: &mut kv_cache,
                    block_ids: &block_ids,
                    positions: &positions,
                    num_computed_tokens: 0,
                    is_prefill: true,
                    aux: Some(LayerAuxMut::Gdn(gdn_states)),
                };
                run_layers(&self.layers, hidden, &mut ctx)?
            };
            let hidden = map_candle(self.norm.forward(&hidden))?;
            let pooled = map_candle(hidden.mean(0)?.flatten_all()?.to_vec1::<f32>())?;
            embeddings.push(pooled);
        }

        Ok(embeddings)
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size()
    }

    fn num_layers(&self) -> usize {
        self.config.num_layers()
    }

    fn num_heads(&self) -> usize {
        self.config.num_kv_heads()
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
            let num_layers = self.layers.len();
            let gdn_states = self
                .gdn_states
                .entry(seq_ids[i])
                .or_insert_with(|| vec![None; num_layers]);

            let hidden = {
                let mut kv_cache = self.kv_cache.lock();
                let mut ctx = LayerCtx {
                    kv_cache: &mut kv_cache,
                    block_ids: &kv_block_ids[i],
                    positions: &positions[i],
                    num_computed_tokens: num_computed_tokens[i],
                    is_prefill: prefill,
                    aux: Some(LayerAuxMut::Gdn(gdn_states)),
                };
                run_layers_upto(&self.layers, hidden, &mut ctx, upto_layer)?
            };
            let hidden = map_candle(self.norm.forward(&hidden))?;
            let logits = forward_lm_head(&self.embed_tokens, self.lm_head.as_ref(), &hidden)?;
            let token = greedy_sample_token(&logits, prefill)?;
            // See comment in causal_lm::model::forward_to_layer:
            // placeholder SampledToken; engine uses forward_logits.
            Ok(SampledToken {
                token,
                logprob: 0.0,
                top_logprobs: Vec::new(),
            })
        })
    }
}

fn forward_lm_head(
    embed_tokens: &Embedding,
    lm_head: Option<&Linear>,
    hidden: &Tensor,
) -> Result<Tensor> {
    lm_head.map_or_else(
        || {
            let embed_w = embed_tokens.embeddings().clone();
            map_candle(Linear::new(embed_w, None).forward(hidden))
        },
        |head| map_candle(head.forward(hidden)),
    )
}
