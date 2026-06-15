#![allow(non_snake_case, clippy::too_many_arguments)]
//! Qwen3.5 hybrid causal language model (GDN + full attention).

use std::collections::HashMap;

use crate::causal_lm::{
    embed_sequence, forward_batch, greedy_sample_token, logits_to_vector, map_candle, run_layers,
    run_layers_upto, LayerAuxMut, LayerCtx,
};
use crate::components::positional::MRoPE;
use crate::paged_tensor::PagedKvCache;
use crate::qwen3_5::block::{FullAttentionBlock35, HybridBlock, LinearAttentionBlock};
use crate::qwen3_5::config::{parse_layer_types, LayerType};
use crate::qwen3_5::gated_delta::GatedDeltaState;
use crate::qwen3_5::weights::load_hybrid_weights;
use crate::qwen3_config::Qwen3Config;
use candle_core::{DType, Device, Module, Result as CandleResult, Tensor};
use candle_nn::{Embedding, LayerNorm, Linear, VarBuilder};
use vllm_traits::{BatchOutput, BlockId, ModelBackend, Result as EngineResult, SeqId, TokenId};

pub struct Qwen35HybridModel {
    pub(crate) config: Qwen3Config,
    pub(crate) embed_tokens: Embedding,
    pub(crate) layers: Vec<HybridBlock>,
    pub(crate) norm: LayerNorm,
    pub(crate) lm_head: Option<Linear>,
    pub(crate) kv_cache: PagedKvCache,
    pub(crate) gdn_states: HashMap<SeqId, Vec<Option<GatedDeltaState>>>,
    pub(crate) device: Device,
    pub(crate) layer_types: Vec<LayerType>,
}

impl Qwen35HybridModel {
    pub fn new(
        config: Qwen3Config,
        device: Device,
        num_kv_blocks: usize,
        kv_quantization: bool,
    ) -> CandleResult<Self> {
        let vocab_size = config.vocab_size();
        let hidden_size = config.hidden_size();

        let embeddings =
            Tensor::zeros((vocab_size, hidden_size), candle_core::DType::F32, &device)?;
        let embed_tokens = Embedding::new(embeddings, hidden_size);

        let layer_types = parse_layer_types(&config);
        let mut layers = Vec::new();

        let rope = MRoPE::from_config(&config);

        for layer_type in &layer_types {
            let layer = match layer_type {
                LayerType::LinearAttention => HybridBlock::Linear(LinearAttentionBlock::new(
                    hidden_size,
                    16,
                    4,
                    2,
                    VarBuilder::zeros(DType::F32, &device),
                )?),
                LayerType::FullAttention => HybridBlock::Full(FullAttentionBlock35::new(
                    hidden_size,
                    config.num_attention_heads(),
                    config.num_key_value_heads(),
                    config.head_dim(),
                    config.intermediate_size(),
                    config.rms_norm_eps(),
                    rope.clone(),
                    VarBuilder::zeros(DType::F32, &device),
                )?),
            };
            layers.push(layer);
        }

        let norm = candle_nn::layer_norm(
            hidden_size,
            config.rms_norm_eps(),
            VarBuilder::zeros(DType::F32, &device),
        )?;

        let kv_cache = PagedKvCache::new(
            config.num_hidden_layers(),
            config.num_key_value_heads(),
            config.head_dim(),
            num_kv_blocks,
            device.clone(),
            kv_quantization,
        )?;

        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
            lm_head: None,
            kv_cache,
            gdn_states: HashMap::new(),
            device,
            layer_types,
        })
    }

    pub fn from_weights(
        config: Qwen3Config,
        device: Device,
        weights: HashMap<String, Tensor>,
        num_kv_blocks: usize,
        kv_quantization: bool,
    ) -> CandleResult<Self> {
        let mut model = Self::new(
            config.clone(),
            device.clone(),
            num_kv_blocks,
            kv_quantization,
        )?;
        load_hybrid_weights(&mut model, &config, &weights)?;
        Ok(model)
    }

    pub fn forward_with_cache(
        &mut self,
        seq_id: SeqId,
        tokens: &[TokenId],
        num_computed_tokens: usize,
        block_ids: &[BlockId],
        positions: &[usize],
        is_prefill: bool,
    ) -> EngineResult<(Tensor, usize)> {
        let vocab_size = self.config.vocab_size();
        if tokens.is_empty() {
            let logits = map_candle(Tensor::zeros(
                (1, 1, vocab_size),
                DType::F32,
                &self.device,
            ))?;
            return Ok((logits, 0));
        }

        let hidden = embed_sequence(&self.embed_tokens, tokens, &self.device, is_prefill)?;
        let num_layers = self.layers.len();
        let gdn_states = self
            .gdn_states
            .entry(seq_id)
            .or_insert_with(|| vec![None; num_layers]);

        let mut ctx = LayerCtx {
            kv_cache: &mut self.kv_cache,
            block_ids,
            positions,
            num_computed_tokens,
            is_prefill,
            aux: Some(LayerAuxMut::Gdn(gdn_states)),
        };
        let hidden = run_layers(&self.layers, hidden, &mut ctx)?;
        let hidden = map_candle(self.norm.forward(&hidden))?;

        let logits = if let Some(ref lm_head) = self.lm_head {
            map_candle(lm_head.forward(&hidden))?
        } else {
            let embed_w = self.embed_tokens.embeddings().clone();
            map_candle(Linear::new(embed_w, None).forward(&hidden))?
        };
        Ok((logits, 0))
    }
}

impl ModelBackend for Qwen35HybridModel {
    fn forward(
        &mut self,
        seq_ids: &[SeqId],
        input_tokens: &[Vec<TokenId>],
        positions: &[Vec<usize>],
        kv_block_ids: &[Vec<usize>],
        num_computed_tokens: &[usize],
        is_prefill: &[bool],
    ) -> EngineResult<BatchOutput> {
        forward_batch(seq_ids, is_prefill, |i, prefill| {
            let (logits, _) = self.forward_with_cache(
                seq_ids[i],
                &input_tokens[i],
                num_computed_tokens[i],
                &kv_block_ids[i],
                &positions[i],
                prefill,
            )?;
            greedy_sample_token(&logits, prefill)
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
    ) -> EngineResult<Vec<Vec<f32>>> {
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
    ) -> EngineResult<Vec<Vec<f32>>> {
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
            self.gdn_states
                .insert(EMBED_SEQ_ID, vec![None; num_layers]);
            let gdn_states = self
                .gdn_states
                .get_mut(&EMBED_SEQ_ID)
                .expect("embed gdn states");

            let mut ctx = LayerCtx {
                kv_cache: &mut self.kv_cache,
                block_ids: &block_ids,
                positions: &positions,
                num_computed_tokens: 0,
                is_prefill: true,
                aux: Some(LayerAuxMut::Gdn(gdn_states)),
            };
            let hidden = run_layers(&self.layers, hidden, &mut ctx)?;
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
        self.config.num_hidden_layers()
    }

    fn num_heads(&self) -> usize {
        self.config.num_key_value_heads()
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
    ) -> EngineResult<BatchOutput> {
        forward_batch(seq_ids, is_prefill, |i, prefill| {
            let tokens = &input_tokens[i];
            if tokens.is_empty() {
                return Ok(0);
            }

            let hidden = embed_sequence(&self.embed_tokens, tokens, &self.device, prefill)?;
            let num_layers = self.layers.len();
            let gdn_states = self
                .gdn_states
                .entry(seq_ids[i])
                .or_insert_with(|| vec![None; num_layers]);

            let mut ctx = LayerCtx {
                kv_cache: &mut self.kv_cache,
                block_ids: &kv_block_ids[i],
                positions: &positions[i],
                num_computed_tokens: num_computed_tokens[i],
                is_prefill: prefill,
                aux: Some(LayerAuxMut::Gdn(gdn_states)),
            };
            let hidden = run_layers_upto(&self.layers, hidden, &mut ctx, upto_layer)?;
            let hidden = map_candle(self.norm.forward(&hidden))?;

            let logits = if let Some(ref lm_head) = self.lm_head {
                map_candle(lm_head.forward(&hidden))?
            } else {
                let embed_w = self.embed_tokens.embeddings().clone();
                map_candle(candle_nn::Linear::new(embed_w, None).forward(&hidden))?
            };

            greedy_sample_token(&logits, prefill)
        })
    }
}

#[cfg(test)]
#[path = "model_tests.rs"]
mod tests;
