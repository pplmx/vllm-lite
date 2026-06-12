#![allow(dead_code)]

use std::collections::HashMap;

use crate::causal_lm::{
    embed_sequence, forward_batch, greedy_sample_token, logits_to_vector, map_candle,
    mean_pool_embeddings,
};
use crate::config::ModelConfig;
use crate::paged_tensor::PagedKvCache;
use candle_core::{Device, Module, Result as CandleResult, Tensor};
use candle_nn::{Embedding, Linear, VarBuilder};
use vllm_traits::{BatchOutput, BlockId, ModelBackend, Result, SeqId, TokenId};

use super::block::MistralBlock;

pub struct MistralModel {
    config: ModelConfig,
    embed_tokens: Embedding,
    layers: Vec<MistralBlock>,
    norm: Linear,
    lm_head: Linear,
    kv_cache: PagedKvCache,
    device: Device,
}

impl MistralModel {
    pub fn new(config: ModelConfig, device: Device, num_kv_blocks: usize) -> CandleResult<Self> {
        let vocab_size = config.vocab_size;
        let hidden_size = config.hidden_size;
        let num_layers = config.num_layers;

        let embeddings =
            Tensor::zeros((vocab_size, hidden_size), candle_core::DType::F32, &device)?;
        let embed_tokens = Embedding::new(embeddings, hidden_size);

        let mut layers = Vec::new();
        for _ in 0..num_layers {
            layers.push(MistralBlock::new(&config, 0)?);
        }

        let vb = VarBuilder::zeros(candle_core::DType::F32, &device);
        let norm = candle_nn::linear(hidden_size, hidden_size, vb.pp("norm"))?;
        let lm_head = candle_nn::linear(hidden_size, vocab_size, vb.pp("lm_head"))?;

        let kv_cache = PagedKvCache::new(
            num_layers,
            config.num_heads,
            config.head_dim,
            num_kv_blocks,
            device.clone(),
            false,
        )?;

        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
            lm_head,
            kv_cache,
            device,
        })
    }

    pub fn from_weights(
        config: ModelConfig,
        device: Device,
        weights: HashMap<String, Tensor>,
        num_kv_blocks: usize,
    ) -> CandleResult<Self> {
        let hidden_size = config.hidden_size;
        let num_layers = config.num_layers;

        let embed_key = "model.embed_tokens.weight";
        let embed_weight = weights
            .get(embed_key)
            .cloned()
            .ok_or_else(|| candle_core::Error::msg(format!("Missing {}", embed_key)))?;
        let embed_tokens = Embedding::new(embed_weight.clone(), hidden_size);

        let mut layers = Vec::new();
        for i in 0..num_layers {
            layers.push(MistralBlock::from_weights(&config, i, &weights)?);
        }

        let norm_key = "model.norm.weight";
        let norm_weight = weights
            .get(norm_key)
            .cloned()
            .ok_or_else(|| candle_core::Error::msg(format!("Missing {}", norm_key)))?;
        let norm = Linear::new(norm_weight, None);

        let lm_head = if config.tie_word_embeddings {
            Linear::new(embed_weight, None)
        } else {
            let lm_key = "lm_head.weight";
            let lm_weight = weights
                .get(lm_key)
                .cloned()
                .or_else(|| weights.get("model.embed_tokens.weight").cloned())
                .ok_or_else(|| candle_core::Error::msg("Missing lm_head.weight"))?;
            Linear::new(lm_weight, None)
        };

        let kv_cache = PagedKvCache::new(
            num_layers,
            config.num_heads,
            config.head_dim,
            num_kv_blocks,
            device.clone(),
            false,
        )?;

        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
            lm_head,
            kv_cache,
            device,
        })
    }

    pub fn forward_with_cache(
        &mut self,
        tokens: &[TokenId],
        num_computed_tokens: usize,
        block_ids: &[BlockId],
        positions: &[usize],
        is_prefill: bool,
    ) -> Result<(Tensor, usize)> {
        if tokens.is_empty() {
            let logits = map_candle(Tensor::zeros(
                (1, 1, self.config.vocab_size),
                candle_core::DType::F32,
                &self.device,
            ))?;
            return Ok((logits, 0));
        }

        let mut hidden = embed_sequence(&self.embed_tokens, tokens, &self.device, is_prefill)?;

        if is_prefill {
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                hidden = map_candle(layer.forward_prefill(
                    &hidden,
                    &mut self.kv_cache,
                    layer_idx,
                    block_ids,
                    positions,
                ))?;
            }
        } else {
            let decode_position = [positions[0]];
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                hidden = map_candle(layer.forward_decode(
                    &hidden,
                    &mut self.kv_cache,
                    layer_idx,
                    block_ids,
                    num_computed_tokens,
                    &decode_position,
                ))?;
            }
        }

        hidden = map_candle(self.norm.forward(&hidden))?;
        let logits = map_candle(self.lm_head.forward(&hidden))?;
        Ok((logits, 0))
    }
}

impl ModelBackend for MistralModel {
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
        _positions: &[Vec<usize>],
    ) -> Result<Vec<Vec<f32>>> {
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

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn num_layers(&self) -> usize {
        self.config.num_layers
    }

    fn num_heads(&self) -> usize {
        self.config.num_heads
    }
}
