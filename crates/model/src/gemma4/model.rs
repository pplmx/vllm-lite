//! Gemma4 causal language model with paged KV cache inference.

use std::collections::HashMap;

use crate::causal_lm::{
    forward_batch, forward_with_paged_kv, greedy_sample_token, logits_to_vector,
    mean_pool_embeddings,
};
use crate::components::RmsNorm;
use crate::config::ModelConfig;
use crate::gemma4::block::Gemma4Block;
use crate::paged_tensor::PagedKvCache;
use candle_core::{Device, Result as CandleResult, Tensor};
use candle_nn::{Embedding, Linear, VarBuilder};
use vllm_traits::{BatchOutput, BlockId, ModelBackend, Result, SeqId, TokenId};

pub struct Gemma4Model {
    config: ModelConfig,
    embed_tokens: Embedding,
    layers: Vec<Gemma4Block>,
    norm: RmsNorm,
    lm_head: Linear,
    kv_cache: PagedKvCache,
    device: Device,
}

impl Gemma4Model {
    pub fn new(
        config: ModelConfig,
        device: Device,
        num_kv_blocks: usize,
        kv_quantization: bool,
    ) -> CandleResult<Self> {
        let vocab_size = config.vocab_size;
        let hidden_size = config.hidden_size;
        let num_layers = config.num_layers;

        let vb = VarBuilder::zeros(candle_core::DType::F32, &device);
        let embeddings =
            Tensor::zeros((vocab_size, hidden_size), candle_core::DType::F32, &device)?;
        let embed_tokens = Embedding::new(embeddings, hidden_size);

        let mut layers = Vec::new();
        for i in 0..num_layers {
            layers.push(Gemma4Block::new(&config, i, vb.clone())?);
        }

        let norm_weight = Tensor::ones(hidden_size, candle_core::DType::F32, &device)?;
        let norm = RmsNorm::new(norm_weight, config.rms_norm_eps);
        let lm_head_weight =
            Tensor::zeros((vocab_size, hidden_size), candle_core::DType::F32, &device)?;
        let lm_head = Linear::new(lm_head_weight, None);

        let kv_cache = PagedKvCache::new(
            num_layers,
            config.num_heads,
            config.head_dim,
            num_kv_blocks,
            device.clone(),
            kv_quantization,
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
        kv_quantization: bool,
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
            layers.push(Gemma4Block::from_weights(&config, i, &weights)?);
        }

        let norm_key = "model.norm.weight";
        let norm_weight = weights
            .get(norm_key)
            .cloned()
            .ok_or_else(|| candle_core::Error::msg(format!("Missing {}", norm_key)))?;
        let norm = RmsNorm::new(norm_weight, config.rms_norm_eps);

        let lm_head = if config.tie_word_embeddings {
            Linear::new(embed_weight, None)
        } else {
            let lm_key = "lm_head.weight";
            let lm_weight = weights
                .get(lm_key)
                .cloned()
                .or_else(|| weights.get(embed_key).cloned())
                .ok_or_else(|| candle_core::Error::msg("Missing lm_head.weight"))?;
            Linear::new(lm_weight, None)
        };

        let kv_cache = PagedKvCache::new(
            num_layers,
            config.num_heads,
            config.head_dim,
            num_kv_blocks,
            device.clone(),
            kv_quantization,
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

impl ModelBackend for Gemma4Model {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Architecture, LayerType, ModelConfig, RoPEConfig};

    fn tiny_config() -> ModelConfig {
        ModelConfig {
            architecture: Architecture::Gemma4,
            hidden_size: 64,
            num_layers: 1,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 16,
            vocab_size: 128,
            intermediate_size: 128,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-6,
            sliding_window: Some(512),
            tie_word_embeddings: true,
            max_position_embeddings: 512,
            layer_types: vec![LayerType::SlidingAttention],
            rope_configs: vec![RoPEConfig {
                rope_theta: 10000.0,
                partial_rotary_factor: 1.0,
            }],
            use_double_wide_mlp: false,
            num_experts: None,
            top_k_experts: None,
            expert_intermediate_size: None,
            has_qk_norm: false,
        }
    }

    #[test]
    fn test_gemma4_model_forward_prefill_and_decode() {
        let config = tiny_config();
        let device = Device::Cpu;
        let mut model = Gemma4Model::new(config, device, 8, false).unwrap();

        let tokens = vec![1u32, 2, 3, 4];
        let positions: Vec<usize> = (0..tokens.len()).collect();
        let block_ids = vec![0usize];

        let (logits, _) = model
            .forward_with_cache(&tokens, 0, &block_ids, &positions, true)
            .unwrap();
        assert_eq!(logits.dims()[2], 128);

        let (logits, _) = model
            .forward_with_cache(&[5], 4, &block_ids, &[4], false)
            .unwrap();
        assert_eq!(logits.dims()[2], 128);
    }
}
