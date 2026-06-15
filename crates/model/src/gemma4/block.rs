//! Gemma4 Transformer Block implementation.

use std::collections::HashMap;

use crate::components::RmsNorm;
use crate::components::decoder_block::PagedDecoderBlock;
use crate::config::{LayerType, ModelConfig, RoPEConfig};
use crate::gemma4::attention::Gemma4Attention;
use crate::gemma4::mlp::GeGLU;
use crate::paged_tensor::PagedKvCache;
use candle_core::{Result, Tensor};

pub struct Gemma4Block {
    attention: Gemma4Attention,
    mlp: GeGLU,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl Gemma4Block {
    pub fn new(config: &ModelConfig, layer_idx: usize, vb: candle_nn::VarBuilder) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_heads;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;
        let intermediate_size = config.intermediate_size;
        let sliding_window = config.sliding_window.unwrap_or(512);

        let layer_type = config
            .layer_types
            .get(layer_idx)
            .copied()
            .unwrap_or(LayerType::SlidingAttention);

        let rope_config = config
            .rope_configs
            .get(layer_idx)
            .cloned()
            .unwrap_or(RoPEConfig {
                rope_theta: config.rope_theta,
                partial_rotary_factor: 1.0,
            });

        let attention = Gemma4Attention::new(
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            sliding_window,
            layer_type,
            &rope_config,
            vb.clone(),
        )?;
        let mlp = GeGLU::new(hidden_size, intermediate_size, vb.clone())?;

        let input_ln_w = Tensor::ones(hidden_size, candle_core::DType::F32, vb.device())?;
        let input_layernorm = RmsNorm::new(input_ln_w, config.rms_norm_eps);
        let post_ln_w = Tensor::ones(hidden_size, candle_core::DType::F32, vb.device())?;
        let post_attention_layernorm = RmsNorm::new(post_ln_w, config.rms_norm_eps);

        Ok(Self {
            attention,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    pub fn from_weights(
        config: &ModelConfig,
        layer_idx: usize,
        weights: &HashMap<String, Tensor>,
    ) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_heads;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;
        let intermediate_size = config.intermediate_size;
        let sliding_window = config.sliding_window.unwrap_or(512);

        let layer_type = config
            .layer_types
            .get(layer_idx)
            .copied()
            .unwrap_or(LayerType::SlidingAttention);

        let rope_config = config
            .rope_configs
            .get(layer_idx)
            .cloned()
            .unwrap_or(RoPEConfig {
                rope_theta: config.rope_theta,
                partial_rotary_factor: 1.0,
            });

        let prefix = format!("model.layers.{}", layer_idx);
        let q_w = weights
            .get(&format!("{}.self_attn.q_proj.weight", prefix))
            .cloned()
            .ok_or_else(|| candle_core::Error::msg("Missing q_proj weight"))?;
        let k_w = weights
            .get(&format!("{}.self_attn.k_proj.weight", prefix))
            .cloned()
            .ok_or_else(|| candle_core::Error::msg("Missing k_proj weight"))?;
        let v_w = weights
            .get(&format!("{}.self_attn.v_proj.weight", prefix))
            .cloned()
            .ok_or_else(|| candle_core::Error::msg("Missing v_proj weight"))?;
        let o_w = weights
            .get(&format!("{}.self_attn.o_proj.weight", prefix))
            .cloned()
            .ok_or_else(|| candle_core::Error::msg("Missing o_proj weight"))?;
        let gate_w = weights
            .get(&format!("{}.mlp.gate_proj.weight", prefix))
            .cloned()
            .ok_or_else(|| candle_core::Error::msg("Missing gate_proj weight"))?;
        let up_w = weights
            .get(&format!("{}.mlp.up_proj.weight", prefix))
            .cloned()
            .ok_or_else(|| candle_core::Error::msg("Missing up_proj weight"))?;
        let down_w = weights
            .get(&format!("{}.mlp.down_proj.weight", prefix))
            .cloned()
            .ok_or_else(|| candle_core::Error::msg("Missing down_proj weight"))?;
        let input_ln_w = weights
            .get(&format!("{}.input_layernorm.weight", prefix))
            .cloned()
            .ok_or_else(|| candle_core::Error::msg("Missing input_layernorm weight"))?;
        let post_ln_w = weights
            .get(&format!("{}.post_attention_layernorm.weight", prefix))
            .cloned()
            .ok_or_else(|| candle_core::Error::msg("Missing post_attention_layernorm weight"))?;

        let attention = Gemma4Attention::new_from_weights(
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            sliding_window,
            layer_type,
            &rope_config,
            q_w,
            k_w,
            v_w,
            o_w,
        )?;
        let mlp = GeGLU::new_with_weights(hidden_size, intermediate_size, gate_w, up_w, down_w)?;
        let input_layernorm = RmsNorm::new(input_ln_w, config.rms_norm_eps);
        let post_attention_layernorm = RmsNorm::new(post_ln_w, config.rms_norm_eps);

        Ok(Self {
            attention,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    pub fn forward(&self, x: &Tensor, positions: &[usize]) -> Result<Tensor> {
        let residual = x.clone();
        let x = self.input_layernorm.forward(x)?;
        let x = self.attention.forward(&x, positions)?;
        let x = x.add(&residual)?;

        let residual = x.clone();
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        x.add(&residual)
    }

    pub fn forward_prefill(
        &self,
        x: &Tensor,
        kv_cache: &mut PagedKvCache,
        layer_idx: usize,
        block_ids: &[usize],
        positions: &[usize],
    ) -> Result<Tensor> {
        let residual = x.clone();
        let x = self.input_layernorm.forward(x)?;
        let x = self
            .attention
            .forward_prefill(&x, kv_cache, layer_idx, block_ids, positions)?;
        let x = x.add(&residual)?;

        let residual = x.clone();
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        x.add(&residual)
    }

    pub fn forward_decode(
        &self,
        x: &Tensor,
        kv_cache: &mut PagedKvCache,
        layer_idx: usize,
        block_ids: &[usize],
        num_computed_tokens: usize,
        positions: &[usize],
    ) -> Result<Tensor> {
        let residual = x.clone();
        let mut x = self.input_layernorm.forward(x)?;
        x = self.attention.forward_decode(
            &x,
            kv_cache,
            layer_idx,
            block_ids,
            num_computed_tokens,
            positions,
        )?;
        if x.dims().len() == 3 && x.dims()[1] == 1 && residual.dims().len() == 2 {
            let dims = x.dims();
            x = x.reshape((dims[0], dims[2]))?;
        }
        x = x.add(&residual)?;

        let residual = x.clone();
        x = self.post_attention_layernorm.forward(&x)?;
        x = self.mlp.forward(&x)?;
        x.add(&residual)
    }
}

impl PagedDecoderBlock for Gemma4Block {
    fn forward_prefill(
        &self,
        x: &Tensor,
        kv_cache: &mut PagedKvCache,
        layer_idx: usize,
        block_ids: &[usize],
        positions: &[usize],
    ) -> Result<Tensor> {
        Gemma4Block::forward_prefill(self, x, kv_cache, layer_idx, block_ids, positions)
    }

    fn forward_decode(
        &self,
        x: &Tensor,
        kv_cache: &mut PagedKvCache,
        layer_idx: usize,
        block_ids: &[usize],
        num_computed_tokens: usize,
        positions: &[usize],
    ) -> Result<Tensor> {
        Gemma4Block::forward_decode(
            self,
            x,
            kv_cache,
            layer_idx,
            block_ids,
            num_computed_tokens,
            positions,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Architecture, LayerType, ModelConfig, RoPEConfig};
    use candle_core::DType;

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
    fn test_gemma4_block_prefill_then_decode() {
        let config = tiny_config();
        let device = candle_core::Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let block = Gemma4Block::new(&config, 0, vb).unwrap();
        let mut kv_cache = PagedKvCache::new(1, 4, 16, 8, device.clone(), false).unwrap();

        let x = Tensor::ones((1, 4, 64), DType::F32, &device).unwrap();
        let block_ids: Vec<usize> = (0..4).map(|i| i / 16).collect();
        let positions: Vec<usize> = (0..4).collect();
        let out = block
            .forward_prefill(&x, &mut kv_cache, 0, &block_ids, &positions)
            .unwrap();
        assert_eq!(out.dims(), &[1, 4, 64]);

        let decode_x = Tensor::ones((1, 64), DType::F32, &device).unwrap();
        let decode_out = block
            .forward_decode(&decode_x, &mut kv_cache, 0, &[0], 4, &[4])
            .unwrap();
        assert_eq!(decode_out.dims(), &[1, 64]);
    }
}
