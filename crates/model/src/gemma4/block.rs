//! Gemma4 Transformer Block implementation.

#![allow(dead_code)]
#![allow(unused_variables)]

use crate::config::{LayerType, ModelConfig, RoPEConfig};
use crate::gemma4::attention::Gemma4Attention;
use crate::gemma4::mlp::GeGLU;
use candle_core::{Result, Tensor};
use candle_nn::{linear, Linear};

pub struct Gemma4Block {
    attention: Gemma4Attention,
    mlp: GeGLU,
    input_layernorm: Linear,
    post_attention_layernorm: Linear,
    _layer_type: LayerType,
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
                rope_theta: 10000.0,
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

        let input_layernorm = linear(hidden_size, hidden_size, vb.pp("input_layernorm"))?;
        let post_attention_layernorm =
            linear(hidden_size, hidden_size, vb.pp("post_attention_layernorm"))?;

        Ok(Self {
            attention,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            _layer_type: layer_type,
        })
    }

    pub fn forward(&self, x: &Tensor, positions: &[usize]) -> Result<Tensor> {
        let residual = x.clone();
        let x = self.rms_norm(x, &self.input_layernorm)?;
        let x = self.attention.forward(&x, positions)?;
        let x = x.add(&residual)?;

        let residual = x.clone();
        let x = self.rms_norm(&x, &self.post_attention_layernorm)?;
        let x = self.mlp.forward(&x)?;
        x.add(&residual)
    }

    fn rms_norm(&self, x: &Tensor, weight: &Linear) -> Result<Tensor> {
        let hidden_size = x.dims().last().unwrap();
        let x_flat = x.reshape(((), *hidden_size))?;
        let weight = weight.weight().reshape((*hidden_size,))?;
        let variance = x_flat.sqr()?.mean(1)?;
        let x = x_flat.broadcast_div(&(variance + 1e-6)?.sqrt()?)?;
        let x = x.broadcast_mul(&weight)?;
        x.reshape(x.dims())
    }
}

#[cfg(test)]
mod tests {
    use crate::config::{Architecture, LayerType, ModelConfig, RoPEConfig};

    #[test]
    fn test_gemma4_block_forward_shape() {
        let config = ModelConfig {
            architecture: Architecture::Gemma4,
            hidden_size: 1536,
            num_layers: 35,
            num_heads: 8,
            num_kv_heads: 1,
            head_dim: 256,
            vocab_size: 262144,
            intermediate_size: 6144,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-6,
            sliding_window: Some(512),
            tie_word_embeddings: true,
            max_position_embeddings: 131072,
            layer_types: vec![LayerType::SlidingAttention; 35],
            rope_configs: vec![
                RoPEConfig {
                    rope_theta: 10000.0,
                    partial_rotary_factor: 1.0
                };
                35
            ],
            use_double_wide_mlp: false,
            num_experts: None,
            top_k_experts: None,
            expert_intermediate_size: None,
        };
    }
}
