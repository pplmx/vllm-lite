#![allow(dead_code)]

use std::collections::HashMap;

use crate::config::ModelConfig;
use crate::qwen3::attention::GqaAttention;
use crate::qwen3::mlp::SwiGLU;
use candle_core::{Module, Result, Tensor};
use candle_nn::{LayerNorm, VarBuilder};

pub struct LlamaBlock {
    input_layernorm: LayerNorm,
    post_attention_layernorm: LayerNorm,
    attention: GqaAttention,
    mlp: SwiGLU,
}

impl LlamaBlock {
    pub fn new(config: &ModelConfig, _layer_idx: usize) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_heads;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;
        let intermediate_size = config.intermediate_size;
        let theta = config.rope_theta;
        let rms_norm_eps = config.rms_norm_eps;

        let vb = VarBuilder::zeros(candle_core::DType::F32, &candle_core::Device::Cpu);

        let input_layernorm =
            candle_nn::layer_norm(hidden_size, rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm =
            candle_nn::layer_norm(hidden_size, rms_norm_eps, vb.pp("post_attention_layernorm"))?;

        let attention = GqaAttention::new(
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            theta,
            None,
            crate::components::AttentionConfig::default(),
            false,
        )?;

        let mlp = SwiGLU::new(hidden_size, intermediate_size, None)?;

        Ok(Self {
            input_layernorm,
            post_attention_layernorm,
            attention,
            mlp,
        })
    }

    pub fn from_weights(
        config: &ModelConfig,
        layer_idx: usize,
        _weights: &HashMap<String, Tensor>,
    ) -> Result<Self> {
        Self::new(config, layer_idx)
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x.clone();
        let x = self.input_layernorm.forward(x)?;
        let x = self.attention.forward(&x)?;
        let x = (x + residual)?;

        let residual = x.clone();
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        x.add(&residual)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;
    use candle_core::{DType, Device, Tensor};

    #[test]
    fn test_llama_block_forward_shape() {
        let config = ModelConfig::llama_7b();
        let block = LlamaBlock::new(&config, 0).unwrap();

        let input = Tensor::ones((2, 10, 4096), DType::F32, &Device::Cpu).unwrap();
        let output = block.forward(&input).unwrap();

        assert_eq!(output.dims(), &[2, 10, 4096]);
    }

    #[test]
    fn test_llama_block_single_token() {
        let config = ModelConfig::llama_7b();
        let block = LlamaBlock::new(&config, 0).unwrap();

        let input = Tensor::ones((1, 1, 4096), DType::F32, &Device::Cpu).unwrap();
        let output = block.forward(&input).unwrap();

        assert_eq!(output.dims(), &[1, 1, 4096]);
    }

    #[test]
    fn test_llama_block_different_batch_sizes() {
        let config = ModelConfig::llama_7b();

        for batch_size in [1usize, 2, 4] {
            let block = LlamaBlock::new(&config, 0).unwrap();
            let input = Tensor::ones((batch_size, 5, 4096), DType::F32, &Device::Cpu).unwrap();
            let output = block.forward(&input).unwrap();
            assert_eq!(output.dims()[0], batch_size);
        }
    }
}
