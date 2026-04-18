#![allow(dead_code)]

use std::collections::HashMap;

use crate::components::GqaAttention;
use crate::config::ModelConfig;
use crate::components::SwiGLU;
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
        weights: &HashMap<String, Tensor>,
    ) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_heads;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;
        let intermediate_size = config.intermediate_size;
        let rms_norm_eps = config.rms_norm_eps;

        let q_w = weights
            .get(&format!(
                "model.layers.{}.self_attn.q_proj.weight",
                layer_idx
            ))
            .cloned();
        let k_w = weights
            .get(&format!(
                "model.layers.{}.self_attn.k_proj.weight",
                layer_idx
            ))
            .cloned();
        let v_w = weights
            .get(&format!(
                "model.layers.{}.self_attn.v_proj.weight",
                layer_idx
            ))
            .cloned();
        let o_w = weights
            .get(&format!(
                "model.layers.{}.self_attn.o_proj.weight",
                layer_idx
            ))
            .cloned();
        let gate_w = weights
            .get(&format!("model.layers.{}.mlp.gate_proj.weight", layer_idx))
            .cloned();
        let up_w = weights
            .get(&format!("model.layers.{}.mlp.up_proj.weight", layer_idx))
            .cloned();
        let down_w = weights
            .get(&format!("model.layers.{}.mlp.down_proj.weight", layer_idx))
            .cloned();
        let input_ln_w = weights
            .get(&format!(
                "model.layers.{}.input_layernorm.weight",
                layer_idx
            ))
            .cloned();
        let post_attn_ln_w = weights
            .get(&format!(
                "model.layers.{}.post_attention_layernorm.weight",
                layer_idx
            ))
            .cloned();

        let q_w = match q_w {
            Some(w) => w,
            None => return Err(candle_core::Error::msg("Missing q_proj weight")),
        };
        let k_w = match k_w {
            Some(w) => w,
            None => return Err(candle_core::Error::msg("Missing k_proj weight")),
        };
        let v_w = match v_w {
            Some(w) => w,
            None => return Err(candle_core::Error::msg("Missing v_proj weight")),
        };
        let o_w = match o_w {
            Some(w) => w,
            None => return Err(candle_core::Error::msg("Missing o_proj weight")),
        };
        let gate_w = match gate_w {
            Some(w) => w,
            None => return Err(candle_core::Error::msg("Missing gate_proj weight")),
        };
        let up_w = match up_w {
            Some(w) => w,
            None => return Err(candle_core::Error::msg("Missing up_proj weight")),
        };
        let down_w = match down_w {
            Some(w) => w,
            None => return Err(candle_core::Error::msg("Missing down_proj weight")),
        };
        let input_ln_w = match input_ln_w {
            Some(w) => w,
            None => return Err(candle_core::Error::msg("Missing input_layernorm weight")),
        };
        let post_attn_ln_w = match post_attn_ln_w {
            Some(w) => w,
            None => {
                return Err(candle_core::Error::msg(
                    "Missing post_attention_layernorm weight",
                ));
            }
        };

        let input_ln_dim = input_ln_w.dim(0).unwrap_or(hidden_size);
        let input_ln_bias = Tensor::zeros(input_ln_dim, input_ln_w.dtype(), input_ln_w.device())?;
        let input_layernorm = LayerNorm::new(input_ln_w, input_ln_bias, rms_norm_eps);

        let post_attn_dim = post_attn_ln_w.dim(0).unwrap_or(hidden_size);
        let post_attn_bias = Tensor::zeros(
            post_attn_dim,
            post_attn_ln_w.dtype(),
            post_attn_ln_w.device(),
        )?;
        let post_attention_layernorm = LayerNorm::new(post_attn_ln_w, post_attn_bias, rms_norm_eps);

        let attention = GqaAttention::new_with_weights(
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            q_w,
            k_w,
            v_w,
            o_w,
            crate::components::AttentionConfig::default(),
            false,
            None,
            None,
        )?;

        let mlp = SwiGLU::new_with_weights(hidden_size, intermediate_size, gate_w, up_w, down_w)?;

        Ok(Self {
            input_layernorm,
            post_attention_layernorm,
            attention,
            mlp,
        })
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

    /// Use tiny config for fast unit tests
    fn test_config() -> ModelConfig {
        ModelConfig::test_tiny()
    }

    #[test]
    fn test_llama_block_forward_shape() {
        let config = test_config();
        let block = LlamaBlock::new(&config, 0).unwrap();

        let input = Tensor::ones((2, 10, config.hidden_size), DType::F32, &Device::Cpu).unwrap();
        let output = block.forward(&input).unwrap();

        assert_eq!(output.dims(), &[2, 10, config.hidden_size]);
    }

    #[test]
    fn test_llama_block_single_token() {
        let config = test_config();
        let block = LlamaBlock::new(&config, 0).unwrap();

        let input = Tensor::ones((1, 1, config.hidden_size), DType::F32, &Device::Cpu).unwrap();
        let output = block.forward(&input).unwrap();

        assert_eq!(output.dims(), &[1, 1, config.hidden_size]);
    }

    #[test]
    fn test_llama_block_different_batch_sizes() {
        let config = test_config();

        for batch_size in [1usize, 2, 4] {
            let block = LlamaBlock::new(&config, 0).unwrap();
            let input = Tensor::ones(
                (batch_size, 5, config.hidden_size),
                DType::F32,
                &Device::Cpu,
            )
            .unwrap();
            let output = block.forward(&input).unwrap();
            assert_eq!(output.dims()[0], batch_size);
        }
    }

    /// Slow integration test with full-size model
    #[test]
    #[ignore = "slow integration test - run with --ignored for full model validation"]
    fn test_llama_block_full_size() {
        let config = ModelConfig::llama_7b();
        let block = LlamaBlock::new(&config, 0).unwrap();

        let input = Tensor::ones((2, 10, 4096), DType::F32, &Device::Cpu).unwrap();
        let output = block.forward(&input).unwrap();

        assert_eq!(output.dims(), &[2, 10, 4096]);
    }
}
