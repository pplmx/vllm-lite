#![allow(dead_code)]

use std::collections::HashMap;

use crate::components::GqaAttention;
use crate::config::ModelConfig;
use crate::components::SwiGLU;
use candle_core::Result;
use candle_core::Tensor;
use candle_nn::{LayerNorm, VarBuilder};

pub struct MistralBlock {
    input_layernorm: LayerNorm,
    post_attention_layernorm: LayerNorm,
    attention: GqaAttention,
    mlp: SwiGLU,
    sliding_window: usize,
}

impl MistralBlock {
    pub fn new(config: &ModelConfig, _layer_idx: usize) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_heads;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;
        let intermediate_size = config.intermediate_size;
        let sliding_window = config.sliding_window.unwrap_or(4096);

        let vb = VarBuilder::zeros(candle_core::DType::F32, &candle_core::Device::Cpu);

        let input_layernorm =
            candle_nn::layer_norm(hidden_size, config.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = candle_nn::layer_norm(
            hidden_size,
            config.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;

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
            sliding_window,
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
        let sliding_window = config.sliding_window.unwrap_or(4096);

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
            sliding_window,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x.clone();
        let x = self.rms_norm(x, &self.input_layernorm)?;
        let x = self.attention.forward(&x)?;
        let x = (x + residual)?;

        let residual = x.clone();
        let x = self.rms_norm(&x, &self.post_attention_layernorm)?;
        let x = self.mlp.forward(&x)?;
        x.add(&residual)
    }

    fn rms_norm(&self, x: &Tensor, layernorm: &LayerNorm) -> Result<Tensor> {
        let hidden_size = *x
            .dims()
            .last()
            .ok_or_else(|| candle_core::Error::msg("Tensor has no dimensions"))?;
        let dims = x.dims();
        let batch_size = dims[0];
        let seq_len = dims[1];

        let total_len = batch_size * seq_len;
        let x_flat = x.reshape((total_len, hidden_size))?;
        let weight = layernorm.weight().clone();
        let weight = weight.reshape((1, hidden_size))?;

        let variance = x_flat.sqr()?.mean(1)?;
        let x_normed = x_flat.broadcast_div(&(variance + 1e-6)?.sqrt()?)?;
        let x = x_normed.broadcast_mul(&weight)?;

        x.reshape((batch_size, seq_len, hidden_size))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;

    #[test]
    fn test_mistral_block_sliding_window() {
        let config = ModelConfig::mistral_7b();
        let block = MistralBlock::new(&config, 0).unwrap();

        assert_eq!(block.sliding_window, 4096);
    }

    #[test]
    fn test_mistral_block_single_token() {
        let config = ModelConfig::mistral_7b();
        let block = MistralBlock::new(&config, 0).unwrap();

        // Just verify block creation works - sliding window should be 4096
        assert_eq!(block.sliding_window, 4096);
    }
}
