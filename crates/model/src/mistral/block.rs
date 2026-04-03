#![allow(dead_code)]

use std::collections::HashMap;

use crate::config::ModelConfig;
use crate::qwen3::attention::GqaAttention;
use crate::qwen3::mlp::SwiGLU;
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
        let theta = config.rope_theta;
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
            sliding_window,
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
        let x = self.rms_norm(x, &self.input_layernorm)?;
        let x = self.attention.forward(&x)?;
        let x = (x + residual)?;

        let residual = x.clone();
        let x = self.rms_norm(&x, &self.post_attention_layernorm)?;
        let x = self.mlp.forward(&x)?;
        x.add(&residual)
    }

    fn rms_norm(&self, x: &Tensor, layernorm: &LayerNorm) -> Result<Tensor> {
        let hidden_size = x.dims().last().unwrap();
        let dims = x.dims();
        let batch_size = dims[0];
        let seq_len = dims[1];

        let total_len = batch_size * seq_len;
        let x_flat = x.reshape((total_len, *hidden_size))?;
        let weight = layernorm.weight().clone();
        let weight = weight.reshape((1, *hidden_size))?;

        let variance = x_flat.sqr()?.mean(1)?;
        let x_normed = x_flat.broadcast_div(&(variance + 1e-6)?.sqrt()?)?;
        let x = x_normed.broadcast_mul(&weight)?;

        x.reshape((batch_size, seq_len, *hidden_size))
    }
}
