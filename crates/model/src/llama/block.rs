#![allow(dead_code)]

use crate::components::AttentionConfig;
use candle_core::{Module, Result, Tensor};
use candle_nn::LayerNorm;

pub struct LlamaBlock {
    input_layernorm: LayerNorm,
    post_attention_layernorm: LayerNorm,
    attention: crate::qwen3::attention::GqaAttention,
    mlp: crate::qwen3::mlp::SwiGLU,
}

impl LlamaBlock {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
        theta: f32,
        rms_norm_eps: f64,
    ) -> Result<Self> {
        let input_layernorm = candle_nn::layer_norm(
            hidden_size,
            rms_norm_eps,
            candle_nn::VarBuilder::zeros(candle_core::DType::F32, &candle_core::Device::Cpu),
        )?;
        let post_attention_layernorm = candle_nn::layer_norm(
            hidden_size,
            rms_norm_eps,
            candle_nn::VarBuilder::zeros(candle_core::DType::F32, &candle_core::Device::Cpu),
        )?;

        let attention = crate::qwen3::attention::GqaAttention::new(
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            theta,
            None,
            AttentionConfig::default(),
            false,
        )?;

        let mlp = crate::qwen3::mlp::SwiGLU::new(hidden_size, intermediate_size, None)?;

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
