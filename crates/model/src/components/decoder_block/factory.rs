//! Shared RoPE-GQA decoder block construction for Llama/Mistral-style checkpoints.

use std::collections::HashMap;

use crate::components::LnLayerNorm;
use crate::components::SwiGLU;
use crate::components::attention::RopeGqaAttention;
use crate::config::ModelConfig;
use candle_core::{Result, Tensor};

use super::RopeGqaDecoderBlock;

/// Runs the operation.
/// # Errors
///
/// Returns `Err` if any required tensor allocation or weight loading fails.
pub fn new_block(config: &ModelConfig, _layer_idx: usize) -> Result<RopeGqaDecoderBlock> {
    let hidden_size = config.hidden_size;
    let num_heads = config.num_heads;
    let num_kv_heads = config.num_kv_heads;
    let head_dim = config.head_dim;
    let intermediate_size = config.intermediate_size;
    let rms_norm_eps = config.rms_norm_eps;

    let device = candle_core::Device::Cpu;

    let input_ln_weight = Tensor::ones(hidden_size, candle_core::DType::F32, &device)?;
    let input_ln_bias = Tensor::zeros(hidden_size, candle_core::DType::F32, &device)?;
    let input_layernorm = LnLayerNorm::new(input_ln_weight, input_ln_bias, rms_norm_eps);

    let post_ln_weight = Tensor::ones(hidden_size, candle_core::DType::F32, &device)?;
    let post_ln_bias = Tensor::zeros(hidden_size, candle_core::DType::F32, &device)?;
    let post_attention_layernorm = LnLayerNorm::new(post_ln_weight, post_ln_bias, rms_norm_eps);

    let attention = RopeGqaAttention::new(
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        config.rope_theta,
        None,
        crate::components::AttentionConfig::default(),
        false,
    )?;

    let mlp = SwiGLU::new(hidden_size, intermediate_size, None)?;

    Ok(RopeGqaDecoderBlock::new(
        input_layernorm,
        post_attention_layernorm,
        attention,
        mlp,
    ))
}

/// Runs the operation.
/// # Errors
///
/// Returns `Err` if the operation fails.
#[allow(clippy::implicit_hasher)]
pub fn block_from_weights(
    config: &ModelConfig,
    layer_idx: usize,
    weights: &HashMap<String, Tensor>,
) -> Result<RopeGqaDecoderBlock> {
    let hidden_size = config.hidden_size;
    let num_heads = config.num_heads;
    let num_kv_heads = config.num_kv_heads;
    let head_dim = config.head_dim;
    let intermediate_size = config.intermediate_size;
    let rms_norm_eps = config.rms_norm_eps;

    let q_w = weights
        .get(&format!("model.layers.{layer_idx}.self_attn.q_proj.weight"))
        .cloned()
        .ok_or_else(|| candle_core::Error::msg("Missing q_proj weight"))?;
    let k_w = weights
        .get(&format!("model.layers.{layer_idx}.self_attn.k_proj.weight"))
        .cloned()
        .ok_or_else(|| candle_core::Error::msg("Missing k_proj weight"))?;
    let v_w = weights
        .get(&format!("model.layers.{layer_idx}.self_attn.v_proj.weight"))
        .cloned()
        .ok_or_else(|| candle_core::Error::msg("Missing v_proj weight"))?;
    let o_w = weights
        .get(&format!("model.layers.{layer_idx}.self_attn.o_proj.weight"))
        .cloned()
        .ok_or_else(|| candle_core::Error::msg("Missing o_proj weight"))?;
    let gate_w = weights
        .get(&format!("model.layers.{layer_idx}.mlp.gate_proj.weight"))
        .cloned()
        .ok_or_else(|| candle_core::Error::msg("Missing gate_proj weight"))?;
    let up_w = weights
        .get(&format!("model.layers.{layer_idx}.mlp.up_proj.weight"))
        .cloned()
        .ok_or_else(|| candle_core::Error::msg("Missing up_proj weight"))?;
    let down_w = weights
        .get(&format!("model.layers.{layer_idx}.mlp.down_proj.weight"))
        .cloned()
        .ok_or_else(|| candle_core::Error::msg("Missing down_proj weight"))?;
    let input_ln_w = weights
        .get(&format!("model.layers.{layer_idx}.input_layernorm.weight"))
        .cloned()
        .ok_or_else(|| candle_core::Error::msg("Missing input_layernorm weight"))?;
    let post_attn_ln_w = weights
        .get(&format!(
            "model.layers.{layer_idx}.post_attention_layernorm.weight"
        ))
        .cloned()
        .ok_or_else(|| candle_core::Error::msg("Missing post_attention_layernorm weight"))?;

    let input_ln_bias = Tensor::zeros(
        input_ln_w.dim(0).unwrap_or(hidden_size),
        input_ln_w.dtype(),
        input_ln_w.device(),
    )?;
    let input_layernorm = LnLayerNorm::new(input_ln_w, input_ln_bias, rms_norm_eps);

    let post_attn_bias = Tensor::zeros(
        post_attn_ln_w.dim(0).unwrap_or(hidden_size),
        post_attn_ln_w.dtype(),
        post_attn_ln_w.device(),
    )?;
    let post_attention_layernorm = LnLayerNorm::new(post_attn_ln_w, post_attn_bias, rms_norm_eps);

    let attention = RopeGqaAttention::new_with_weights(
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        config.rope_theta,
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

    Ok(RopeGqaDecoderBlock::new(
        input_layernorm,
        post_attention_layernorm,
        attention,
        mlp,
    ))
}
