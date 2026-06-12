use std::collections::HashMap;

use crate::components::LnLayerNorm;
use crate::components::RopeGqaDecoderBlock;
use crate::components::SwiGLU;
use crate::components::attention::RopeGqaAttention;
use crate::config::ModelConfig;
use candle_core::{Result, Tensor};

/// Mistral decoder layer (alias for the shared RoPE-GQA block).
pub type MistralBlock = RopeGqaDecoderBlock;

pub fn new_block(config: &ModelConfig, _layer_idx: usize) -> Result<MistralBlock> {
    let hidden_size = config.hidden_size;
    let num_heads = config.num_heads;
    let num_kv_heads = config.num_kv_heads;
    let head_dim = config.head_dim;
    let intermediate_size = config.intermediate_size;

    let device = candle_core::Device::Cpu;

    let input_ln_weight = Tensor::ones(hidden_size, candle_core::DType::F32, &device)?;
    let input_ln_bias = Tensor::zeros(hidden_size, candle_core::DType::F32, &device)?;
    let input_layernorm = LnLayerNorm::new(input_ln_weight, input_ln_bias, config.rms_norm_eps);

    let post_ln_weight = Tensor::ones(hidden_size, candle_core::DType::F32, &device)?;
    let post_ln_bias = Tensor::zeros(hidden_size, candle_core::DType::F32, &device)?;
    let post_attention_layernorm =
        LnLayerNorm::new(post_ln_weight, post_ln_bias, config.rms_norm_eps);

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

pub fn block_from_weights(
    config: &ModelConfig,
    layer_idx: usize,
    weights: &HashMap<String, Tensor>,
) -> Result<MistralBlock> {
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
        .cloned()
        .ok_or_else(|| candle_core::Error::msg("Missing q_proj weight"))?;
    let k_w = weights
        .get(&format!(
            "model.layers.{}.self_attn.k_proj.weight",
            layer_idx
        ))
        .cloned()
        .ok_or_else(|| candle_core::Error::msg("Missing k_proj weight"))?;
    let v_w = weights
        .get(&format!(
            "model.layers.{}.self_attn.v_proj.weight",
            layer_idx
        ))
        .cloned()
        .ok_or_else(|| candle_core::Error::msg("Missing v_proj weight"))?;
    let o_w = weights
        .get(&format!(
            "model.layers.{}.self_attn.o_proj.weight",
            layer_idx
        ))
        .cloned()
        .ok_or_else(|| candle_core::Error::msg("Missing o_proj weight"))?;
    let gate_w = weights
        .get(&format!("model.layers.{}.mlp.gate_proj.weight", layer_idx))
        .cloned()
        .ok_or_else(|| candle_core::Error::msg("Missing gate_proj weight"))?;
    let up_w = weights
        .get(&format!("model.layers.{}.mlp.up_proj.weight", layer_idx))
        .cloned()
        .ok_or_else(|| candle_core::Error::msg("Missing up_proj weight"))?;
    let down_w = weights
        .get(&format!("model.layers.{}.mlp.down_proj.weight", layer_idx))
        .cloned()
        .ok_or_else(|| candle_core::Error::msg("Missing down_proj weight"))?;
    let input_ln_w = weights
        .get(&format!(
            "model.layers.{}.input_layernorm.weight",
            layer_idx
        ))
        .cloned()
        .ok_or_else(|| candle_core::Error::msg("Missing input_layernorm weight"))?;
    let post_attn_ln_w = weights
        .get(&format!(
            "model.layers.{}.post_attention_layernorm.weight",
            layer_idx
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;

    #[test]
    fn test_mistral_block_forward() {
        let config = ModelConfig::test_tiny_for(crate::config::Architecture::Mistral);
        let block = new_block(&config, 0).unwrap();
        let input = Tensor::ones(
            (1, 4, config.hidden_size),
            candle_core::DType::F32,
            &candle_core::Device::Cpu,
        )
        .unwrap();
        let output = block.forward(&input).unwrap();
        assert_eq!(output.dims(), &[1, 4, config.hidden_size]);
    }

    #[test]
    fn test_mistral_block_sliding_window_config() {
        let config = ModelConfig::test_tiny_for(crate::config::Architecture::Mistral);
        let _block = new_block(&config, 0).unwrap();
        assert_eq!(config.sliding_window, Some(4096));
    }
}
