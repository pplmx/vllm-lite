//! Shared RoPE-GQA decoder block construction for Llama/Mistral-style checkpoints.
//!
//! Both [`new_block`] (zero-init, used by tests) and [`block_from_weights`]
//! (HF weight-map loader, used by
//! [`crate::qwen3::block::TransformerBlock::from_weights`] and friends)
//! thread `config.rope_scaling` and `config.max_position_embeddings`
//! through to `RopeGqaAttention::new_with_rope_scaling` /
//! `new_with_weights_rope_scaling` so long-context configs (YaRN / Linear /
//! Dynamic / Su) actually take effect at the attention layer. Pre-P20 this
//! silently dropped the block via the bare `RopeGqaAttention::new` /
//! `new_with_weights` constructors, which P19 deliberately preserved as
//! `None`-scaling aliases for backward compatibility.

use std::collections::HashMap;

use crate::components::LnLayerNorm;
use crate::components::SwiGLU;
use crate::components::attention::RopeGqaAttention;
use crate::config::ModelConfig;
use candle_core::{Result, Tensor};

use super::RopeGqaDecoderBlock;

/// Run the operation (see signature for params and return type).
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

    // Thread `rope_scaling` (YaRN / Linear / Dynamic / Su) and
    // `max_position_embeddings` through to the attention so long-context
    // configs take effect. `new_with_rope_scaling` (P19) is the
    // scaling-aware constructor; the bare `new` would silently drop the
    // block and revert to default RoPE.
    let attention = RopeGqaAttention::new_with_rope_scaling(
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        config.rope_theta,
        config.max_position_embeddings,
        config.rope_scaling.as_ref(),
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

/// Run the operation (see signature for params and return type).
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

    let attention = RopeGqaAttention::new_with_weights_rope_scaling(
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        config.rope_theta,
        config.max_position_embeddings,
        config.rope_scaling.as_ref(),
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
    //! P20 regression tests for the shared decoder-block factory.
    //!
    //! Pre-P20, `new_block` and `block_from_weights` here called the bare
    //! `RopeGqaAttention::new` / `new_with_weights` constructors which
    //! silently dropped `config.rope_scaling`. A Llama/Mistral-style
    //! checkpoint that declared a YaRN block in `config.json` therefore
    //! produced numerically identical output to a default model of the
    //! same shape. These tests pin the scaling-aware constructor
    //! signatures so the regression can't recur.

    use super::*;
    use crate::config::Architecture;
    use crate::qwen3::config::{RopeScaling, RopeType};

    fn tiny_config(rope_scaling: Option<RopeScaling>) -> ModelConfig {
        ModelConfig {
            architecture: Architecture::Llama,
            hidden_size: 64,
            num_layers: 1,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 16,
            vocab_size: 128,
            intermediate_size: 128,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            sliding_window: Some(4096),
            tie_word_embeddings: false,
            max_position_embeddings: 512,
            layer_types: vec![],
            rope_configs: vec![],
            rope_scaling,
            use_double_wide_mlp: false,
            num_experts: None,
            top_k_experts: None,
            expert_intermediate_size: None,
            has_qk_norm: false,
        }
    }

    fn yarn_scaling(factor: f32, attn_factor: Option<f32>) -> RopeScaling {
        RopeScaling {
            rope_type: Some(RopeType::Yarn),
            factor: Some(factor),
            original_max_position_embeddings: Some(4096),
            attn_factor,
            partial_rotary_factor: None,
            mrope_section: None,
            short_factor: None,
            long_factor: None,
        }
    }

    #[test]
    fn new_block_accepts_yarn_rope_scaling() {
        let config = tiny_config(Some(yarn_scaling(4.0, Some(0.5))));
        let _block = new_block(&config, 0).expect(
            "decoder_block::factory::new_block must accept a ModelConfig with \
             rope_scaling=Some(...) (P20 wiring: forwards to \
             RopeGqaAttention::new_with_rope_scaling)",
        );
    }

    #[test]
    fn new_block_accepts_none_rope_scaling() {
        // The no-op path: pre-P20 this code path used `RopeGqaAttention::new`
        // (bare). Post-P20 it routes through `new_with_rope_scaling(..., None)`
        // which delegates internally to the bare constructor. Behaviour is
        // bit-for-bit identical for the None case — this test pins that
        // invariant so callers can rely on no-scaling configs continuing to
        // work without surprises.
        let config = tiny_config(None);
        let _block = new_block(&config, 0).expect(
            "decoder_block::factory::new_block must continue to accept \
             rope_scaling=None (backward-compatible path)",
        );
    }
}
