//! Weight loading for Qwen3.5 hybrid models.

use std::collections::HashMap;

use crate::components::positional::MRoPE;
use crate::qwen3_5::block::{FullAttentionBlock35, HybridBlock, LinearAttentionBlock};
use crate::qwen3_5::config::LayerType;
use crate::qwen3_5::model::Qwen35HybridModel;
use crate::qwen3_config::Qwen3Config;
use candle_core::{Result as CandleResult, Tensor};
use candle_nn::{Embedding, Linear};

pub fn load_hybrid_weights(
    model: &mut Qwen35HybridModel,
    config: &Qwen3Config,
    weights: &HashMap<String, Tensor>,
) -> CandleResult<()> {
    let embed_key = if weights.contains_key("model.language_model.embed_tokens.weight") {
        "model.language_model.embed_tokens.weight"
    } else if weights.contains_key("model.embed_tokens.weight") {
        "model.embed_tokens.weight"
    } else if weights.contains_key("language_model.embed_tokens.weight") {
        "language_model.embed_tokens.weight"
    } else {
        return Err(candle_core::Error::msg("Missing embed_tokens weight"));
    };

    if let Some(w) = weights.get(embed_key) {
        model.embed_tokens = Embedding::new(w.clone(), w.dims()[1]);
    }

    let num_layers = config.num_hidden_layers();
    let hidden_size = config.hidden_size();
    let rope = MRoPE::from_config(config);

    for i in 0..num_layers {
        let prefix = format!("model.layers.{}", i);
        let layer_type = &model.layer_types[i];

        let layer = match layer_type {
            LayerType::LinearAttention => {
                let linear_block = LinearAttentionBlock::from_weights(
                    &prefix,
                    weights,
                    hidden_size,
                    16,
                    4,
                    2,
                )?;
                HybridBlock::Linear(linear_block)
            }
            LayerType::FullAttention => {
                let full_block = FullAttentionBlock35::from_weights(
                    &prefix,
                    weights,
                    hidden_size,
                    config.num_attention_heads(),
                    config.num_key_value_heads(),
                    config.head_dim(),
                    config.intermediate_size(),
                    config.rms_norm_eps(),
                    rope.clone(),
                )?;
                HybridBlock::Full(full_block)
            }
        };

        model.layers[i] = layer;
    }

    if let Some(w) = weights.get("model.norm.weight") {
        let bias = Tensor::zeros(w.dim(0).unwrap_or(hidden_size), w.dtype(), w.device())?;
        model.norm = candle_nn::LayerNorm::new(w.clone(), bias, config.rms_norm_eps());
    } else if let Some(w) = weights.get("model.language_model.norm.weight") {
        let bias = Tensor::zeros(w.dim(0).unwrap_or(hidden_size), w.dtype(), w.device())?;
        model.norm = candle_nn::LayerNorm::new(w.clone(), bias, config.rms_norm_eps());
    } else if let Some(w) = weights.get("model.final_layernorm.weight") {
        let bias = Tensor::zeros(w.dim(0).unwrap_or(hidden_size), w.dtype(), w.device())?;
        model.norm = candle_nn::LayerNorm::new(w.clone(), bias, config.rms_norm_eps());
    }

    let lm_head_key: Option<&str> = if weights.contains_key("lm_head.weight") {
        Some("lm_head.weight")
    } else if weights.contains_key("model.lm_head.weight") {
        Some("model.lm_head.weight")
    } else {
        None
    };

    if let Some(key) = lm_head_key {
        if let Some(w) = weights.get(key) {
            model.lm_head = Some(Linear::new(w.clone(), None));
        }
    }

    Ok(())
}
