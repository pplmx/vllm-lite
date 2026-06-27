//! Weight loading for Qwen3.5 hybrid models.

use std::collections::HashMap;

use crate::causal_lm::HybridLm;
use crate::causal_lm::weights::{load_final_norm_weight, load_lm_head};
use crate::components::positional::MRoPE;
use crate::qwen3::config::Qwen3Config;
use crate::qwen3_5::block::{FullAttentionBlock35, HybridBlock, LinearAttentionBlock};
use crate::qwen3_5::config::{LayerType, parse_layer_types};
use candle_core::{Result as CandleResult, Tensor};
use candle_nn::{Embedding, LayerNorm};

/// load_hybrid_weights: load hybrid weights.
pub fn load_hybrid_weights(
    model: &mut HybridLm<HybridBlock, LayerNorm, Qwen3Config>,
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
    let layer_types = parse_layer_types(config);

    for (i, layer_type) in layer_types.iter().enumerate().take(num_layers) {
        let prefix = format!("model.layers.{}", i);

        let layer = match layer_type {
            LayerType::LinearAttention => {
                let linear_block =
                    LinearAttentionBlock::from_weights(&prefix, weights, hidden_size)?;
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

    if let Some(w) = load_final_norm_weight(weights) {
        let out_features = w.dim(0).unwrap_or(hidden_size);
        let bias = Tensor::zeros(out_features, w.dtype(), w.device())?;
        model.norm = LayerNorm::new(w, bias, config.rms_norm_eps());
    }

    let embed_weight = model.embed_tokens.embeddings().clone();
    if config.tie_word_embeddings() {
        model.lm_head = None;
    } else {
        model.lm_head = Some(load_lm_head(weights, embed_weight, false)?);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_load_hybrid_weights_tied_lm_head() {
        let device = Device::Cpu;
        let hidden = 32;
        let config = Qwen3Config {
            hidden_size: Some(hidden),
            num_hidden_layers: Some(0),
            tie_word_embeddings: Some(true),
            ..Default::default()
        };

        let mut weights = HashMap::new();
        weights.insert(
            "model.embed_tokens.weight".to_string(),
            Tensor::zeros((64, hidden), DType::F32, &device).unwrap(),
        );
        weights.insert(
            "model.norm.weight".to_string(),
            Tensor::zeros(hidden, DType::F32, &device).unwrap(),
        );

        let mut model =
            crate::qwen3_5::model::Qwen35HybridModel::new(config.clone(), device, 4, false)
                .unwrap();
        load_hybrid_weights(&mut model, &config, &weights).unwrap();
        assert!(model.lm_head.is_none());
    }

    #[test]
    fn test_load_hybrid_weights_lm_head_fallback_keys() {
        let device = Device::Cpu;
        let hidden = 32;
        let vocab = 64;
        let config = Qwen3Config {
            vocab_size: Some(vocab),
            hidden_size: Some(hidden),
            num_hidden_layers: Some(0),
            tie_word_embeddings: Some(false),
            ..Default::default()
        };

        let mut weights = HashMap::new();
        weights.insert(
            "model.embed_tokens.weight".to_string(),
            Tensor::zeros((vocab, hidden), DType::F32, &device).unwrap(),
        );
        weights.insert(
            "output.weight".to_string(),
            Tensor::zeros((vocab, hidden), DType::F32, &device).unwrap(),
        );
        weights.insert(
            "model.final_layernorm.weight".to_string(),
            Tensor::zeros(hidden, DType::F32, &device).unwrap(),
        );

        let mut model =
            crate::qwen3_5::model::Qwen35HybridModel::new(config.clone(), device, 4, false)
                .unwrap();
        load_hybrid_weights(&mut model, &config, &weights).unwrap();
        assert!(model.lm_head.is_some());
    }
}
