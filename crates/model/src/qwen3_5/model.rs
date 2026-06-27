//! model: model.

#![allow(non_snake_case, clippy::too_many_arguments)]
//! Qwen3.5 hybrid causal language model (GDN + full attention).

use std::collections::HashMap;

use crate::causal_lm::{HybridLm, HybridLmConfig};
use crate::components::positional::MRoPE;
use crate::paged_tensor::PagedKvCache;
use crate::qwen3_5::block::{FullAttentionBlock35, HybridBlock, LinearAttentionBlock};
use crate::qwen3_5::config::{GdnLinearConfig, LayerType, parse_layer_types};
use crate::qwen3_5::weights::load_hybrid_weights;
use crate::qwen3_config::Qwen3Config;
use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_nn::{Embedding, LayerNorm, VarBuilder};

/// Qwen35HybridModel: qwen35 hybrid model.
pub type Qwen35HybridModel = HybridLm<HybridBlock, LayerNorm, Qwen3Config>;

impl HybridLmConfig for Qwen3Config {
    fn vocab_size(&self) -> usize {
        Qwen3Config::vocab_size(self)
    }

    fn hidden_size(&self) -> usize {
        Qwen3Config::hidden_size(self)
    }

    fn num_layers(&self) -> usize {
        Qwen3Config::num_hidden_layers(self)
    }

    fn num_kv_heads(&self) -> usize {
        Qwen3Config::num_key_value_heads(self)
    }
}

impl Qwen35HybridModel {
/// new: new.
    pub fn new(
        config: Qwen3Config,
        device: Device,
        num_kv_blocks: usize,
        kv_quantization: bool,
    ) -> CandleResult<Self> {
        let vocab_size = config.vocab_size();
        let hidden_size = config.hidden_size();

        let embeddings =
            Tensor::zeros((vocab_size, hidden_size), candle_core::DType::F32, &device)?;
        let embed_tokens = Embedding::new(embeddings, hidden_size);

        let gdn_config = GdnLinearConfig::from_qwen3_config(&config);
        let layer_types = parse_layer_types(&config);
        let mut layers = Vec::new();
        let rope = MRoPE::from_config(&config);

        for layer_type in &layer_types {
            let layer = match layer_type {
                LayerType::LinearAttention => HybridBlock::Linear(LinearAttentionBlock::new(
                    hidden_size,
                    gdn_config,
                    VarBuilder::zeros(DType::F32, &device),
                )?),
                LayerType::FullAttention => HybridBlock::Full(FullAttentionBlock35::new(
                    hidden_size,
                    config.num_attention_heads(),
                    config.num_key_value_heads(),
                    config.head_dim(),
                    config.intermediate_size(),
                    config.rms_norm_eps(),
                    rope.clone(),
                    VarBuilder::zeros(DType::F32, &device),
                )?),
            };
            layers.push(layer);
        }

        let norm = candle_nn::layer_norm(
            hidden_size,
            config.rms_norm_eps(),
            VarBuilder::zeros(DType::F32, &device),
        )?;

        let kv_cache = PagedKvCache::new(
            config.num_hidden_layers(),
            config.num_key_value_heads(),
            config.head_dim(),
            num_kv_blocks,
            device.clone(),
            kv_quantization,
        )?;

        Ok(HybridLm::from_parts(
            config,
            embed_tokens,
            layers,
            norm,
            None,
            kv_cache,
            device,
        ))
    }

/// from_weights: from weights.
    pub fn from_weights(
        config: Qwen3Config,
        device: Device,
        weights: HashMap<String, Tensor>,
        num_kv_blocks: usize,
        kv_quantization: bool,
    ) -> CandleResult<Self> {
        let mut model = Self::new(
            config.clone(),
            device.clone(),
            num_kv_blocks,
            kv_quantization,
        )?;
        load_hybrid_weights(&mut model, &config, &weights)?;
        Ok(model)
    }
}
