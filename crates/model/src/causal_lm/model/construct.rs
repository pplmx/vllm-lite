//! Construction helpers for [`CausalLm`].
//!
//! These four entry points differ only in (a) which final norm layer
//! they wire up (`LnLayerNorm` vs `RmsNorm`) and (b) whether the
//! weights come from an `Embedding` factory closure or a `HuggingFace`
//! weight map. Split out of `mod.rs` so the facade file stays focused
//! on the `ModelBackend` trait surface.

use std::collections::HashMap;

use super::CausalLm;
use crate::components::decoder_block::PagedDecoderBlock;
use crate::components::{LnLayerNorm, RmsNorm};
use crate::config::ModelConfig;
use crate::paged_tensor::PagedKvCache;
use candle_core::{Device, Result as CandleResult, Tensor};
use candle_nn::{Embedding, Linear, VarBuilder};

impl<B> CausalLm<B, LnLayerNorm, Linear>
where
    B: PagedDecoderBlock + Send + Sync,
{
    /// Construct a `CausalLm` with `LnLayerNorm` and zero-initialized
    /// embeddings. The `block_fn` closure supplies each decoder layer.
    /// # Errors
    ///
    /// Returns `Err` if any required tensor allocation or weight loading fails.
    pub fn new_with_block_fn<F>(
        config: ModelConfig,
        device: Device,
        num_kv_blocks: usize,
        kv_quantization: bool,
        mut block_fn: F,
    ) -> CandleResult<Self>
    where
        F: FnMut(&ModelConfig, usize) -> CandleResult<B>,
    {
        let vocab_size = config.vocab_size;
        let hidden_size = config.hidden_size;
        let num_layers = config.num_layers;

        let embeddings =
            Tensor::zeros((vocab_size, hidden_size), candle_core::DType::F32, &device)?;
        let embed_tokens = Embedding::new(embeddings, hidden_size);

        let mut layers = Vec::with_capacity(num_layers);
        for layer_idx in 0..num_layers {
            layers.push(block_fn(&config, layer_idx)?);
        }

        let norm_weight = Tensor::ones(hidden_size, candle_core::DType::F32, &device)?;
        let norm_bias = Tensor::zeros(hidden_size, candle_core::DType::F32, &device)?;
        let norm = LnLayerNorm::new(norm_weight, norm_bias, config.rms_norm_eps);

        let vb = VarBuilder::zeros(candle_core::DType::F32, &device);
        let lm_head = candle_nn::linear(hidden_size, vocab_size, vb.pp("lm_head"))?;

        let kv_cache = PagedKvCache::new(
            num_layers,
            config.num_heads,
            config.head_dim,
            num_kv_blocks,
            device.clone(),
            kv_quantization,
        )?;

        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
            lm_head,
            kv_cache,
            device,
            embed_through_layers: false,
        })
    }

    /// Build a `CausalLm` with `LnLayerNorm` from a `HuggingFace` weight map.
    /// # Errors
    ///
    /// Returns `Err` if reading or parsing the source fails.
    #[allow(clippy::needless_pass_by_value)]
    pub fn from_hf_weights_ln<F>(
        config: ModelConfig,
        device: Device,
        weights: HashMap<String, Tensor>,
        num_kv_blocks: usize,
        kv_quantization: bool,
        block_from_weights: F,
    ) -> CandleResult<Self>
    where
        F: Fn(&ModelConfig, usize, &HashMap<String, Tensor>) -> CandleResult<B>,
    {
        let hidden_size = config.hidden_size;
        let num_layers = config.num_layers;

        let embed_key = "model.embed_tokens.weight";
        let embed_weight = weights
            .get(embed_key)
            .cloned()
            .ok_or_else(|| candle_core::Error::msg(format!("Missing {embed_key}")))?;
        let embed_tokens = Embedding::new(embed_weight.clone(), hidden_size);

        let mut layers = Vec::with_capacity(num_layers);
        for layer_idx in 0..num_layers {
            layers.push(block_from_weights(&config, layer_idx, &weights)?);
        }

        let norm_key = "model.norm.weight";
        let norm_weight = weights
            .get(norm_key)
            .cloned()
            .ok_or_else(|| candle_core::Error::msg(format!("Missing {norm_key}")))?;
        let norm_bias = Tensor::zeros(
            norm_weight.dim(0).unwrap_or(hidden_size),
            norm_weight.dtype(),
            norm_weight.device(),
        )?;
        let norm = LnLayerNorm::new(norm_weight, norm_bias, config.rms_norm_eps);

        let lm_head =
            super::super::weights::load_lm_head(&weights, embed_weight, config.tie_word_embeddings)?;

        let kv_cache = PagedKvCache::new(
            num_layers,
            config.num_heads,
            config.head_dim,
            num_kv_blocks,
            device.clone(),
            kv_quantization,
        )?;

        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
            lm_head,
            kv_cache,
            device,
            embed_through_layers: false,
        })
    }
}

impl<B> CausalLm<B, RmsNorm, Linear>
where
    B: PagedDecoderBlock + Send + Sync,
{
    /// Construct a `CausalLm` with `RmsNorm` and zero-initialized
    /// embeddings. The `block_fn` closure supplies each decoder layer.
    /// # Errors
    ///
    /// Returns `Err` if any required tensor allocation or weight loading fails.
    pub fn new_rms<F>(
        config: ModelConfig,
        device: Device,
        num_kv_blocks: usize,
        kv_quantization: bool,
        mut block_fn: F,
    ) -> CandleResult<Self>
    where
        F: FnMut(&ModelConfig, usize) -> CandleResult<B>,
    {
        let vocab_size = config.vocab_size;
        let hidden_size = config.hidden_size;
        let num_layers = config.num_layers;

        let embeddings =
            Tensor::zeros((vocab_size, hidden_size), candle_core::DType::F32, &device)?;
        let embed_tokens = Embedding::new(embeddings, hidden_size);

        let mut layers = Vec::with_capacity(num_layers);
        for layer_idx in 0..num_layers {
            layers.push(block_fn(&config, layer_idx)?);
        }

        let norm_weight = Tensor::ones(hidden_size, candle_core::DType::F32, &device)?;
        let norm = RmsNorm::new(norm_weight, config.rms_norm_eps);

        let vb = VarBuilder::zeros(candle_core::DType::F32, &device);
        let lm_head = candle_nn::linear(hidden_size, vocab_size, vb.pp("lm_head"))?;

        let kv_cache = PagedKvCache::new(
            num_layers,
            config.num_heads,
            config.head_dim,
            num_kv_blocks,
            device.clone(),
            kv_quantization,
        )?;

        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
            lm_head,
            kv_cache,
            device,
            embed_through_layers: false,
        })
    }

    /// Build a `CausalLm` with `RmsNorm` from a `HuggingFace` weight map.
    /// # Errors
    ///
    /// Returns `Err` if reading or parsing the source fails.
    #[allow(clippy::needless_pass_by_value)]
    pub fn from_hf_weights_rms<F>(
        config: ModelConfig,
        device: Device,
        weights: HashMap<String, Tensor>,
        num_kv_blocks: usize,
        kv_quantization: bool,
        block_from_weights: F,
    ) -> CandleResult<Self>
    where
        F: Fn(&ModelConfig, usize, &HashMap<String, Tensor>) -> CandleResult<B>,
    {
        let hidden_size = config.hidden_size;
        let num_layers = config.num_layers;

        let embed_key = "model.embed_tokens.weight";
        let embed_weight = weights
            .get(embed_key)
            .cloned()
            .ok_or_else(|| candle_core::Error::msg(format!("Missing {embed_key}")))?;
        let embed_tokens = Embedding::new(embed_weight.clone(), hidden_size);

        let mut layers = Vec::with_capacity(num_layers);
        for layer_idx in 0..num_layers {
            layers.push(block_from_weights(&config, layer_idx, &weights)?);
        }

        let norm_key = "model.norm.weight";
        let norm_weight = weights
            .get(norm_key)
            .cloned()
            .ok_or_else(|| candle_core::Error::msg(format!("Missing {norm_key}")))?;
        let norm = RmsNorm::new(norm_weight, config.rms_norm_eps);

        let lm_head =
            super::super::weights::load_lm_head(&weights, embed_weight, config.tie_word_embeddings)?;

        let kv_cache = PagedKvCache::new(
            num_layers,
            config.num_heads,
            config.head_dim,
            num_kv_blocks,
            device.clone(),
            kv_quantization,
        )?;

        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
            lm_head,
            kv_cache,
            device,
            embed_through_layers: false,
        })
    }
}
