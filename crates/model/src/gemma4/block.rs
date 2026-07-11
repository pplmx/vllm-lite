#![allow(clippy::module_name_repetitions)]
//! Gemma4 Transformer Block implementation.

use std::collections::HashMap;

use crate::components::RmsNorm;
use crate::components::decoder_block::PagedDecoderBlock;
use crate::config::{LayerType, ModelConfig, RoPEConfig};
use crate::gemma4::attention::Gemma4Attention;
use crate::gemma4::mlp::GeGLU;
use crate::paged_tensor::PagedKvCache;
use candle_core::{Result, Tensor};

#[derive(Debug)]
/// Block abstraction for Gemma4. Groups a contiguous range of work (e.g. one transformer layer, one pipeline stage).
pub struct Gemma4Block {
    attention: Gemma4Attention,
    mlp: GeGLU,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl Gemma4Block {
    /// Construct a new instance from the given configuration.
    /// # Errors
    ///
    /// Returns `Err` if any required tensor allocation or weight loading fails.
    #[allow(clippy::needless_pass_by_value)]
    pub fn new(
        config: &ModelConfig,
        layer_idx: usize,
        vb: candle_nn::VarBuilder<'_>,
    ) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_heads;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;
        let intermediate_size = config.intermediate_size;
        let sliding_window = config.sliding_window.unwrap_or(512);

        let layer_type = config
            .layer_types
            .get(layer_idx)
            .copied()
            .unwrap_or(LayerType::SlidingAttention);

        let rope_config = config
            .rope_configs
            .get(layer_idx)
            .cloned()
            .unwrap_or(RoPEConfig {
                rope_theta: config.rope_theta,
                partial_rotary_factor: 1.0,
            });

        let attention = Gemma4Attention::new(
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            sliding_window,
            layer_type,
            &rope_config,
            vb.clone(),
        )?;
        let mlp = GeGLU::new(hidden_size, intermediate_size, vb.clone())?;

        let input_ln_w = Tensor::ones(hidden_size, candle_core::DType::F32, vb.device())?;
        let input_layernorm = RmsNorm::new(input_ln_w, config.rms_norm_eps);
        let post_ln_w = Tensor::ones(hidden_size, candle_core::DType::F32, vb.device())?;
        let post_attention_layernorm = RmsNorm::new(post_ln_w, config.rms_norm_eps);

        Ok(Self {
            attention,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    /// Build from weights.
    /// # Errors
    ///
    /// Returns `Err` if reading or parsing the source fails.
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
        let sliding_window = config.sliding_window.unwrap_or(512);

        let layer_type = config
            .layer_types
            .get(layer_idx)
            .copied()
            .unwrap_or(LayerType::SlidingAttention);

        let rope_config = config
            .rope_configs
            .get(layer_idx)
            .cloned()
            .unwrap_or(RoPEConfig {
                rope_theta: config.rope_theta,
                partial_rotary_factor: 1.0,
            });

        let prefix = format!("model.layers.{layer_idx}");
        let q_w = weights
            .get(&format!("{prefix}.self_attn.q_proj.weight"))
            .cloned()
            .ok_or_else(|| candle_core::Error::msg("Missing q_proj weight"))?;
        let k_w = weights
            .get(&format!("{prefix}.self_attn.k_proj.weight"))
            .cloned()
            .ok_or_else(|| candle_core::Error::msg("Missing k_proj weight"))?;
        let v_w = weights
            .get(&format!("{prefix}.self_attn.v_proj.weight"))
            .cloned()
            .ok_or_else(|| candle_core::Error::msg("Missing v_proj weight"))?;
        let o_w = weights
            .get(&format!("{prefix}.self_attn.o_proj.weight"))
            .cloned()
            .ok_or_else(|| candle_core::Error::msg("Missing o_proj weight"))?;
        let gate_w = weights
            .get(&format!("{prefix}.mlp.gate_proj.weight"))
            .cloned()
            .ok_or_else(|| candle_core::Error::msg("Missing gate_proj weight"))?;
        let up_w = weights
            .get(&format!("{prefix}.mlp.up_proj.weight"))
            .cloned()
            .ok_or_else(|| candle_core::Error::msg("Missing up_proj weight"))?;
        let down_w = weights
            .get(&format!("{prefix}.mlp.down_proj.weight"))
            .cloned()
            .ok_or_else(|| candle_core::Error::msg("Missing down_proj weight"))?;
        let input_ln_w = weights
            .get(&format!("{prefix}.input_layernorm.weight"))
            .cloned()
            .ok_or_else(|| candle_core::Error::msg("Missing input_layernorm weight"))?;
        let post_ln_w = weights
            .get(&format!("{prefix}.post_attention_layernorm.weight"))
            .cloned()
            .ok_or_else(|| candle_core::Error::msg("Missing post_attention_layernorm weight"))?;

        let attention = Gemma4Attention::new_from_weights(
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            sliding_window,
            layer_type,
            &rope_config,
            q_w,
            k_w,
            v_w,
            o_w,
        );
        let mlp = GeGLU::new_with_weights(hidden_size, intermediate_size, gate_w, up_w, down_w);
        let input_layernorm = RmsNorm::new(input_ln_w, config.rms_norm_eps);
        let post_attention_layernorm = RmsNorm::new(post_ln_w, config.rms_norm_eps);

        Ok(Self {
            attention,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    /// Run the layer forward pass over the input.
    /// # Errors
    ///
    /// Returns `Err` if any tensor operation fails (shape mismatch, out-of-memory, dtype incompatibility, or kernel error).
    pub fn forward(&self, x: &Tensor, positions: &[usize]) -> Result<Tensor> {
        let residual = x.clone();
        let x = self.input_layernorm.forward(x)?;
        let x = self.attention.forward(&x, positions)?;
        let x = x.add(&residual)?;

        let residual = x.clone();
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        x.add(&residual)
    }

    /// Run the prefill path: process the full prompt and cache its KV.
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    pub fn forward_prefill(
        &self,
        x: &Tensor,
        kv_cache: &mut PagedKvCache,
        layer_idx: usize,
        block_ids: &[usize],
        positions: &[usize],
    ) -> Result<Tensor> {
        let residual = x.clone();
        let x = self.input_layernorm.forward(x)?;
        let x = self
            .attention
            .forward_prefill(&x, kv_cache, layer_idx, block_ids, positions)?;
        let x = x.add(&residual)?;

        let residual = x.clone();
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        x.add(&residual)
    }

    /// Run a chunked-prefill continuation against an existing KV prefix.
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    pub fn forward_prefill_continue(
        &self,
        x: &Tensor,
        kv_cache: &mut PagedKvCache,
        layer_idx: usize,
        block_ids: &[usize],
        positions: &[usize],
        num_computed_tokens: usize,
    ) -> Result<Tensor> {
        let residual = x.clone();
        let x = self.input_layernorm.forward(x)?;
        let x = self.attention.forward_prefill_continue(
            &x,
            kv_cache,
            layer_idx,
            block_ids,
            positions,
            num_computed_tokens,
        )?;
        let x = x.add(&residual)?;

        let residual = x.clone();
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        x.add(&residual)
    }

    /// Run the decode path: process one new token against cached KV.
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    pub fn forward_decode(
        &self,
        x: &Tensor,
        kv_cache: &mut PagedKvCache,
        layer_idx: usize,
        block_ids: &[usize],
        num_computed_tokens: usize,
        positions: &[usize],
    ) -> Result<Tensor> {
        let residual = x.clone();
        let mut x = self.input_layernorm.forward(x)?;
        x = self.attention.forward_decode(
            &x,
            kv_cache,
            layer_idx,
            block_ids,
            num_computed_tokens,
            positions,
        )?;
        if x.dims().len() == 3 && x.dims()[1] == 1 && residual.dims().len() == 2 {
            let dims = x.dims();
            x = x.reshape((dims[0], dims[2]))?;
        }
        x = x.add(&residual)?;

        let residual = x.clone();
        x = self.post_attention_layernorm.forward(&x)?;
        x = self.mlp.forward(&x)?;
        x.add(&residual)
    }
}

impl PagedDecoderBlock for Gemma4Block {
    fn forward_prefill(
        &self,
        x: &Tensor,
        kv_cache: &mut PagedKvCache,
        layer_idx: usize,
        block_ids: &[usize],
        positions: &[usize],
    ) -> Result<Tensor> {
        Self::forward_prefill(self, x, kv_cache, layer_idx, block_ids, positions)
    }

    fn forward_prefill_continue(
        &self,
        x: &Tensor,
        kv_cache: &mut PagedKvCache,
        layer_idx: usize,
        block_ids: &[usize],
        positions: &[usize],
        num_computed_tokens: usize,
    ) -> Result<Tensor> {
        Self::forward_prefill_continue(
            self,
            x,
            kv_cache,
            layer_idx,
            block_ids,
            positions,
            num_computed_tokens,
        )
    }

    fn forward_decode(
        &self,
        x: &Tensor,
        kv_cache: &mut PagedKvCache,
        layer_idx: usize,
        block_ids: &[usize],
        num_computed_tokens: usize,
        positions: &[usize],
    ) -> Result<Tensor> {
        Self::forward_decode(
            self,
            x,
            kv_cache,
            layer_idx,
            block_ids,
            num_computed_tokens,
            positions,
        )
    }
}

/// # Errors
///
/// Returns `Err` if any required tensor allocation or weight loading fails.
/// Build a zero-initialized Gemma4 block for `CausalLm::new_rms`.
pub fn new_block(config: &ModelConfig, layer_idx: usize) -> Result<Gemma4Block> {
    let vb = candle_nn::VarBuilder::zeros(candle_core::DType::F32, &candle_core::Device::Cpu);
    Gemma4Block::new(config, layer_idx, vb)
}

/// Load a Gemma4 block from HuggingFace-style weight keys.
pub(crate) fn block_from_weights(
    config: &ModelConfig,
    layer_idx: usize,
    weights: &HashMap<String, Tensor>,
) -> Result<Gemma4Block> {
    Gemma4Block::from_weights(config, layer_idx, weights)
}

// Unit test is extracted to `tests.rs` (sibling) to keep this
// block module under the 800-line soft cap. It exercises the
// prefill + decode shape contract through a shared
// `PagedKvCache` using the Gemma4 sliding-attention path.
#[cfg(test)]
mod tests;
