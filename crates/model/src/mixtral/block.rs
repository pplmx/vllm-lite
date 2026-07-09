#![allow(clippy::module_name_repetitions)]
//! Mixtral block (Transformer layer with `MoE`).

use std::collections::HashMap;

use crate::components::LnLayerNorm;
use crate::components::attention::RopeGqaAttention;
use crate::components::decoder_block::PagedDecoderBlock;
use crate::config::ModelConfig;
use crate::mixtral::sparse_moe::MixtralSparseMoe;
use crate::paged_tensor::PagedKvCache;
use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

#[derive(Debug)]
/// Block abstraction for Mixtral. Groups a contiguous range of work (e.g. one transformer layer, one pipeline stage).
pub struct MixtralBlock {
    attention: RopeGqaAttention,
    mlp: MixtralSparseMoe,
    input_layernorm: LnLayerNorm,
    post_attention_layernorm: LnLayerNorm,
}

impl MixtralBlock {
    /// Construct a new instance from the given configuration.
    /// # Errors
    ///
    /// Returns `Err` if any required tensor allocation or weight loading fails.
    pub fn new(config: &ModelConfig, _layer_idx: usize) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_heads;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;
        let rms_norm_eps = config.rms_norm_eps;

        let num_experts = config.num_experts.unwrap_or(8);
        let expert_intermediate_size = config
            .expert_intermediate_size
            .unwrap_or(config.intermediate_size);
        let top_k = config.top_k_experts.unwrap_or(2);

        let vb = VarBuilder::zeros(candle_core::DType::F32, &candle_core::Device::Cpu);
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
            Some(vb.clone()),
            crate::components::AttentionConfig::default(),
            false,
        )?;

        let mlp = MixtralSparseMoe::new(
            hidden_size,
            num_experts,
            expert_intermediate_size,
            top_k,
            vb,
        )?;

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
    #[allow(clippy::too_many_lines)] // weight-loader linear sequence; splitting would obscure the per-tensor lookups
    pub fn from_weights(
        config: &ModelConfig,
        layer_idx: usize,
        weights: &HashMap<String, Tensor>,
    ) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_heads;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;
        let rms_norm_eps = config.rms_norm_eps;

        let num_experts = config.num_experts.unwrap_or(8);
        let expert_intermediate_size = config
            .expert_intermediate_size
            .unwrap_or(config.intermediate_size);
        let top_k = config.top_k_experts.unwrap_or(2);

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
        let post_attention_layernorm =
            LnLayerNorm::new(post_attn_ln_w, post_attn_bias, rms_norm_eps);

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

        let mut expert_weights = Vec::new();
        for i in 0..num_experts {
            let gate_w = weights
                .get(&format!(
                    "model.layers.{layer_idx}.block_sparse_moe.experts.{i}.gate_proj.weight"
                ))
                .cloned()
                .ok_or_else(|| {
                    candle_core::Error::msg(format!("Missing expert {i}.gate_proj weight"))
                })?;
            let up_w = weights
                .get(&format!(
                    "model.layers.{layer_idx}.block_sparse_moe.experts.{i}.up_proj.weight"
                ))
                .cloned()
                .ok_or_else(|| {
                    candle_core::Error::msg(format!("Missing expert {i}.up_proj weight"))
                })?;
            let down_w = weights
                .get(&format!(
                    "model.layers.{layer_idx}.block_sparse_moe.experts.{i}.down_proj.weight"
                ))
                .cloned()
                .ok_or_else(|| {
                    candle_core::Error::msg(format!("Missing expert {i}.down_proj weight"))
                })?;
            expert_weights.push((gate_w, up_w, down_w));
        }

        let gate_w = weights
            .get(&format!(
                "model.layers.{layer_idx}.block_sparse_moe.gate.weight"
            ))
            .cloned()
            .ok_or_else(|| candle_core::Error::msg("Missing gate weight"))?;

        let mlp = MixtralSparseMoe::new_with_weights(
            hidden_size,
            num_experts,
            expert_intermediate_size,
            top_k,
            gate_w,
            expert_weights,
        )?;

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
    pub fn forward(&self, x: &Tensor, _positions: &[usize]) -> Result<Tensor> {
        let residual = x.clone();
        let x = self.input_layernorm.forward(x)?;
        let x = self.attention.forward(&x)?;
        let x = (x + residual)?;

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
        let x = (&x + &residual)?;

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
        x = (&x + &residual)?;

        let residual = x.clone();
        x = self.post_attention_layernorm.forward(&x)?;
        x = self.mlp.forward(&x)?;
        x.add(&residual)
    }
}

impl PagedDecoderBlock for MixtralBlock {
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

// Unit test is extracted to `tests.rs` (sibling) to keep this
// block module under the 800-line soft cap. It exercises the
// prefill + decode shape contract through a shared
// `PagedKvCache`.
#[cfg(test)]
mod tests;
