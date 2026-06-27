//! Mixtral block (Transformer layer with MoE).

use std::collections::HashMap;

use crate::components::LnLayerNorm;
use crate::components::attention::RopeGqaAttention;
use crate::components::decoder_block::PagedDecoderBlock;
use crate::config::ModelConfig;
use crate::mixtral::sparse_moe::MixtralSparseMoe;
use crate::paged_tensor::PagedKvCache;
use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

/// MixtralBlock: mixtral block.
pub struct MixtralBlock {
    attention: RopeGqaAttention,
    mlp: MixtralSparseMoe,
    input_layernorm: LnLayerNorm,
    post_attention_layernorm: LnLayerNorm,
}

impl MixtralBlock {
/// new: new.
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

/// from_weights: from weights.
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
                    "model.layers.{}.block_sparse_moe.experts.{}.gate_proj.weight",
                    layer_idx, i
                ))
                .cloned()
                .ok_or_else(|| {
                    candle_core::Error::msg(format!("Missing expert {}.gate_proj weight", i))
                })?;
            let up_w = weights
                .get(&format!(
                    "model.layers.{}.block_sparse_moe.experts.{}.up_proj.weight",
                    layer_idx, i
                ))
                .cloned()
                .ok_or_else(|| {
                    candle_core::Error::msg(format!("Missing expert {}.up_proj weight", i))
                })?;
            let down_w = weights
                .get(&format!(
                    "model.layers.{}.block_sparse_moe.experts.{}.down_proj.weight",
                    layer_idx, i
                ))
                .cloned()
                .ok_or_else(|| {
                    candle_core::Error::msg(format!("Missing expert {}.down_proj weight", i))
                })?;
            expert_weights.push((gate_w, up_w, down_w));
        }

        let gate_w = weights
            .get(&format!(
                "model.layers.{}.block_sparse_moe.gate.weight",
                layer_idx
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

/// forward: forward.
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

/// forward_prefill: forward prefill.
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

/// forward_decode: forward decode.
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
        MixtralBlock::forward_prefill(self, x, kv_cache, layer_idx, block_ids, positions)
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
        MixtralBlock::forward_decode(
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Architecture, ModelConfig};
    use candle_core::DType;

    fn tiny_config() -> ModelConfig {
        ModelConfig {
            architecture: Architecture::Mixtral,
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
            use_double_wide_mlp: false,
            num_experts: Some(4),
            top_k_experts: Some(2),
            expert_intermediate_size: Some(128),
            has_qk_norm: false,
        }
    }

    #[test]
    fn test_mixtral_block_prefill_then_decode() {
        let config = tiny_config();
        let device = candle_core::Device::Cpu;
        let block = MixtralBlock::new(&config, 0).unwrap();
        let mut kv_cache = PagedKvCache::new(1, 4, 16, 8, device.clone(), false).unwrap();

        let x = Tensor::ones((1, 4, 64), DType::F32, &device).unwrap();
        let block_ids: Vec<usize> = (0..4).map(|i| i / 16).collect();
        let positions: Vec<usize> = (0..4).collect();
        let out = block
            .forward_prefill(&x, &mut kv_cache, 0, &block_ids, &positions)
            .unwrap();
        assert_eq!(out.dims(), &[1, 4, 64]);

        let decode_x = Tensor::ones((1, 64), DType::F32, &device).unwrap();
        let decode_out = block
            .forward_decode(&decode_x, &mut kv_cache, 0, &[0], 4, &[4])
            .unwrap();
        assert_eq!(decode_out.dims(), &[1, 64]);
    }
}
