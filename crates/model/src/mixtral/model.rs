//! Mixtral causal language model with paged KV cache and sparse MoE.

use std::collections::HashMap;

use crate::causal_lm::CausalLm;
use crate::components::LnLayerNorm;
use crate::config::ModelConfig;
use candle_core::{Device, Result as CandleResult, Tensor};
use candle_nn::Linear;

use super::block::MixtralBlock;

/// MixtralModel: mixtral model.
pub type MixtralModel = CausalLm<MixtralBlock, LnLayerNorm, Linear>;

impl MixtralModel {
/// new: new.
    pub fn new(
        config: ModelConfig,
        device: Device,
        num_kv_blocks: usize,
        kv_quantization: bool,
    ) -> CandleResult<Self> {
        Self::new_with_block_fn(
            config,
            device,
            num_kv_blocks,
            kv_quantization,
            MixtralBlock::new,
        )
    }

/// from_weights: from weights.
    pub fn from_weights(
        config: ModelConfig,
        device: Device,
        weights: HashMap<String, Tensor>,
        num_kv_blocks: usize,
        kv_quantization: bool,
    ) -> CandleResult<Self> {
        Self::from_hf_weights_ln(
            config,
            device,
            weights,
            num_kv_blocks,
            kv_quantization,
            MixtralBlock::from_weights,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Architecture;

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
    fn test_mixtral_model_forward_prefill_and_decode() {
        let config = tiny_config();
        let device = Device::Cpu;
        let mut model = MixtralModel::new(config, device, 8, false).unwrap();

        let tokens = vec![1u32, 2, 3, 4];
        let positions: Vec<usize> = (0..tokens.len()).collect();
        let block_ids = vec![0usize];

        let (logits, _) = model
            .forward_with_cache(&tokens, 0, &block_ids, &positions, true)
            .unwrap();
        assert_eq!(logits.dims()[2], 128);

        let (logits, _) = model
            .forward_with_cache(&[5], 4, &block_ids, &[4], false)
            .unwrap();
        assert_eq!(logits.dims()[2], 128);
    }
}
