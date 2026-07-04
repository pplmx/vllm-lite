#![allow(clippy::module_name_repetitions)]
//! Gemma4 causal language model with paged KV cache inference.

use std::collections::HashMap;

use crate::causal_lm::CausalLm;
use crate::components::RmsNorm;
use crate::config::ModelConfig;
use candle_core::{Device, Result as CandleResult, Tensor};
use candle_nn::Linear;

use super::block::{Gemma4Block, block_from_weights, new_block};

/// `Gemma4Model`. See the type definition for fields and behavior.
pub type Gemma4Model = CausalLm<Gemma4Block, RmsNorm, Linear>;

impl Gemma4Model {
    /// Construct a new instance from the given configuration.
    /// # Errors
    ///
    /// Returns `Err` if any required tensor allocation or weight loading fails.
    pub fn new(
        config: ModelConfig,
        device: Device,
        num_kv_blocks: usize,
        kv_quantization: bool,
    ) -> CandleResult<Self> {
        Self::new_rms(config, device, num_kv_blocks, kv_quantization, new_block)
    }

    /// Build from weights.
    /// # Errors
    ///
    /// Returns `Err` if reading or parsing the source fails.
    pub fn from_weights(
        config: ModelConfig,
        device: Device,
        weights: HashMap<String, Tensor>,
        num_kv_blocks: usize,
        kv_quantization: bool,
    ) -> CandleResult<Self> {
        Self::from_hf_weights_rms(
            config,
            device,
            weights,
            num_kv_blocks,
            kv_quantization,
            block_from_weights,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Architecture, LayerType, ModelConfig, RoPEConfig};

    fn tiny_config() -> ModelConfig {
        ModelConfig {
            architecture: Architecture::Gemma4,
            hidden_size: 64,
            num_layers: 1,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 16,
            vocab_size: 128,
            intermediate_size: 128,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-6,
            sliding_window: Some(512),
            tie_word_embeddings: true,
            max_position_embeddings: 512,
            layer_types: vec![LayerType::SlidingAttention],
            rope_configs: vec![RoPEConfig {
                rope_theta: 10000.0,
                partial_rotary_factor: 1.0,
            }],
            use_double_wide_mlp: false,
            num_experts: None,
            top_k_experts: None,
            expert_intermediate_size: None,
            has_qk_norm: false,
        }
    }

    #[test]
    fn test_gemma4_model_forward_prefill_and_decode() {
        let config = tiny_config();
        let device = Device::Cpu;
        let mut model = Gemma4Model::new(config, device, 8, false).unwrap();

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
