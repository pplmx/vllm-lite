use crate::causal_lm::CausalLm;
use crate::components::LnLayerNorm;
use crate::config::ModelConfig;
use crate::qwen3::config::Qwen3Config;
use candle_core::{Device, Result as CandleResult, Tensor};
use candle_nn::Linear;
use std::collections::HashMap;
#[cfg(feature = "multi-node")]
use vllm_dist::TensorParallelConfig;

use super::block::{block_from_weights, new_block};

/// `Qwen3Model`: qwen3 model.
pub type Qwen3Model = CausalLm<super::block::TransformerBlock, LnLayerNorm, Linear>;

impl Qwen3Model {
    pub fn new(config: Qwen3Config, device: Device, num_kv_blocks: usize) -> CandleResult<Self> {
        let model_config = ModelConfig::from(&config);
        Self::new_with_block_fn(model_config, device, num_kv_blocks, false, |c, idx| {
            new_block(c, idx)
        })
        .map(|m| m.with_embed_through_layers(true))
    }

    #[cfg(feature = "multi-node")]
    pub fn new_with_tp(
        config: Qwen3Config,
        tp_config: Option<TensorParallelConfig>,
        num_kv_blocks: usize,
    ) -> CandleResult<Self> {
        super::tp::new_with_tp(config, tp_config, num_kv_blocks)
    }

    pub fn from_weights(
        config: Qwen3Config,
        device: Device,
        weights: HashMap<String, Tensor>,
        num_kv_blocks: usize,
        kv_quantization: bool,
    ) -> CandleResult<Self> {
        let model_config = ModelConfig::from(&config);
        Self::from_hf_weights_ln(
            model_config,
            device,
            weights,
            num_kv_blocks,
            kv_quantization,
            block_from_weights,
        )
        .map(|m| m.with_embed_through_layers(true))
    }
}
