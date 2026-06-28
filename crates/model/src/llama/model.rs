use crate::causal_lm::CausalLm;
use crate::components::RmsNorm;
use crate::components::decoder_block::RopeGqaDecoderBlock;
use crate::config::ModelConfig;
use candle_core::{Device, Result as CandleResult, Tensor};
use candle_nn::Linear;
use std::collections::HashMap;

use super::block::{block_from_weights, new_block};

/// `LlamaBlock`: llama block.
pub type LlamaBlock = RopeGqaDecoderBlock;
/// `LlamaModel`: llama model.
pub type LlamaModel = CausalLm<LlamaBlock, RmsNorm, Linear>;

impl LlamaModel {
    pub fn new(config: ModelConfig, device: Device, num_kv_blocks: usize) -> CandleResult<Self> {
        Self::new_rms(config, device, num_kv_blocks, false, new_block)
    }

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
