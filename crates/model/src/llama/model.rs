//! Llama architecture: 1:1 GQA + `SwiGLU` + `RoPE` transformer block, with the `Llama` `Architecture` trait impl.
//!
//! Concrete model wrapper exposed to the registry; selected when
//! `config.json` reports `"architectures": ["LlamaForCausalLM"]`.
#![allow(clippy::module_name_repetitions)]
use crate::causal_lm::CausalLm;
use crate::components::RmsNorm;
use crate::components::decoder_block::RopeGqaDecoderBlock;
use crate::config::ModelConfig;
use candle_core::{Device, Result as CandleResult, Tensor};
use candle_nn::Linear;
use std::collections::HashMap;

use super::block::{block_from_weights, new_block};

/// Block abstraction for Llama. Groups a contiguous range of work (e.g. one transformer layer, one pipeline stage).
pub type LlamaBlock = RopeGqaDecoderBlock;
/// `LlamaModel`. See the type definition for fields and behavior.
pub type LlamaModel = CausalLm<LlamaBlock, RmsNorm, Linear>;

impl LlamaModel {
    /// Construct a new instance from the given configuration.
    /// # Errors
    ///
    /// Returns `Err` if any required tensor allocation or weight loading fails.
    pub fn new(config: ModelConfig, device: Device, num_kv_blocks: usize) -> CandleResult<Self> {
        Self::new_rms(config, device, num_kv_blocks, false, new_block)
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
