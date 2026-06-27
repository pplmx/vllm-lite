//! Mistral architecture implementation.

use crate::arch::{ArchCapabilities, Architecture};
use crate::causal_lm::BlockWrapper;
use crate::components::TransformerBlock;
use crate::config::ModelConfig;
use candle_core::{Device, Result, Tensor};
use std::collections::HashMap;
use vllm_traits::ModelBackend;

use super::block::{MistralBlock, block_from_weights};
use super::model::MistralModel;

/// MistralArchitecture: mistral architecture.
pub struct MistralArchitecture;

impl MistralArchitecture {
    pub fn new() -> Self {
        Self
    }
}

impl Default for MistralArchitecture {
    fn default() -> Self {
        Self::new()
    }
}

/// MistralBlockWrapper: mistral block wrapper.
pub type MistralBlockWrapper = BlockWrapper<MistralBlock>;

impl Architecture for MistralArchitecture {
    fn name(&self) -> &'static str {
        "mistral"
    }

    fn detect(&self, config_json: &serde_json::Value) -> bool {
        // PERF-02: avoid per-load `String` allocation.
        config_json
            .get("model_type")
            .and_then(|v| v.as_str())
            .map(|s| s.eq_ignore_ascii_case("mistral"))
            .unwrap_or(false)
    }

    fn capabilities(&self) -> ArchCapabilities {
        ArchCapabilities::PRODUCTION
    }

    fn create_block(
        &self,
        config: &ModelConfig,
        layer_idx: usize,
        weights: &HashMap<String, Tensor>,
        _device: &Device,
    ) -> Result<Box<dyn TransformerBlock>> {
        let block = block_from_weights(config, layer_idx, weights)?;
        Ok(Box::new(BlockWrapper::new(block, config)))
    }

    fn create_model(
        &self,
        config: ModelConfig,
        device: Device,
        weights: HashMap<String, Tensor>,
        num_kv_blocks: usize,
        kv_quantization: bool,
    ) -> Result<Box<dyn ModelBackend>> {
        let model =
            MistralModel::from_weights(config, device, weights, num_kv_blocks, kv_quantization)?;
        Ok(Box::new(model))
    }
}
