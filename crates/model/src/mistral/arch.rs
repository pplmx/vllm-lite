//! Mistral architecture implementation.

use crate::arch::{ArchCapabilities, Architecture};
use crate::causal_lm::BlockWrapper;
use crate::components::TransformerBlock;
use crate::config::ModelConfig;
use crate::paged_tensor::PagedKvCache;
use candle_core::{Device, Result, Tensor};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;
use vllm_traits::ModelBackend;

use super::block::block_from_weights;
use super::model::MistralModel;

/// `MistralArchitecture`. See the type definition for fields and behavior.
pub(crate) struct MistralArchitecture;

impl MistralArchitecture {
    pub const fn new() -> Self {
        Self
    }
}

impl Default for MistralArchitecture {
    fn default() -> Self {
        Self::new()
    }
}

impl Architecture for MistralArchitecture {
    fn name(&self) -> &'static str {
        "mistral"
    }

    fn detect(&self, config_json: &serde_json::Value) -> bool {
        // PERF-02: avoid per-load `String` allocation.
        config_json
            .get("model_type")
            .and_then(|v| v.as_str())
            .is_some_and(|s| s.eq_ignore_ascii_case("mistral"))
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
    ) -> Result<(Box<dyn ModelBackend>, Option<Arc<Mutex<PagedKvCache>>>)> {
        let model =
            MistralModel::from_weights(config, device, weights, num_kv_blocks, kv_quantization)?;
        let kv_cache = model.paged_kv_cache();
        Ok((Box::new(model), Some(kv_cache)))
    }
}
