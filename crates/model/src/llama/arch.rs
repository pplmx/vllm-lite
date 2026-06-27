//! Llama architecture implementation.

use crate::arch::{ArchCapabilities, Architecture};
use crate::causal_lm::BlockWrapper;
use crate::components::TransformerBlock;
use crate::config::ModelConfig;
use candle_core::{Device, Result, Tensor};
use std::collections::HashMap;
use vllm_traits::ModelBackend;

use super::block::{LlamaBlock, block_from_weights};
use super::model::LlamaModel;

/// LlamaArchitecture: llama architecture.
pub struct LlamaArchitecture;

impl LlamaArchitecture {
/// new: new.
    pub fn new() -> Self {
        Self
    }
}

impl Default for LlamaArchitecture {
    fn default() -> Self {
        Self::new()
    }
}

/// LlamaBlockWrapper: llama block wrapper.
pub type LlamaBlockWrapper = BlockWrapper<LlamaBlock>;

impl Architecture for LlamaArchitecture {
    fn name(&self) -> &'static str {
        "llama"
    }

    fn detect(&self, config_json: &serde_json::Value) -> bool {
        let model_type = config_json
            .get("model_type")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        matches!(
            model_type.to_lowercase().as_str(),
            "llama" | "llama2" | "llama3"
        )
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
            LlamaModel::from_weights(config, device, weights, num_kv_blocks, kv_quantization)?;
        Ok(Box::new(model))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_llama_architecture_detect() {
        let arch = LlamaArchitecture::new();
        for model_type in ["llama", "llama2", "llama3", "LLAMA"] {
            let config = json!({"model_type": model_type});
            assert!(
                arch.detect(&config),
                "Failed to detect model_type: {}",
                model_type
            );
        }
    }

    #[test]
    fn test_llama_architecture_not_detect_others() {
        let arch = LlamaArchitecture::new();
        for model_type in ["mistral", "qwen3", "gpt2"] {
            let config = json!({"model_type": model_type});
            assert!(
                !arch.detect(&config),
                "Should not detect model_type: {}",
                model_type
            );
        }
    }

    #[test]
    fn test_llama_architecture_name() {
        let arch = LlamaArchitecture::new();
        assert_eq!(arch.name(), "llama");
    }
}
