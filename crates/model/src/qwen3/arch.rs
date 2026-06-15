//! Qwen3 architecture implementation.

use crate::arch::{ArchCapabilities, Architecture};
use crate::causal_lm::BlockWrapper;
use crate::components::TransformerBlock;
use crate::config::ModelConfig;
use crate::qwen3_config::Qwen3Config;
use candle_core::{Device, Result, Tensor};
use std::collections::HashMap;
use vllm_traits::ModelBackend;

use super::block::TransformerBlock as Qwen3Block;
use super::model::Qwen3Model;

pub struct Qwen3Architecture;

impl Qwen3Architecture {
    pub fn new() -> Self {
        Self
    }
}

impl Default for Qwen3Architecture {
    fn default() -> Self {
        Self::new()
    }
}

pub type Qwen3BlockWrapper = BlockWrapper<Qwen3Block>;

impl Architecture for Qwen3Architecture {
    fn name(&self) -> &'static str {
        "qwen3"
    }

    fn detect(&self, config_json: &serde_json::Value) -> bool {
        let model_type = config_json
            .get("model_type")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        matches!(
            model_type.to_lowercase().as_str(),
            "qwen2" | "qwen2.5" | "qwen2_5" | "qwen3" | "qwen_3"
        )
    }

    fn capabilities(&self) -> ArchCapabilities {
        ArchCapabilities::PRODUCTION_SPECULATIVE
    }

    fn create_block(
        &self,
        config: &ModelConfig,
        layer_idx: usize,
        weights: &HashMap<String, Tensor>,
        _device: &Device,
    ) -> Result<Box<dyn TransformerBlock>> {
        let block = Qwen3Block::from_weights(config, layer_idx, weights)?;
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
        let qwen_config = Qwen3Config::from(config);
        let model =
            Qwen3Model::from_weights(qwen_config, device, weights, num_kv_blocks, kv_quantization)?;
        Ok(Box::new(model))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_qwen3_architecture_detect() {
        let arch = Qwen3Architecture::new();
        for model_type in ["qwen2", "qwen2.5", "qwen2_5", "qwen3", "qwen_3"] {
            let config = json!({"model_type": model_type});
            assert!(
                arch.detect(&config),
                "Failed to detect model_type: {}",
                model_type
            );
        }
    }

    #[test]
    fn test_qwen3_architecture_detect_case_insensitive() {
        let arch = Qwen3Architecture::new();
        for model_type in ["QWEN2", "Qwen2.5", "Qwen3"] {
            let config = json!({"model_type": model_type});
            assert!(
                arch.detect(&config),
                "Failed to detect case-insensitive model_type: {}",
                model_type
            );
        }
    }

    #[test]
    fn test_qwen3_architecture_not_detect_others() {
        let arch = Qwen3Architecture::new();
        for model_type in ["llama", "mistral", "gpt2", "bert"] {
            let config = json!({"model_type": model_type});
            assert!(
                !arch.detect(&config),
                "Should not detect model_type: {}",
                model_type
            );
        }
    }

    #[test]
    fn test_qwen3_architecture_detect_missing_model_type() {
        let arch = Qwen3Architecture::new();
        let config = json!({"hidden_size": 4096});
        assert!(
            !arch.detect(&config),
            "Should not detect when model_type is missing"
        );
    }

    #[test]
    fn test_qwen3_architecture_name() {
        let arch = Qwen3Architecture::new();
        assert_eq!(arch.name(), "qwen3");
    }

    #[test]
    fn test_qwen3_architecture_default() {
        let arch = Qwen3Architecture::new();
        assert_eq!(arch.name(), "qwen3");
    }
}
