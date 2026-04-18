//! Qwen3.5 architecture implementation (Mamba SSM hybrid).

use std::collections::HashMap;

use candle_core::{Device, Result, Tensor};
use vllm_traits::ModelBackend;

use crate::arch::Architecture;
use crate::components::TransformerBlock;
use crate::config::ModelConfig;
use crate::qwen3_config::Qwen3Config;

use super::hybrid::Qwen35HybridModel;

pub struct Qwen35Architecture;

impl Qwen35Architecture {
    pub fn new() -> Self {
        Self
    }
}

impl Default for Qwen35Architecture {
    fn default() -> Self {
        Self::new()
    }
}

impl Architecture for Qwen35Architecture {
    fn name(&self) -> &'static str {
        "qwen3.5"
    }

    fn detect(&self, config_json: &serde_json::Value) -> bool {
        let model_type = config_json
            .get("model_type")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        matches!(
            model_type.to_lowercase().as_str(),
            "qwen3.5" | "qwen3_5" | "mamba"
        )
    }

    fn create_block(
        &self,
        _config: &ModelConfig,
        _layer_idx: usize,
        _weights: &HashMap<String, Tensor>,
        _device: &Device,
    ) -> Result<Box<dyn TransformerBlock>> {
        todo!("Qwen3.5 hybrid block not yet implemented - using model-level integration")
    }

    fn create_model(
        &self,
        config: ModelConfig,
        device: Device,
        weights: HashMap<String, Tensor>,
        num_kv_blocks: usize,
    ) -> Result<Box<dyn ModelBackend>> {
        let remapped_weights = crate::loader::remap_qwen35_weight_keys(weights);

        let qwen_config = Qwen3Config {
            vocab_size: Some(config.vocab_size),
            hidden_size: Some(config.hidden_size),
            num_hidden_layers: Some(config.num_layers),
            num_attention_heads: Some(config.num_heads),
            num_key_value_heads: Some(config.num_kv_heads),
            intermediate_size: Some(config.intermediate_size),
            rope_theta: Some(config.rope_theta),
            max_position_embeddings: Some(config.max_position_embeddings),
            rms_norm_eps: Some(config.rms_norm_eps as f32),
            tie_word_embeddings: Some(config.tie_word_embeddings),
            ..Default::default()
        };

        let model = Qwen35HybridModel::from_weights(qwen_config, device, remapped_weights, num_kv_blocks)?;
        Ok(Box::new(model))
    }

    fn remap_weights(&self, weights: HashMap<String, Tensor>) -> HashMap<String, Tensor> {
        crate::loader::remap_qwen35_weight_keys(weights)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_qwen35_architecture_detect() {
        let arch = Qwen35Architecture::new();
        for model_type in ["qwen3.5", "qwen3_5", "mamba"] {
            let config = json!({"model_type": model_type});
            assert!(
                arch.detect(&config),
                "Failed to detect model_type: {}",
                model_type
            );
        }
    }

    #[test]
    fn test_qwen35_architecture_detect_case_insensitive() {
        let arch = Qwen35Architecture::new();
        for model_type in ["Qwen3.5", "QWEN3_5", "Mamba"] {
            let config = json!({"model_type": model_type});
            assert!(
                arch.detect(&config),
                "Failed to detect case-insensitive model_type: {}",
                model_type
            );
        }
    }

    #[test]
    fn test_qwen35_architecture_not_detect_others() {
        let arch = Qwen35Architecture::new();
        for model_type in ["llama", "mistral", "qwen3", "gpt2", "bert"] {
            let config = json!({"model_type": model_type});
            assert!(
                !arch.detect(&config),
                "Should not detect model_type: {}",
                model_type
            );
        }
    }

    #[test]
    fn test_qwen35_architecture_detect_missing_model_type() {
        let arch = Qwen35Architecture::new();
        let config = json!({"hidden_size": 4096});
        assert!(
            !arch.detect(&config),
            "Should not detect when model_type is missing"
        );
    }

    #[test]
    fn test_qwen35_architecture_name() {
        let arch = Qwen35Architecture::new();
        assert_eq!(arch.name(), "qwen3.5");
    }

    #[test]
    fn test_qwen35_architecture_default() {
        let arch = Qwen35Architecture::default();
        assert_eq!(arch.name(), "qwen3.5");
    }
}
