//! Qwen3.5 architecture implementation (Mamba SSM hybrid).

use std::collections::HashMap;

use candle_core::{Device, Result, Tensor};
use vllm_traits::ModelBackend;

use crate::arch::{ArchCapabilities, Architecture};
use crate::components::TransformerBlock;
use crate::config::ModelConfig;
use crate::qwen3_config::Qwen3Config;

use super::model::Qwen35HybridModel;

/// Normalize Qwen3.5 / Qwen3-VL hybrid checkpoint keys to the loader layout.
///
/// HF checkpoints may nest the language model under `model.language_model.*` and use
/// alternate final-norm / lm_head prefixes. After remapping, `load_hybrid_weights` expects:
///
/// - Embeddings: `model.embed_tokens.weight` (or pre-remap `model.language_model.embed_tokens.weight`)
/// - Per-layer blocks: `model.layers.{i}.*` with `linear_attn.*` (GDN) or `self_attn.*` (full)
/// - Final norm: `model.final_layernorm.weight` (from `model.norm.weight` in VL checkpoints)
/// - LM head: `model.lm_head.weight` (from top-level `lm_head.weight`)
pub fn remap_qwen35_weight_keys(weights: HashMap<String, Tensor>) -> HashMap<String, Tensor> {
    let mut remapped = HashMap::new();

    for (key, value) in weights {
        let new_key = if key.starts_with("model.language_model.") {
            key.replace("model.language_model.", "model.")
        } else if key.starts_with("lm_head.") || key.starts_with("model.lm_head.") {
            if key.starts_with("lm_head.") {
                key.replace("lm_head.", "model.lm_head.")
            } else {
                key
            }
        } else if key == "model.norm.weight" {
            "model.final_layernorm.weight".to_string()
        } else {
            key
        };
        remapped.insert(new_key, value);
    }

    remapped
}

/// Qwen35Architecture: qwen35 architecture.
pub struct Qwen35Architecture;

impl Qwen35Architecture {
/// new: new.
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

    fn capabilities(&self) -> ArchCapabilities {
        ArchCapabilities::PRODUCTION_SPECULATIVE
    }

    fn create_block(
        &self,
        _config: &ModelConfig,
        _layer_idx: usize,
        _weights: &HashMap<String, Tensor>,
        _device: &Device,
    ) -> Result<Box<dyn TransformerBlock>> {
        Err(candle_core::Error::Msg(
            "Qwen3.5 hybrid blocks are only available via create_model (SSM+attention layers)"
                .into(),
        ))
    }

    fn create_model(
        &self,
        config: ModelConfig,
        device: Device,
        weights: HashMap<String, Tensor>,
        num_kv_blocks: usize,
        kv_quantization: bool,
    ) -> Result<Box<dyn ModelBackend>> {
        let remapped_weights = remap_qwen35_weight_keys(weights);

        let qwen_config = Qwen3Config::from(config);

        let model = Qwen35HybridModel::from_weights(
            qwen_config,
            device,
            remapped_weights,
            num_kv_blocks,
            kv_quantization,
        )?;
        Ok(Box::new(model))
    }

    fn remap_weights(&self, weights: HashMap<String, Tensor>) -> HashMap<String, Tensor> {
        remap_qwen35_weight_keys(weights)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use serde_json::json;

    #[test]
    fn test_qwen35_capabilities_speculative() {
        let arch = Qwen35Architecture::new();
        assert_eq!(
            arch.capabilities(),
            ArchCapabilities::PRODUCTION_SPECULATIVE
        );
        assert!(arch.capabilities().speculative);
    }

    #[test]
    fn test_remap_qwen35_weight_keys() {
        let device = Device::Cpu;
        let t = Tensor::zeros(4, DType::F32, &device).unwrap();

        let mut weights = HashMap::new();
        weights.insert(
            "model.language_model.embed_tokens.weight".to_string(),
            t.clone(),
        );
        weights.insert("lm_head.weight".to_string(), t.clone());
        weights.insert("model.norm.weight".to_string(), t.clone());

        let remapped = remap_qwen35_weight_keys(weights);
        assert!(remapped.contains_key("model.embed_tokens.weight"));
        assert!(remapped.contains_key("model.lm_head.weight"));
        assert!(remapped.contains_key("model.final_layernorm.weight"));
    }

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
        let arch = Qwen35Architecture;
        assert_eq!(arch.name(), "qwen3.5");
    }
}
