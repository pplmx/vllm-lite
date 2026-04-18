//! Gemma4 architecture implementation.

use crate::arch::Architecture;
use crate::components::TransformerBlock;
use crate::config::ModelConfig;
use candle_core::{Device, Result, Tensor};
use std::collections::HashMap;
use vllm_traits::ModelBackend;

use super::block::Gemma4Block;
use super::model::Gemma4Model;

pub struct Gemma4Architecture;

impl Gemma4Architecture {
    pub fn new() -> Self {
        Self
    }
}

impl Default for Gemma4Architecture {
    fn default() -> Self {
        Self::new()
    }
}

pub struct Gemma4BlockWrapper {
    inner: Gemma4Block,
    inner_dim: usize,
    num_kv_heads: usize,
}

impl Gemma4BlockWrapper {
    pub fn new(block: Gemma4Block, config: &ModelConfig) -> Self {
        Self {
            inner_dim: config.head_dim,
            num_kv_heads: config.num_kv_heads,
            inner: block,
        }
    }
}

impl TransformerBlock for Gemma4BlockWrapper {
    fn forward(
        &mut self,
        hidden_states: &Tensor,
        positions: &[usize],
        _kv_block_ids: &[usize],
        _num_computed: usize,
        _is_prefill: bool,
    ) -> Result<Tensor> {
        self.inner.forward(hidden_states, positions)
    }

    fn inner_dim(&self) -> usize {
        self.inner_dim
    }

    fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }
}

impl Architecture for Gemma4Architecture {
    fn name(&self) -> &'static str {
        "gemma4"
    }

    fn detect(&self, config_json: &serde_json::Value) -> bool {
        let model_type = config_json
            .get("model_type")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        matches!(
            model_type.to_lowercase().as_str(),
            "gemma2" | "gemma3" | "gemma4"
        )
    }

    fn create_block(
        &self,
        config: &ModelConfig,
        layer_idx: usize,
        _weights: &HashMap<String, Tensor>,
        device: &Device,
    ) -> Result<Box<dyn TransformerBlock>> {
        let vb = candle_nn::VarBuilder::zeros(candle_core::DType::F32, device);
        let block = Gemma4Block::new(config, layer_idx, vb)?;
        Ok(Box::new(Gemma4BlockWrapper::new(block, config)))
    }

    fn create_model(
        &self,
        config: ModelConfig,
        device: Device,
        weights: HashMap<String, Tensor>,
        num_kv_blocks: usize,
    ) -> Result<Box<dyn ModelBackend>> {
        let model = Gemma4Model::from_weights(config, device, weights, num_kv_blocks)?;
        Ok(Box::new(model))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_gemma4_architecture_detect() {
        let arch = Gemma4Architecture::new();
        for model_type in ["gemma2", "gemma3", "gemma4"] {
            let config = json!({"model_type": model_type});
            assert!(
                arch.detect(&config),
                "Failed to detect model_type: {}",
                model_type
            );
        }
    }

    #[test]
    fn test_gemma4_architecture_detect_case_insensitive() {
        let arch = Gemma4Architecture::new();
        for model_type in ["Gemma2", "GEMMA3", "Gemma4"] {
            let config = json!({"model_type": model_type});
            assert!(
                arch.detect(&config),
                "Failed to detect case-insensitive model_type: {}",
                model_type
            );
        }
    }

    #[test]
    fn test_gemma4_architecture_not_detect_others() {
        let arch = Gemma4Architecture::new();
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
    fn test_gemma4_architecture_detect_missing_model_type() {
        let arch = Gemma4Architecture::new();
        let config = json!({"hidden_size": 4096});
        assert!(
            !arch.detect(&config),
            "Should not detect when model_type is missing"
        );
    }

    #[test]
    fn test_gemma4_architecture_name() {
        let arch = Gemma4Architecture::new();
        assert_eq!(arch.name(), "gemma4");
    }
}
