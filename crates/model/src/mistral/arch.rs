//! Mistral architecture implementation.

use crate::arch::Architecture;
use crate::components::TransformerBlock;
use crate::config::ModelConfig;
use candle_core::{Device, Result, Tensor};
use std::collections::HashMap;
use vllm_traits::ModelBackend;

use super::block::MistralBlock;
use super::model::MistralModel;

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

pub struct MistralBlockWrapper {
    inner: MistralBlock,
    inner_dim: usize,
    num_kv_heads: usize,
}

impl MistralBlockWrapper {
    pub fn new(block: MistralBlock, config: &ModelConfig) -> Self {
        Self {
            inner_dim: config.head_dim,
            num_kv_heads: config.num_kv_heads,
            inner: block,
        }
    }
}

impl TransformerBlock for MistralBlockWrapper {
    fn forward(
        &mut self,
        hidden_states: &Tensor,
        _positions: &[usize],
        _kv_block_ids: &[usize],
        _num_computed: usize,
        _is_prefill: bool,
    ) -> Result<Tensor> {
        self.inner.forward(hidden_states)
    }

    fn inner_dim(&self) -> usize {
        self.inner_dim
    }

    fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }
}

impl Architecture for MistralArchitecture {
    fn name(&self) -> &'static str {
        "mistral"
    }

    fn detect(&self, config_json: &serde_json::Value) -> bool {
        config_json
            .get("model_type")
            .and_then(|v| v.as_str())
            .map(|s| s.to_lowercase() == "mistral")
            .unwrap_or(false)
    }

    fn create_block(
        &self,
        config: &ModelConfig,
        layer_idx: usize,
        weights: &HashMap<String, Tensor>,
        _device: &Device,
    ) -> Result<Box<dyn TransformerBlock>> {
        let block = MistralBlock::from_weights(config, layer_idx, weights)?;
        Ok(Box::new(MistralBlockWrapper::new(block, config)))
    }

    fn create_model(
        &self,
        config: ModelConfig,
        device: Device,
        weights: HashMap<String, Tensor>,
        num_kv_blocks: usize,
    ) -> Result<Box<dyn ModelBackend>> {
        let model = MistralModel::from_weights(config, device, weights, num_kv_blocks)?;
        Ok(Box::new(model))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_mistral_architecture_detect() {
        let arch = MistralArchitecture::new();
        let config = json!({"model_type": "mistral"});
        assert!(arch.detect(&config));
    }

    #[test]
    fn test_mistral_architecture_detect_case_insensitive() {
        let arch = MistralArchitecture::new();
        let config = json!({"model_type": "Mistral"});
        assert!(arch.detect(&config));
    }

    #[test]
    fn test_mistral_architecture_not_detect_others() {
        let arch = MistralArchitecture::new();
        for model_type in ["llama", "qwen2", "gpt2"] {
            let config = json!({"model_type": model_type});
            assert!(!arch.detect(&config), "Should not detect {}", model_type);
        }
    }

    #[test]
    fn test_mistral_architecture_name() {
        let arch = MistralArchitecture::new();
        assert_eq!(arch.name(), "mistral");
    }

    #[test]
    fn test_mistral_architecture_default() {
        let arch = MistralArchitecture::new();
        assert_eq!(arch.name(), "mistral");
    }
}
