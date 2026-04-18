//! Mixtral architecture implementation (Sparse MoE).

use crate::arch::Architecture;
use crate::components::TransformerBlock;
use crate::config::ModelConfig;
use candle_core::{Device, Result, Tensor};
use std::collections::HashMap;
use vllm_traits::ModelBackend;

use super::block::MixtralBlock;
use super::model::MixtralModel;

pub struct MixtralArchitecture;

impl MixtralArchitecture {
    pub fn new() -> Self {
        Self
    }
}

impl Default for MixtralArchitecture {
    fn default() -> Self {
        Self::new()
    }
}

pub struct MixtralBlockWrapper {
    inner: MixtralBlock,
    inner_dim: usize,
    num_kv_heads: usize,
}

impl MixtralBlockWrapper {
    pub fn new(block: MixtralBlock, config: &ModelConfig) -> Self {
        Self {
            inner_dim: config.head_dim,
            num_kv_heads: config.num_kv_heads,
            inner: block,
        }
    }
}

impl TransformerBlock for MixtralBlockWrapper {
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

impl Architecture for MixtralArchitecture {
    fn name(&self) -> &'static str {
        "mixtral"
    }

    fn detect(&self, config_json: &serde_json::Value) -> bool {
        config_json
            .get("model_type")
            .and_then(|v| v.as_str())
            .map(|s| s.to_lowercase() == "mixtral")
            .unwrap_or(false)
    }

    fn create_block(
        &self,
        config: &ModelConfig,
        layer_idx: usize,
        weights: &HashMap<String, Tensor>,
        _device: &Device,
    ) -> Result<Box<dyn TransformerBlock>> {
        let block = MixtralBlock::from_weights(config, layer_idx, weights)?;
        Ok(Box::new(MixtralBlockWrapper::new(block, config)))
    }

    fn create_model(
        &self,
        config: ModelConfig,
        device: Device,
        weights: HashMap<String, Tensor>,
        num_kv_blocks: usize,
    ) -> Result<Box<dyn ModelBackend>> {
        let model = MixtralModel::from_weights(config, device, weights, num_kv_blocks)?;
        Ok(Box::new(model))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_mixtral_architecture_detect() {
        let arch = MixtralArchitecture::new();
        let config = json!({"model_type": "mixtral"});
        assert!(arch.detect(&config));
    }

    #[test]
    fn test_mixtral_architecture_detect_case_insensitive() {
        let arch = MixtralArchitecture::new();
        let config = json!({"model_type": "Mixtral"});
        assert!(arch.detect(&config));
    }

    #[test]
    fn test_mixtral_architecture_not_detect_others() {
        let arch = MixtralArchitecture::new();
        for model_type in ["llama", "mistral", "qwen3", "gpt2"] {
            let config = json!({"model_type": model_type});
            assert!(!arch.detect(&config), "Should not detect {}", model_type);
        }
    }

    #[test]
    fn test_mixtral_architecture_name() {
        let arch = MixtralArchitecture::new();
        assert_eq!(arch.name(), "mixtral");
    }

    #[test]
    fn test_mixtral_architecture_default() {
        let arch = MixtralArchitecture::new();
        assert_eq!(arch.name(), "mixtral");
    }
}
