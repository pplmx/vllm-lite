//! Phi-4 architecture implementation.

use crate::arch::Architecture;
use crate::components::TransformerBlock;
use crate::config::ModelConfig;
use candle_core::{Device, Result, Tensor};
use std::collections::HashMap;
use vllm_traits::ModelBackend;
use vllm_traits::types::BatchOutput;

pub struct Phi4Architecture;

impl Phi4Architecture {
    pub fn new() -> Self {
        Self
    }
}

impl Default for Phi4Architecture {
    fn default() -> Self {
        Self::new()
    }
}

pub struct Phi4BlockWrapper {
    inner_dim: usize,
    num_kv_heads: usize,
}

impl Phi4BlockWrapper {
    pub fn new(config: &ModelConfig) -> Self {
        Self {
            inner_dim: config.head_dim * config.num_heads,
            num_kv_heads: config.num_kv_heads,
        }
    }
}

impl TransformerBlock for Phi4BlockWrapper {
    fn forward(
        &mut self,
        hidden_states: &Tensor,
        _positions: &[usize],
        _kv_block_ids: &[usize],
        _num_computed: usize,
        _is_prefill: bool,
    ) -> Result<Tensor> {
        Ok(hidden_states.clone())
    }

    fn inner_dim(&self) -> usize {
        self.inner_dim
    }

    fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }
}

impl Architecture for Phi4Architecture {
    fn name(&self) -> &'static str {
        "phi4"
    }

    fn detect(&self, config_json: &serde_json::Value) -> bool {
        let model_type = config_json
            .get("model_type")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let hidden_size = config_json
            .get("hidden_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);

        let is_phi = model_type.to_lowercase().starts_with("phi");

        is_phi && hidden_size > 0
    }

    fn create_block(
        &self,
        config: &ModelConfig,
        _layer_idx: usize,
        _weights: &HashMap<String, Tensor>,
        _device: &Device,
    ) -> Result<Box<dyn TransformerBlock>> {
        Ok(Box::new(Phi4BlockWrapper::new(config)))
    }

    fn create_model(
        &self,
        config: ModelConfig,
        device: Device,
        _weights: HashMap<String, Tensor>,
        num_kv_blocks: usize,
    ) -> Result<Box<dyn ModelBackend>> {
        let model = Phi4Model::new(config, device, num_kv_blocks)?;
        Ok(Box::new(model))
    }
}

pub struct Phi4Model {
    config: ModelConfig,
    #[allow(dead_code)]
    device: Device,
    #[allow(dead_code)]
    num_kv_blocks: usize,
}

impl Phi4Model {
    pub fn new(config: ModelConfig, device: Device, num_kv_blocks: usize) -> Result<Self> {
        Ok(Self {
            config,
            device,
            num_kv_blocks,
        })
    }
}

impl ModelBackend for Phi4Model {
    fn forward(
        &mut self,
        seq_ids: &[vllm_traits::SeqId],
        _input_tokens: &[Vec<vllm_traits::TokenId>],
        _positions: &[Vec<usize>],
        _kv_block_ids: &[Vec<usize>],
        _num_computed_tokens: &[usize],
        _is_prefill: &[bool],
    ) -> vllm_traits::Result<BatchOutput> {
        let next_tokens: Vec<vllm_traits::TokenId> = seq_ids.iter().map(|_| 0).collect();
        Ok(BatchOutput {
            seq_ids: seq_ids.to_vec(),
            next_tokens,
        })
    }

    fn forward_logits(
        &mut self,
        seq_ids: &[vllm_traits::SeqId],
        _input_tokens: &[Vec<vllm_traits::TokenId>],
        _positions: &[Vec<usize>],
        _kv_block_ids: &[Vec<usize>],
        _num_computed_tokens: &[usize],
        _is_prefill: &[bool],
    ) -> vllm_traits::Result<Vec<Vec<f32>>> {
        Ok(vec![vec![0.0_f32; self.config.vocab_size]; seq_ids.len()])
    }

    fn embed(
        &mut self,
        input_tokens: &[Vec<vllm_traits::TokenId>],
        _positions: &[Vec<usize>],
    ) -> vllm_traits::Result<Vec<Vec<f32>>> {
        Ok(vec![
            vec![0.0_f32; self.config.hidden_size];
            input_tokens.len()
        ])
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn num_layers(&self) -> usize {
        self.config.num_layers
    }

    fn num_heads(&self) -> usize {
        self.config.num_heads
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_phi4_architecture_detect() {
        let arch = Phi4Architecture::new();
        for model_type in ["phi", "phi2", "phi3", "phi4"] {
            let config = json!({
                "model_type": model_type,
                "hidden_size": 5120
            });
            assert!(
                arch.detect(&config),
                "Failed to detect model_type: {}",
                model_type
            );
        }
    }

    #[test]
    fn test_phi4_architecture_not_detect_others() {
        let arch = Phi4Architecture::new();
        for model_type in ["llama", "mistral", "gemma", "qwen2"] {
            let config = json!({
                "model_type": model_type,
                "hidden_size": 4096
            });
            assert!(
                !arch.detect(&config),
                "Should not detect model_type: {}",
                model_type
            );
        }
    }

    #[test]
    fn test_phi4_architecture_name() {
        let arch = Phi4Architecture::new();
        assert_eq!(arch.name(), "phi4");
    }
}
