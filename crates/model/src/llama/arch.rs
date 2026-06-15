//! Llama architecture implementation.

use crate::arch::{ArchCapabilities, Architecture};
use crate::components::TransformerBlock;
use crate::components::decoder_block::PagedDecoderBlock;
use crate::config::ModelConfig;
use crate::paged_tensor::PagedKvCache;
use candle_core::{Device, Result, Tensor};
use std::collections::HashMap;
use vllm_traits::ModelBackend;

use super::block::LlamaBlock;
use super::model::LlamaModel;

pub struct LlamaArchitecture;

impl LlamaArchitecture {
    pub fn new() -> Self {
        Self
    }
}

impl Default for LlamaArchitecture {
    fn default() -> Self {
        Self::new()
    }
}

pub struct LlamaBlockWrapper {
    inner: LlamaBlock,
    inner_dim: usize,
    num_kv_heads: usize,
}

impl LlamaBlockWrapper {
    pub fn new(block: LlamaBlock, config: &ModelConfig) -> Self {
        Self {
            inner_dim: config.head_dim,
            num_kv_heads: config.num_kv_heads,
            inner: block,
        }
    }
}

impl PagedDecoderBlock for LlamaBlockWrapper {
    fn forward_prefill(
        &self,
        x: &Tensor,
        kv_cache: &mut PagedKvCache,
        layer_idx: usize,
        block_ids: &[usize],
        positions: &[usize],
    ) -> Result<Tensor> {
        self.inner
            .forward_prefill(x, kv_cache, layer_idx, block_ids, positions)
    }

    fn forward_decode(
        &self,
        x: &Tensor,
        kv_cache: &mut PagedKvCache,
        layer_idx: usize,
        block_ids: &[usize],
        num_computed_tokens: usize,
        positions: &[usize],
    ) -> Result<Tensor> {
        self.inner.forward_decode(
            x,
            kv_cache,
            layer_idx,
            block_ids,
            num_computed_tokens,
            positions,
        )
    }
}

impl TransformerBlock for LlamaBlockWrapper {
    fn inner_dim(&self) -> usize {
        self.inner_dim
    }

    fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }
}

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
        let block = super::block::block_from_weights(config, layer_idx, weights)?;
        Ok(Box::new(LlamaBlockWrapper::new(block, config)))
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
        for model_type in ["llama", "llama2", "llama3"] {
            let config = json!({"model_type": model_type});
            assert!(
                arch.detect(&config),
                "Failed to detect model_type: {}",
                model_type
            );
        }
    }

    #[test]
    fn test_llama_architecture_detect_case_insensitive() {
        let arch = LlamaArchitecture::new();
        for model_type in ["LLAMA", "Llama2", "LLAMA3"] {
            let config = json!({"model_type": model_type});
            assert!(
                arch.detect(&config),
                "Failed to detect case-insensitive model_type: {}",
                model_type
            );
        }
    }

    #[test]
    fn test_llama_architecture_not_detect_others() {
        let arch = LlamaArchitecture::new();
        for model_type in ["mistral", "qwen2", "gpt2", "bert"] {
            let config = json!({"model_type": model_type});
            assert!(
                !arch.detect(&config),
                "Should not detect model_type: {}",
                model_type
            );
        }
    }

    #[test]
    fn test_llama_architecture_detect_missing_model_type() {
        let arch = LlamaArchitecture::new();
        let config = json!({"hidden_size": 4096});
        assert!(
            !arch.detect(&config),
            "Should not detect when model_type is missing"
        );
    }

    #[test]
    fn test_llama_architecture_name() {
        let arch = LlamaArchitecture::new();
        assert_eq!(arch.name(), "llama");
    }

    #[test]
    fn test_llama_architecture_default() {
        let arch = LlamaArchitecture::new();
        assert_eq!(arch.name(), "llama");
    }
}
