//! Gemma3 architecture implementation.

use crate::arch::{ArchCapabilities, Architecture};
use crate::components::block::{
    TransformerBlock, passthrough_paged_decode, passthrough_paged_prefill,
};
use crate::components::decoder_block::PagedDecoderBlock;
use crate::config::ModelConfig;
use crate::paged_tensor::PagedKvCache;
use candle_core::{Device, Result, Tensor};
use std::collections::HashMap;
use vllm_traits::ModelBackend;
use vllm_traits::types::BatchOutput;

/// Gemma3Architecture: gemma3 architecture.
pub struct Gemma3Architecture {
    sliding_window: usize,
}

impl Gemma3Architecture {
/// new: new.
    pub fn new() -> Self {
        Self {
            sliding_window: 4096,
        }
    }
}

impl Default for Gemma3Architecture {
    fn default() -> Self {
        Self::new()
    }
}

/// Gemma3BlockWrapper: gemma3 block wrapper.
pub struct Gemma3BlockWrapper {
    inner_dim: usize,
    num_kv_heads: usize,
}

impl Gemma3BlockWrapper {
/// new: new.
    pub fn new(config: &ModelConfig) -> Self {
        Self {
            inner_dim: config.head_dim * config.num_heads,
            num_kv_heads: config.num_kv_heads,
        }
    }
}

impl PagedDecoderBlock for Gemma3BlockWrapper {
    fn forward_prefill(
        &self,
        x: &Tensor,
        kv_cache: &mut PagedKvCache,
        layer_idx: usize,
        block_ids: &[usize],
        positions: &[usize],
    ) -> Result<Tensor> {
        passthrough_paged_prefill(x, kv_cache, layer_idx, block_ids, positions)
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
        passthrough_paged_decode(
            x,
            kv_cache,
            layer_idx,
            block_ids,
            num_computed_tokens,
            positions,
        )
    }
}

impl TransformerBlock for Gemma3BlockWrapper {
    fn inner_dim(&self) -> usize {
        self.inner_dim
    }

    fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }
}

impl Architecture for Gemma3Architecture {
    fn name(&self) -> &'static str {
        "gemma3"
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

        let is_gemma = matches!(
            model_type.to_lowercase().as_str(),
            "gemma" | "gemma2" | "gemma3"
        );

        is_gemma && hidden_size > 0
    }

    fn capabilities(&self) -> ArchCapabilities {
        ArchCapabilities::STUB
    }

    fn create_block(
        &self,
        config: &ModelConfig,
        _layer_idx: usize,
        _weights: &HashMap<String, Tensor>,
        _device: &Device,
    ) -> Result<Box<dyn TransformerBlock>> {
        Ok(Box::new(Gemma3BlockWrapper::new(config)))
    }

    fn create_model(
        &self,
        config: ModelConfig,
        device: Device,
        _weights: HashMap<String, Tensor>,
        num_kv_blocks: usize,
        _kv_quantization: bool,
    ) -> Result<Box<dyn ModelBackend>> {
        let model = Gemma3Model::new(config, device, num_kv_blocks, self.sliding_window)?;
        Ok(Box::new(model))
    }
}

/// Gemma3Model: gemma3 model.
pub struct Gemma3Model {
    config: ModelConfig,
    #[allow(dead_code)] // audited 2026-06-26 (Wave 1): stub arch (Option C)
    device: Device,
    #[allow(dead_code)] // audited 2026-06-26 (Wave 1): stub arch (Option C)
    num_kv_blocks: usize,
    #[allow(dead_code)] // audited 2026-06-26 (Wave 1): stub arch (Option C)
    sliding_window: usize,
}

impl Gemma3Model {
/// new: new.
    pub fn new(
        config: ModelConfig,
        device: Device,
        num_kv_blocks: usize,
        sliding_window: usize,
    ) -> Result<Self> {
        Ok(Self {
            config,
            device,
            num_kv_blocks,
            sliding_window,
        })
    }
}

impl ModelBackend for Gemma3Model {
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
    fn test_gemma3_architecture_detect() {
        let arch = Gemma3Architecture::new();
        for model_type in ["gemma", "gemma2", "gemma3"] {
            let config = json!({
                "model_type": model_type,
                "hidden_size": 3072
            });
            assert!(
                arch.detect(&config),
                "Failed to detect model_type: {}",
                model_type
            );
        }
    }

    #[test]
    fn test_gemma3_architecture_not_detect_others() {
        let arch = Gemma3Architecture::new();
        for model_type in ["llama", "mistral", "qwen2", "phi"] {
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
    fn test_gemma3_architecture_name() {
        let arch = Gemma3Architecture::new();
        assert_eq!(arch.name(), "gemma3");
    }
}
