//! Llama 4 architecture implementation.

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

/// `Llama4Architecture`: llama4 architecture.
pub struct Llama4Architecture {
    #[allow(dead_code)] // audited 2026-06-26 (Wave 1): stub arch (Option C)
    is_moe: bool,
    #[allow(dead_code)] // audited 2026-06-26 (Wave 1): stub arch (Option C)
    num_experts: usize,
    #[allow(dead_code)] // audited 2026-06-26 (Wave 1): stub arch (Option C)
    num_active_experts: usize,
}

impl Llama4Architecture {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            is_moe: false,
            num_experts: 8,
            num_active_experts: 2,
        }
    }

    #[must_use]
    pub const fn with_moe(num_experts: usize, num_active_experts: usize) -> Self {
        Self {
            is_moe: true,
            num_experts,
            num_active_experts,
        }
    }
}

impl Default for Llama4Architecture {
    fn default() -> Self {
        Self::new()
    }
}

/// `Llama4BlockWrapper`: llama4 block wrapper.
pub(crate) struct Llama4BlockWrapper {
    inner_dim: usize,
    num_kv_heads: usize,
    #[allow(dead_code)] // audited 2026-06-26 (Wave 1): stub arch (Option C)
    is_moe: bool,
    #[allow(dead_code)] // audited 2026-06-26 (Wave 1): stub arch (Option C)
    num_experts: usize,
}

impl Llama4BlockWrapper {
    pub const fn new(config: &ModelConfig, is_moe: bool, num_experts: usize) -> Self {
        Self {
            inner_dim: config.head_dim * config.num_heads,
            num_kv_heads: config.num_kv_heads,
            is_moe,
            num_experts,
        }
    }
}

impl PagedDecoderBlock for Llama4BlockWrapper {
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

impl TransformerBlock for Llama4BlockWrapper {
    fn inner_dim(&self) -> usize {
        self.inner_dim
    }

    fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }
}

impl Architecture for Llama4Architecture {
    fn name(&self) -> &'static str {
        "llama4"
    }

    fn detect(&self, config_json: &serde_json::Value) -> bool {
        let model_type = config_json
            .get("model_type")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let hidden_size = config_json
            .get("hidden_size")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);

        let is_llama4 = matches!(
            model_type.to_lowercase().as_str(),
            "llama4" | "llama-4" | "meta-llama4"
        );

        is_llama4 && hidden_size > 0
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
        Ok(Box::new(Llama4BlockWrapper::new(
            config,
            self.is_moe,
            self.num_experts,
        )))
    }

    fn create_model(
        &self,
        config: ModelConfig,
        device: Device,
        _weights: HashMap<String, Tensor>,
        num_kv_blocks: usize,
        _kv_quantization: bool,
    ) -> Result<Box<dyn ModelBackend>> {
        let model = Llama4Model::new(config, device, num_kv_blocks, self.is_moe, self.num_experts)?;
        Ok(Box::new(model))
    }
}

/// `Llama4Model`: llama4 model.
pub(crate) struct Llama4Model {
    config: ModelConfig,
    #[allow(dead_code)] // audited 2026-06-26 (Wave 1): stub arch (Option C)
    device: Device,
    #[allow(dead_code)] // audited 2026-06-26 (Wave 1): stub arch (Option C)
    num_kv_blocks: usize,
    #[allow(dead_code)] // audited 2026-06-26 (Wave 1): stub arch (Option C)
    is_moe: bool,
    #[allow(dead_code)] // audited 2026-06-26 (Wave 1): stub arch (Option C)
    num_experts: usize,
}

impl Llama4Model {
    pub const fn new(
        config: ModelConfig,
        device: Device,
        num_kv_blocks: usize,
        is_moe: bool,
        num_experts: usize,
    ) -> Result<Self> {
        Ok(Self {
            config,
            device,
            num_kv_blocks,
            is_moe,
            num_experts,
        })
    }
}

impl ModelBackend for Llama4Model {
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

    #[allow(clippy::misnamed_getters)]
    fn num_heads(&self) -> usize {
        self.config.num_kv_heads
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_llama4_architecture_detect() {
        let arch = Llama4Architecture::new();
        for model_type in ["llama4", "llama-4", "meta-llama4"] {
            let config = json!({
                "model_type": model_type,
                "hidden_size": 8192
            });
            assert!(
                arch.detect(&config),
                "Failed to detect model_type: {model_type}"
            );
        }
    }

    #[test]
    fn test_llama4_architecture_not_detect_others() {
        let arch = Llama4Architecture::new();
        for model_type in ["llama", "llama2", "llama3", "mistral", "gemma"] {
            let config = json!({
                "model_type": model_type,
                "hidden_size": 4096
            });
            assert!(
                !arch.detect(&config),
                "Should not detect model_type: {model_type}"
            );
        }
    }

    #[test]
    fn test_llama4_architecture_name() {
        let arch = Llama4Architecture::new();
        assert_eq!(arch.name(), "llama4");
    }

    #[test]
    fn test_llama4_moe_architecture() {
        let arch = Llama4Architecture::with_moe(16, 2);
        assert!(arch.is_moe);
        assert_eq!(arch.num_experts, 16);
        assert_eq!(arch.num_active_experts, 2);
    }
}
