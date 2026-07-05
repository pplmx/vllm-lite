//! Mistral Small architecture implementation.

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

#[derive(Debug)]
/// `MistralSmallArchitecture`. See the type definition for fields and behavior.
pub struct MistralSmallArchitecture {
    #[allow(dead_code)] // audited 2026-06-26 (Wave 1): stub arch (Option C)
    num_experts: usize,
    #[allow(dead_code)] // audited 2026-06-26 (Wave 1): stub arch (Option C)
    num_active_experts: usize,
}

impl MistralSmallArchitecture {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            num_experts: 8,
            num_active_experts: 2,
        }
    }

    #[must_use]
    pub const fn with_experts(num_experts: usize, num_active_experts: usize) -> Self {
        Self {
            num_experts,
            num_active_experts,
        }
    }
}

impl Default for MistralSmallArchitecture {
    fn default() -> Self {
        Self::new()
    }
}

/// Wrapper around `MistralSmallBlock`. Adds a single responsibility (caching, locking, observation) on top of the inner type.
pub(crate) struct MistralSmallBlockWrapper {
    inner_dim: usize,
    num_kv_heads: usize,
    #[allow(dead_code)] // audited 2026-06-26 (Wave 1): stub arch (Option C)
    num_experts: usize,
    #[allow(dead_code)] // audited 2026-06-26 (Wave 1): stub arch (Option C)
    num_active_experts: usize,
}

impl MistralSmallBlockWrapper {
    pub const fn new(config: &ModelConfig, num_experts: usize, num_active_experts: usize) -> Self {
        Self {
            inner_dim: config.head_dim * config.num_heads,
            num_kv_heads: config.num_kv_heads,
            num_experts,
            num_active_experts,
        }
    }
}

impl PagedDecoderBlock for MistralSmallBlockWrapper {
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

impl TransformerBlock for MistralSmallBlockWrapper {
    fn inner_dim(&self) -> usize {
        self.inner_dim
    }

    fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }
}

impl Architecture for MistralSmallArchitecture {
    fn name(&self) -> &'static str {
        "mistral-small"
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

        let num_experts = config_json
            .get("num_local_experts")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);

        let is_mistral_small = model_type.to_lowercase().contains("mistral")
            && (model_type.to_lowercase().contains("small")
                || model_type.to_lowercase().contains("mistral-small"));

        is_mistral_small && hidden_size > 0 && num_experts > 1
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
        Ok(Box::new(MistralSmallBlockWrapper::new(
            config,
            self.num_experts,
            self.num_active_experts,
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
        let model = MistralSmallModel::new(
            config,
            device,
            num_kv_blocks,
            self.num_experts,
            self.num_active_experts,
        );
        Ok(Box::new(model))
    }
}

/// `MistralSmallModel`. See the type definition for fields and behavior.
pub(crate) struct MistralSmallModel {
    config: ModelConfig,
    #[allow(dead_code)] // audited 2026-06-26 (Wave 1): stub arch (Option C)
    device: Device,
    #[allow(dead_code)] // audited 2026-06-26 (Wave 1): stub arch (Option C)
    num_kv_blocks: usize,
    #[allow(dead_code)] // audited 2026-06-26 (Wave 1): stub arch (Option C)
    num_experts: usize,
    #[allow(dead_code)] // audited 2026-06-26 (Wave 1): stub arch (Option C)
    num_active_experts: usize,
}

impl MistralSmallModel {
    pub const fn new(
        config: ModelConfig,
        device: Device,
        num_kv_blocks: usize,
        num_experts: usize,
        num_active_experts: usize,
    ) -> Self {
        Self {
            config,
            device,
            num_kv_blocks,
            num_experts,
            num_active_experts,
        }
    }
}

impl ModelBackend for MistralSmallModel {
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
    fn test_mistral_small_architecture_detect() {
        let arch = MistralSmallArchitecture::new();
        for model_type in ["mistral-small", "Mistral-Small-Instruct-2407"] {
            let config = json!({
                "model_type": model_type,
                "hidden_size": 4096,
                "num_local_experts": 8
            });
            assert!(
                arch.detect(&config),
                "Failed to detect model_type: {model_type}"
            );
        }
    }

    #[test]
    fn test_mistral_small_architecture_not_detect_others() {
        let arch = MistralSmallArchitecture::new();
        for model_type in ["mistral", "mistral-7b", "llama"] {
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
    fn test_mistral_small_architecture_name() {
        let arch = MistralSmallArchitecture::new();
        assert_eq!(arch.name(), "mistral-small");
    }

    #[test]
    fn test_mistral_small_expert_config() {
        let arch = MistralSmallArchitecture::with_experts(16, 4);
        assert_eq!(arch.num_experts, 16);
        assert_eq!(arch.num_active_experts, 4);
    }
}
