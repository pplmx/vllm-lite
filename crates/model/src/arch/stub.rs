//! Unified stub `Architecture` for architectures that are registered for
//! `detect()` but whose full implementation is not yet shipped.
//!
//! Replaces four near-identical stubs (gemma3, llama4, phi4,
//! mistral_small) that previously each carried their own ~250-line
//! `arch.rs` file. The unified struct is parameterised by:
//!
//! - `name` — the wire-format name (e.g. `"gemma3"`, `"phi4"`,
//!   `"mistral-small"`).
//! - `detect` — a `fn(&Value) -> bool` that decides whether this
//!   stub should claim a given config JSON. Each of the 4 former
//!   stubs had its own detection pattern; that logic is preserved
//!   verbatim in the constructor helpers below.
//!
//! `StubArchitecture::STUB_CAPABILITIES` reports
//! `ArchCapabilities::STUB`; `create_block` / `create_model` return
//! the shared [`StubBlockWrapper`] / [`StubModel`], both of which
//! always return zero tokens / passthrough KV (intentionally — see
//! `.planning/MODEL-ARCHITECTURE-REFACTOR.md` Phase 4.4 Option C).
//!
//! `ModelLoader` still rejects stub models unless `--allow-stub` is
//! passed (see `crates/server/src/main.rs` and Phase 12d policy).

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

/// Detection predicate for a stub architecture: takes a parsed config
/// JSON and returns whether this stub should claim it.
pub type StubDetectFn = fn(&serde_json::Value) -> bool;

/// Unified stub `Architecture` covering all currently-stubbed model
/// families (gemma3, llama4, phi4, mistral_small).
#[derive(Debug, Clone, Copy)]
pub struct StubArchitecture {
    name: &'static str,
    detect: StubDetectFn,
}

impl StubArchitecture {
    /// Construct a new stub architecture with the given wire-format
    /// name and detection predicate.
    #[must_use]
    pub const fn new(name: &'static str, detect: StubDetectFn) -> Self {
        Self { name, detect }
    }

    /// `gemma3` / `gemma2` / `gemma` with `hidden_size > 0`.
    #[must_use]
    pub const fn gemma3() -> Self {
        Self::new("gemma3", detect_gemma3)
    }

    /// `llama4` / `llama-4` / `meta-llama4` with `hidden_size > 0`.
    #[must_use]
    pub const fn llama4() -> Self {
        Self::new("llama4", detect_llama4)
    }

    /// `phi*` with `hidden_size > 0`.
    #[must_use]
    pub const fn phi4() -> Self {
        Self::new("phi4", detect_phi4)
    }

    /// `mistral*small*` with `hidden_size > 0` and `num_local_experts > 1`.
    #[must_use]
    pub const fn mistral_small() -> Self {
        Self::new("mistral-small", detect_mistral_small)
    }
}

fn detect_gemma3(config_json: &serde_json::Value) -> bool {
    let model_type = config_json
        .get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let hidden_size = config_json
        .get("hidden_size")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let is_gemma = matches!(
        model_type.to_lowercase().as_str(),
        "gemma" | "gemma2" | "gemma3"
    );
    is_gemma && hidden_size > 0
}

fn detect_llama4(config_json: &serde_json::Value) -> bool {
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

fn detect_phi4(config_json: &serde_json::Value) -> bool {
    let model_type = config_json
        .get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let hidden_size = config_json
        .get("hidden_size")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let is_phi = model_type.to_lowercase().starts_with("phi");
    is_phi && hidden_size > 0
}

fn detect_mistral_small(config_json: &serde_json::Value) -> bool {
    let model_type = config_json
        .get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_lowercase();
    let hidden_size = config_json
        .get("hidden_size")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let num_experts = config_json
        .get("num_local_experts")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let is_mistral_small = model_type.contains("mistral")
        && (model_type.contains("small") || model_type.contains("mistral-small"));
    is_mistral_small && hidden_size > 0 && num_experts > 1
}

impl Architecture for StubArchitecture {
    fn name(&self) -> &'static str {
        self.name
    }

    fn detect(&self, config_json: &serde_json::Value) -> bool {
        (self.detect)(config_json)
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
        Ok(Box::new(StubBlockWrapper::new(config)))
    }

    fn create_model(
        &self,
        config: ModelConfig,
        device: Device,
        _weights: HashMap<String, Tensor>,
        num_kv_blocks: usize,
        _kv_quantization: bool,
    ) -> Result<Box<dyn ModelBackend>> {
        Ok(Box::new(StubModel::new(config, device, num_kv_blocks)))
    }
}

/// Shared block wrapper used by all stub architectures. Forwards the
/// prefill / decode tensors to the passthrough paged implementations;
/// the actual model body is the [`StubModel`] which returns zero tokens.
pub(crate) struct StubBlockWrapper {
    inner_dim: usize,
    num_kv_heads: usize,
}

impl StubBlockWrapper {
    pub const fn new(config: &ModelConfig) -> Self {
        Self {
            inner_dim: config.head_dim * config.num_heads,
            num_kv_heads: config.num_kv_heads,
        }
    }
}

impl PagedDecoderBlock for StubBlockWrapper {
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

impl TransformerBlock for StubBlockWrapper {
    fn inner_dim(&self) -> usize {
        self.inner_dim
    }

    fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }
}

/// Shared backend used by all stub architectures. Returns zero tokens
/// / zero logits / zero embeddings on every forward call. The paged KV
/// cache is still updated by [`StubBlockWrapper::forward_prefill`] /
/// [`StubBlockWrapper::forward_decode`] so downstream code can validate
/// the KV plumbing without actually running a model.
pub(crate) struct StubModel {
    config: ModelConfig,
    #[allow(dead_code)] // stub: device recorded for completeness
    device: Device,
    #[allow(dead_code)] // stub: kv_blocks recorded for completeness
    num_kv_blocks: usize,
}

impl StubModel {
    pub const fn new(config: ModelConfig, device: Device, num_kv_blocks: usize) -> Self {
        Self {
            config,
            device,
            num_kv_blocks,
        }
    }
}

impl ModelBackend for StubModel {
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
    fn stub_architecture_gemma3_detect() {
        let arch = StubArchitecture::gemma3();
        for model_type in ["gemma", "gemma2", "gemma3"] {
            let config = json!({
                "model_type": model_type,
                "hidden_size": 3072
            });
            assert!(
                arch.detect(&config),
                "Failed to detect model_type: {model_type}"
            );
        }
        // No hidden_size → should NOT detect.
        assert!(!arch.detect(&json!({"model_type": "gemma3"})));
    }

    #[test]
    fn stub_architecture_gemma3_rejects_others() {
        let arch = StubArchitecture::gemma3();
        for model_type in ["llama", "mistral", "qwen2", "phi"] {
            let config = json!({"model_type": model_type, "hidden_size": 4096});
            assert!(
                !arch.detect(&config),
                "Should not detect model_type: {model_type}"
            );
        }
    }

    #[test]
    fn stub_architecture_llama4_detect() {
        let arch = StubArchitecture::llama4();
        for model_type in ["llama4", "llama-4", "meta-llama4"] {
            let config = json!({"model_type": model_type, "hidden_size": 4096});
            assert!(
                arch.detect(&config),
                "Failed to detect model_type: {model_type}"
            );
        }
        // "llama" (without 4) should NOT detect.
        assert!(!arch.detect(&json!({"model_type": "llama", "hidden_size": 4096})));
    }

    #[test]
    fn stub_architecture_phi4_detect() {
        let arch = StubArchitecture::phi4();
        for model_type in ["phi", "phi2", "phi3", "phi4"] {
            let config = json!({"model_type": model_type, "hidden_size": 5120});
            assert!(
                arch.detect(&config),
                "Failed to detect model_type: {model_type}"
            );
        }
        // Non-phi should not detect.
        assert!(!arch.detect(&json!({"model_type": "llama", "hidden_size": 4096})));
    }

    #[test]
    fn stub_architecture_mistral_small_detect() {
        let arch = StubArchitecture::mistral_small();
        let config = json!({
            "model_type": "mistral-small",
            "hidden_size": 4096,
            "num_local_experts": 8
        });
        assert!(arch.detect(&config));
        // Missing experts → should NOT detect (moe-only stub).
        assert!(!arch.detect(&json!({
            "model_type": "mistral-small",
            "hidden_size": 4096
        })));
    }

    #[test]
    fn stub_architecture_name() {
        assert_eq!(StubArchitecture::gemma3().name(), "gemma3");
        assert_eq!(StubArchitecture::llama4().name(), "llama4");
        assert_eq!(StubArchitecture::phi4().name(), "phi4");
        assert_eq!(StubArchitecture::mistral_small().name(), "mistral-small");
    }

    #[test]
    fn stub_architecture_capabilities_is_stub() {
        assert_eq!(
            StubArchitecture::gemma3().capabilities(),
            ArchCapabilities::STUB
        );
    }
}
