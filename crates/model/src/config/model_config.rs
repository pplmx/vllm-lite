//! Unified model configuration.
//!
//! Provides ModelConfig struct that works across different model architectures.

use super::architecture::{Architecture, LayerType, RoPEConfig};
use crate::loader::detect_architecture;

pub struct ModelConfig {
    pub architecture: Architecture,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub intermediate_size: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f64,
    pub sliding_window: Option<usize>,
    pub tie_word_embeddings: bool,
    pub max_position_embeddings: usize,
    pub layer_types: Vec<LayerType>,
    pub rope_configs: Vec<RoPEConfig>,
    pub use_double_wide_mlp: bool,
    pub num_experts: Option<usize>,
    pub top_k_experts: Option<usize>,
    pub expert_intermediate_size: Option<usize>,
}

impl ModelConfig {
    pub fn llama_7b() -> Self {
        Self {
            architecture: Architecture::Llama,
            hidden_size: 4096,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 32,
            head_dim: 128,
            vocab_size: 32000,
            intermediate_size: 11008,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            sliding_window: None,
            tie_word_embeddings: false,
            max_position_embeddings: 2048,
            layer_types: vec![],
            rope_configs: vec![],
            use_double_wide_mlp: false,
            num_experts: None,
            top_k_experts: None,
            expert_intermediate_size: None,
        }
    }

    pub fn mistral_7b() -> Self {
        Self {
            architecture: Architecture::Mistral,
            hidden_size: 4096,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            vocab_size: 32000,
            intermediate_size: 14336,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            sliding_window: Some(4096),
            tie_word_embeddings: false,
            max_position_embeddings: 32768,
            layer_types: vec![],
            rope_configs: vec![],
            use_double_wide_mlp: false,
            num_experts: None,
            top_k_experts: None,
            expert_intermediate_size: None,
        }
    }

    pub fn mixtral_8x7b() -> Self {
        Self {
            architecture: Architecture::Mixtral,
            hidden_size: 4096,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            vocab_size: 32000,
            intermediate_size: 14336,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            sliding_window: Some(4096),
            tie_word_embeddings: false,
            max_position_embeddings: 32768,
            layer_types: vec![],
            rope_configs: vec![],
            use_double_wide_mlp: false,
            num_experts: Some(8),
            top_k_experts: Some(2),
            expert_intermediate_size: Some(14336),
        }
    }

    pub fn from_config_json(value: &serde_json::Value) -> Result<Self, Box<dyn std::error::Error>> {
        let architecture = detect_architecture(value);

        let hidden_size = value
            .get("hidden_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(4096) as usize;
        let num_layers = value
            .get("num_hidden_layers")
            .and_then(|v| v.as_u64())
            .unwrap_or(32) as usize;
        let num_heads = value
            .get("num_attention_heads")
            .and_then(|v| v.as_u64())
            .unwrap_or(32) as usize;
        let num_kv_heads = value
            .get("num_key_value_heads")
            .or_else(|| value.get("num_local_heads"))
            .and_then(|v| v.as_u64())
            .unwrap_or(num_heads as u64) as usize;
        let head_dim = hidden_size / num_heads;
        let vocab_size = value
            .get("vocab_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(32000) as usize;
        let intermediate_size = value
            .get("intermediate_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(11008) as usize;
        let rope_theta = value
            .get("rope_theta")
            .or_else(|| value.get("rotary_base"))
            .and_then(|v| v.as_f64())
            .unwrap_or(10000.0) as f32;
        let rms_norm_eps = value
            .get("rms_norm_eps")
            .or_else(|| value.get("layer_norm_eps"))
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-5);
        let sliding_window = value
            .get("sliding_window")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);
        let tie_word_embeddings = value
            .get("tie_word_embeddings")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let max_position_embeddings = value
            .get("max_position_embeddings")
            .and_then(|v| v.as_u64())
            .unwrap_or(2048) as usize;
        let num_experts = value
            .get("num_local_experts")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);
        let top_k_experts = value
            .get("num_experts_per_tok")
            .or_else(|| value.get("top_k_experts"))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);
        let expert_intermediate_size = value
            .get("expert_intermediate_size")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);

        Ok(Self {
            architecture,
            hidden_size,
            num_layers,
            num_heads,
            num_kv_heads,
            head_dim,
            vocab_size,
            intermediate_size,
            rope_theta,
            rms_norm_eps,
            sliding_window,
            tie_word_embeddings,
            max_position_embeddings,
            layer_types: vec![],
            rope_configs: vec![],
            use_double_wide_mlp: false,
            num_experts,
            top_k_experts,
            expert_intermediate_size,
        })
    }
}
