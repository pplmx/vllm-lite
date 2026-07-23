//! Unified model configuration.
//!
//! Provides `ModelConfig` struct that works across different model architectures.

use super::architecture::{Architecture, LayerType, RoPEConfig};
use super::errors::ConfigResult;
use crate::arch::ARCHITECTURE_REGISTRY;

#[derive(Debug)]
/// Configuration for Model. Constructed via the `builder()` associated function or by deserializing from JSON / TOML. Pass-by-value to construction APIs.
pub struct ModelConfig {
    /// Resolved architecture (`Llama`, `Qwen3`, `Mamba`, etc.).
    pub architecture: Architecture,
    /// Residual stream dimensionality.
    pub hidden_size: usize,
    /// Number of stacked decoder layers.
    pub num_layers: usize,
    /// Number of query attention heads per layer.
    pub num_heads: usize,
    /// Number of key/value heads (GQA grouping).
    pub num_kv_heads: usize,
    /// Per-head dimension; for most models `hidden_size / num_heads`.
    pub head_dim: usize,
    /// Tokeniser vocabulary size.
    pub vocab_size: usize,
    /// MLP hidden dimension (gate/up/down projections).
    pub intermediate_size: usize,
    /// `RoPE` base wavelength.
    pub rope_theta: f32,
    /// `RMSNorm` numerical-stability epsilon.
    pub rms_norm_eps: f64,
    /// Sliding-window attention span (`None` = full causal).
    pub sliding_window: Option<usize>,
    /// Whether the LM head shares weights with the embedding table.
    pub tie_word_embeddings: bool,
    /// Maximum sequence length the model was trained on.
    pub max_position_embeddings: usize,
    /// Per-layer type list (sliding vs. full attention, SSM, etc.).
    pub layer_types: Vec<LayerType>,
    /// Per-layer `RoPE` overrides (NTK scaling, `MRoPE` axes, etc.).
    pub rope_configs: Vec<RoPEConfig>,
    /// Long-context `RoPE` scaling block (YaRN, Linear, Dynamic, Su).
    /// `None` means default (no scaling). Populated from
    /// `config.json["rope_scaling"]` in `from_config_json`, and
    /// forwarded into `RopeGqaAttention::new_with_rope_scaling` so the
    /// scaling actually reaches the `RoPE` inverse-frequency table and the
    /// attention-temperature factor. See `qwen3::config::RopeScaling`.
    pub rope_scaling: Option<crate::qwen3::config::RopeScaling>,
    /// Use the "double-wide" MLP variant (Phi-3 style).
    pub use_double_wide_mlp: bool,
    /// Number of experts for sparse `MoE` (`None` = dense model).
    pub num_experts: Option<usize>,
    /// Top-k routing for sparse `MoE` (`None` = dense model).
    pub top_k_experts: Option<usize>,
    /// Per-expert intermediate size for sparse `MoE`.
    pub expert_intermediate_size: Option<usize>,
    /// Qwen3-style Q/K `RMSNorm` before `RoPE` (default false for other architectures).
    pub has_qk_norm: bool,
}

impl ModelConfig {
    /// Tiny configuration for fast unit tests
    /// Hidden size: 128, Heads: 4, Head dim: 32
    /// Tiny config for a specific architecture (fast CPU smoke tests).
    #[must_use]
    pub fn test_tiny_for(architecture: Architecture) -> Self {
        let sliding_window = if architecture == Architecture::Mistral {
            Some(4096)
        } else {
            None
        };
        Self {
            architecture,
            sliding_window,
            ..Self::test_tiny()
        }
    }

    #[must_use]
    pub const fn test_tiny() -> Self {
        Self {
            architecture: Architecture::Llama,
            hidden_size: 128,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 4,
            head_dim: 32,
            vocab_size: 1000,
            intermediate_size: 256,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            sliding_window: None,
            tie_word_embeddings: false,
            max_position_embeddings: 512,
            layer_types: vec![],
            rope_configs: vec![],
            rope_scaling: None,
            use_double_wide_mlp: false,
            num_experts: None,
            top_k_experts: None,
            expert_intermediate_size: None,
            has_qk_norm: false,
        }
    }

    /// Small configuration for faster integration tests
    /// Hidden size: 256, Heads: 4, Head dim: 64
    #[must_use]
    pub const fn test_small() -> Self {
        Self {
            architecture: Architecture::Llama,
            hidden_size: 256,
            num_layers: 4,
            num_heads: 4,
            num_kv_heads: 4,
            head_dim: 64,
            vocab_size: 1000,
            intermediate_size: 512,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            sliding_window: None,
            tie_word_embeddings: false,
            max_position_embeddings: 512,
            layer_types: vec![],
            rope_configs: vec![],
            rope_scaling: None,
            use_double_wide_mlp: false,
            num_experts: None,
            top_k_experts: None,
            expert_intermediate_size: None,
            has_qk_norm: false,
        }
    }

    /// Medium configuration for more realistic tests
    #[must_use]
    pub const fn test_medium() -> Self {
        Self {
            architecture: Architecture::Llama,
            hidden_size: 512,
            num_layers: 8,
            num_heads: 8,
            num_kv_heads: 8,
            head_dim: 64,
            vocab_size: 5000,
            intermediate_size: 1024,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            sliding_window: None,
            tie_word_embeddings: false,
            max_position_embeddings: 2048,
            layer_types: vec![],
            rope_configs: vec![],
            rope_scaling: None,
            use_double_wide_mlp: false,
            num_experts: None,
            top_k_experts: None,
            expert_intermediate_size: None,
            has_qk_norm: false,
        }
    }

    /// `llama_7b`: llama 7b.
    #[must_use]
    pub const fn llama_7b() -> Self {
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
            rope_scaling: None,
            use_double_wide_mlp: false,
            num_experts: None,
            top_k_experts: None,
            expert_intermediate_size: None,
            has_qk_norm: false,
        }
    }

    /// `mistral_7b`: mistral 7b.
    #[must_use]
    pub const fn mistral_7b() -> Self {
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
            rope_scaling: None,
            use_double_wide_mlp: false,
            num_experts: None,
            top_k_experts: None,
            expert_intermediate_size: None,
            has_qk_norm: false,
        }
    }

    /// `mixtral_8x7b`: mixtral 8x7b.
    #[must_use]
    pub const fn mixtral_8x7b() -> Self {
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
            rope_scaling: None,
            use_double_wide_mlp: false,
            num_experts: Some(8),
            top_k_experts: Some(2),
            expert_intermediate_size: Some(14336),
            has_qk_norm: false,
        }
    }

    /// Build from config json.
    /// # Errors
    ///
    /// Returns `Err` if reading or parsing the source fails.
    pub fn from_config_json(value: &serde_json::Value) -> ConfigResult<Self> {
        let architecture = ARCHITECTURE_REGISTRY
            .detect(value)
            .and_then(|name| Architecture::from_name(&name))
            // Fallback to `Unknown` rather than a guessed architecture
            // (previously this returned `Architecture::Llama`, which silently
            // misclassified unrecognised configs as Llama).
            .unwrap_or(Architecture::Unknown);

        let hidden_size = read_usize(value, "hidden_size", 4096);
        let num_layers = read_usize(value, "num_hidden_layers", 32);
        let num_heads = read_usize(value, "num_attention_heads", 32);
        let num_kv_heads = read_usize_with_alt(
            value,
            &["num_key_value_heads", "num_local_heads"],
            num_heads as u64,
        );
        let head_dim = read_usize_with_alt(value, &["head_dim"], (hidden_size / num_heads) as u64);
        let vocab_size = read_usize(value, "vocab_size", 32_000);
        let intermediate_size = read_usize(value, "intermediate_size", 11_008);
        let rope_theta = read_f32_with_alt(value, &["rope_theta", "rotary_base"], 10_000.0);
        let rms_norm_eps = value
            .get("rms_norm_eps")
            .or_else(|| value.get("layer_norm_eps"))
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(1e-5);
        let sliding_window = read_optional_usize(value, "sliding_window");
        let tie_word_embeddings = value
            .get("tie_word_embeddings")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        let max_position_embeddings = read_usize(value, "max_position_embeddings", 2048);
        let num_experts = read_optional_usize(value, "num_local_experts");
        let top_k_experts =
            read_optional_usize_with_alt(value, &["num_experts_per_tok", "top_k_experts"]);
        let expert_intermediate_size = read_optional_usize(value, "expert_intermediate_size");
        let has_qk_norm = value
            .get("has_qk_norm")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        // Long-context RoPE scaling block (YaRN / Linear / Dynamic / Su).
        // Malformed values silently fall back to None — same conservative
        // pattern used for other optional fields; the warning is logged
        // once the engine attempts to use it (currently unreachable because
        // the Qwen3 path is the only one that consumes the field, and a
        // Qwen3 model with a malformed `rope_scaling` block will already
        // fail to load via the dedicated `Qwen3Config::from_value` path).
        let rope_scaling = value.get("rope_scaling").and_then(|v| {
            serde_json::from_value::<crate::qwen3::config::RopeScaling>(v.clone()).ok()
        });

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
            rope_scaling,
            use_double_wide_mlp: false,
            num_experts,
            top_k_experts,
            expert_intermediate_size,
            has_qk_norm,
        })
    }
}

/// Read a required JSON field as `usize`, falling back to `default` when missing.
fn read_usize(value: &serde_json::Value, field: &str, default: usize) -> usize {
    // invariant: model config dimensions come from JSON and are always small
    // (architectural constants); u64 -> usize truncation is not reachable in practice.
    value
        .get(field)
        .and_then(serde_json::Value::as_u64)
        .map_or(default, |v| usize::try_from(v).unwrap_or(default))
}

/// Read a JSON field as `usize` from the first key present, falling back to `default`.
fn read_usize_with_alt(value: &serde_json::Value, fields: &[&str], default: u64) -> usize {
    // invariant: model config dimensions come from JSON and are always small
    // (architectural constants); u64 -> usize truncation is not reachable in practice.
    for field in fields {
        if let Some(v) = value.get(field).and_then(serde_json::Value::as_u64) {
            return usize::try_from(v).unwrap_or(0);
        }
    }
    usize::try_from(default).unwrap_or(0)
}

/// Read an optional JSON field as `Option<usize>`.
fn read_optional_usize(value: &serde_json::Value, field: &str) -> Option<usize> {
    // invariant: model config dimensions come from JSON and are always small.
    value
        .get(field)
        .and_then(serde_json::Value::as_u64)
        .and_then(|v| usize::try_from(v).ok())
}

/// Read an optional JSON field as `Option<usize>` from the first key present.
fn read_optional_usize_with_alt(value: &serde_json::Value, fields: &[&str]) -> Option<usize> {
    // invariant: model config dimensions come from JSON and are always small.
    for field in fields {
        if let Some(v) = value.get(field).and_then(serde_json::Value::as_u64) {
            return usize::try_from(v).ok();
        }
    }
    None
}

/// Read a JSON field as `f32` from the first key present, falling back to `default`.
// invariant: f64 -> f32 truncation is acceptable for model config floats.
#[allow(clippy::cast_possible_truncation)]
fn read_f32_with_alt(value: &serde_json::Value, fields: &[&str], default: f64) -> f32 {
    for field in fields {
        if let Some(v) = value.get(field).and_then(serde_json::Value::as_f64) {
            return v as f32;
        }
    }
    default as f32
}
