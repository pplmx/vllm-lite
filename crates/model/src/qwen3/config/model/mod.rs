//! Qwen3 model config: hidden size, head counts, `RoPE` theta, sliding-window, QK-norm, MLA latent dim.
//!
//! Deserialized from `config.json`. The same struct serves Qwen3-`0.6B/4B/8B/32B`;
//! the field set is the union so smaller models just leave optionals empty.
//!
//! Tests for these types live in `tests.rs` (sibling file) to keep this
//! module under the 800-line soft cap.

// crates/model/src/qwen3/config/model/mod.rs
//
// Model-specific configuration for Qwen3:
// - `TextConfig` struct (the text-only sub-config nested under
//   `text_config` for multimodal / hybrid variants)
// - `Qwen3Config` struct (the top-level Qwen3 model config)
// - `AttentionType` enum (MHA / MQA / GQA / MLA classifier)
//
// Accessor impls are split into:
// - `text_config.rs` — `impl TextConfig { ... }` (defaults + getters)
// - `qwen3_config.rs` — `impl Qwen3Config { ... }` (`from_file` +
//   defaults with text_config fallback + `attention_type` classifier)

use super::rope::{RopeParameters, RopeScaling};
use serde::Deserialize;

mod qwen3_config;
mod text_config;

/// Configuration for Text. Constructed via the `builder()` associated function or by deserializing from JSON / TOML. Pass-by-value to construction APIs.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct TextConfig {
    /// Tokeniser vocab size. Defaults to `151_936` (Qwen3).
    #[serde(default)]
    pub vocab_size: Option<usize>,
    /// Dimensionality of residual stream. Defaults to 4096.
    #[serde(default)]
    pub hidden_size: Option<usize>,
    /// Number of decoder layers stacked in the transformer.
    #[serde(default)]
    pub num_hidden_layers: Option<usize>,
    /// Number of query attention heads per layer.
    #[serde(default)]
    pub num_attention_heads: Option<usize>,
    /// Number of key/value attention heads (GQA grouping denominator).
    /// Smaller than `num_attention_heads` for grouped-query variants.
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    /// MLP hidden dimension (gate/up/down projections).
    #[serde(default)]
    pub intermediate_size: Option<usize>,
    /// Sliding-window attention span; `None` means full causal.
    #[serde(default)]
    pub sliding_window: Option<usize>,
    /// `RoPE` base wavelength. Defaults to `10_000` (extended for long-context models).
    #[serde(default)]
    pub rope_theta: Option<f32>,
    /// Maximum sequence length the model was trained on.
    #[serde(default)]
    pub max_position_embeddings: Option<usize>,
    /// Epsilon for `RMSNorm` numerical stability.
    #[serde(default)]
    pub rms_norm_eps: Option<f32>,
    /// Per-layer type strings (e.g. `["full_attention", "sliding_attention"]`).
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,
    /// Gated `DeltaNet` (linear attention) head counts — HF `text_config` fields.
    #[serde(default)]
    pub linear_num_key_heads: Option<usize>,
    #[serde(default)]
    pub linear_num_value_heads: Option<usize>,
    #[serde(default)]
    pub linear_key_head_dim: Option<usize>,
    #[serde(default)]
    pub linear_value_head_dim: Option<usize>,
    /// Convolution kernel size preceding the SSM recurrence.
    #[serde(default)]
    pub linear_conv_kernel_dim: Option<usize>,
    /// Every N-th layer is full attention when `layer_types` is absent (default 4).
    #[serde(default)]
    pub full_attention_interval: Option<usize>,
}

/// Configuration for Qwen3. Constructed via the `builder()` associated function or by deserializing from JSON / TOML. Pass-by-value to construction APIs.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct Qwen3Config {
    /// Tokeniser vocab size at the top level (overrides `text_config.vocab_size` when set).
    #[serde(default)]
    pub vocab_size: Option<usize>,
    /// Residual stream dimensionality.
    #[serde(default)]
    pub hidden_size: Option<usize>,
    /// Number of decoder layers.
    #[serde(default)]
    pub num_hidden_layers: Option<usize>,
    /// Query attention heads per layer.
    #[serde(default)]
    pub num_attention_heads: Option<usize>,
    /// Key/value attention heads (GQA grouping).
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    /// MLP hidden dimension.
    #[serde(default)]
    pub intermediate_size: Option<usize>,
    /// Sliding-window attention span (`None` = full causal).
    #[serde(default)]
    pub sliding_window: Option<usize>,
    /// `RoPE` base wavelength.
    #[serde(default)]
    pub rope_theta: Option<f32>,
    /// Maximum sequence length.
    #[serde(default)]
    pub max_position_embeddings: Option<usize>,
    /// `RMSNorm` epsilon.
    #[serde(default)]
    pub rms_norm_eps: Option<f32>,
    /// Nested text-model config (Qwen3 multimodal / hybrid variants).
    #[serde(default)]
    pub text_config: Option<TextConfig>,
    /// Query length dimension for MLA latent projection.
    #[serde(default)]
    pub q_len: Option<usize>,
    /// Non-RoPE portion of the QK head dimension (MLA).
    #[serde(default)]
    pub qk_nope_dim: Option<usize>,
    /// `RoPE` portion of the QK head dimension (MLA).
    #[serde(default)]
    pub qk_rope_dim: Option<usize>,
    /// KV length dimension for MLA.
    #[serde(default)]
    pub kv_len: Option<usize>,
    /// Whether the LM head shares weights with the embedding table.
    #[serde(default)]
    pub tie_word_embeddings: Option<bool>,
    /// Whether Q and K projections carry a per-head `RMSNorm` (`Qwen3`-style).
    #[serde(default)]
    pub has_qk_norm: Option<bool>,
    /// Legacy `RoPE` scaling block (NTK-aware, dynamic, etc.).
    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,
    /// Modern `RoPE` parameters block (HF format).
    #[serde(default)]
    pub rope_parameters: Option<RopeParameters>,
    /// Explicit per-head dimension override; derived from `hidden_size/num_attention_heads` if absent.
    #[serde(default)]
    pub head_dim: Option<usize>,
}

/// `AttentionType`. See the type definition for fields and behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionType {
    MHA,
    MQA,
    GQA,
    MLA,
}

#[cfg(test)]
mod tests;
