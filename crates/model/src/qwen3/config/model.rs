//! Qwen3 model config: hidden size, head counts, `RoPE` theta, sliding-window, QK-norm, MLA latent dim.
//!
//! Deserialized from `config.json`. The same struct serves Qwen3-`0.6B/4B/8B/32B`;
//! the field set is the union so smaller models just leave optionals empty.
//!
//! Tests for these types live in `tests.rs` (sibling file) to keep this
//! module under the 800-line soft cap.

// crates/model/src/qwen3/config/model.rs
//
// Model-specific configuration for Qwen3:
// `TextConfig`, `Qwen3Config`, `AttentionType`.

use super::rope::{RopeParameters, RopeScaling};
use crate::config::errors::{ConfigError, ConfigResult};
use serde::Deserialize;

#[cfg(test)]
mod tests;

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

impl TextConfig {
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.vocab_size.unwrap_or(151_936)
    }

    #[must_use]
    pub fn hidden_size(&self) -> usize {
        self.hidden_size.unwrap_or(4096)
    }

    #[must_use]
    pub fn num_hidden_layers(&self) -> usize {
        self.num_hidden_layers.unwrap_or(32)
    }

    #[must_use]
    pub fn num_attention_heads(&self) -> usize {
        self.num_attention_heads.unwrap_or(32)
    }

    #[must_use]
    pub fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(32)
    }

    #[must_use]
    pub fn intermediate_size(&self) -> usize {
        self.intermediate_size.unwrap_or(11008)
    }

    #[must_use]
    pub fn rope_theta(&self) -> f32 {
        self.rope_theta.unwrap_or(10000.0)
    }

    #[must_use]
    pub fn max_position_embeddings(&self) -> usize {
        self.max_position_embeddings.unwrap_or(8192)
    }

    #[must_use]
    pub fn rms_norm_eps(&self) -> f32 {
        self.rms_norm_eps.unwrap_or(1e-6)
    }

    #[must_use]
    pub fn layer_types(&self) -> Option<&[String]> {
        self.layer_types.as_deref()
    }

    #[must_use]
    pub fn linear_num_key_heads(&self) -> usize {
        self.linear_num_key_heads.unwrap_or(16)
    }

    #[must_use]
    pub fn linear_num_value_heads(&self) -> usize {
        self.linear_num_value_heads.unwrap_or(64)
    }

    #[must_use]
    pub fn linear_key_head_dim(&self) -> usize {
        self.linear_key_head_dim.unwrap_or(128)
    }

    #[must_use]
    pub fn linear_value_head_dim(&self) -> usize {
        self.linear_value_head_dim.unwrap_or(128)
    }

    #[must_use]
    pub fn linear_conv_kernel_dim(&self) -> usize {
        self.linear_conv_kernel_dim.unwrap_or(4)
    }

    #[must_use]
    pub fn full_attention_interval(&self) -> usize {
        self.full_attention_interval.unwrap_or(4).max(1)
    }

    #[must_use]
    pub const fn has_explicit_gdn_config(&self) -> bool {
        self.linear_num_key_heads.is_some()
            || self.linear_num_value_heads.is_some()
            || self.linear_key_head_dim.is_some()
            || self.linear_value_head_dim.is_some()
            || self.linear_conv_kernel_dim.is_some()
    }
}

impl Qwen3Config {
    /// Construct a tokenizer from a tokenizer.json file.
    /// # Errors
    ///
    /// Returns `Err` if reading or parsing the source fails.
    pub fn from_file(path: &str) -> ConfigResult<Self> {
        let content = std::fs::read_to_string(path).map_err(|source| ConfigError::Io {
            path: path.to_string(),
            source,
        })?;
        let config: Self = serde_json::from_str(&content)?;
        Ok(config)
    }

    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
            .or_else(|| self.text_config.as_ref().map(TextConfig::vocab_size))
            .unwrap_or(151_936)
    }

    #[must_use]
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
            .or_else(|| self.text_config.as_ref().map(TextConfig::hidden_size))
            .unwrap_or(4096)
    }

    #[must_use]
    pub fn num_hidden_layers(&self) -> usize {
        self.num_hidden_layers
            .or_else(|| self.text_config.as_ref().map(TextConfig::num_hidden_layers))
            .unwrap_or(32)
    }

    #[must_use]
    pub fn num_attention_heads(&self) -> usize {
        self.num_attention_heads
            .or_else(|| {
                self.text_config
                    .as_ref()
                    .map(TextConfig::num_attention_heads)
            })
            .unwrap_or(32)
    }

    #[must_use]
    pub fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads
            .or_else(|| {
                self.text_config
                    .as_ref()
                    .map(TextConfig::num_key_value_heads)
            })
            .unwrap_or(32)
    }

    #[must_use]
    pub fn intermediate_size(&self) -> usize {
        self.intermediate_size
            .or_else(|| self.text_config.as_ref().map(TextConfig::intermediate_size))
            .unwrap_or(11008)
    }

    #[must_use]
    pub fn rope_theta(&self) -> f32 {
        self.rope_theta
            .or_else(|| self.text_config.as_ref().map(TextConfig::rope_theta))
            .unwrap_or(10000.0)
    }

    #[must_use]
    pub fn max_position_embeddings(&self) -> usize {
        self.max_position_embeddings
            .or_else(|| {
                self.text_config
                    .as_ref()
                    .map(TextConfig::max_position_embeddings)
            })
            .unwrap_or(8192)
    }

    #[must_use]
    pub fn rms_norm_eps(&self) -> f64 {
        f64::from(
            self.rms_norm_eps
                .or_else(|| self.text_config.as_ref().map(TextConfig::rms_norm_eps))
                .unwrap_or(1e-6),
        )
    }

    #[must_use]
    pub fn tie_word_embeddings(&self) -> bool {
        self.tie_word_embeddings.unwrap_or(false)
    }

    #[must_use]
    pub fn has_qk_norm(&self) -> bool {
        self.has_qk_norm.unwrap_or(false)
    }

    #[must_use]
    pub const fn rope_scaling(&self) -> Option<&RopeScaling> {
        self.rope_scaling.as_ref()
    }

    #[must_use]
    pub const fn rope_parameters(&self) -> Option<&RopeParameters> {
        self.rope_parameters.as_ref()
    }

    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or_else(|| self.hidden_size() / self.num_attention_heads())
    }

    #[must_use]
    pub fn layer_types(&self) -> Option<&[String]> {
        self.text_config.as_ref().and_then(|c| c.layer_types())
    }

    #[must_use]
    pub fn full_attention_interval(&self) -> usize {
        self.text_config
            .as_ref()
            .map_or(4, TextConfig::full_attention_interval)
    }

    #[must_use]
    pub fn attention_type(&self) -> AttentionType {
        if self.q_len.is_some() || self.kv_len.is_some() {
            AttentionType::MLA
        } else if self.num_key_value_heads() == 1 {
            AttentionType::MQA
        } else if self.num_attention_heads() == self.num_key_value_heads() {
            AttentionType::MHA
        } else {
            AttentionType::GQA
        }
    }
}
