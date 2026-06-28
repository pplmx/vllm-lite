//! qwen3_config: qwen3 config.

use crate::config::errors::{ConfigError, ConfigResult};
use serde::Deserialize;

/// RoPE scaling type (HuggingFace-compatible).
///
/// Maps the `rope_type` string field found in HuggingFace
/// `RopeScaling` and `RopeParameters` JSON blobs to a typed enum.
/// Unknown values deserialize to [`RopeType::Other`] for graceful
/// forward compatibility with new HF rope types.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Default, serde::Deserialize, serde::Serialize,
)]
#[serde(rename_all = "lowercase")]
pub enum RopeType {
    /// Default standard RoPE without scaling (Qwen3 baseline).
    #[default]
    Default,
    /// Linear position interpolation.
    Linear,
    /// Dynamic NTK-aware scaling.
    Dynamic,
    /// YaRN (Yet another RoPE extensioN).
    Yarn,
    /// Su RoPE (RoPE in any precision).
    Su,
    /// Other / unknown rope type (serde fallback for forward compat).
    #[serde(other)]
    Other,
}

impl RopeType {
    /// Canonical string representation (matches the HF wire value).
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Linear => "linear",
            Self::Dynamic => "dynamic",
            Self::Yarn => "yarn",
            Self::Su => "su",
            Self::Other => "other",
        }
    }

    /// Parse from a string (case-insensitive). Unknown values map to
    /// [`RopeType::Other`] rather than `None`, so callers can
    /// distinguish "missing" from "unknown".
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "default" => Some(Self::Default),
            "linear" => Some(Self::Linear),
            "dynamic" => Some(Self::Dynamic),
            "yarn" => Some(Self::Yarn),
            "su" => Some(Self::Su),
            _ => Some(Self::Other),
        }
    }
}

impl std::fmt::Display for RopeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// RopeScaling: rope scaling.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct RopeScaling {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rope_type: Option<RopeType>,
    #[serde(default)]
    pub factor: Option<f32>,
    #[serde(default)]
    pub original_max_position_embeddings: Option<usize>,
    #[serde(default)]
    pub attn_factor: Option<f32>,
    #[serde(default)]
    pub partial_rotary_factor: Option<f32>,
    #[serde(default)]
    pub mrope_section: Option<Vec<usize>>,
}

/// RopeParameters: rope parameters.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct RopeParameters {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rope_type: Option<RopeType>,
    #[serde(default)]
    pub rope_theta: Option<f32>,
    #[serde(default)]
    pub partial_rotary_factor: Option<f32>,
    #[serde(default)]
    pub mrope_section: Option<Vec<usize>>,
    #[serde(default)]
    pub mrope_interleaved: Option<bool>,
}

/// TextConfig: text configuration.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct TextConfig {
    #[serde(default)]
    pub vocab_size: Option<usize>,
    #[serde(default)]
    pub hidden_size: Option<usize>,
    #[serde(default)]
    pub num_hidden_layers: Option<usize>,
    #[serde(default)]
    pub num_attention_heads: Option<usize>,
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    #[serde(default)]
    pub intermediate_size: Option<usize>,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default)]
    pub rope_theta: Option<f32>,
    #[serde(default)]
    pub max_position_embeddings: Option<usize>,
    #[serde(default)]
    pub rms_norm_eps: Option<f32>,
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,
    /// Gated DeltaNet (linear attention) head counts — HF `text_config` fields.
    #[serde(default)]
    pub linear_num_key_heads: Option<usize>,
    #[serde(default)]
    pub linear_num_value_heads: Option<usize>,
    #[serde(default)]
    pub linear_key_head_dim: Option<usize>,
    #[serde(default)]
    pub linear_value_head_dim: Option<usize>,
    #[serde(default)]
    pub linear_conv_kernel_dim: Option<usize>,
    /// Every N-th layer is full attention when `layer_types` is absent (default 4).
    #[serde(default)]
    pub full_attention_interval: Option<usize>,
}

/// Qwen3Config: qwen3 configuration.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct Qwen3Config {
    #[serde(default)]
    pub vocab_size: Option<usize>,
    #[serde(default)]
    pub hidden_size: Option<usize>,
    #[serde(default)]
    pub num_hidden_layers: Option<usize>,
    #[serde(default)]
    pub num_attention_heads: Option<usize>,
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    #[serde(default)]
    pub intermediate_size: Option<usize>,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default)]
    pub rope_theta: Option<f32>,
    #[serde(default)]
    pub max_position_embeddings: Option<usize>,
    #[serde(default)]
    pub rms_norm_eps: Option<f32>,
    #[serde(default)]
    pub text_config: Option<TextConfig>,
    #[serde(default)]
    pub q_len: Option<usize>,
    #[serde(default)]
    pub qk_nope_dim: Option<usize>,
    #[serde(default)]
    pub qk_rope_dim: Option<usize>,
    #[serde(default)]
    pub kv_len: Option<usize>,
    #[serde(default)]
    pub tie_word_embeddings: Option<bool>,
    #[serde(default)]
    pub has_qk_norm: Option<bool>,
    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,
    #[serde(default)]
    pub rope_parameters: Option<RopeParameters>,
    #[serde(default)]
    pub head_dim: Option<usize>,
}

impl TextConfig {
    pub fn vocab_size(&self) -> usize {
        self.vocab_size.unwrap_or(151936)
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size.unwrap_or(4096)
    }

    pub fn num_hidden_layers(&self) -> usize {
        self.num_hidden_layers.unwrap_or(32)
    }

    pub fn num_attention_heads(&self) -> usize {
        self.num_attention_heads.unwrap_or(32)
    }

    pub fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(32)
    }

    pub fn intermediate_size(&self) -> usize {
        self.intermediate_size.unwrap_or(11008)
    }

    pub fn rope_theta(&self) -> f32 {
        self.rope_theta.unwrap_or(10000.0)
    }

    pub fn max_position_embeddings(&self) -> usize {
        self.max_position_embeddings.unwrap_or(8192)
    }

    pub fn rms_norm_eps(&self) -> f32 {
        self.rms_norm_eps.unwrap_or(1e-6)
    }

    pub fn layer_types(&self) -> Option<&[String]> {
        self.layer_types.as_deref()
    }

    pub fn linear_num_key_heads(&self) -> usize {
        self.linear_num_key_heads.unwrap_or(16)
    }

    pub fn linear_num_value_heads(&self) -> usize {
        self.linear_num_value_heads.unwrap_or(64)
    }

    pub fn linear_key_head_dim(&self) -> usize {
        self.linear_key_head_dim.unwrap_or(128)
    }

    pub fn linear_value_head_dim(&self) -> usize {
        self.linear_value_head_dim.unwrap_or(128)
    }

    pub fn linear_conv_kernel_dim(&self) -> usize {
        self.linear_conv_kernel_dim.unwrap_or(4)
    }

    pub fn full_attention_interval(&self) -> usize {
        self.full_attention_interval.unwrap_or(4).max(1)
    }

    pub fn has_explicit_gdn_config(&self) -> bool {
        self.linear_num_key_heads.is_some()
            || self.linear_num_value_heads.is_some()
            || self.linear_key_head_dim.is_some()
            || self.linear_value_head_dim.is_some()
            || self.linear_conv_kernel_dim.is_some()
    }
}

impl Qwen3Config {
    pub fn from_file(path: &str) -> ConfigResult<Self> {
        let content = std::fs::read_to_string(path).map_err(|source| ConfigError::Io {
            path: path.to_string(),
            source,
        })?;
        let config: Qwen3Config = serde_json::from_str(&content)?;
        Ok(config)
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
            .or(self.text_config.as_ref().map(|c| c.vocab_size()))
            .unwrap_or(151936)
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
            .or(self.text_config.as_ref().map(|c| c.hidden_size()))
            .unwrap_or(4096)
    }

    pub fn num_hidden_layers(&self) -> usize {
        self.num_hidden_layers
            .or(self.text_config.as_ref().map(|c| c.num_hidden_layers()))
            .unwrap_or(32)
    }

    pub fn num_attention_heads(&self) -> usize {
        self.num_attention_heads
            .or(self.text_config.as_ref().map(|c| c.num_attention_heads()))
            .unwrap_or(32)
    }

    pub fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads
            .or(self.text_config.as_ref().map(|c| c.num_key_value_heads()))
            .unwrap_or(32)
    }

    pub fn intermediate_size(&self) -> usize {
        self.intermediate_size
            .or(self.text_config.as_ref().map(|c| c.intermediate_size()))
            .unwrap_or(11008)
    }

    pub fn rope_theta(&self) -> f32 {
        self.rope_theta
            .or(self.text_config.as_ref().map(|c| c.rope_theta()))
            .unwrap_or(10000.0)
    }

    pub fn max_position_embeddings(&self) -> usize {
        self.max_position_embeddings
            .or(self
                .text_config
                .as_ref()
                .map(|c| c.max_position_embeddings()))
            .unwrap_or(8192)
    }

    pub fn rms_norm_eps(&self) -> f64 {
        self.rms_norm_eps
            .or(self.text_config.as_ref().map(|c| c.rms_norm_eps()))
            .unwrap_or(1e-6) as f64
    }

    pub fn tie_word_embeddings(&self) -> bool {
        self.tie_word_embeddings.unwrap_or(false)
    }

    pub fn has_qk_norm(&self) -> bool {
        self.has_qk_norm.unwrap_or(false)
    }

    pub fn rope_scaling(&self) -> Option<&RopeScaling> {
        self.rope_scaling.as_ref()
    }

    pub fn rope_parameters(&self) -> Option<&RopeParameters> {
        self.rope_parameters.as_ref()
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or_else(|| self.hidden_size() / self.num_attention_heads())
    }

    pub fn layer_types(&self) -> Option<&[String]> {
        self.text_config.as_ref().and_then(|c| c.layer_types())
    }

    pub fn full_attention_interval(&self) -> usize {
        self.text_config
            .as_ref()
            .map(|c| c.full_attention_interval())
            .unwrap_or(4)
    }

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

/// AttentionType: attention type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionType {
    MHA,
    MQA,
    GQA,
    MLA,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qwen3_config_defaults() {
        let config = Qwen3Config {
            vocab_size: None,
            hidden_size: None,
            num_hidden_layers: None,
            num_attention_heads: None,
            num_key_value_heads: None,
            intermediate_size: None,
            sliding_window: None,
            rope_theta: None,
            max_position_embeddings: None,
            rms_norm_eps: None,
            text_config: None,
            q_len: None,
            qk_nope_dim: None,
            qk_rope_dim: None,
            kv_len: None,
            tie_word_embeddings: None,
            has_qk_norm: None,
            rope_scaling: None,
            rope_parameters: None,
            head_dim: None,
        };

        assert_eq!(config.vocab_size(), 151936);
        assert!(!config.tie_word_embeddings());
        assert_eq!(config.hidden_size(), 4096);
        assert_eq!(config.num_hidden_layers(), 32);
        assert_eq!(config.num_attention_heads(), 32);
        assert_eq!(config.intermediate_size(), 11008);
    }

    #[test]
    fn test_qwen3_config_explicit_values() {
        let config = Qwen3Config {
            vocab_size: Some(1000),
            hidden_size: Some(512),
            num_hidden_layers: Some(4),
            num_attention_heads: Some(8),
            num_key_value_heads: Some(2),
            intermediate_size: Some(1024),
            sliding_window: None,
            rope_theta: Some(20000.0),
            max_position_embeddings: Some(4096),
            rms_norm_eps: Some(1e-5),
            text_config: None,
            q_len: None,
            qk_nope_dim: None,
            qk_rope_dim: None,
            kv_len: None,
            tie_word_embeddings: Some(true),
            has_qk_norm: Some(true),
            rope_scaling: None,
            rope_parameters: None,
            head_dim: None,
        };

        assert_eq!(config.vocab_size(), 1000);
        assert_eq!(config.hidden_size(), 512);
        assert!(config.tie_word_embeddings());
        assert_eq!(config.num_hidden_layers(), 4);
        assert_eq!(config.num_attention_heads(), 8);
        assert_eq!(config.num_key_value_heads(), 2);
        assert_eq!(config.intermediate_size(), 1024);
    }

    #[test]
    fn test_text_config_fallback() {
        let text_config = TextConfig {
            vocab_size: Some(500),
            hidden_size: Some(256),
            num_hidden_layers: Some(2),
            num_attention_heads: Some(4),
            num_key_value_heads: Some(4),
            intermediate_size: Some(512),
            sliding_window: None,
            rope_theta: None,
            max_position_embeddings: None,
            rms_norm_eps: None,
            layer_types: None,
            ..Default::default()
        };

        let config = Qwen3Config {
            vocab_size: None,
            hidden_size: None,
            num_hidden_layers: None,
            num_attention_heads: None,
            num_key_value_heads: None,
            intermediate_size: None,
            sliding_window: None,
            rope_theta: None,
            max_position_embeddings: None,
            rms_norm_eps: None,
            rope_parameters: None,
            text_config: Some(text_config),
            q_len: None,
            qk_nope_dim: None,
            qk_rope_dim: None,
            kv_len: None,
            tie_word_embeddings: None,
            has_qk_norm: None,
            rope_scaling: None,
            head_dim: None,
        };

        assert_eq!(config.vocab_size(), 500);
        assert_eq!(config.hidden_size(), 256);
        assert_eq!(config.num_attention_heads(), 4);
    }

    #[test]
    fn test_attention_type_mha() {
        let config = Qwen3Config {
            num_attention_heads: Some(8),
            num_key_value_heads: Some(8),
            ..Default::default()
        };
        assert_eq!(config.attention_type(), AttentionType::MHA);
    }

    #[test]
    fn test_attention_type_mqa() {
        let config = Qwen3Config {
            num_attention_heads: Some(8),
            num_key_value_heads: Some(1),
            ..Default::default()
        };
        assert_eq!(config.attention_type(), AttentionType::MQA);
    }

    #[test]
    fn test_attention_type_gqa() {
        let config = Qwen3Config {
            num_attention_heads: Some(8),
            num_key_value_heads: Some(2),
            ..Default::default()
        };
        assert_eq!(config.attention_type(), AttentionType::GQA);
    }

    #[test]
    fn test_attention_type_mla() {
        let config = Qwen3Config {
            num_attention_heads: Some(8),
            num_key_value_heads: Some(2),
            q_len: Some(4),
            ..Default::default()
        };
        assert_eq!(config.attention_type(), AttentionType::MLA);
    }

    #[test]
    fn test_head_dim_default_computed() {
        // When head_dim not specified, compute from hidden_size / num_attention_heads
        let config = Qwen3Config {
            hidden_size: Some(1024),
            num_attention_heads: Some(16),
            ..Default::default()
        };
        assert_eq!(config.head_dim(), 64); // 1024 / 16
    }

    #[test]
    fn test_head_dim_from_config() {
        // Qwen3-0.6B specifies head_dim=128 explicitly
        let config = Qwen3Config {
            hidden_size: Some(1024),
            num_attention_heads: Some(16),
            head_dim: Some(128),
            ..Default::default()
        };
        assert_eq!(config.head_dim(), 128); // Uses explicit value, not 1024/16=64
    }

    #[test]
    fn rope_type_serde_lowercase() {
        // Known variants serialize as their lowercase HF wire name.
        assert_eq!(
            serde_json::to_string(&RopeType::Default).unwrap(),
            "\"default\""
        );
        assert_eq!(
            serde_json::to_string(&RopeType::Linear).unwrap(),
            "\"linear\""
        );
        assert_eq!(
            serde_json::to_string(&RopeType::Dynamic).unwrap(),
            "\"dynamic\""
        );
        assert_eq!(serde_json::to_string(&RopeType::Yarn).unwrap(), "\"yarn\"");
        assert_eq!(serde_json::to_string(&RopeType::Su).unwrap(), "\"su\"");

        // Known lowercase strings deserialize to the matching variant.
        let parsed: RopeType = serde_json::from_str("\"default\"").unwrap();
        assert_eq!(parsed, RopeType::Default);
        let parsed: RopeType = serde_json::from_str("\"linear\"").unwrap();
        assert_eq!(parsed, RopeType::Linear);
        let parsed: RopeType = serde_json::from_str("\"yarn\"").unwrap();
        assert_eq!(parsed, RopeType::Yarn);

        // Unknown values map to Other (graceful forward compat).
        let other: RopeType = serde_json::from_str("\"future_unknown\"").unwrap();
        assert_eq!(other, RopeType::Other);
    }

    #[test]
    fn rope_type_parse_roundtrip() {
        for kind in [
            RopeType::Default,
            RopeType::Linear,
            RopeType::Dynamic,
            RopeType::Yarn,
            RopeType::Su,
        ] {
            assert_eq!(RopeType::parse(kind.as_str()), Some(kind));
            assert_eq!(RopeType::parse(&kind.to_string()), Some(kind));
        }
    }

    #[test]
    fn rope_type_default_is_default_variant() {
        assert_eq!(RopeType::default(), RopeType::Default);
    }

    #[test]
    fn rope_type_optional_field_default_is_none() {
        let json = "{}";
        let parsed: RopeScaling = serde_json::from_str(json).unwrap();
        assert!(parsed.rope_type.is_none());

        let parsed: RopeParameters = serde_json::from_str(json).unwrap();
        assert!(parsed.rope_type.is_none());
    }

    #[test]
    fn rope_type_optional_field_known_string_deserializes() {
        let json = r#"{"rope_type": "default", "factor": 2.0}"#;
        let parsed: RopeScaling = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.rope_type, Some(RopeType::Default));
        assert_eq!(parsed.factor, Some(2.0));

        let json = r#"{"rope_type": "yarn"}"#;
        let parsed: RopeParameters = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.rope_type, Some(RopeType::Yarn));
    }
}
