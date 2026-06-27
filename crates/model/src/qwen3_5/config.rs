//! Qwen3.5 hybrid layer-type configuration.

use crate::qwen3::config::{Qwen3Config, TextConfig};

/// GdnLinearConfig: gdn linear configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GdnLinearConfig {
    pub num_k_heads: usize,
    pub num_v_heads: usize,
    pub key_head_dim: usize,
    pub value_head_dim: usize,
    pub conv_kernel_size: usize,
}

impl GdnLinearConfig {
    /// HF Qwen3.5 production defaults (e.g. Qwen3.5-397B-A17B `text_config`).
    pub fn production_defaults() -> Self {
        Self {
            num_k_heads: 16,
            num_v_heads: 64,
            key_head_dim: 128,
            value_head_dim: 128,
            conv_kernel_size: 4,
        }
    }

    /// Zero-init fallback when no explicit GDN fields are present in config.
    pub fn legacy_heuristic(hidden_size: usize) -> Self {
        let num_v_heads = if hidden_size >= 512 {
            16
        } else {
            (hidden_size / 8).max(4)
        };
        let num_k_heads = (num_v_heads / 2).max(1);
        Self {
            num_k_heads,
            num_v_heads,
            key_head_dim: 16,
            value_head_dim: 16,
            conv_kernel_size: 4,
        }
    }

    /// from_text_config: from text config.
    pub fn from_text_config(tc: &TextConfig) -> Self {
        Self {
            num_k_heads: tc.linear_num_key_heads(),
            num_v_heads: tc.linear_num_value_heads(),
            key_head_dim: tc.linear_key_head_dim(),
            value_head_dim: tc.linear_value_head_dim(),
            conv_kernel_size: tc.linear_conv_kernel_dim(),
        }
    }

    /// from_qwen3_config: from qwen3 config.
    pub fn from_qwen3_config(config: &Qwen3Config) -> Self {
        if let Some(tc) = config.text_config.as_ref() {
            if tc.has_explicit_gdn_config() {
                return Self::from_text_config(tc);
            }
        }
        Self::legacy_heuristic(config.hidden_size())
    }
}

/// LayerType: layer type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    LinearAttention,
    FullAttention,
}

/// parse_layer_types: parse layer types.
pub fn parse_layer_types(config: &Qwen3Config) -> Vec<LayerType> {
    if let Some(types) = config.layer_types() {
        types
            .iter()
            .map(|t| match t.as_str() {
                "linear_attention" => LayerType::LinearAttention,
                "full_attention" => LayerType::FullAttention,
                _ => LayerType::LinearAttention,
            })
            .collect()
    } else {
        let num_layers = config.num_hidden_layers();
        let interval = config.full_attention_interval();
        (0..num_layers)
            .map(|i| {
                if (i + 1) % interval == 0 {
                    LayerType::FullAttention
                } else {
                    LayerType::LinearAttention
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen3::config::Qwen3Config;

    #[test]
    fn test_gdn_linear_config_from_hf_fields() {
        let config = Qwen3Config {
            text_config: Some(TextConfig {
                linear_num_key_heads: Some(16),
                linear_num_value_heads: Some(64),
                linear_key_head_dim: Some(128),
                linear_value_head_dim: Some(128),
                linear_conv_kernel_dim: Some(4),
                ..Default::default()
            }),
            ..Default::default()
        };

        let gdn = GdnLinearConfig::from_qwen3_config(&config);
        assert_eq!(gdn.num_k_heads, 16);
        assert_eq!(gdn.num_v_heads, 64);
        assert_eq!(gdn.key_head_dim, 128);
        assert_eq!(gdn.value_head_dim, 128);
        assert_eq!(gdn.conv_kernel_size, 4);
    }

    #[test]
    fn test_gdn_linear_config_legacy_heuristic() {
        let config = Qwen3Config {
            hidden_size: Some(64),
            ..Default::default()
        };
        let gdn = GdnLinearConfig::from_qwen3_config(&config);
        assert_eq!(gdn.num_v_heads, 8);
        assert_eq!(gdn.num_k_heads, 4);
        assert_eq!(gdn.key_head_dim, 16);
    }

    #[test]
    fn test_full_attention_interval_from_config() {
        let config = Qwen3Config {
            text_config: Some(TextConfig {
                num_hidden_layers: Some(6),
                full_attention_interval: Some(3),
                ..Default::default()
            }),
            ..Default::default()
        };

        let layer_types = parse_layer_types(&config);
        assert_eq!(layer_types.len(), 6);
        assert_eq!(layer_types[2], LayerType::FullAttention);
        assert_eq!(layer_types[5], LayerType::FullAttention);
        assert_eq!(layer_types[0], LayerType::LinearAttention);
    }

    #[test]
    fn test_layer_type_parsing() {
        let config = Qwen3Config {
            text_config: Some(crate::qwen3::config::TextConfig {
                layer_types: Some(vec![
                    "linear_attention".to_string(),
                    "linear_attention".to_string(),
                    "linear_attention".to_string(),
                    "full_attention".to_string(),
                ]),
                ..Default::default()
            }),
            ..Default::default()
        };

        let layer_types = parse_layer_types(&config);
        assert_eq!(layer_types.len(), 4);
        assert_eq!(layer_types[0], LayerType::LinearAttention);
        assert_eq!(layer_types[3], LayerType::FullAttention);
    }

    #[test]
    fn test_layer_type_default_pattern() {
        let config = Qwen3Config {
            text_config: Some(crate::qwen3::config::TextConfig {
                num_hidden_layers: Some(8),
                ..Default::default()
            }),
            ..Default::default()
        };

        let layer_types = parse_layer_types(&config);
        assert_eq!(layer_types.len(), 8);

        for (i, lt) in layer_types.iter().enumerate() {
            let expected = if i % 4 == 3 {
                LayerType::FullAttention
            } else {
                LayerType::LinearAttention
            };
            assert_eq!(*lt, expected, "Layer {} type mismatch", i);
        }
    }

    #[test]
    fn test_layer_type_parsing_mixed() {
        let config = Qwen3Config {
            text_config: Some(crate::qwen3::config::TextConfig {
                num_hidden_layers: Some(8),
                layer_types: Some(vec![
                    "linear_attention".to_string(),
                    "linear_attention".to_string(),
                    "linear_attention".to_string(),
                    "full_attention".to_string(),
                    "linear_attention".to_string(),
                    "linear_attention".to_string(),
                    "linear_attention".to_string(),
                    "full_attention".to_string(),
                ]),
                ..Default::default()
            }),
            ..Default::default()
        };

        let layer_types = parse_layer_types(&config);
        assert_eq!(layer_types.len(), 8);
        assert_eq!(layer_types[0], LayerType::LinearAttention);
        assert_eq!(layer_types[3], LayerType::FullAttention);
        assert_eq!(layer_types[7], LayerType::FullAttention);
    }
}
