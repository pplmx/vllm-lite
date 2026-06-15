//! Qwen3.5 hybrid layer-type configuration.

use crate::qwen3_config::Qwen3Config;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    LinearAttention,
    FullAttention,
}

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
        (0..num_layers)
            .map(|i| {
                if i % 4 == 3 {
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
    use crate::qwen3_config::Qwen3Config;

    #[test]
    fn test_layer_type_parsing() {
        let config = Qwen3Config {
            text_config: Some(crate::qwen3_config::TextConfig {
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
            text_config: Some(crate::qwen3_config::TextConfig {
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
            text_config: Some(crate::qwen3_config::TextConfig {
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
