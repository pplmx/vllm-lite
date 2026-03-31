use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
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
}

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
}

impl Qwen3Config {
    pub fn from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
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
        };

        assert_eq!(config.vocab_size(), 151936);
        assert_eq!(config.tie_word_embeddings(), false);
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
        };

        assert_eq!(config.vocab_size(), 1000);
        assert_eq!(config.hidden_size(), 512);
        assert_eq!(config.tie_word_embeddings(), true);
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
            text_config: Some(text_config),
            q_len: None,
            qk_nope_dim: None,
            qk_rope_dim: None,
            kv_len: None,
            tie_word_embeddings: None,
            has_qk_norm: None,
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
}
