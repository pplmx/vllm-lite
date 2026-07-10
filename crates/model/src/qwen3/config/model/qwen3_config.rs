// crates/model/src/qwen3/config/model/qwen3_config.rs
//
// Accessor impls for `Qwen3Config`. Each getter falls back to the nested
// `TextConfig` if the top-level field is `None`, then to the canonical
// Qwen3 default. Also includes `from_file` (JSON deserialization) and
// `attention_type` (MHA / MQA / GQA / MLA classifier).

use super::{AttentionType, Qwen3Config, TextConfig};
use crate::config::errors::{ConfigError, ConfigResult};

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
    pub const fn rope_scaling(&self) -> Option<&super::super::rope::RopeScaling> {
        self.rope_scaling.as_ref()
    }

    #[must_use]
    pub const fn rope_parameters(&self) -> Option<&super::super::rope::RopeParameters> {
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
