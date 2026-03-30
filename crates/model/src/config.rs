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

#[derive(Debug, Clone, Deserialize)]
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
        self.rms_norm_eps.unwrap_or(1e-6) as f32
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
            .or(self.text_config.as_ref().map(|c| c.rms_norm_eps() as f32))
            .unwrap_or(1e-6) as f64
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
