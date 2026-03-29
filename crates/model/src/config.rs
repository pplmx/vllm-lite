use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
}

impl Qwen3Config {
    pub fn from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let config: Qwen3Config = serde_json::from_str(&content)?;
        Ok(config)
    }
}
