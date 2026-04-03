use super::Architecture;

pub struct ModelConfig {
    pub architecture: Architecture,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub intermediate_size: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f64,
    pub sliding_window: Option<usize>,
    pub tie_word_embeddings: bool,
    pub max_position_embeddings: usize,
}

impl ModelConfig {
    pub fn llama_7b() -> Self {
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
        }
    }

    pub fn mistral_7b() -> Self {
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
        }
    }
}
