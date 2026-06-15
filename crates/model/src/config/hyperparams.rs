//! Shared hyperparameter accessors for unified and Qwen-specific configs.

use super::ModelConfig;
use crate::qwen3_config::Qwen3Config;

/// Common model dimensions shared by [`ModelConfig`] and [`Qwen3Config`].
pub trait ModelHyperparams {
    fn vocab_size(&self) -> usize;
    fn hidden_size(&self) -> usize;
    fn num_layers(&self) -> usize;
    fn num_heads(&self) -> usize;
    fn num_kv_heads(&self) -> usize;
    fn head_dim(&self) -> usize;
    fn intermediate_size(&self) -> usize;
    fn rope_theta(&self) -> f32;
    fn rms_norm_eps(&self) -> f64;
    fn tie_word_embeddings(&self) -> bool;
    fn max_position_embeddings(&self) -> usize;
}

impl ModelHyperparams for ModelConfig {
    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    fn num_layers(&self) -> usize {
        self.num_layers
    }

    fn num_heads(&self) -> usize {
        self.num_heads
    }

    fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }

    fn head_dim(&self) -> usize {
        self.head_dim
    }

    fn intermediate_size(&self) -> usize {
        self.intermediate_size
    }

    fn rope_theta(&self) -> f32 {
        self.rope_theta
    }

    fn rms_norm_eps(&self) -> f64 {
        self.rms_norm_eps
    }

    fn tie_word_embeddings(&self) -> bool {
        self.tie_word_embeddings
    }

    fn max_position_embeddings(&self) -> usize {
        self.max_position_embeddings
    }
}

impl ModelHyperparams for Qwen3Config {
    fn vocab_size(&self) -> usize {
        Qwen3Config::vocab_size(self)
    }

    fn hidden_size(&self) -> usize {
        Qwen3Config::hidden_size(self)
    }

    fn num_layers(&self) -> usize {
        Qwen3Config::num_hidden_layers(self)
    }

    fn num_heads(&self) -> usize {
        Qwen3Config::num_attention_heads(self)
    }

    fn num_kv_heads(&self) -> usize {
        Qwen3Config::num_key_value_heads(self)
    }

    fn head_dim(&self) -> usize {
        Qwen3Config::head_dim(self)
    }

    fn intermediate_size(&self) -> usize {
        Qwen3Config::intermediate_size(self)
    }

    fn rope_theta(&self) -> f32 {
        Qwen3Config::rope_theta(self)
    }

    fn rms_norm_eps(&self) -> f64 {
        Qwen3Config::rms_norm_eps(self)
    }

    fn tie_word_embeddings(&self) -> bool {
        Qwen3Config::tie_word_embeddings(self)
    }

    fn max_position_embeddings(&self) -> usize {
        Qwen3Config::max_position_embeddings(self)
    }
}

impl From<&ModelConfig> for Qwen3Config {
    fn from(config: &ModelConfig) -> Self {
        Self {
            vocab_size: Some(config.vocab_size),
            hidden_size: Some(config.hidden_size),
            num_hidden_layers: Some(config.num_layers),
            num_attention_heads: Some(config.num_heads),
            num_key_value_heads: Some(config.num_kv_heads),
            head_dim: Some(config.head_dim),
            intermediate_size: Some(config.intermediate_size),
            rope_theta: Some(config.rope_theta),
            max_position_embeddings: Some(config.max_position_embeddings),
            rms_norm_eps: Some(config.rms_norm_eps as f32),
            tie_word_embeddings: Some(config.tie_word_embeddings),
            has_qk_norm: Some(config.has_qk_norm),
            ..Default::default()
        }
    }
}

impl From<&Qwen3Config> for ModelConfig {
    fn from(config: &Qwen3Config) -> Self {
        Self {
            architecture: crate::config::Architecture::Qwen3,
            hidden_size: config.hidden_size(),
            num_layers: config.num_hidden_layers(),
            num_heads: config.num_attention_heads(),
            num_kv_heads: config.num_key_value_heads(),
            head_dim: config.head_dim(),
            vocab_size: config.vocab_size(),
            intermediate_size: config.intermediate_size(),
            rope_theta: config.rope_theta(),
            rms_norm_eps: config.rms_norm_eps(),
            tie_word_embeddings: config.tie_word_embeddings(),
            max_position_embeddings: config.max_position_embeddings(),
            sliding_window: None,
            layer_types: vec![],
            rope_configs: vec![],
            use_double_wide_mlp: false,
            num_experts: None,
            top_k_experts: None,
            expert_intermediate_size: None,
            has_qk_norm: config.has_qk_norm(),
        }
    }
}

impl From<ModelConfig> for Qwen3Config {
    fn from(config: ModelConfig) -> Self {
        Qwen3Config::from(&config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Architecture;

    #[test]
    fn test_model_config_to_qwen3_config() {
        let model = ModelConfig::test_tiny();
        let qwen: Qwen3Config = (&model).into();
        assert_eq!(qwen.vocab_size(), model.vocab_size());
        assert_eq!(qwen.hidden_size(), model.hidden_size());
        assert_eq!(qwen.num_hidden_layers(), model.num_layers);
        assert_eq!(qwen.head_dim(), model.head_dim);
    }

    #[test]
    fn test_hyperparams_trait_parity() {
        let model = ModelConfig::test_tiny_for(Architecture::Qwen3);
        let qwen: Qwen3Config = (&model).into();
        assert_eq!(model.vocab_size(), qwen.vocab_size());
        assert_eq!(model.num_kv_heads(), qwen.num_kv_heads());
        assert!(
            (model.rms_norm_eps() - qwen.rms_norm_eps()).abs() < 1e-6,
            "rms_norm_eps: {} vs {}",
            model.rms_norm_eps(),
            qwen.rms_norm_eps()
        );
    }
}
