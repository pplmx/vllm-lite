//! Qwen3 configuration unit tests (no checkpoint required).

use vllm_model::qwen3::config::Qwen3Config;

#[test]
fn test_qwen3_config_default() {
    let config = Qwen3Config::default();
    assert_eq!(config.vocab_size(), 151936);
    assert_eq!(config.hidden_size(), 4096);
    assert_eq!(config.num_hidden_layers(), 32);
}

#[test]
fn test_qwen3_config_builder() {
    let config = Qwen3Config {
        vocab_size: Some(1000),
        hidden_size: Some(512),
        num_hidden_layers: Some(2),
        num_attention_heads: Some(8),
        num_key_value_heads: Some(2),
        intermediate_size: Some(1024),
        ..Default::default()
    };

    assert_eq!(config.vocab_size(), 1000);
    assert_eq!(config.hidden_size(), 512);
    assert_eq!(config.num_hidden_layers(), 2);
    assert_eq!(config.intermediate_size(), 1024);
}

#[test]
fn test_qwen3_config_text_config_fallback() {
    let text_config = vllm_model::qwen3::config::TextConfig {
        vocab_size: Some(5000),
        hidden_size: Some(256),
        num_hidden_layers: Some(4),
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
        text_config: Some(text_config),
        ..Default::default()
    };

    assert_eq!(config.vocab_size(), 5000);
    assert_eq!(config.hidden_size(), 256);
    assert_eq!(config.num_hidden_layers(), 4);
}
