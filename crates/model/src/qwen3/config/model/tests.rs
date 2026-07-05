//! Unit tests for the Qwen3 config types (`TextConfig`, `Qwen3Config`,
//! `AttentionType`).
//!
//! Extracted from `model.rs` to keep the implementation file under the
//! project's 800-line soft cap. The tests exercise the real production
//! getters (`vocab_size`, `hidden_size`, `attention_type`, `head_dim`,
//! …) including the `text_config` fallback path and the explicit-overrides
//! path.

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

    assert_eq!(config.vocab_size(), 151_936);
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
