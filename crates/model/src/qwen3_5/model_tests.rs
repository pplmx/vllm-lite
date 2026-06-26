use super::Qwen35HybridModel;
use crate::components::SwiGLU;
use crate::components::positional::MRoPE;
use crate::qwen3_5::attention35::Attention35WithRoPE;
use crate::qwen3_config::Qwen3Config;
use candle_core::{DType, Device, Tensor};
use vllm_traits::ModelBackend;
use candle_nn::VarBuilder;

#[test]
fn test_swiglu_forward() {
    let device = Device::Cpu;
    let mlp = SwiGLU::new(128, 512, None).unwrap();

    let x = Tensor::ones((1, 2, 128), DType::F32, &device).unwrap();
    let out = mlp.forward(&x).unwrap();

    assert_eq!(out.dims(), &[1, 2, 128]);
}

#[test]
fn test_swiglu_output_shape_different_intermediate_size() {
    let device = Device::Cpu;
    let mlp = SwiGLU::new(256, 1024, None).unwrap();

    let x = Tensor::ones((1, 3, 256), DType::F32, &device).unwrap();
    let out = mlp.forward(&x).unwrap();

    assert_eq!(out.dims(), &[1, 3, 256]);
}

#[test]
fn test_attention35_rope_preserves_head_dim() {
    let device = Device::Cpu;
    let rope = MRoPE::new(64, 10000.0, vec![21, 21, 22], 0.25);
    let attn = Attention35WithRoPE::new(
        256,
        4,
        4,
        64,
        rope.clone(),
        VarBuilder::zeros(DType::F32, &device),
    )
    .unwrap();

    assert_eq!(attn.head_dim, 64);
    assert_eq!(attn.num_heads, 4);
    assert_eq!(attn.num_kv_heads, 4);
    assert_eq!(rope.dim, 64);
}

#[test]
fn test_qwen35_hybrid_model_kv_cache_init() {
    let config = Qwen3Config {
        text_config: Some(crate::qwen3_config::TextConfig {
            num_hidden_layers: Some(4),
            num_key_value_heads: Some(2),
            ..Default::default()
        }),
        head_dim: Some(64),
        ..Default::default()
    };

    let device = Device::Cpu;
    let model = Qwen35HybridModel::new(config.clone(), device, 16, false).unwrap();

    assert_eq!(model.num_layers(), 4);
}

#[test]
fn test_qwen35_hybrid_model_forward_prefill_and_decode() {
    let config = Qwen3Config {
        text_config: Some(crate::qwen3_config::TextConfig {
            num_hidden_layers: Some(2),
            num_attention_heads: Some(2),
            num_key_value_heads: Some(2),
            hidden_size: Some(64),
            intermediate_size: Some(128),
            layer_types: Some(vec![
                "linear_attention".to_string(),
                "full_attention".to_string(),
            ]),
            ..Default::default()
        }),
        head_dim: Some(32),
        vocab_size: Some(128),
        ..Default::default()
    };

    let device = Device::Cpu;
    let mut model = Qwen35HybridModel::new(config, device, 16, false).unwrap();
    let seq_id = 1u64;
    let tokens = vec![1u32, 2, 3, 4];
    let positions: Vec<usize> = (0..tokens.len()).collect();
    let block_ids = vec![0usize; tokens.len()];

    let (prefill_logits, _) = model
        .forward_with_cache(seq_id, &tokens, 0, &block_ids, &positions, true)
        .unwrap();
    assert_eq!(prefill_logits.dims(), &[1, tokens.len(), 128]);

    let decode_positions = vec![tokens.len()];
    let (decode_logits, _) = model
        .forward_with_cache(seq_id, &tokens, tokens.len(), &[0], &decode_positions, false)
        .unwrap();
    assert_eq!(decode_logits.dims(), &[1, 1, 128]);
}

#[test]
fn test_qwen35_hybrid_model_layer_count() {
    let config = Qwen3Config {
        text_config: Some(crate::qwen3_config::TextConfig {
            num_hidden_layers: Some(12),
            layer_types: Some(vec!["linear_attention".to_string(); 12]),
            ..Default::default()
        }),
        ..Default::default()
    };

    let device = Device::Cpu;
    let model = Qwen35HybridModel::new(config.clone(), device, 8, false).unwrap();

    assert_eq!(model.num_layers(), 12);
}
