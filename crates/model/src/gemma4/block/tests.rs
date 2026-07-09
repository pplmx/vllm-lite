//! Unit tests for `Gemma4Block`.
//!
//! One end-to-end test that drives the block through a prefill pass
//! followed by a decode step on the same `PagedKvCache`. Locks in
//! the shape contract:
//!
//! - prefill input `[1, 4, 64]` → output `[1, 4, 64]`
//! - decode input `[1, 64]` → output `[1, 64]`
//!
//! The block uses the Gemma4 sliding-attention path (per
//! `tiny_config.layer_types`); the test stays on the same
//! shared cache to validate that decode continues from the
//! prefill's KV writes.
use super::*;
use crate::config::{Architecture, LayerType, ModelConfig, RoPEConfig};
use candle_core::DType;

fn tiny_config() -> ModelConfig {
    ModelConfig {
        architecture: Architecture::Gemma4,
        hidden_size: 64,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 16,
        vocab_size: 128,
        intermediate_size: 128,
        rope_theta: 10000.0,
        rms_norm_eps: 1e-6,
        sliding_window: Some(512),
        tie_word_embeddings: true,
        max_position_embeddings: 512,
        layer_types: vec![LayerType::SlidingAttention],
        rope_configs: vec![RoPEConfig {
            rope_theta: 10000.0,
            partial_rotary_factor: 1.0,
        }],
        use_double_wide_mlp: false,
        num_experts: None,
        top_k_experts: None,
        expert_intermediate_size: None,
        has_qk_norm: false,
    }
}

#[test]
fn test_gemma4_block_prefill_then_decode() {
    let config = tiny_config();
    let device = candle_core::Device::Cpu;
    let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
    let block = Gemma4Block::new(&config, 0, vb).unwrap();
    let mut kv_cache = PagedKvCache::new(1, 4, 16, 8, device.clone(), false).unwrap();

    let x = Tensor::ones((1, 4, 64), DType::F32, &device).unwrap();
    let block_ids: Vec<usize> = (0..4).map(|i| i / 16).collect();
    let positions: Vec<usize> = (0..4).collect();
    let out = block
        .forward_prefill(&x, &mut kv_cache, 0, &block_ids, &positions)
        .unwrap();
    assert_eq!(out.dims(), &[1, 4, 64]);

    let decode_x = Tensor::ones((1, 64), DType::F32, &device).unwrap();
    let decode_out = block
        .forward_decode(&decode_x, &mut kv_cache, 0, &[0], 4, &[4])
        .unwrap();
    assert_eq!(decode_out.dims(), &[1, 64]);
}
