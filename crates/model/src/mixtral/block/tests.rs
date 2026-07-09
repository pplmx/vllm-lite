//! Unit tests for `MixtralBlock`.
//!
//! One end-to-end test that drives the block through a prefill pass
//! followed by a decode step on the same `PagedKvCache`. Locks in
//! the shape contract:
//!
//! - prefill input `[1, 4, 64]` → output `[1, 4, 64]`
//! - decode input `[1, 64]` → output `[1, 64]`
use super::*;
use crate::config::{Architecture, ModelConfig};
use candle_core::DType;

fn tiny_config() -> ModelConfig {
    ModelConfig {
        architecture: Architecture::Mixtral,
        hidden_size: 64,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 16,
        vocab_size: 128,
        intermediate_size: 128,
        rope_theta: 10000.0,
        rms_norm_eps: 1e-5,
        sliding_window: Some(4096),
        tie_word_embeddings: false,
        max_position_embeddings: 512,
        layer_types: vec![],
        rope_configs: vec![],
        use_double_wide_mlp: false,
        num_experts: Some(4),
        top_k_experts: Some(2),
        expert_intermediate_size: Some(128),
        has_qk_norm: false,
    }
}

#[test]
fn test_mixtral_block_prefill_then_decode() {
    let config = tiny_config();
    let device = candle_core::Device::Cpu;
    let block = MixtralBlock::new(&config, 0).unwrap();
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
