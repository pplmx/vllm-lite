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
    let block_ids = vec![0usize];
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

#[test]
fn test_mixtral_block_prefill_continue_matches_full_prefill() {
    let config = tiny_config();
    let device = candle_core::Device::Cpu;
    let block = MixtralBlock::new(&config, 0).unwrap();

    let seq_len = 6usize;
    let x = Tensor::ones((1, seq_len, 64), DType::F32, &device).unwrap();
    let block_ids = vec![0usize];
    let positions: Vec<usize> = (0..seq_len).collect();

    let mut full_cache = PagedKvCache::new(1, 4, 16, 8, device.clone(), false).unwrap();
    let full_out = block
        .forward_prefill(&x, &mut full_cache, 0, &block_ids, &positions)
        .unwrap();

    let mut chunked_cache = PagedKvCache::new(1, 4, 16, 8, device.clone(), false).unwrap();
    let chunk1 = x.narrow(1, 0, 3).unwrap();
    let pos1: Vec<usize> = (0..3).collect();
    block
        .forward_prefill(&chunk1, &mut chunked_cache, 0, &block_ids, &pos1)
        .unwrap();

    let chunk2 = x.narrow(1, 3, 3).unwrap();
    let pos2: Vec<usize> = (3..6).collect();
    let cont_out = block
        .forward_prefill_continue(&chunk2, &mut chunked_cache, 0, &block_ids, &pos2, 3)
        .unwrap();

    let full_last = full_out.narrow(1, seq_len - 1, 1).unwrap();
    let cont_last = cont_out.narrow(1, 2, 1).unwrap();
    let diff = (full_last - cont_last)
        .unwrap()
        .abs()
        .unwrap()
        .sum_all()
        .unwrap()
        .to_vec0::<f32>()
        .unwrap();
    assert!(diff < 1e-4, "chunked continuation diverged: diff={diff}");
}
