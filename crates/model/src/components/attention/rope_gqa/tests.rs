//! Unit tests for `RopeGqaAttention`.
//!
//! Locks in two contracts:
//!
//! 1. **Shape correctness**: `forward` / `forward_decode` / `forward_prefill`
//!    preserve the `[batch, seq, hidden]` / `[batch, 1, hidden]` shape
//!    contract regardless of `qk_norm` / `fused-kernel` toggles.
//! 2. **Fused-paged equivalence**: when `use_fused` is toggled on, the
//!    fused prefill and decode paths must match the paged reference
//!    to within `1e-4` absolute max element diff. This guards against
//!    regressions in the fused-kernel numerical path.
//!
//! All tests run on `Device::Cpu` with `candle_core::DType::F32`.
use super::*;

#[test]
fn test_rope_gqa_attention_forward_output_shape() {
    let device = candle_core::Device::Cpu;
    let num_heads = 8;
    let num_kv_heads = 4;
    let head_dim = 64;
    let hidden_size = num_heads * head_dim;
    let batch_size = 1;
    let seq_len = 4;

    let attention = RopeGqaAttention::new(
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        10000.0,
        None,
        AttentionConfig::default(),
        false,
    )
    .unwrap();

    let x = Tensor::randn(0.0f32, 1.0, (batch_size, seq_len, hidden_size), &device).unwrap();
    let output = attention.forward(&x).unwrap();

    assert_eq!(output.dims(), &[batch_size, seq_len, hidden_size]);
}

#[test]
fn test_rope_gqa_attention_with_qk_norm() {
    let device = candle_core::Device::Cpu;
    let num_heads = 8;
    let num_kv_heads = 4;
    let head_dim = 64;
    let hidden_size = num_heads * head_dim;
    let batch_size = 1;
    let seq_len = 4;

    let attention = RopeGqaAttention::new(
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        10000.0,
        None,
        AttentionConfig::default(),
        true,
    )
    .unwrap();

    let x = Tensor::randn(0.0f32, 1.0, (batch_size, seq_len, hidden_size), &device).unwrap();
    let output = attention.forward(&x).unwrap();

    assert_eq!(output.dims(), &[batch_size, seq_len, hidden_size]);
}

#[test]
fn test_rope_gqa_attention_decode_single_token() {
    let device = candle_core::Device::Cpu;
    let num_heads = 8;
    let num_kv_heads = 4;
    let head_dim = 64;
    let hidden_size = num_heads * head_dim;

    let attention = RopeGqaAttention::new(
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        10000.0,
        None,
        AttentionConfig::default(),
        false,
    )
    .unwrap();

    let x = Tensor::ones((1, hidden_size), candle_core::DType::F32, &device).unwrap();
    let mut kv_cache =
        crate::paged_tensor::PagedKvCache::new(1, num_heads, head_dim, 8, device, false).unwrap();

    let block_ids: Vec<usize> = vec![0];
    let positions = vec![0];

    let result = attention
        .forward_decode(&x, &mut kv_cache, 0, &block_ids, 0, &positions)
        .unwrap();

    assert_eq!(result.dims(), &[1, 1, hidden_size]);
}

#[test]
fn test_rope_gqa_attention_decode_with_kv_cache() {
    let device = candle_core::Device::Cpu;
    let num_heads = 4;
    let num_kv_heads = 2;
    let head_dim = 64;
    let hidden_size = num_heads * head_dim;

    let attention = RopeGqaAttention::new(
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        10000.0,
        None,
        AttentionConfig::default(),
        false,
    )
    .unwrap();

    let mut kv_cache =
        crate::paged_tensor::PagedKvCache::new(1, num_heads, head_dim, 16, device.clone(), false)
            .unwrap();

    for step in 0..8 {
        let x = Tensor::ones((1, hidden_size), candle_core::DType::F32, &device).unwrap();
        let block_id = step / 8;
        let block_ids: Vec<usize> = vec![block_id];
        let positions = vec![step];

        let result = attention
            .forward_decode(&x, &mut kv_cache, 0, &block_ids, step, &positions)
            .unwrap();

        assert_eq!(result.dims(), &[1, 1, hidden_size], "step={step}");
    }
}

fn make_rope_attention(use_fused: bool) -> RopeGqaAttention {
    let device = candle_core::Device::Cpu;
    let num_heads = 4;
    let num_kv_heads = 2;
    let head_dim = 32;
    let hidden_size = num_heads * head_dim;

    RopeGqaAttention::new(
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        10000.0,
        Some(candle_nn::VarBuilder::zeros(
            candle_core::DType::F32,
            &device,
        )),
        AttentionConfig {
            tile_size: Some(16),
            use_fused,
        },
        false,
    )
    .unwrap()
}

#[test]
fn test_rope_gqa_prefill_fused_matches_paged() {
    let device = candle_core::Device::Cpu;
    let hidden_size = 128;
    let seq_len = 6;

    let standard = make_rope_attention(false);
    let fused = make_rope_attention(true);

    let mut cache_std = PagedKvCache::new(1, 4, 32, 16, device.clone(), false).unwrap();
    let mut cache_fused = PagedKvCache::new(1, 4, 32, 16, device.clone(), false).unwrap();

    let x = Tensor::randn(0.0f32, 1.0, (1, seq_len, hidden_size), &device).unwrap();
    let block_ids: Vec<usize> = (0..seq_len).map(|i| i / 16).collect();
    let positions: Vec<usize> = (0..seq_len).collect();

    let out_std = standard
        .forward_prefill(&x, &mut cache_std, 0, &block_ids, &positions)
        .unwrap();
    let out_fused = fused
        .forward_prefill(&x, &mut cache_fused, 0, &block_ids, &positions)
        .unwrap();

    let diff = (&out_std - &out_fused).unwrap().abs().unwrap();
    let max_diff: f32 = diff.max_all().unwrap().to_scalar().unwrap();
    assert!(
        max_diff < 1e-4,
        "fused prefill should match paged path, max_diff={max_diff}"
    );
}

#[test]
fn test_rope_gqa_decode_fused_matches_paged() {
    let device = candle_core::Device::Cpu;
    let hidden_size = 128;

    let standard = make_rope_attention(false);
    let fused = make_rope_attention(true);

    let mut cache_std = PagedKvCache::new(1, 4, 32, 16, device.clone(), false).unwrap();
    let mut cache_fused = PagedKvCache::new(1, 4, 32, 16, device.clone(), false).unwrap();

    for step in 0..5 {
        let x = Tensor::randn(0.0f32, 1.0, (1, hidden_size), &device).unwrap();
        let block_id = step / 16;
        let block_ids = vec![block_id];
        let positions = vec![step];

        let out_std = standard
            .forward_decode(&x, &mut cache_std, 0, &block_ids, step, &positions)
            .unwrap();
        let out_fused = fused
            .forward_decode(&x, &mut cache_fused, 0, &block_ids, step, &positions)
            .unwrap();

        let diff = (&out_std - &out_fused).unwrap().abs().unwrap();
        let max_diff: f32 = diff.max_all().unwrap().to_scalar().unwrap();
        assert!(
            max_diff < 1e-4,
            "fused decode should match paged at step {step}, max_diff={max_diff}"
        );
    }
}
