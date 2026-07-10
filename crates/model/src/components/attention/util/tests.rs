//! Unit tests for the attention helper functions (`expand_kv`,
//! `causal_mask`, `paged_attention`, `tiled_attention`).
//!
//! Extracted from `util.rs` to keep the implementation file under the
//! project's 800-line soft cap. Exercises:
//!
//! - `paged_attention` output shape (basic + single-token decode)
//! - `tiled_attention` output shape parity vs. `paged_attention`
//! - `tiled_attention` single-tile path (`tile_size > seq_len`)
//! - `expand_kv` GQA expansion (basic, no-expansion, invalid head count,
//!   exact division)
//! - `causal_mask` shape and causal values (0 below diagonal,
//!   -inf above)

use super::*;

const DEVICE: &candle_core::Device = &candle_core::Device::Cpu;

#[test]
fn test_paged_attention_output_shape() {
    let batch_size = 2;
    let seq_len = 4;
    let num_heads = 8;
    let head_dim = 64;

    let q = Tensor::randn(
        0.0f32,
        1.0,
        (batch_size, num_heads, seq_len, head_dim),
        DEVICE,
    )
    .unwrap();
    let k = Tensor::randn(
        0.0f32,
        1.0,
        (batch_size, num_heads, seq_len, head_dim),
        DEVICE,
    )
    .unwrap();
    let v = Tensor::randn(
        0.0f32,
        1.0,
        (batch_size, num_heads, seq_len, head_dim),
        DEVICE,
    )
    .unwrap();

    let output = paged_attention(&q, &k, &v, num_heads, head_dim).unwrap();

    assert_eq!(output.dims(), &[batch_size, seq_len, num_heads * head_dim]);
}

#[test]
fn test_tiled_attention_output_shape_matches_paged_attention() {
    let batch_size = 1;
    let seq_len = 20;
    let num_heads = 4;
    let head_dim = 32;
    let tile_size = 8;

    let q = Tensor::randn(
        0.0f32,
        1.0,
        (batch_size, num_heads, seq_len, head_dim),
        DEVICE,
    )
    .unwrap();
    let k = Tensor::randn(
        0.0f32,
        1.0,
        (batch_size, num_heads, seq_len, head_dim),
        DEVICE,
    )
    .unwrap();
    let v = Tensor::randn(
        0.0f32,
        1.0,
        (batch_size, num_heads, seq_len, head_dim),
        DEVICE,
    )
    .unwrap();

    let paged_output = paged_attention(&q, &k, &v, num_heads, head_dim).unwrap();
    let tiled_output = tiled_attention(&q, &k, &v, num_heads, tile_size).unwrap();

    let expected = [batch_size, seq_len, num_heads * head_dim];
    assert_eq!(
        paged_output.dims(),
        &expected[..],
        "paged_attention output shape mismatch"
    );
    assert_eq!(
        tiled_output.dims(),
        &expected[..],
        "tiled_attention output shape mismatch"
    );
}

#[test]
fn test_tiled_attention_single_tile() {
    let batch_size = 1;
    let seq_len = 8;
    let num_heads = 4;
    let head_dim = 32;
    let tile_size = 16;

    let q = Tensor::ones(
        (batch_size, num_heads, seq_len, head_dim),
        candle_core::DType::F32,
        DEVICE,
    )
    .unwrap();
    let k = Tensor::ones(
        (batch_size, num_heads, seq_len, head_dim),
        candle_core::DType::F32,
        DEVICE,
    )
    .unwrap();
    let v = Tensor::ones(
        (batch_size, num_heads, seq_len, head_dim),
        candle_core::DType::F32,
        DEVICE,
    )
    .unwrap();

    let output = tiled_attention(&q, &k, &v, num_heads, tile_size).unwrap();

    assert_eq!(output.dims(), &[batch_size, seq_len, num_heads * head_dim]);
}

#[test]
fn test_expand_kv_gqa_basic() {
    let batch_size = 1;
    let seq_len = 4;
    let num_kv_heads = 2;
    let num_q_heads = 14;
    let head_dim = 64;

    let kv = Tensor::randn(
        0.0f32,
        1.0,
        (batch_size, seq_len, num_kv_heads, head_dim),
        DEVICE,
    )
    .unwrap();

    let expanded = expand_kv(&kv, num_q_heads, num_kv_heads).unwrap();

    assert_eq!(
        expanded.dims(),
        &[batch_size, seq_len, num_q_heads, head_dim]
    );
}

#[test]
fn test_expand_kv_no_expansion_needed() {
    let batch_size = 1;
    let seq_len = 4;
    let num_heads = 8;
    let head_dim = 64;

    let kv = Tensor::randn(
        0.0f32,
        1.0,
        (batch_size, seq_len, num_heads, head_dim),
        DEVICE,
    )
    .unwrap();

    let expanded = expand_kv(&kv, num_heads, num_heads).unwrap();

    assert_eq!(expanded.dims(), kv.dims());
}

#[test]
fn test_expand_kv_invalid_head_count() {
    let batch_size = 1;
    let seq_len = 4;
    let wrong_kv_heads = 4;
    let expected_kv_heads = 2;
    let num_q_heads = 14;
    let head_dim = 64;

    let kv = Tensor::randn(
        0.0f32,
        1.0,
        (batch_size, seq_len, wrong_kv_heads, head_dim),
        DEVICE,
    )
    .unwrap();

    let result = expand_kv(&kv, num_q_heads, expected_kv_heads);
    assert!(result.is_err());
}

#[test]
fn test_causal_mask_shape() {
    let seq_len = 16;
    let mask = causal_mask(seq_len, DEVICE).unwrap();

    assert_eq!(mask.dims(), &[1, 1, seq_len, seq_len]);
}

#[test]
fn test_causal_mask_causality() {
    let seq_len = 4;
    let mask = causal_mask(seq_len, DEVICE).unwrap();
    let mask_data: Vec<f32> = mask.flatten_all().unwrap().to_vec1().unwrap();

    for i in 0..seq_len {
        for j in 0..seq_len {
            let idx = i * seq_len + j;
            if j <= i {
                assert!(
                    mask_data[idx].abs() < 1e-6,
                    "Position ({i}, {j}) should be 0"
                );
            } else {
                assert!(
                    mask_data[idx] == f32::NEG_INFINITY
                        || mask_data[idx].is_infinite() && mask_data[idx] < 0.0,
                    "Position ({i}, {j}) should be -inf, got {}",
                    mask_data[idx]
                );
            }
        }
    }
}

#[test]
fn test_expand_kv_exact_division() {
    let batch_size = 2;
    let seq_len = 4;
    let num_kv_heads = 2;
    let num_q_heads = 16;
    let head_dim = 64;

    let kv = Tensor::randn(
        0.0f32,
        1.0,
        (batch_size, seq_len, num_kv_heads, head_dim),
        DEVICE,
    )
    .unwrap();

    let expanded = expand_kv(&kv, num_q_heads, num_kv_heads).unwrap();

    assert_eq!(
        expanded.dims(),
        &[batch_size, seq_len, num_q_heads, head_dim]
    );
}

#[test]
fn test_paged_attention_single_token_decode() {
    let batch_size = 1;
    let seq_q = 1;
    let seq_kv = 8;
    let num_heads = 16;
    let head_dim = 128;

    let q = Tensor::randn(
        0.0f32,
        1.0,
        (batch_size, num_heads, seq_q, head_dim),
        DEVICE,
    )
    .unwrap();
    let k = Tensor::randn(
        0.0f32,
        1.0,
        (batch_size, num_heads, seq_kv, head_dim),
        DEVICE,
    )
    .unwrap();
    let v = Tensor::randn(
        0.0f32,
        1.0,
        (batch_size, num_heads, seq_kv, head_dim),
        DEVICE,
    )
    .unwrap();

    let output = paged_attention(&q, &k, &v, num_heads, head_dim).unwrap();

    assert_eq!(output.dims(), &[batch_size, seq_q, num_heads * head_dim]);
}
