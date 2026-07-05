//! Unit tests for the `FlashAttentionV3` family (`FlashAttentionV3`,
//! `MqaFlashAttention`, `GqaFlashAttention`).
//!
//! Extracted from `flash_attention_v3.rs` to keep the implementation file
//! under the project's 800-line soft cap. Exercises the production forward
//! paths across basic / causal / sliding-window variants, plus
//! determinism and causal-vs-full regression checks.

use super::*;

const DEVICE: &candle_core::Device = &candle_core::Device::Cpu;

#[test]
fn test_flash_attention_v3_basic() {
    let batch_size = 1;
    let seq_len = 4;
    let num_heads = 4;
    let head_dim = 32;

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

    let flash = FlashAttentionV3::new(FlashAttentionV3Config {
        num_heads,
        head_dim,
        dropout_p: 0.0,
        causal: false,
        window_size: None,
    });

    let output = flash.forward(&q, &k, &v).unwrap();
    assert_eq!(output.dims(), q.dims());
}

#[test]
fn test_flash_attention_v3_causal() {
    let batch_size = 1;
    let seq_len = 4;
    let num_heads = 4;
    let head_dim = 32;

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

    let flash = FlashAttentionV3::new(FlashAttentionV3Config {
        num_heads,
        head_dim,
        dropout_p: 0.0,
        causal: true,
        window_size: None,
    });

    let output = flash.forward(&q, &k, &v).unwrap();
    assert_eq!(output.dims(), q.dims());
}

#[test]
fn test_flash_attention_v3_with_sliding_window() {
    let batch_size = 1;
    let seq_len = 16;
    let num_heads = 4;
    let head_dim = 32;

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

    let flash = FlashAttentionV3::new(FlashAttentionV3Config {
        num_heads,
        head_dim,
        dropout_p: 0.0,
        causal: false,
        window_size: Some((8, 8)),
    });

    let output = flash.forward_with_swa(&q, &k, &v).unwrap();
    assert_eq!(output.dims(), q.dims());
}

#[test]
fn test_mqa_flash_attention() {
    let batch_size = 1;
    let seq_len = 8;
    let num_heads = 16;
    let num_kv_heads = 1;
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
        (batch_size, num_kv_heads, seq_len, head_dim),
        DEVICE,
    )
    .unwrap();
    let v = Tensor::randn(
        0.0f32,
        1.0,
        (batch_size, num_kv_heads, seq_len, head_dim),
        DEVICE,
    )
    .unwrap();

    let mqa = MqaFlashAttention::new(num_heads, num_kv_heads, head_dim, true);
    let output = mqa.forward(&q, &k, &v).unwrap();

    assert_eq!(output.dims(), &[batch_size, num_heads, seq_len, head_dim]);
}

#[test]
fn test_gqa_flash_attention() {
    let batch_size = 1;
    let seq_len = 8;
    let num_heads = 16;
    let num_kv_heads = 4;
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
        (batch_size, num_kv_heads, seq_len, head_dim),
        DEVICE,
    )
    .unwrap();
    let v = Tensor::randn(
        0.0f32,
        1.0,
        (batch_size, num_kv_heads, seq_len, head_dim),
        DEVICE,
    )
    .unwrap();

    let gqa = GqaFlashAttention::new(num_heads, num_kv_heads, head_dim, true);
    let output = gqa.forward(&q, &k, &v).unwrap();

    assert_eq!(output.dims(), &[batch_size, num_heads, seq_len, head_dim]);
}

#[test]
fn test_gqa_flash_attention_non_divisible() {
    let batch_size = 1;
    let seq_len = 8;
    let num_heads = 14;
    let num_kv_heads = 7;
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
        (batch_size, num_kv_heads, seq_len, head_dim),
        DEVICE,
    )
    .unwrap();
    let v = Tensor::randn(
        0.0f32,
        1.0,
        (batch_size, num_kv_heads, seq_len, head_dim),
        DEVICE,
    )
    .unwrap();

    let gqa = GqaFlashAttention::new(num_heads, num_kv_heads, head_dim, false);
    let output = gqa.forward(&q, &k, &v).unwrap();

    assert_eq!(output.dims(), &[batch_size, num_heads, seq_len, head_dim]);
}

#[test]
fn test_flash_attention_v3_output_finite() {
    let batch_size = 2;
    let seq_len = 16;
    let num_heads = 8;
    let head_dim = 64;

    let q = Tensor::randn(
        -2.0f32,
        2.0,
        (batch_size, num_heads, seq_len, head_dim),
        DEVICE,
    )
    .unwrap();
    let k = Tensor::randn(
        -2.0f32,
        2.0,
        (batch_size, num_heads, seq_len, head_dim),
        DEVICE,
    )
    .unwrap();
    let v = Tensor::randn(
        -2.0f32,
        2.0,
        (batch_size, num_heads, seq_len, head_dim),
        DEVICE,
    )
    .unwrap();

    let flash = FlashAttentionV3::new(FlashAttentionV3Config {
        num_heads,
        head_dim,
        dropout_p: 0.0,
        causal: true,
        window_size: Some((8, 0)),
    });

    let output = flash.forward(&q, &k, &v).unwrap();
    let data: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();
    assert!(data.iter().all(|v| v.is_finite()));
}

#[test]
fn test_gqa_flash_attention_causal_changes_output() {
    let batch_size = 1;
    let seq_len = 6;
    let num_heads = 4;
    let num_kv_heads = 2;
    let head_dim = 16;

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
        (batch_size, num_kv_heads, seq_len, head_dim),
        DEVICE,
    )
    .unwrap();
    let v = Tensor::randn(
        0.0f32,
        1.0,
        (batch_size, num_kv_heads, seq_len, head_dim),
        DEVICE,
    )
    .unwrap();

    let causal = GqaFlashAttention::new(num_heads, num_kv_heads, head_dim, true);
    let full = GqaFlashAttention::new(num_heads, num_kv_heads, head_dim, false);

    let causal_out = causal.forward(&q, &k, &v).unwrap();
    let full_out = full.forward(&q, &k, &v).unwrap();

    let diff = (&causal_out - &full_out).unwrap().abs().unwrap();
    let max_diff: f32 = diff.max_all().unwrap().to_scalar().unwrap();
    assert!(
        max_diff > 1e-6,
        "causal GQA flash attention should differ from unmasked, max_diff={max_diff}"
    );
}

#[test]
fn test_mqa_flash_attention_deterministic() {
    let batch_size = 1;
    let seq_len = 8;
    let num_heads = 8;
    let num_kv_heads = 1;
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
        (batch_size, num_kv_heads, seq_len, head_dim),
        DEVICE,
    )
    .unwrap();
    let v = Tensor::randn(
        0.0f32,
        1.0,
        (batch_size, num_kv_heads, seq_len, head_dim),
        DEVICE,
    )
    .unwrap();

    let mqa = MqaFlashAttention::new(num_heads, num_kv_heads, head_dim, false);

    let out1 = mqa.forward(&q, &k, &v).unwrap();
    let out2 = mqa.forward(&q, &k, &v).unwrap();

    let diff = (&out1 - &out2).unwrap().abs().unwrap();
    let max_diff: f32 = diff
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap()
        .iter()
        .copied()
        .fold(0.0f32, f32::max);
    assert!(max_diff < 1e-6);
}
