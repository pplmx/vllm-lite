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

/// Regression (RIL ISS-010 / TASK-010): the GQA flash-path `expand_kv`
/// (layout `[batch, heads, seq, dim]`) must produce BLOCKED grouping along
/// the head axis (axis 1), not tiled. Pre-fix it used `Tensor::repeat`,
/// pairing query head `h` with the wrong KV head `h % num_kv_heads`.
#[test]
fn test_flash_gqa_expand_kv_blocked_grouping() {
    // KV [batch, heads, seq, dim] = [1, 2, 1, 1]; head 0 = 0.0, head 1 = 1.0.
    let kv = Tensor::from_vec(vec![0.0f32, 1.0], (1, 2, 1, 1), DEVICE).unwrap();
    let gqa = GqaFlashAttention::new(8, 2, 1, true);
    let expanded = gqa.expand_kv(&kv, 8).unwrap();
    assert_eq!(expanded.dims(), &[1, 8, 1, 1]);
    let vals: Vec<f32> = expanded.flatten_all().unwrap().to_vec1().unwrap();
    assert_eq!(
        vals,
        vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        "flash GQA expand_kv must be blocked [K0x4, K1x4]; got {vals:?} (RIL ISS-010)"
    );
}

/// Regression (RIL ISS-010 / TASK-010): the MQA flash-path `expand_kv`
/// must also be blocked. MQA = single KV head shared by all query heads.
#[test]
fn test_flash_mqa_expand_kv_blocked_grouping() {
    // KV [1, 1, 1, 1]; the lone KV head = 0.5. All 4 query heads must get it.
    let kv = Tensor::from_vec(vec![0.5f32], (1, 1, 1, 1), DEVICE).unwrap();
    let mqa = MqaFlashAttention::new(4, 1, 1, true);
    let expanded = mqa.expand_kv(&kv, 4).unwrap();
    assert_eq!(expanded.dims(), &[1, 4, 1, 1]);
    let vals: Vec<f32> = expanded.flatten_all().unwrap().to_vec1().unwrap();
    assert_eq!(
        vals,
        vec![0.5, 0.5, 0.5, 0.5],
        "flash MQA expand_kv must replicate the single KV head to all query heads; got {vals:?}"
    );
}
