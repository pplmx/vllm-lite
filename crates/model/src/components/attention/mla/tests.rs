//! Unit tests for `MlaAttention` (Multi-Latent Attention).
//!
//! Extracted from `mla.rs` to keep the implementation file under the
//! project's 800-line soft cap. The tests cover construction, accessor
//! round-trips, intermediate-projection shape checks, `RoPE` application,
//! and forward-pass determinism.

use super::*;

#[test]
fn test_mla_attention_new_creation() {
    let attn = MlaAttention::new(
        2048, // hidden_size
        16,   // num_heads
        16,   // num_kv_heads
        512,  // q_lora_rank
        512,  // kv_lora_rank
        128,  // qk_nope_dim
        64,   // qk_rope_dim
        128,  // v_head_dim
        None, // vb
        AttentionConfig::default(),
    )
    .unwrap();

    assert_eq!(attn.num_heads(), 16);
    assert_eq!(attn.kv_lora_rank(), 512);
}

#[test]
fn test_mla_attention_accessors() {
    let attn = MlaAttention::new(
        2048,
        16,
        16,
        512,
        512,
        128,
        64,
        128,
        None,
        AttentionConfig::default(),
    )
    .unwrap();

    assert_eq!(attn.head_dim(), 128 + 64); // qk_nope_dim + qk_rope_dim
    assert_eq!(attn.num_kv_heads(), 16);
    assert_eq!(attn.q_lora_rank(), 512);
}

#[test]
fn test_mla_q_projection_shape() {
    let attn = MlaAttention::new(
        2048,
        16,
        16,
        512,
        512,
        128,
        64,
        128,
        None,
        AttentionConfig::default(),
    )
    .unwrap();

    let x = Tensor::randn(0.0f32, 1.0, (1, 4, 2048), &candle_core::Device::Cpu).unwrap();
    let q_compressed = attn.q_proj_test().forward(&x).unwrap();

    assert_eq!(q_compressed.dims(), &[1, 4, 512]); // [batch, seq, q_lora_rank]
}

#[test]
fn test_mla_split_q_shape() {
    let q_lora_rank = 16 * (128 + 64);
    let attn = MlaAttention::new(
        2048,
        16,
        16,
        q_lora_rank,
        512,
        128,
        64,
        128,
        None,
        AttentionConfig::default(),
    )
    .unwrap();

    let x = Tensor::randn(0.0f32, 1.0, (1, 4, 2048), &candle_core::Device::Cpu).unwrap();
    let q_compressed = attn.q_proj_test().forward(&x).unwrap();

    let (q_nope, q_rope) = attn.split_q(&q_compressed, 4).unwrap();

    // q_nope: [batch, seq, num_heads * qk_nope_dim] = [1, 4, 16 * 128]
    assert_eq!(q_nope.dims(), &[1, 4, 2048]);
    // q_rope: [batch, seq, num_heads * qk_rope_dim] = [1, 4, 16 * 64]
    assert_eq!(q_rope.dims(), &[1, 4, 1024]);
}

#[test]
fn test_mla_kv_compression_shape() {
    let attn = MlaAttention::new(
        2048,
        16,
        16,
        3072,
        512,
        128,
        64,
        128,
        None,
        AttentionConfig::default(),
    )
    .unwrap();

    let x = Tensor::randn(0.0f32, 1.0, (1, 4, 2048), &candle_core::Device::Cpu).unwrap();
    let kv_compressed = attn.kv_proj_test().forward(&x).unwrap();

    assert_eq!(kv_compressed.dims(), &[1, 4, 512]); // [batch, seq, kv_lora_rank]
}

#[test]
fn test_mla_k_decompression_shape() {
    let attn = MlaAttention::new(
        2048,
        16,
        16,
        3072,
        512,
        128,
        64,
        128,
        None,
        AttentionConfig::default(),
    )
    .unwrap();

    let kv_compressed = Tensor::randn(0.0f32, 1.0, (1, 4, 512), &candle_core::Device::Cpu).unwrap();
    let k_decompressed = attn.k_decompress_test().forward(&kv_compressed).unwrap();

    // [batch, seq, num_kv_heads * v_head_dim] = [1, 4, 16 * 128]
    assert_eq!(k_decompressed.dims(), &[1, 4, 2048]);
}

#[test]
fn test_mla_v_decompression_shape() {
    let attn = MlaAttention::new(
        2048,
        16,
        16,
        3072,
        512,
        128,
        64,
        128,
        None,
        AttentionConfig::default(),
    )
    .unwrap();

    let kv_compressed = Tensor::randn(0.0f32, 1.0, (1, 4, 512), &candle_core::Device::Cpu).unwrap();
    let v_decompressed = attn.v_decompress_test().forward(&kv_compressed).unwrap();

    // [batch, seq, num_kv_heads * v_head_dim] = [1, 4, 16 * 128]
    assert_eq!(v_decompressed.dims(), &[1, 4, 2048]);
}

#[test]
fn test_mla_concat_q_nope_rope_shape() {
    let attn = MlaAttention::new(
        2048,
        16,
        16,
        3072,
        512,
        128,
        64,
        128,
        None,
        AttentionConfig::default(),
    )
    .unwrap();

    let batch_size = 1;
    let seq_len = 4;

    let q_nope = Tensor::randn(
        0.0f32,
        1.0,
        (batch_size, seq_len, 16 * 128),
        &candle_core::Device::Cpu,
    )
    .unwrap();
    let q_rope = Tensor::randn(
        0.0f32,
        1.0,
        (batch_size, seq_len, 16 * 64),
        &candle_core::Device::Cpu,
    )
    .unwrap();

    let q = attn.concat_q_nope_rope(&q_nope, &q_rope).unwrap();

    // Q: [batch, num_heads, seq, head_dim] = [1, 16, 4, 192]
    assert_eq!(q.dims(), &[1, 16, 4, 192]);
}

#[test]
fn test_mla_rope_application() {
    use crate::components::positional::rope::{apply_rope_with_scaling, RopeScalingContext};

    let batch_size = 1;
    let seq_len = 4;
    let num_heads = 16;
    let rope_dim = 64;

    let q_rope = Tensor::randn(
        0.0f32,
        1.0,
        (batch_size, seq_len, num_heads, rope_dim),
        &candle_core::Device::Cpu,
    )
    .unwrap();
    let positions: Vec<i64> = vec![0, 1, 2, 3];

    let q_rope_rotated =
        apply_rope_with_scaling(&q_rope, &positions, 10000.0, RopeScalingContext::default())
            .unwrap();

    assert_eq!(q_rope_rotated.dims(), q_rope.dims());

    let diff = (&q_rope_rotated - &q_rope).unwrap().abs().unwrap();
    let sum_diff: f32 = diff.sum_all().unwrap().to_scalar().unwrap();
    assert!(sum_diff > 1e-5, "RoPE should modify the tensor");
}

#[test]
fn test_mla_reshape_kv() {
    let attn = MlaAttention::new(
        2048,
        16,
        16,
        3072,
        512,
        128,
        64,
        128,
        None,
        AttentionConfig::default(),
    )
    .unwrap();

    let batch_size = 1;
    let seq_len = 4;
    let kv_compressed = Tensor::randn(
        0.0f32,
        1.0,
        (batch_size, seq_len, 512),
        &candle_core::Device::Cpu,
    )
    .unwrap();

    let k_decompressed = attn.k_decompress_test().forward(&kv_compressed).unwrap();
    let k = attn
        .reshape_k(&k_decompressed, batch_size, seq_len)
        .unwrap();

    // K: [batch, num_kv_heads, seq, v_head_dim] = [1, 16, 4, 128]
    assert_eq!(k.dims(), &[1, 16, 4, 128]);

    let v_decompressed = attn.v_decompress_test().forward(&kv_compressed).unwrap();
    let v = attn
        .reshape_v(&v_decompressed, batch_size, seq_len)
        .unwrap();

    // V: [batch, num_kv_heads, seq, v_head_dim] = [1, 16, 4, 128]
    assert_eq!(v.dims(), &[1, 16, 4, 128]);
}

#[test]
fn test_mla_forward_output_shape() {
    let qk_nope_dim = 128;
    let qk_rope_dim = 64;
    let v_head_dim = qk_nope_dim;
    let q_lora_rank = 16 * (qk_nope_dim + qk_rope_dim);
    let attn = MlaAttention::new(
        2048,
        16,
        16,
        q_lora_rank,
        512,
        qk_nope_dim,
        qk_rope_dim,
        v_head_dim,
        None,
        AttentionConfig::default(),
    )
    .unwrap();

    let x = Tensor::randn(0.0f32, 1.0, (1, 4, 2048), &candle_core::Device::Cpu).unwrap();
    let positions: Vec<i64> = vec![0, 1, 2, 3];

    let output = attn.forward(&x, &positions).unwrap();

    assert_eq!(output.dims(), &[1, 4, 2048]);
}

#[test]
fn test_mla_forward_decode_mode() {
    let qk_nope_dim = 128;
    let qk_rope_dim = 64;
    let v_head_dim = qk_nope_dim;
    let q_lora_rank = 16 * (qk_nope_dim + qk_rope_dim);
    let attn = MlaAttention::new(
        2048,
        16,
        16,
        q_lora_rank,
        512,
        qk_nope_dim,
        qk_rope_dim,
        v_head_dim,
        None,
        AttentionConfig::default(),
    )
    .unwrap();

    let x = Tensor::randn(0.0f32, 1.0, (1, 1, 2048), &candle_core::Device::Cpu).unwrap();
    let positions: Vec<i64> = vec![100];

    let output = attn.forward(&x, &positions).unwrap();
    assert_eq!(output.dims(), &[1, 1, 2048]);
}

#[test]
fn test_mla_output_finite() {
    let qk_nope_dim = 128;
    let qk_rope_dim = 64;
    let v_head_dim = qk_nope_dim;
    let q_lora_rank = 16 * (qk_nope_dim + qk_rope_dim);
    let attn = MlaAttention::new(
        2048,
        16,
        16,
        q_lora_rank,
        512,
        qk_nope_dim,
        qk_rope_dim,
        v_head_dim,
        None,
        AttentionConfig::default(),
    )
    .unwrap();

    let x = Tensor::randn(-2.0f32, 2.0, (1, 4, 2048), &candle_core::Device::Cpu).unwrap();
    let positions: Vec<i64> = vec![0, 1, 2, 3];

    let output = attn.forward(&x, &positions).unwrap();
    let data: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();
    assert!(data.iter().all(|v: &f32| v.is_finite()));
}

#[test]
fn test_mla_deterministic() {
    let qk_nope_dim = 128;
    let qk_rope_dim = 64;
    let v_head_dim = qk_nope_dim;
    let q_lora_rank = 16 * (qk_nope_dim + qk_rope_dim);
    let attn = MlaAttention::new(
        2048,
        16,
        16,
        q_lora_rank,
        512,
        qk_nope_dim,
        qk_rope_dim,
        v_head_dim,
        None,
        AttentionConfig::default(),
    )
    .unwrap();

    let x = Tensor::randn(0.0f32, 1.0, (1, 4, 2048), &candle_core::Device::Cpu).unwrap();
    let positions: Vec<i64> = vec![0, 1, 2, 3];

    let out1 = attn.forward(&x, &positions).unwrap();
    let out2 = attn.forward(&x, &positions).unwrap();

    let diff = (&out1 - &out2).unwrap().abs().unwrap();
    let max_diff: f32 = diff
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap()
        .iter()
        .copied()
        .fold(0.0f32, f32::max);
    assert!(max_diff < 1e-5);
}

#[test]
fn test_mla_rope_affects_different_positions() {
    use crate::components::positional::rope::{apply_rope_with_scaling, RopeScalingContext};

    let batch_size = 1;
    let seq_len = 2;
    let num_heads = 16;
    let rope_dim = 64;

    let q_rope = Tensor::randn(
        0.0f32,
        1.0,
        (batch_size, seq_len, num_heads, rope_dim),
        &candle_core::Device::Cpu,
    )
    .unwrap();

    let pos1: Vec<i64> = vec![0, 1];
    let pos2: Vec<i64> = vec![10, 11];

    let q_rope_rotated1 = apply_rope_with_scaling(
        &q_rope,
        &pos1,
        10000.0,
        RopeScalingContext::default(),
    )
    .unwrap();
    let q_rope_rotated2 = apply_rope_with_scaling(
        &q_rope,
        &pos2,
        10000.0,
        RopeScalingContext::default(),
    )
    .unwrap();

    let diff = (&q_rope_rotated1 - &q_rope_rotated2)
        .unwrap()
        .abs()
        .unwrap();
    let sum_diff: f32 = diff.sum_all().unwrap().to_scalar().unwrap();
    assert!(
        sum_diff > 1e-5,
        "RoPE should produce different outputs for different positions"
    );
}
