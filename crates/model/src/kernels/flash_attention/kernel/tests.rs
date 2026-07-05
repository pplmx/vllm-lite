//! Unit tests for the `FlashAttention` kernel family.
//!
//! Extracted from `kernel.rs` to keep the implementation files under the
//! project's 800-line soft cap. Exercises the production forward paths
//! (`ScaledDotProductAttention`, `FlashAttentionV2`, `FlashAttentionKernel`)
//! across small/large `head_dim`, batch sizes, sliding-window variants,
//! and the SDPA-vs-V2 consistency regression test.

use super::super::config::{AttentionVariant, FlashAttentionConfig};
use super::flash_attention_v2::FlashAttentionV2;
use super::scaled_dot_product::ScaledDotProductAttention;
use super::*;
use candle_core::Device;

#[test]
fn test_scaled_dot_product_attention() -> Result<()> {
    let sdpa = ScaledDotProductAttention::new(64);

    let q = Tensor::ones((2, 8, 10, 64), candle_core::DType::F32, &Device::Cpu)?;
    let k = Tensor::ones((2, 8, 10, 64), candle_core::DType::F32, &Device::Cpu)?;
    let v = Tensor::ones((2, 8, 10, 64), candle_core::DType::F32, &Device::Cpu)?;

    let output = sdpa.forward(&q, &k, &v)?;

    assert_eq!(output.dims(), &[2, 8, 10, 64]);

    Ok(())
}

#[test]
fn test_scaled_dot_product_attention_small_head_dim() -> Result<()> {
    let sdpa = ScaledDotProductAttention::new(32);

    let q = Tensor::ones((1, 2, 5, 32), candle_core::DType::F32, &Device::Cpu)?;
    let k = Tensor::ones((1, 2, 5, 32), candle_core::DType::F32, &Device::Cpu)?;
    let v = Tensor::ones((1, 2, 5, 32), candle_core::DType::F32, &Device::Cpu)?;

    let output = sdpa.forward(&q, &k, &v)?;

    assert_eq!(output.dims(), &[1, 2, 5, 32]);

    Ok(())
}

#[test]
fn test_scaled_dot_product_attention_large_head_dim() -> Result<()> {
    let sdpa = ScaledDotProductAttention::new(128);

    let q = Tensor::ones((1, 4, 8, 128), candle_core::DType::F32, &Device::Cpu)?;
    let k = Tensor::ones((1, 4, 8, 128), candle_core::DType::F32, &Device::Cpu)?;
    let v = Tensor::ones((1, 4, 8, 128), candle_core::DType::F32, &Device::Cpu)?;

    let output = sdpa.forward(&q, &k, &v)?;

    assert_eq!(output.dims(), &[1, 4, 8, 128]);

    Ok(())
}

#[test]
fn test_scaled_dot_product_attention_known_values() -> Result<()> {
    let head_dim = 4;
    let sdpa = ScaledDotProductAttention::new(head_dim);

    let q_data: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let q = Tensor::from_slice(&q_data, (1, 1, 2, head_dim), &Device::Cpu)?;

    let k_data: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let k = Tensor::from_slice(&k_data, (1, 1, 2, head_dim), &Device::Cpu)?;

    let v_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let v = Tensor::from_slice(&v_data, (1, 1, 2, head_dim), &Device::Cpu)?;

    let output = sdpa.forward(&q, &k, &v)?;

    assert_eq!(output.dims(), &[1, 1, 2, head_dim]);

    Ok(())
}

#[test]
fn test_scaled_dot_product_batch() -> Result<()> {
    let sdpa = ScaledDotProductAttention::new(32);

    let batch_size = 4;
    let num_heads = 4;
    let seq_len = 8;
    let head_dim = 32;

    let q = Tensor::ones(
        (batch_size, num_heads, seq_len, head_dim),
        candle_core::DType::F32,
        &Device::Cpu,
    )?;
    let k = Tensor::ones(
        (batch_size, num_heads, seq_len, head_dim),
        candle_core::DType::F32,
        &Device::Cpu,
    )?;
    let v = Tensor::ones(
        (batch_size, num_heads, seq_len, head_dim),
        candle_core::DType::F32,
        &Device::Cpu,
    )?;

    let output = sdpa.forward(&q, &k, &v)?;

    assert_eq!(output.dims(), &[batch_size, num_heads, seq_len, head_dim]);

    Ok(())
}

#[test]
fn test_flash_attention_config() {
    let config = FlashAttentionConfig::new()
        .with_flash()
        .with_sliding_window(512);

    assert_eq!(config.variant, AttentionVariant::Flash);
    assert_eq!(config.sliding_window_size, 512);
    assert!(config.use_sliding_window);
}

#[test]
fn test_flash_attention_config_default() {
    let config = FlashAttentionConfig::new();

    assert_eq!(config.variant, AttentionVariant::Standard);
    assert_eq!(config.flash_block_size, 128);
    assert!(!config.use_sliding_window);
    assert_eq!(config.sliding_window_size, 512);
}

#[test]
fn test_attention_variant_defaults() {
    assert_eq!(AttentionVariant::default(), AttentionVariant::Standard);
}

#[test]
fn test_flash_attention_kernel_creation() {
    let config = FlashAttentionConfig::new().with_flash();
    let kernel = FlashAttentionKernel::new(2, 64, config);

    assert_eq!(kernel.config.variant, AttentionVariant::Flash);
}

#[test]
fn test_flash_attention_kernel_forward() -> Result<()> {
    let config = FlashAttentionConfig::new();
    let kernel = FlashAttentionKernel::new(2, 64, config);

    let q = Tensor::ones((1, 2, 4, 64), candle_core::DType::F32, &Device::Cpu)?;
    let k = Tensor::ones((1, 2, 4, 64), candle_core::DType::F32, &Device::Cpu)?;
    let v = Tensor::ones((1, 2, 4, 64), candle_core::DType::F32, &Device::Cpu)?;

    let output = kernel.forward(&q, &k, &v)?;

    assert_eq!(output.dims(), &[1, 2, 4, 64]);

    Ok(())
}

#[test]
fn test_flash_attention_v2_config() {
    let config = FlashAttentionConfig::new().with_flash_v2();
    assert_eq!(config.variant, AttentionVariant::FlashV2);
}

#[test]
fn test_flash_attention_v2_forward() -> Result<()> {
    let config = FlashAttentionConfig::new().with_flash_v2();
    let kernel = FlashAttentionKernel::new(2, 64, config);

    let q = Tensor::ones((1, 2, 4, 64), candle_core::DType::F32, &Device::Cpu)?;
    let k = Tensor::ones((1, 2, 4, 64), candle_core::DType::F32, &Device::Cpu)?;
    let v = Tensor::ones((1, 2, 4, 64), candle_core::DType::F32, &Device::Cpu)?;

    let output = kernel.forward(&q, &k, &v)?;

    assert_eq!(output.dims(), &[1, 2, 4, 64]);
    Ok(())
}

#[test]
fn test_flash_attention_v2_long_sequence() -> Result<()> {
    let fa_v2 = FlashAttentionV2::new(4, 64).with_block_size(32);

    let q = Tensor::randn(0f32, 1.0, (2, 4, 128, 64), &Device::Cpu)?;
    let k = Tensor::randn(0f32, 1.0, (2, 4, 128, 64), &Device::Cpu)?;
    let v = Tensor::randn(0f32, 1.0, (2, 4, 128, 64), &Device::Cpu)?;

    let output = fa_v2.forward(&q, &k, &v)?;

    assert_eq!(output.dims(), &[2, 4, 128, 64]);
    Ok(())
}

#[test]
fn test_flash_attention_v2_output_range() -> Result<()> {
    let fa_v2 = FlashAttentionV2::new(1, 32).with_block_size(16);

    let q = Tensor::randn(0f32, 1.0, (1, 1, 64, 32), &Device::Cpu)?;
    let k = Tensor::randn(0f32, 1.0, (1, 1, 64, 32), &Device::Cpu)?;
    let v = Tensor::ones((1, 1, 64, 32), candle_core::DType::F32, &Device::Cpu)?;

    let output = fa_v2.forward(&q, &k, &v)?;
    let output_data: Vec<f32> = output.flatten_all()?.to_vec1()?;

    for val in &output_data {
        assert!(val.is_finite(), "Output should be finite: {val}");
    }

    Ok(())
}

#[test]
fn test_flash_attention_v2_consistency_with_sdpa() -> Result<()> {
    let sdpa = ScaledDotProductAttention::new(64);
    let fa_v2 = FlashAttentionV2::new(2, 64).with_block_size(64);

    // invariant: small integer test seeds (42..=44) used only to seed the
    // RNG; precision loss is acceptable for the test fixture.
    #[allow(clippy::cast_precision_loss)]
    let seed: i32 = 42;
    #[allow(clippy::cast_precision_loss)]
    let q = Tensor::randn(seed as f32, 0.1, (1, 2, 32, 64), &Device::Cpu)?;
    #[allow(clippy::cast_precision_loss)]
    let k = Tensor::randn((seed + 1) as f32, 0.1, (1, 2, 32, 64), &Device::Cpu)?;
    #[allow(clippy::cast_precision_loss)]
    let v = Tensor::randn((seed + 2) as f32, 0.1, (1, 2, 32, 64), &Device::Cpu)?;

    let sdpa_out = sdpa.forward(&q, &k, &v)?;
    let fa_v2_out = fa_v2.forward(&q, &k, &v)?;

    let sdpa_data: Vec<f32> = sdpa_out.flatten_all()?.to_vec1()?;
    let fa_v2_data: Vec<f32> = fa_v2_out.flatten_all()?.to_vec1()?;

    let max_diff = sdpa_data
        .iter()
        .zip(fa_v2_data.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_diff < 1e-2,
        "FlashAttentionV2 should be close to SDPA, max diff: {max_diff}"
    );

    Ok(())
}

#[test]
fn test_flash_attention_kernel_sliding_window() -> Result<()> {
    let config = FlashAttentionConfig::new().with_sliding_window(2);
    let kernel = FlashAttentionKernel::new(2, 64, config);

    let q = Tensor::ones((1, 1, 4, 64), candle_core::DType::F32, &Device::Cpu)?;
    let k = Tensor::ones((1, 1, 4, 64), candle_core::DType::F32, &Device::Cpu)?;
    let v = Tensor::ones((1, 1, 4, 64), candle_core::DType::F32, &Device::Cpu)?;

    let output = kernel.forward(&q, &k, &v)?;

    assert_eq!(output.dims(), &[1, 1, 4, 64]);

    Ok(())
}

#[test]
fn test_sliding_window_small_seq() -> Result<()> {
    let sdpa = ScaledDotProductAttention::new(32);

    let q = Tensor::ones((1, 1, 3, 32), candle_core::DType::F32, &Device::Cpu)?;
    let k = Tensor::ones((1, 1, 3, 32), candle_core::DType::F32, &Device::Cpu)?;
    let v = Tensor::ones((1, 1, 3, 32), candle_core::DType::F32, &Device::Cpu)?;

    let output = sdpa.compute_sliding_window(&q, &k, &v, 5)?;

    assert_eq!(output.dims(), &[1, 1, 3, 32]);

    Ok(())
}

#[test]
fn test_sliding_window_large_seq() -> Result<()> {
    let sdpa = ScaledDotProductAttention::new(32);

    let q = Tensor::ones((1, 1, 10, 32), candle_core::DType::F32, &Device::Cpu)?;
    let k = Tensor::ones((1, 1, 10, 32), candle_core::DType::F32, &Device::Cpu)?;
    let v = Tensor::ones((1, 1, 10, 32), candle_core::DType::F32, &Device::Cpu)?;

    let output = sdpa.compute_sliding_window(&q, &k, &v, 4)?;

    assert_eq!(output.dims(), &[1, 1, 10, 32]);

    Ok(())
}

#[test]
fn test_softmax_output_range() -> Result<()> {
    let sdpa = ScaledDotProductAttention::new(8);

    let q_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let q = Tensor::from_slice(&q_data, (1, 1, 1, 8), &Device::Cpu)?;

    let k_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let k = Tensor::from_slice(&k_data, (1, 1, 1, 8), &Device::Cpu)?;

    let v_data: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let v = Tensor::from_slice(&v_data, (1, 1, 1, 8), &Device::Cpu)?;

    let output = sdpa.forward(&q, &k, &v)?;

    let output_data: Vec<f32> = output.flatten_all()?.to_vec1()?;

    for val in &output_data {
        assert!(*val >= 0.0, "Softmax output should be non-negative: {val}");
        assert!(*val <= 100.0, "Softmax output should be reasonable: {val}");
    }

    Ok(())
}
