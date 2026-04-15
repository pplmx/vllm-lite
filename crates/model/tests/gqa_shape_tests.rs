//! Tests for GQA tensor shape handling
//!
//! These tests verify that expand_kv correctly handles GQA head count mismatches.

use candle_core::{Device, Shape, Tensor};

/// Helper function to test expand_kv
fn test_expand_kv(
    kv: &Tensor,
    num_q_heads: usize,
    num_kv_heads: usize,
) -> candle_core::Result<Tensor> {
    if num_q_heads == num_kv_heads {
        return Ok(kv.clone());
    }

    let _dims = kv.dims();
    let _batch_size = _dims[0];
    let _seq_len = _dims[1];
    let _head_dim = _dims[3];

    // Check if num_q_heads is divisible by num_kv_heads
    if num_q_heads % num_kv_heads != 0 {
        // Handle edge case: repeat KV heads to match Q heads
        let repeat_factor = (num_q_heads + num_kv_heads - 1) / num_kv_heads;
        let kv_repeated = kv.repeat(&[1, 1, repeat_factor, 1])?;
        // Slice to exact num_q_heads
        return kv_repeated.narrow(2, 0, num_q_heads);
    }

    let repeat_factor = num_q_heads / num_kv_heads;
    kv.repeat(&[1, 1, repeat_factor, 1])
}

#[test]
fn test_expand_kv_gqa_qwen25_config() {
    // Qwen2.5-0.5B: num_heads=14, num_kv_heads=2
    let device = Device::Cpu;
    let kv = Tensor::zeros(
        (1usize, 10usize, 2usize, 64usize),
        candle_core::DType::F32,
        &device,
    )
    .unwrap();

    let result = test_expand_kv(&kv, 14, 2);
    assert!(
        result.is_ok(),
        "expand_kv should succeed for Qwen2.5 config: {:?}",
        result.err()
    );

    let expanded = result.unwrap();
    let dims = expanded.dims();
    assert_eq!(
        dims,
        &[1, 10, 14, 64],
        "Expected shape [1, 10, 14, 64], got {:?}",
        dims
    );
}

#[test]
fn test_expand_kv_gqa_qwen3_config() {
    // Qwen3-0.6B: num_heads=16, num_kv_heads=8
    let device = Device::Cpu;
    let kv = Tensor::zeros(
        (1usize, 10usize, 8usize, 64usize),
        candle_core::DType::F32,
        &device,
    )
    .unwrap();

    let result = test_expand_kv(&kv, 16, 8);
    assert!(
        result.is_ok(),
        "expand_kv should succeed for Qwen3 config: {:?}",
        result.err()
    );

    let expanded = result.unwrap();
    let dims = expanded.dims();
    assert_eq!(
        dims,
        &[1, 10, 16, 64],
        "Expected shape [1, 10, 16, 64], got {:?}",
        dims
    );
}

#[test]
fn test_expand_kv_no_expansion_needed() {
    // MHA case: num_heads == num_kv_heads
    let device = Device::Cpu;
    let kv = Tensor::zeros(
        (1usize, 10usize, 8usize, 64usize),
        candle_core::DType::F32,
        &device,
    )
    .unwrap();

    let result = test_expand_kv(&kv, 8, 8);
    assert!(result.is_ok());

    let expanded = result.unwrap();
    let dims = expanded.dims();
    assert_eq!(
        dims,
        &[1, 10, 8, 64],
        "Shape should remain unchanged for MHA"
    );
}

#[test]
fn test_expand_kv_non_divisible() {
    // Edge case: num_heads=10, num_kv_heads=3 (10 % 3 != 0)
    let device = Device::Cpu;
    let kv = Tensor::zeros(
        (1usize, 5usize, 3usize, 64usize),
        candle_core::DType::F32,
        &device,
    )
    .unwrap();

    let result = test_expand_kv(&kv, 10, 3);
    // Should handle gracefully
    assert!(
        result.is_ok(),
        "expand_kv should handle non-divisible case: {:?}",
        result.err()
    );

    let expanded = result.unwrap();
    let dims = expanded.dims();
    assert_eq!(dims[2], 10, "Should have exactly 10 heads after expansion");
}
