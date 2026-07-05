//! Unit tests for `TransformerBlock` (Qwen3 decoder layer).
//!
//! Extracted from `block.rs` to keep the implementation file under the
//! project's 800-line soft cap. Exercises prefill, decode, and decode
//! across multiple KV blocks with/without QK-norm and custom head-dim.

use super::*;
use candle_core::{DType, Device};

#[test]
fn test_transformer_block_forward() -> Result<()> {
    let device = Device::Cpu;
    let block = TransformerBlock::new(256, 4, 2, 64, 512, 10000.0, 1e-6, None, false)?;

    let x = Tensor::ones((1, 2, 256), DType::F32, &device)?;
    let output = block.forward(&x)?;

    assert_eq!(output.dims(), &[1, 2, 256]);
    Ok(())
}

#[test]
fn test_transformer_block_batch_forward() -> Result<()> {
    let device = Device::Cpu;
    let block = TransformerBlock::new(128, 4, 2, 32, 256, 10000.0, 1e-6, None, false)?;

    let x = Tensor::ones((4, 3, 128), DType::F32, &device)?;
    let output = block.forward(&x)?;

    assert_eq!(output.dims(), &[4, 3, 128]);
    Ok(())
}

#[test]
fn test_transformer_block_output_shape() -> Result<()> {
    let device = Device::Cpu;
    let block = TransformerBlock::new(128, 4, 2, 32, 256, 10000.0, 1e-6, None, false)?;

    let x = Tensor::zeros((2, 1, 128), DType::F32, &device)?;
    let output = block.forward(&x)?;

    assert_eq!(output.dims(), &[2, 1, 128]);
    Ok(())
}

#[test]
fn test_transformer_block_with_qk_norm() -> Result<()> {
    // Qwen3-0.6B uses q_norm/k_norm
    let device = Device::Cpu;
    let block = TransformerBlock::new(128, 4, 2, 32, 256, 10000.0, 1e-6, None, true)?;

    let x = Tensor::ones((1, 2, 128), DType::F32, &device)?;
    let output = block.forward(&x)?;

    assert_eq!(output.dims(), &[1, 2, 128]);
    Ok(())
}

#[test]
fn test_transformer_block_with_custom_head_dim() -> Result<()> {
    // Qwen3-0.6B has head_dim=128 but hidden/heads=1024/16=64
    // This test verifies custom head_dim works
    let device = Device::Cpu;
    // hidden=1024, heads=16, kv_heads=8, head_dim=128, intermediate=3072
    let block = TransformerBlock::new(1024, 16, 8, 128, 3072, 10000.0, 1e-6, None, true)?;

    let x = Tensor::ones((1, 4, 1024), DType::F32, &device)?;
    let output = block.forward(&x)?;

    assert_eq!(output.dims(), &[1, 4, 1024]);
    Ok(())
}

#[test]
fn test_transformer_block_forward_decode_3d_output_shape() -> Result<()> {
    let device = Device::Cpu;
    let hidden_size = 256;
    let num_heads = 4;
    let num_kv_heads = 2;
    let head_dim = 64;

    let block = TransformerBlock::new(
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        512,
        10000.0,
        1e-6,
        None,
        false,
    )?;
    let mut kv_cache =
        crate::paged_tensor::PagedKvCache::new(1, num_heads, head_dim, 8, device.clone(), false)?;

    let x = Tensor::ones((1, hidden_size), DType::F32, &device)?;

    let block_ids: Vec<usize> = vec![0];
    let num_computed = 0;
    let positions = vec![0];

    let output =
        block.forward_decode(&x, &mut kv_cache, 0, &block_ids, num_computed, &positions)?;

    assert_eq!(output.dims(), &[1, hidden_size]);
    Ok(())
}

#[test]
fn test_transformer_block_forward_decode_with_qk_norm() -> Result<()> {
    let device = Device::Cpu;
    let hidden_size = 1024;
    let num_heads = 16;
    let num_kv_heads = 8;
    let head_dim = 128;

    let block = TransformerBlock::new(
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        3072,
        10000.0,
        1e-6,
        None,
        true,
    )?;
    let mut kv_cache =
        crate::paged_tensor::PagedKvCache::new(1, num_heads, head_dim, 8, device.clone(), false)?;

    let x = Tensor::ones((1, hidden_size), DType::F32, &device)?;
    let block_ids: Vec<usize> = vec![0];
    let positions = vec![0];

    let output = block.forward_decode(&x, &mut kv_cache, 0, &block_ids, 0, &positions)?;

    assert_eq!(output.dims(), &[1, hidden_size]);
    Ok(())
}

#[test]
fn test_transformer_block_decode_sequential_tokens() -> Result<()> {
    let device = Device::Cpu;
    let hidden_size = 512;
    let num_heads = 8;
    let num_kv_heads = 4;
    let head_dim = 64;

    let block = TransformerBlock::new(
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        1024,
        10000.0,
        1e-6,
        None,
        false,
    )?;
    let mut kv_cache =
        crate::paged_tensor::PagedKvCache::new(1, num_heads, head_dim, 16, device.clone(), false)?;

    for step in 0..5 {
        let x = Tensor::ones((1, hidden_size), DType::F32, &device)?;
        let block_ids: Vec<usize> = vec![step / 8];
        let positions = vec![step];

        let output = block.forward_decode(&x, &mut kv_cache, 0, &block_ids, step, &positions)?;

        assert_eq!(
            output.dims(),
            &[1, hidden_size],
            "Step {step} output shape mismatch"
        );
    }

    Ok(())
}

#[test]
fn test_transformer_block_decode_with_multiple_kv_blocks() -> Result<()> {
    let device = Device::Cpu;
    let hidden_size = 256;
    let num_heads = 4;
    let num_kv_heads = 2;
    let head_dim = 64;
    let block_size = 8;

    let block = TransformerBlock::new(
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        512,
        10000.0,
        1e-6,
        None,
        false,
    )?;
    let mut kv_cache =
        crate::paged_tensor::PagedKvCache::new(1, num_heads, head_dim, 4, device.clone(), false)?;

    for step in 0..24 {
        let x = Tensor::ones((1, hidden_size), DType::F32, &device)?;
        let block_id = step / block_size;
        let block_ids: Vec<usize> = vec![block_id];
        let positions = vec![step];

        let output = block.forward_decode(&x, &mut kv_cache, 0, &block_ids, step, &positions)?;

        assert_eq!(
            output.dims(),
            &[1, hidden_size],
            "Step {step} output shape mismatch"
        );
    }

    Ok(())
}
