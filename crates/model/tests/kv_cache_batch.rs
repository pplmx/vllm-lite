use candle_core::{DType, Device, Result, Tensor};
use vllm_model::kv_cache::PagedKvCache;

#[test]
fn test_write_kv_batch_basic() -> Result<()> {
    let device = Device::Cpu;
    let mut cache = PagedKvCache::new(1, 2, 4, 4, device.clone(), false)?;

    // Create batch of 4 tokens
    let k_batch = Tensor::ones((1, 4, 2, 4), DType::F32, &device)?;
    let v_batch = Tensor::ones((1, 4, 2, 4), DType::F32, &device)?;

    // Write batch at once
    cache.write_kv_batch(0, 0, 0, &k_batch, &v_batch)?;

    // Read back and verify
    let (k_out, _v_out) = cache.read_kv(0, &[0], 4)?;
    assert_eq!(k_out.dims(), &[4, 2, 4]);

    Ok(())
}

#[test]
fn test_write_kv_batch_multiple_blocks() -> Result<()> {
    let device = Device::Cpu;
    let mut cache = PagedKvCache::new(1, 2, 4, 4, device.clone(), false)?;

    // Write 32 tokens across 2 blocks
    let k_batch = Tensor::ones((1, 32, 2, 4), DType::F32, &device)?;
    let v_batch = Tensor::ones((1, 32, 2, 4), DType::F32, &device)?;

    cache.write_kv_batch(0, 0, 0, &k_batch, &v_batch)?;

    let (k_out, _v_out) = cache.read_kv(0, &[0, 1], 32)?;
    assert_eq!(k_out.dims(), &[32, 2, 4]);

    Ok(())
}

#[test]
fn test_write_kv_batch_dimension_mismatch() -> Result<()> {
    let device = Device::Cpu;
    let mut cache = PagedKvCache::new(1, 2, 4, 4, device.clone(), false)?;

    let k_batch = Tensor::ones((1, 4, 2, 4), DType::F32, &device)?;
    let v_batch = Tensor::ones((1, 4, 3, 4), DType::F32, &device)?;

    let result = cache.write_kv_batch(0, 0, 0, &k_batch, &v_batch);
    assert!(result.is_err());

    Ok(())
}

#[test]
fn test_write_kv_batch_wrong_head_dim() -> Result<()> {
    let device = Device::Cpu;
    let mut cache = PagedKvCache::new(1, 2, 4, 4, device.clone(), false)?;

    let k_batch = Tensor::ones((1, 4, 2, 8), DType::F32, &device)?;
    let v_batch = Tensor::ones((1, 4, 2, 8), DType::F32, &device)?;

    let result = cache.write_kv_batch(0, 0, 0, &k_batch, &v_batch);
    assert!(result.is_err());

    Ok(())
}
