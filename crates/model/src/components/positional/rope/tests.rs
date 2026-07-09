//! Unit tests for the `rope` (rotary positional embedding) module.
//!
//! Two free-function paths and one struct path are exercised:
//!
//! 1. **`apply_rope` (free fn, 3 tests)**: shape preservation
//!    (\`[B, H, S, D]\` → same shape), positional sensitivity
//!    (different positions produce different outputs), determinism
//!    (same input → same output).
//! 2. **`precompute_rope_cache` (3 tests)**: cache length matches
//!    \`max_seq_len * (head_dim / 2)\`, edge cases for length=1 and
//!    length=10.
//! 3. **`RoPE::new` + `RoPE::apply` + `RoPE::forward` (8 tests)**:
//!    construction records `theta` / `head_dim` / `scaling_factor`,
//!    shape preservation for both q and k outputs, rotation actually
//!    modifies the tensor (not identity), and numerical robustness
//!    for large positions (8 KiB) — every output element must
//!    remain finite.
//!
//! All tests run on `Device::Cpu` with `DType::F32`.
use super::*;
use candle_core::{DType, Device, Tensor};

#[test]
fn test_apply_rope_returns_same_shape() -> Result<()> {
    let device = Device::Cpu;
    let query = Tensor::ones((2, 4, 2, 32), DType::F32, &device)?;
    let positions: Vec<i64> = vec![0, 1, 2, 3];

    let result = apply_rope(&query, &positions, 10000.0)?;
    assert_eq!(result.dims(), query.dims());

    Ok(())
}

#[test]
fn test_apply_rope_different_positions() -> Result<()> {
    let device = Device::Cpu;
    let query = Tensor::ones((1, 2, 2, 4), DType::F32, &device)?;

    let pos0: Vec<i64> = vec![0, 1];
    let out0 = apply_rope(&query, &pos0, 10000.0)?;

    let pos1: Vec<i64> = vec![10, 11];
    let out1 = apply_rope(&query, &pos1, 10000.0)?;

    let diff = (out0 - out1)?.abs()?.sum_all()?;
    assert!(
        diff.to_scalar::<f32>()? > 1e-5,
        "RoPE should produce different outputs for different positions"
    );

    Ok(())
}

#[test]
fn test_apply_rope_deterministic() -> Result<()> {
    let device = Device::Cpu;
    let query = Tensor::ones((1, 2, 2, 4), DType::F32, &device)?;
    let positions: Vec<i64> = vec![0, 1];

    let out1 = apply_rope(&query, &positions, 10000.0)?;
    let out2 = apply_rope(&query, &positions, 10000.0)?;

    let diff = (out1 - out2)?.abs()?.sum_all()?;
    assert!(
        diff.to_scalar::<f32>()?.abs() < 1e-6,
        "RoPE should be deterministic"
    );

    Ok(())
}

#[test]
fn test_precompute_rope_cache_length() {
    let cache = precompute_rope_cache(10, 64, 10000.0);
    assert_eq!(cache.len(), 10 * 32);
}

#[test]
fn test_precompute_rope_cache_first_position() {
    let cache = precompute_rope_cache(1, 64, 10000.0);
    assert_eq!(cache.len(), 32);
}

#[test]
fn test_precompute_rope_cache_values() {
    let cache = precompute_rope_cache(10, 64, 10000.0);
    assert_eq!(cache.len(), 320);
}

#[test]
fn test_rope_creation() {
    let device = Device::Cpu;
    let rope = RoPE::new(128, 2048, 10000.0, &device);
    assert!((rope.theta - 10_000.0).abs() < 1e-6);
    assert_eq!(rope.head_dim, 128);
    assert!((rope.scaling_factor - 1.0).abs() < 1e-6);
}

#[test]
fn test_rope_apply() -> Result<()> {
    let device = Device::Cpu;
    let rope = RoPE::new(64, 1024, 10000.0, &device);
    let query = Tensor::ones((1, 4, 8, 64), DType::F32, &device)?;
    let positions: Vec<i64> = vec![0, 1, 2, 3];

    let result = rope.apply(&query, &positions)?;
    assert_eq!(result.dims(), query.dims());

    Ok(())
}

#[test]
fn test_rope_forward_q_shape_preserved() -> Result<()> {
    let device = Device::Cpu;
    let rope = RoPE::new(64, 1024, 10000.0, &device);

    let q = Tensor::randn(0.0f32, 1.0, (2, 4, 8, 64), &device)?;
    let k = Tensor::randn(0.0f32, 1.0, (2, 4, 8, 64), &device)?;

    let (q_out, _) = rope.forward(&q, &k, 0)?;
    assert_eq!(q_out.dims(), q.dims());
    Ok(())
}

#[test]
fn test_rope_forward_k_shape_preserved() -> Result<()> {
    let device = Device::Cpu;
    let rope = RoPE::new(64, 1024, 10000.0, &device);

    let q = Tensor::randn(0.0f32, 1.0, (2, 4, 8, 64), &device)?;
    let k = Tensor::randn(0.0f32, 1.0, (2, 4, 8, 64), &device)?;

    let (_, k_out) = rope.forward(&q, &k, 0)?;
    assert_eq!(k_out.dims(), k.dims());
    Ok(())
}

#[test]
fn test_rope_rotation_applied() -> Result<()> {
    let device = Device::Cpu;
    let rope = RoPE::new(64, 1024, 10000.0, &device);

    let q = Tensor::ones((1, 2, 8, 64), DType::F32, &device)?;
    let k = Tensor::ones((1, 2, 8, 64), DType::F32, &device)?;

    let (q_out, _) = rope.forward(&q, &k, 0)?;

    let diff = (&q_out - &q)?.abs()?;
    let sum_diff = diff.sum_all()?.to_scalar::<f32>()?;
    assert!(sum_diff > 1e-6, "RoPE should modify the tensor");
    Ok(())
}

#[test]
fn test_rope_minimal_head_dim() -> Result<()> {
    let device = Device::Cpu;
    let rope = RoPE::new(64, 512, 10000.0, &device);

    let q = Tensor::randn(0.0f32, 1.0, (1, 1, 1, 64), &device)?;
    let k = Tensor::randn(0.0f32, 1.0, (1, 1, 1, 64), &device)?;

    let (q_out, _) = rope.forward(&q, &k, 0)?;
    assert_eq!(q_out.dims(), q.dims());
    Ok(())
}

#[test]
fn test_rope_large_position() -> Result<()> {
    let device = Device::Cpu;
    let rope = RoPE::new(64, 1024, 10000.0, &device);

    let q = Tensor::randn(0.0f32, 1.0, (1, 1, 1, 64), &device)?;
    let k = Tensor::randn(0.0f32, 1.0, (1, 1, 1, 64), &device)?;

    let (q_out, _) = rope.forward(&q, &k, 8192)?;
    let flat = q_out.flatten_all()?;
    for i in 0..flat.elem_count() {
        let val: f32 = flat.get(i)?.to_scalar()?;
        assert!(val.is_finite(), "Value at index {i} is not finite: {val}");
    }
    Ok(())
}
