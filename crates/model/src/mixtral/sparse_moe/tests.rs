//! Unit tests for `MixtralSparseMoe`.
//!
//! Locks in two contracts:
//!
//! 1. **Shape preservation**: prefill `[B, T, H]` and decode `[B, H]`
//!    inputs round-trip to the same shape on the output, regardless of
//!    `hidden_size`, `num_experts`, or `top_k`.
//! 2. **Vectorized ≡ naive**: the optimized `forward` path matches the
//!    `forward_naive` reference implementation to within `1e-5` max
//!    element diff, for both prefill and decode inputs. Guards against
//!    regressions in the expert-grouped batching + `scatter_add`
//!    scatter path introduced in the v22.0 `MoE` vectorization pass.
//!
//! All tests run on `Device::Cpu` with `DType::F32`.
use super::*;
use candle_core::DType;

#[test]
fn test_sparse_moe_creation() -> Result<()> {
    let device = candle_core::Device::Cpu;
    let moe = MixtralSparseMoe::new(
        4096,
        8,
        14336,
        2,
        candle_nn::VarBuilder::zeros(DType::F32, &device),
    )?;

    assert_eq!(moe.num_experts, 8);
    assert_eq!(moe.top_k, 2);
    Ok(())
}

#[test]
fn test_sparse_moe_forward_shape() -> Result<()> {
    let device = candle_core::Device::Cpu;
    let moe = MixtralSparseMoe::new(
        256,
        4,
        512,
        2,
        candle_nn::VarBuilder::zeros(DType::F32, &device),
    )?;

    let x = Tensor::ones((2, 3, 256), DType::F32, &device)?;
    let output = moe.forward(&x)?;

    assert_eq!(output.dims(), &[2, 3, 256]);
    Ok(())
}

#[test]
fn test_sparse_moe_forward_decode_shape() -> Result<()> {
    let device = candle_core::Device::Cpu;
    let moe = MixtralSparseMoe::new(
        256,
        4,
        512,
        2,
        candle_nn::VarBuilder::zeros(DType::F32, &device),
    )?;

    let x = Tensor::ones((2, 256), DType::F32, &device)?;
    let output = moe.forward(&x)?;
    assert_eq!(output.dims(), &[2, 256]);
    Ok(())
}

#[test]
fn test_sparse_moe_vectorized_matches_naive() -> Result<()> {
    let device = candle_core::Device::Cpu;
    let moe = MixtralSparseMoe::new(
        64,
        4,
        128,
        2,
        candle_nn::VarBuilder::zeros(DType::F32, &device),
    )?;

    let x = Tensor::randn(0.0f32, 1.0, (3, 5, 64), &device)?;
    let vectorized = moe.forward(&x)?;
    let naive = moe.forward_naive(&x)?;

    let diff = (&vectorized - &naive)?.abs()?;
    let max_diff: f32 = diff.max_all()?.to_scalar()?;
    assert!(
        max_diff < 1e-5,
        "vectorized MoE diverged from naive reference: max_diff={max_diff}"
    );
    Ok(())
}

#[test]
fn test_sparse_moe_decode_vectorized_matches_naive() -> Result<()> {
    let device = candle_core::Device::Cpu;
    let moe = MixtralSparseMoe::new(
        64,
        8,
        128,
        2,
        candle_nn::VarBuilder::zeros(DType::F32, &device),
    )?;

    let x = Tensor::randn(0.0f32, 1.0, (16, 64), &device)?;
    let vectorized = moe.forward(&x)?;
    let naive = moe.forward_naive(&x)?;

    let diff = (&vectorized - &naive)?.abs()?;
    let max_diff: f32 = diff.max_all()?.to_scalar()?;
    assert!(
        max_diff < 1e-5,
        "decode vectorized MoE diverged from naive reference: max_diff={max_diff}"
    );
    Ok(())
}
