//! Unit tests for the FP8 KV-cache quantization module.
//!
//! Covers three layers of the FP8 surface:
//!
//! 1. **Dtype / quantizer basics (3)**: `Fp8Quantizer::new(Fp8E4m3)`
//!    reports 1 byte/element; `memory_reduction_ratio` is 1.0 for
//!    `Fp16`, 2.0 for `Fp8E4m3`; `estimate_memory_savings` returns
//! 2. **Round-trip + precision (2)**: quantizing then dequantizing
//!    normal-range values preserves them within ~15% relative error
//!    (3-bit mantissa); sub-normal values (< 2^-6) flush to ~0 rather
//!    than saturating.
//! 3. **Special values + Fp16 pass-through (2)**: 0.0 → 0 in `E4M3`;
//!    1.0 → non-zero; `Fp16` quantization is identity on the `F16`
//!    tensor shape (no-op).
//!
//! All tests run on `Device::Cpu`.
use super::*;

#[test]
fn test_fp8_quantizer_creation() {
    let quantizer = Fp8Quantizer::new(KvCacheDtype::Fp8E4m3);
    assert_eq!(quantizer.dtype.bytes_per_element(), 1);
}

#[test]
fn test_fp8_memory_reduction() {
    let fp16 = KvCacheDtype::Fp16;
    let fp8 = KvCacheDtype::Fp8E4m3;

    assert!((fp16.memory_reduction_ratio() - 1.0).abs() < 1e-6);
    assert!((fp8.memory_reduction_ratio() - 2.0).abs() < 1e-6);
}

#[test]
#[allow(clippy::similar_names)]
fn test_fp8_roundtrip_quantization() {
    let device = candle_core::Device::Cpu;
    // Values across the representable normal range (clear of the subnormal
    // flush floor and the overflow saturation).
    let values: Vec<f32> = vec![
        1.0, 2.0, 0.5, -3.0, 0.1, -0.1, 100.0, -100.0, 50.0, 0.02, -0.02, 7.5,
    ];
    let tensor = Tensor::from_vec(values.clone(), (values.len(),), &device).unwrap();

    let quantizer = Fp8Quantizer::new(KvCacheDtype::Fp8E4m3);
    let quantized = quantizer.quantize(&tensor).unwrap();
    let dequantized = quantizer.dequantize(&quantized).unwrap();
    assert_eq!(dequantized.dims(), tensor.dims());

    // Regression (RIL ISS-015): the round-trip must preserve VALUES, not just
    // shape. A 3-bit mantissa bounds the relative rounding error well under
    // ~13%; allow 15% headroom. Pre-fix the dequant exponent was off by 3, so
    // every value came back 8x too small — invisible to a shape-only assert.
    let recovered: Vec<f32> = dequantized
        .to_vec1::<half::f16>()
        .unwrap()
        .iter()
        .map(|h| h.to_f32())
        .collect();
    for (orig, rec) in values.iter().zip(recovered.iter()) {
        let rel_err = (orig - rec).abs() / orig.abs();
        assert!(
            rel_err < 0.15,
            "FP8 round-trip must preserve value within ~15% relative error:              orig={orig}, recovered={rec}, rel_err={rel_err} (8x error => RIL ISS-015)"
        );
    }
}

#[test]
#[allow(clippy::similar_names)]
fn test_fp8_preserves_small_values() {
    let device = candle_core::Device::Cpu;
    // All below the smallest E4M3 normal (2^-6 ~= 0.0156), so they flush to
    // zero (subnormals are not encoded).
    let tensor = Tensor::new(&[0.0001f32, 0.0005f32, 0.001f32, -0.001f32], &device).unwrap();

    let quantizer = Fp8Quantizer::new(KvCacheDtype::Fp8E4m3);
    let quantized = quantizer.quantize(&tensor).unwrap();
    let dequantized = quantizer.dequantize(&quantized).unwrap();

    let recovered: Vec<f32> = dequantized
        .to_vec1::<half::f16>()
        .unwrap()
        .iter()
        .map(|h| h.to_f32())
        .collect();

    // Regression (RIL ISS-015): sub-normal inputs must flush to ~0, NOT
    // saturate to a large value. Pre-fix, a negative biased_exp wrapped to a
    // huge u8 and tripped overflow saturation, so 0.001 dequantized to ~44;
    // the old assertion's `|| o.abs() <= 0.001` clause made it vacuous.
    for (i, r) in recovered.iter().enumerate() {
        assert!(
            r.abs() < 0.02,
            "sub-normal input #{i} must flush to ~0, got {r} (saturation => RIL ISS-015)"
        );
    }
}

#[test]
fn test_memory_savings_estimation() {
    let savings = Fp8Quantizer::estimate_memory_savings(1000, 16, 32, 128);

    assert!(
        (savings - 0.5).abs() < 0.01,
        "FP8 should save approximately 50% memory"
    );
}

#[test]
fn test_fp8_special_values() {
    let zero = Fp8Quantizer::float32_to_fp8_e4m3(0.0);
    assert_eq!(zero, 0);

    let one = Fp8Quantizer::float32_to_fp8_e4m3(1.0);
    assert_ne!(one, 0);
}

#[test]
fn test_fp16_unchanged() {
    let device = candle_core::Device::Cpu;
    let tensor = Tensor::randn(-1.0f32, 1.0, (2, 4), &device)
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap();

    let quantizer = Fp8Quantizer::new(KvCacheDtype::Fp16);
    let result = quantizer.quantize(&tensor).unwrap();

    assert_eq!(result.dims(), tensor.dims());
}
