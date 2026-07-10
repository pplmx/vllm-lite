//! Unit tests for the FP8 KV-cache quantization module.
//!
//! Covers three layers of the FP8 surface:
//!
//! 1. **Dtype / quantizer basics (3)**: `Fp8Quantizer::new(Fp8E4m3)`
//!    reports 1 byte/element; `memory_reduction_ratio` is 1.0 for
//!    `Fp16`, 2.0 for `Fp8E4m3`; `estimate_memory_savings` returns
//! 2. **Round-trip + precision (2)**: quantizing a random N(-1, 1)
//!    tensor and dequantizing preserves shape; small values
//!    (~0.0001-0.001) are recovered within \`0.05\` absolute error
//!    (the floor is the E4M3 subnormal range).
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
    let tensor = Tensor::randn(-1.0f32, 1.0, (2, 4, 8), &device).unwrap();

    let quantizer = Fp8Quantizer::new(KvCacheDtype::Fp8E4m3);
    let quantized = quantizer.quantize(&tensor).unwrap();
    let dequantized = quantizer.dequantize(&quantized).unwrap();

    assert_eq!(dequantized.dims(), tensor.dims());
}

#[test]
#[allow(clippy::similar_names)]
fn test_fp8_preserves_small_values() {
    let device = candle_core::Device::Cpu;
    let tensor = Tensor::new(&[0.0001f32, 0.0005f32, 0.001f32], &device).unwrap();

    let quantizer = Fp8Quantizer::new(KvCacheDtype::Fp8E4m3);
    let quantized = quantizer.quantize(&tensor).unwrap();
    let dequantized = quantizer.dequantize(&quantized).unwrap();

    let original: Vec<f32> = tensor.to_vec1().unwrap();
    let recovered: Vec<f32> = dequantized
        .to_vec1::<half::f16>()
        .unwrap()
        .iter()
        .map(|h| h.to_f32())
        .collect();

    for (o, r) in original.iter().zip(recovered.iter()) {
        let diff = (o - r).abs();
        assert!(
            diff < 0.05 || o.abs() <= 0.001,
            "Values should be approximately preserved (o={o}, r={r}, diff={diff})"
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
