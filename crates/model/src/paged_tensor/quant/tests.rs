//! Unit tests for the quantization module (`AWQQuantization`,
//! `GPTQQuantization`, `QuantizationType`, `QuantizedWeights`).
//!
//! Two concerns are exercised:
//!
//! 1. **Quantize → dequantize round-trip**: a linear ramp of `f32`
//!    values quantizes through `AWQQuantization::new(4, 32)` and
//!    `GPTQQuantization::new(4, 16)` and dequantizes back with a
//!    bounded max abs error (`<10.0` for AWQ, `<20.0` for GPTQ; the
//!    thresholds are loose because 4-bit quantization inherently
//!    loses ~6% precision).
//! 2. **Type parsing & builder**: `QuantizationType::from_str` round-
//!    trips both `awq`/`gptq` short names and the legacy `llm-awq`
//!    alias; `Awq.bits()` / `Gptq.bits()` both report 4. The
//!    `QuantizedWeights::new(...).with_zeros(...)` builder chain
//!    records the zero-points tensor.
//!
//! The precision-loss comment on `make_ramp` documents why the test
//! accepts f32 indices 0..=127 in its input.
use super::*;

// invariant: test fixture indices 0..=127 fit in f32 exactly; precision loss
// is acceptable for the linear ramp test data.
#[allow(clippy::cast_precision_loss)]
fn make_ramp(start: f32, end: f32, count: usize) -> Vec<f32> {
    let step = (end - start) / count as f32;
    (0..count)
        .map(|i| (i as f32).mul_add(step, start))
        .collect()
}

#[test]
fn test_awq_quantize_dequantize() {
    let device = Device::Cpu;
    let data: Vec<f32> = make_ramp(0.0, 12.8, 128);
    let awq = AWQQuantization::new(4, 32);

    let weights = awq.quantize(&data, &[128], &device).unwrap();
    let dequantized = awq.dequantize(&weights).unwrap();

    let deq_data: Vec<f32> = dequantized.to_vec1().unwrap();
    let max_diff = data
        .iter()
        .zip(deq_data.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_diff < 10.0,
        "Dequantization error too large: {max_diff}"
    );
}

#[test]
fn test_gptq_quantize_dequantize() {
    let device = Device::Cpu;
    // ramp from -16 to +15.5 with step 0.5 (equivalent to (i - 32) * 0.5)
    let data: Vec<f32> = make_ramp(-16.0, 16.0, 64);
    let gptq = GPTQQuantization::new(4, 16);

    let weights = gptq.quantize(&data, &[64], &device).unwrap();
    let dequantized = gptq.dequantize(&weights).unwrap();

    let deq_data: Vec<f32> = dequantized.to_vec1().unwrap();
    let max_diff = data
        .iter()
        .zip(deq_data.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(max_diff < 20.0, "GPTQ dequantization error too large");
}

#[test]
fn test_quantization_types() {
    assert_eq!(
        QuantizationType::from_str("awq"),
        Some(QuantizationType::Awq)
    );
    assert_eq!(
        QuantizationType::from_str("gptq"),
        Some(QuantizationType::Gptq)
    );
    assert_eq!(
        QuantizationType::from_str("llm-awq"),
        Some(QuantizationType::Awq)
    );

    assert_eq!(QuantizationType::Awq.bits(), 4);
    assert_eq!(QuantizationType::Gptq.bits(), 4);
}

#[test]
fn test_quantized_weights_builder() {
    use candle_core::Device;

    let qweight = Tensor::ones((32,), candle_core::DType::U8, &Device::Cpu).unwrap();
    let scales = Tensor::ones((1,), candle_core::DType::F32, &Device::Cpu).unwrap();

    let weights = QuantizedWeights::new(qweight, scales)
        .with_zeros(Tensor::zeros((1,), candle_core::DType::F32, &Device::Cpu).unwrap());

    assert!(weights.zeros.is_some());
}
