//! Symmetric 8-bit quantization (`QuantizedTensor`): `f32 → i8 → f32` round-trip with scale + zero-point metadata.
//!
//! Memory footprint is 4× smaller than FP32 at the cost of ~3% accuracy
//! loss on attention scores. Used by the on-disk KV cache format and
//! by the INT8 weight loader for select architectures.
// invariant: quantization scalar math operates on bounded quantization levels
// and tensor dimensions; precision loss / truncation / wrap is intentional in
// the quantization rounding math.
/// Symmetric 8-bit quantized tensor (int8 values stored as `f32` for
/// compatibility with candle tensor APIs).
///
/// The values are normalized by [`Self::scale`] (which maps the
/// original tensor max-magnitude to 127.0). To recover the original
/// tensor, multiply element-wise by `scale` — see [`dequantize`].
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// Quantized integer values normalized to the range `-127..=127`.
    /// Stored as `f32` rather than `i8` because candle tensor types do not
    /// expose an `i8` dtype; multiply by [`Self::scale`] to recover the
    /// original dequantized magnitude.
    pub values: Vec<f32>,
    /// Per-tensor scale factor: dividing the original max-magnitude by 127.
    /// Use [`dequantize`] to apply this scale to the quantized values.
    pub scale: f32,
}

/// Quantize a flat `f32` slice into a [`QuantizedTensor`] using symmetric
/// 8-bit scaling.
///
/// The output `scale` is `max(|x|) / 127` (or `1.0` when all inputs are
/// zero so we still produce a no-op dequantization).
pub fn quantize(data: &[f32]) -> QuantizedTensor {
    let max_val = data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);

    let scale = if max_val > 0.0 { max_val / 127.0 } else { 1.0 };

    let quantized_values: Vec<f32> = data.iter().map(|v| (*v / scale).round()).collect();

    QuantizedTensor {
        values: quantized_values,
        scale,
    }
}

/// Reverse of [`quantize`]: multiply each quantized value by `scale` to
/// recover the dequantized magnitude. This is a per-element multiply; for
/// batch dequantization of an entire tensor, call once per row.
#[must_use]
pub fn dequantize(values: &[f32], scale: f32) -> Vec<f32> {
    values.iter().map(|x| x * scale).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_basic() {
        let data = vec![10.0f32, 20.0, 30.0, -10.0];
        let result = quantize(&data);

        assert!(result.scale > 0.0);
        assert_eq!(result.values.len(), data.len());
    }

    #[test]
    fn test_quantize_zero() {
        let data = vec![0.0f32, 0.0, 0.0];
        let result = quantize(&data);

        assert!((result.scale - 1.0).abs() < 1e-6);
        for val in &result.values {
            assert!(val.abs() < 1e-6);
        }
    }

    #[test]
    fn test_dequantize_basic() {
        let values = vec![10.0f32, 20.0, 30.0];
        let scale = 2.0;
        let result = dequantize(&values, scale);

        assert_eq!(result, vec![20.0, 40.0, 60.0]);
    }

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        let original = vec![10.0f32, 20.0, 30.0, -15.0, 5.0];
        let quantized = quantize(&original);
        let dequantized = dequantize(&quantized.values, quantized.scale);

        for (orig, deq) in original.iter().zip(dequantized.iter()) {
            assert!((orig - deq).abs() < 1.0);
        }
    }

    #[test]
    fn test_quantized_tensor_fields() {
        let data = vec![100.0f32; 8];
        let result = quantize(&data);

        assert!(result.scale > 0.0);
        assert_eq!(result.values.len(), 8);
    }
}
