#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    pub data: Vec<f32>,
    pub scale: f32,
}

pub fn quantize(data: &[f32]) -> QuantizedTensor {
    let max_val = data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);

    let scale = if max_val > 0.0 { max_val / 127.0 } else { 1.0 };

    let quantized_data: Vec<f32> = data.iter().map(|v| (*v / scale).round()).collect();

    QuantizedTensor {
        data: quantized_data,
        scale,
    }
}

pub fn dequantize(data: &[f32], scale: f32) -> Vec<f32> {
    data.iter().map(|x| x * scale).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_basic() {
        let data = vec![10.0f32, 20.0, 30.0, -10.0];
        let result = quantize(&data);

        assert!(result.scale > 0.0);
        assert_eq!(result.data.len(), data.len());
    }

    #[test]
    fn test_quantize_zero() {
        let data = vec![0.0f32, 0.0, 0.0];
        let result = quantize(&data);

        assert_eq!(result.scale, 1.0);
        for val in &result.data {
            assert_eq!(*val, 0.0);
        }
    }

    #[test]
    fn test_dequantize_basic() {
        let data = vec![10.0f32, 20.0, 30.0];
        let scale = 2.0;
        let result = dequantize(&data, scale);

        assert_eq!(result, vec![20.0, 40.0, 60.0]);
    }

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        let original = vec![10.0f32, 20.0, 30.0, -15.0, 5.0];
        let quantized = quantize(&original);
        let dequantized = dequantize(&quantized.data, quantized.scale);

        for (orig, deq) in original.iter().zip(dequantized.iter()) {
            assert!((orig - deq).abs() < 1.0);
        }
    }

    #[test]
    fn test_quantized_tensor_fields() {
        let data = vec![100.0f32; 8];
        let result = quantize(&data);

        assert!(result.scale > 0.0);
        assert_eq!(result.data.len(), 8);
    }
}
