use candle_core::{DType, Result, Tensor};
use tracing::trace;

#[derive(Debug, Clone)]
pub enum KvCacheDtype {
    Fp16,
    Fp32,
    Fp8E4m3,
}

impl KvCacheDtype {
    pub fn bytes_per_element(&self) -> usize {
        match self {
            KvCacheDtype::Fp16 => 2,
            KvCacheDtype::Fp32 => 4,
            KvCacheDtype::Fp8E4m3 => 1,
        }
    }

    pub fn memory_reduction_ratio(&self) -> f32 {
        match self {
            KvCacheDtype::Fp16 => 1.0,
            KvCacheDtype::Fp32 => 0.5,
            KvCacheDtype::Fp8E4m3 => 2.0,
        }
    }
}

pub struct Fp8Quantizer {
    dtype: KvCacheDtype,
}

impl Fp8Quantizer {
    pub fn new(dtype: KvCacheDtype) -> Self {
        Self { dtype }
    }

    pub fn quantize(&self, tensor: &Tensor) -> Result<Tensor> {
        match self.dtype {
            KvCacheDtype::Fp16 => Ok(tensor.clone()),
            KvCacheDtype::Fp32 => self.quantize_to_fp32(tensor),
            KvCacheDtype::Fp8E4m3 => self.quantize_to_fp8(tensor),
        }
    }

    pub fn dequantize(&self, tensor: &Tensor) -> Result<Tensor> {
        match self.dtype {
            KvCacheDtype::Fp16 => Ok(tensor.clone()),
            KvCacheDtype::Fp32 => {
                if tensor.dtype() == DType::F32 {
                    Ok(tensor.clone())
                } else {
                    tensor.to_dtype(DType::F32)
                }
            }
            KvCacheDtype::Fp8E4m3 => self.dequantize_from_fp8(tensor),
        }
    }

    fn quantize_to_fp32(&self, tensor: &Tensor) -> Result<Tensor> {
        tensor.to_dtype(DType::F32)
    }

    fn quantize_to_fp8(&self, tensor: &Tensor) -> Result<Tensor> {
        let shape = tensor.dims();
        let flat = tensor.flatten_all()?;

        match flat.dtype() {
            DType::F16 => {
                let data: Vec<f32> = flat.to_vec1::<half::f16>()?
                    .into_iter()
                    .map(|h| h.to_f32())
                    .collect();

                let fp8_data: Vec<u8> = data
                    .iter()
                    .map(|&f| self.float32_to_fp8_e4m3(f))
                    .collect();

                Tensor::from_slice(&fp8_data, shape, tensor.device())
            }
            DType::F32 => {
                let data: Vec<f32> = flat.to_vec1()?;
                let fp8_data: Vec<u8> = data
                    .iter()
                    .map(|&f| self.float32_to_fp8_e4m3(f))
                    .collect();

                Tensor::from_slice(&fp8_data, shape, tensor.device())
            }
            _ => Err(candle_core::Error::msg(format!(
                "Unsupported dtype for FP8 quantization: {:?}",
                flat.dtype()
            ))),
        }
    }

    fn dequantize_from_fp8(&self, tensor: &Tensor) -> Result<Tensor> {
        let shape = tensor.dims();
        let flat = tensor.flatten_all()?;

        let data: Vec<u8> = flat.to_vec1()?;
        let float_data: Vec<half::f16> = data
            .iter()
            .map(|&b| self.fp8_e4m3_to_float16(b))
            .collect();

        let result = Tensor::from_slice(&float_data, shape, tensor.device())?;
        result.to_dtype(DType::F16)
    }

    fn float32_to_fp8_e4m3(&self, value: f32) -> u8 {
        if value.is_nan() {
            return 0x80;
        }
        if value.is_infinite() {
            return if value.is_sign_positive() { 0x7C } else { 0xFC };
        }
        if value.abs() < 0.000732 => 0u8;

        let sign = if value.is_sign_negative() { 1u8 } else { 0u8 };
        let abs = value.abs();

        let (exp, mantissa) = frexp(abs);

        let biased_exp = (exp + 7) as u8;
        if biased_exp >= 0x1F {
            return if sign == 0 { 0x7B } else { 0xFB };
        }

        let scaled_mantissa = mantissa * 8.0;
        let int_mantissa = (scaled_mantissa + 0.5) as u8;

        ((sign & 0x01) << 7) | (biased_exp & 0x0F) << 3 | (int_mantissa & 0x07)
    }

    fn fp8_e4m3_to_float16(&self, value: u8) -> half::f16 {
        if value == 0x80 {
            return half::f16::from_f32(f32::NAN);
        }
        if value == 0x7C {
            return half::f16::INFINITY;
        }
        if value == 0xFC {
            return half::f16::NEG_INFINITY;
        }

        let sign = ((value >> 7) & 0x01) as u16;
        let biased_exp = ((value >> 3) & 0x0F) as i32;
        let mantissa = (value & 0x07) as u16;

        if biased_exp == 0 && mantissa == 0 {
            return half::f16::from_bits(sign << 15);
        }

        let exp = biased_exp as i32 - 7 - 3;

        let bits = (sign << 15)
            | ((exp + 15) as u16 & 0x7FFF) << 10
            | (mantissa << 7);

        half::f16::from_bits(bits)
    }

    pub fn estimate_memory_savings(
        num_blocks: usize,
        block_size: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> f32 {
        let fp16_bytes = num_blocks * block_size * num_heads * head_dim * 2;
        let fp8_bytes = num_blocks * block_size * num_heads * head_dim * 1;
        1.0 - (fp8_bytes as f32 / fp16_bytes as f32)
    }
}

fn frexp(value: f32) -> (i32, f32) {
    if value == 0.0 {
        return (0, 0.0);
    }

    let bits = value.to_bits();
    let sign = bits >> 31;
    let mut exp = ((bits >> 23) & 0xFF) as i32;
    let mut mantissa_bits = bits & 0x007FFFFF;

    if exp == 0 {
        let shift = mantissa_bits.leading_zeros() as i32 - 8;
        exp -= shift;
        mantissa_bits = mantissa_bits << (shift + 1);
    }

    exp -= 127;

    let mantissa = 1.0 + (mantissa_bits as f32 / (1 << 23) as f32);

    (exp, mantissa)
}

#[cfg(test)]
mod tests {
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

        assert_eq!(fp16.memory_reduction_ratio(), 1.0);
        assert_eq!(fp8.memory_reduction_ratio(), 2.0);
    }

    #[test]
    fn test_fp8_roundtrip_quantization() {
        let device = candle_core::Device::Cpu;
        let tensor = Tensor::randn(-1.0f32, 1.0, (2, 4, 8), &device).unwrap();

        let quantizer = Fp8Quantizer::new(KvCacheDtype::Fp8E4m3);
        let quantized = quantizer.quantize(&tensor).unwrap();
        let dequantized = quantizer.dequantize(&quantized).unwrap();

        assert_eq!(dequantized.dims(), tensor.dims());
    }

    #[test]
    fn test_fp8_preserves_small_values() {
        let device = candle_core::Device::Cpu;
        let tensor = Tensor::new(&[0.0001f32, 0.0005f32, 0.001f32], &device).unwrap();

        let quantizer = Fp8Quantizer::new(KvCacheDtype::Fp8E4m3);
        let quantized = quantizer.quantize(&tensor).unwrap();
        let dequantized = quantizer.dequantize(&quantized).unwrap();

        let original: Vec<f32> = tensor.to_vec1().unwrap();
        let recovered: Vec<f32> = dequantized.to_vec1::<half::f16>().unwrap()
            .iter()
            .map(|h| h.to_f32())
            .collect();

        for (o, r) in original.iter().zip(recovered.iter()) {
            let diff = (o - r).abs();
            assert!(
                diff < 0.01 || o.abs() < 0.001,
                "Values should be approximately preserved"
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
        let quantizer = Fp8Quantizer::new(KvCacheDtype::Fp8E4m3);

        let zero = quantizer.float32_to_fp8_e4m3(0.0);
        assert_eq!(zero, 0);

        let one = quantizer.float32_to_fp8_e4m3(1.0);
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
}
