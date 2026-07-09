//! kv_cache_fp8: kv cache fp8.

// invariant: quantization scalar math operates on bounded values within fp8
// range; precision loss / truncation / wrap is intentional in the quantization
// rounding math.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]

use candle_core::{DType, Result, Tensor};

/// `KvCacheDtype`. See the type definition for fields and behavior.
#[derive(Debug, Clone)]
pub enum KvCacheDtype {
    Fp16,
    Fp32,
    Fp8E4m3,
}

impl KvCacheDtype {
    #[must_use]
    pub const fn bytes_per_element(&self) -> usize {
        match self {
            Self::Fp16 => 2,
            Self::Fp32 => 4,
            Self::Fp8E4m3 => 1,
        }
    }

    #[must_use]
    pub const fn memory_reduction_ratio(&self) -> f32 {
        match self {
            Self::Fp16 => 1.0,
            Self::Fp32 => 0.5,
            Self::Fp8E4m3 => 2.0,
        }
    }
}

#[derive(Debug)]
/// `Fp8Quantizer`. See the type definition for fields and behavior.
pub struct Fp8Quantizer {
    dtype: KvCacheDtype,
}

impl Fp8Quantizer {
    #[must_use]
    pub const fn new(dtype: KvCacheDtype) -> Self {
        Self { dtype }
    }

    /// Quantize the input tensor into the configured storage dtype.
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    pub fn quantize(&self, tensor: &Tensor) -> Result<Tensor> {
        match self.dtype {
            KvCacheDtype::Fp16 => Ok(tensor.clone()),
            KvCacheDtype::Fp32 => Self::quantize_to_fp32(tensor),
            KvCacheDtype::Fp8E4m3 => Self::quantize_to_fp8(tensor),
        }
    }

    /// Dequantize the input tensor from its stored dtype back to F32/F16.
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
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
            KvCacheDtype::Fp8E4m3 => Self::dequantize_from_fp8(tensor),
        }
    }

    fn quantize_to_fp32(tensor: &Tensor) -> Result<Tensor> {
        tensor.to_dtype(DType::F32)
    }

    fn quantize_to_fp8(tensor: &Tensor) -> Result<Tensor> {
        let shape = tensor.dims();
        let flat = tensor.flatten_all()?;

        match flat.dtype() {
            DType::F16 => {
                let float_values: Vec<f32> = flat
                    .to_vec1::<half::f16>()?
                    .into_iter()
                    .map(half::f16::to_f32)
                    .collect();

                let fp8_data: Vec<u8> = float_values
                    .iter()
                    .map(|&f| Self::float32_to_fp8_e4m3(f))
                    .collect();

                Tensor::from_slice(&fp8_data, shape, tensor.device())
            }
            DType::F32 => {
                let float_values: Vec<f32> = flat.to_vec1()?;
                let fp8_data: Vec<u8> = float_values
                    .iter()
                    .map(|&f| Self::float32_to_fp8_e4m3(f))
                    .collect();

                Tensor::from_slice(&fp8_data, shape, tensor.device())
            }
            _ => Err(candle_core::Error::msg(format!(
                "Unsupported dtype for FP8 quantization: {:?}",
                flat.dtype()
            ))),
        }
    }

    fn dequantize_from_fp8(tensor: &Tensor) -> Result<Tensor> {
        let shape = tensor.dims();
        let flat = tensor.flatten_all()?;

        let fp8_bytes: Vec<u8> = flat.to_vec1()?;
        let float_data: Vec<half::f16> = fp8_bytes
            .iter()
            .map(|&b| Self::fp8_e4m3_to_float16(b))
            .collect();

        let result = Tensor::from_slice(&float_data, shape, tensor.device())?;
        result.to_dtype(DType::F16)
    }

    fn float32_to_fp8_e4m3(value: f32) -> u8 {
        if value.is_nan() {
            return 0x80;
        }
        if value.is_infinite() {
            return if value.is_sign_positive() { 0x7C } else { 0xFC };
        }
        if value.abs() < 0.000_732 {
            return 0u8;
        }

        let sign = u8::from(value.is_sign_negative());
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

    fn fp8_e4m3_to_float16(value: u8) -> half::f16 {
        if value == 0x80 {
            return half::f16::from_f32(f32::NAN);
        }
        if value == 0x7C {
            return half::f16::INFINITY;
        }
        if value == 0xFC {
            return half::f16::NEG_INFINITY;
        }

        let sign = u16::from((value >> 7) & 0x01);
        let biased_exp = i32::from((value >> 3) & 0x0F);
        let mantissa = u16::from(value & 0x07);

        if biased_exp == 0 && mantissa == 0 {
            return half::f16::from_bits(sign << 15);
        }

        let exp = biased_exp - 7 - 3;

        let bits = (sign << 15) | ((exp + 15) as u16 & 0x7FFF) << 10 | (mantissa << 7);

        half::f16::from_bits(bits)
    }

    #[must_use]
    pub fn estimate_memory_savings(
        num_blocks: usize,
        block_size: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> f32 {
        let fp16_bytes = num_blocks * block_size * num_heads * head_dim * 2;
        let fp8_bytes = num_blocks * block_size * num_heads * head_dim;
        1.0 - (fp8_bytes as f32 / fp16_bytes as f32)
    }
}

fn frexp(value: f32) -> (i32, f32) {
    if value == 0.0 {
        return (0, 0.0);
    }

    let bits = value.to_bits();
    let _ = bits >> 31;
    let mut exp = ((bits >> 23) & 0xFF) as i32;
    let mut mantissa_bits = bits & 0x007F_FFFF;

    if exp == 0 {
        let shift = mantissa_bits.leading_zeros() as i32 - 8;
        exp -= shift;
        mantissa_bits <<= shift + 1;
    }

    exp -= 127;

    let mantissa = 1.0 + (mantissa_bits as f32 / (1 << 23) as f32);

    (exp, mantissa)
}

// Unit tests are extracted to `tests.rs` (sibling) to keep this
// FP8 KV-cache module under the 800-line soft cap. They cover
// dtype basics, the round-trip precision contract (including the
// small-value E4M3 subnormal floor), and the Fp16 pass-through.
#[cfg(test)]
mod tests;
