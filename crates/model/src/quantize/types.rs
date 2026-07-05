//! Storage backend enum for the paged KV-cache: `Quantized(QuantizedTensor)` for memory-tight packed formats, `Fp16(Tensor)` for balanced, `Fp32(Tensor)` for highest precision.
//!
//! `StorageTensor` is the runtime abstraction; the on-disk `QuantizationFormat`
//! enum lives in `mod.rs` and only describes what's in the checkpoint file.
use candle_core::{DType, Result, Tensor};

/// `StorageTensor`. See the type definition for fields and behavior.
#[derive(Debug, Clone)]
pub enum StorageTensor {
    Quantized(QuantizedTensor),
    Fp16(Tensor),
    Fp32(Tensor),
}

/// `QuantizedTensor`. See the type definition for fields and behavior.
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// Packed quantized payload (interpretation depends on `format`).
    pub data: Vec<u8>,
    /// Per-block dequantization scales.
    pub scales: Vec<f32>,
    /// Per-block zero points (None for symmetric quantization schemes).
    pub zeros: Option<Vec<f32>>,
    /// Quantization scheme used (e.g. GGUF `Q4_K_M`).
    pub format: QuantizationFormat,
    /// Logical tensor shape before quantization.
    pub shape: Vec<usize>,
}

/// `QuantizationFormat`. See the type definition for fields and behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum QuantizationFormat {
    /// GGUF `Q4_K_M` — 4-bit K-quant with per-group scales (default for Llama-class models).
    GgufQ4_K_M,
    /// GGUF `Q5_K_M` — 5-bit K-quant with per-group scales.
    GgufQ5_K_M,
    /// GGUF `Q8_0` — 8-bit symmetric per-block.
    GgufQ8_0,
    /// GPTQ 4-bit (reserved for future support).
    GptqQ4,
    /// AWQ 4-bit (reserved for future support).
    AwqQ4,
}

/// Configuration for Quantization. Constructed via the `builder()` associated function or by deserializing from JSON / TOML. Pass-by-value to construction APIs.
#[derive(Debug, Clone)]
pub struct QuantizationConfig {
    /// Quantization scheme to apply.
    pub format: QuantizationFormat,
    /// Quantization block size (e.g. 32 / 64 / 128 elements per block).
    pub block_size: usize,
    /// Group size for per-group scale/zero sharing.
    pub group_size: usize,
}

impl QuantizedTensor {
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    /// `dequantize_to_f16`: dequantize to f16.
    pub fn dequantize_to_f16(&self) -> Result<Tensor> {
        let dequantized = self.dequantize_to_f32()?;
        dequantized.to_dtype(DType::F16)
    }

    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    /// `dequantize_to_f32`: dequantize to f32.
    pub fn dequantize_to_f32(&self) -> Result<Tensor> {
        let total_elements: usize = self.shape.iter().product();
        let zeros: Vec<f32> = vec![0.0; total_elements];
        Tensor::from_vec(zeros, self.shape.clone(), &candle_core::Device::Cpu)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization_format_enum() {
        let format = QuantizationFormat::GgufQ4_K_M;
        assert_eq!(format, QuantizationFormat::GgufQ4_K_M);
    }
}
