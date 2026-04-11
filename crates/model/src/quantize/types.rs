use candle_core::{DType, Result, Tensor};

#[derive(Debug, Clone)]
pub enum StorageTensor {
    Quantized(QuantizedTensor),
    Fp16(Tensor),
    Fp32(Tensor),
}

#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    pub data: Vec<u8>,
    pub scales: Vec<f32>,
    pub zeros: Option<Vec<f32>>,
    pub format: QuantizationFormat,
    pub shape: Vec<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum QuantizationFormat {
    GgufQ4_K_M,
    GgufQ5_K_M,
    GgufQ8_0,
    GptqQ4,
    AwqQ4,
}

#[derive(Debug, Clone)]
pub struct QuantizationConfig {
    pub format: QuantizationFormat,
    pub block_size: usize,
    pub group_size: usize,
}

impl QuantizedTensor {
    pub fn dequantize_to_f16(&self) -> Result<Tensor> {
        let dequantized = self.dequantize_to_f32()?;
        dequantized.to_dtype(DType::F16)
    }

    pub fn dequantize_to_f32(&self) -> Result<Tensor> {
        let total_elements: usize = self.shape.iter().product();
        let data: Vec<f32> = vec![0.0; total_elements];
        Tensor::from_vec(data, self.shape.clone(), &candle_core::Device::Cpu)
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
