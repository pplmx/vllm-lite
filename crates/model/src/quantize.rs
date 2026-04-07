use candle_core::{Device, Module, Result, Tensor};
use std::collections::HashMap;

pub struct QuantizationCalibrator {
    max_values: HashMap<String, f32>,
}

impl QuantizationCalibrator {
    pub fn new() -> Self {
        Self {
            max_values: HashMap::new(),
        }
    }

    pub fn observe(&mut self, name: &str, tensor: &Tensor) -> Result<()> {
        let data: Vec<f32> = tensor.flatten_all()?.to_vec1()?;
        let max_abs = data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);

        let current_max = self.max_values.get(name).copied().unwrap_or(0.0);
        if max_abs > current_max {
            self.max_values.insert(name.to_string(), max_abs);
        }
        Ok(())
    }

    pub fn compute_scales(&self) -> HashMap<String, f32> {
        self.max_values
            .iter()
            .map(|(k, v)| (k.clone(), v / 127.0))
            .collect()
    }

    pub fn reset(&mut self) {
        self.max_values.clear();
    }
}

impl Default for QuantizationCalibrator {
    fn default() -> Self {
        Self::new()
    }
}

pub struct QuantizedTensor {
    pub data: Vec<i8>,
    pub scale: f32,
    pub shape: Vec<usize>,
}

pub fn quantize(tensor: &Tensor) -> Result<QuantizedTensor> {
    let shape = tensor.dims().to_vec();
    let flat = tensor.flatten_all()?;
    let data_fp32: Vec<f32> = flat.to_vec1()?;

    let max_abs = data_fp32.iter().map(|v| v.abs()).fold(0.0f32, f32::max);

    if max_abs == 0.0 {
        return Ok(QuantizedTensor {
            data: vec![],
            scale: 1.0,
            shape,
        });
    }

    let scale = max_abs / 127.0;
    let data_int8: Vec<i8> = data_fp32
        .iter()
        .map(|v| (v / scale).round() as i8)
        .collect();

    Ok(QuantizedTensor {
        data: data_int8,
        scale,
        shape,
    })
}

pub fn quantize_2d(tensor: &Tensor) -> Result<QuantizedTensor> {
    let shape = tensor.dims().to_vec();
    if shape.len() < 2 {
        return quantize(tensor);
    }

    let data_fp32: Vec<f32> = tensor.to_vec2()?.into_iter().flatten().collect();

    let max_abs = data_fp32.iter().map(|v| v.abs()).fold(0.0f32, f32::max);

    if max_abs == 0.0 {
        return Ok(QuantizedTensor {
            data: vec![],
            scale: 1.0,
            shape,
        });
    }

    let scale = max_abs / 127.0;
    let data_int8: Vec<i8> = data_fp32
        .iter()
        .map(|v| (v / scale).round() as i8)
        .collect();

    Ok(QuantizedTensor {
        data: data_int8,
        scale,
        shape,
    })
}

pub fn dequantize(quant: &QuantizedTensor) -> Result<Tensor> {
    if quant.data.is_empty() {
        let shape: Vec<usize> = quant.shape.clone();
        return Tensor::zeros(shape, candle_core::DType::F32, &Device::Cpu);
    }

    let data_fp32: Vec<f32> = quant.data.iter().map(|&v| v as f32 * quant.scale).collect();

    Tensor::from_slice(&data_fp32, &quant.shape[..], &Device::Cpu)
}

pub struct QuantizedLinear {
    weight: QuantizedTensor,
    bias: Option<Tensor>,
}

impl QuantizedLinear {
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Result<Self> {
        let weight = quantize_2d(&weight)?;
        Ok(Self { weight, bias })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let weight_fp32 = dequantize(&self.weight)?;
        let weight_t = weight_fp32.t()?;

        let input_2d = if input.dims().len() == 1 {
            input.unsqueeze(0)?
        } else {
            input.clone()
        };

        let bias = self.bias.clone();
        let result = input_2d.matmul(&weight_t)?;

        if let Some(bias) = bias {
            result.broadcast_add(&bias)
        } else {
            if input.dims().len() == 1 {
                result.squeeze(0)
            } else {
                Ok(result)
            }
        }
    }
}

impl Module for QuantizedLinear {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_dequantize() -> Result<()> {
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let tensor = Tensor::from_slice(&data, &[4, 4], &Device::Cpu)?;

        let quantized = quantize_2d(&tensor)?;
        assert_eq!(quantized.shape, vec![4, 4]);
        assert!(quantized.scale > 0.0);

        let dequantized = dequantize(&quantized)?;
        let dequantized_data: Vec<f32> = dequantized.flatten_all()?.to_vec1()?;

        for (orig, deq) in data.iter().zip(dequantized_data.iter()) {
            let diff = (orig - deq).abs();
            assert!(
                diff < 1.0,
                "Expected diff < 1.0, got {} for {} vs {}",
                diff,
                orig,
                deq
            );
        }

        Ok(())
    }

    #[test]
    fn test_quantized_linear() -> Result<()> {
        let weight_data: Vec<f32> = (0..6).map(|i| i as f32 * 0.1).collect();
        let weight = Tensor::from_slice(&weight_data, &[2, 3], &Device::Cpu)?;

        let qlinear = QuantizedLinear::new(weight, None)?;

        let input: Vec<f32> = vec![1.0, 2.0, 3.0];
        let input = Tensor::from_slice(&input, &[3], &Device::Cpu)?;
        let output = qlinear.forward(&input)?;

        assert_eq!(output.dims(), &[2]);

        Ok(())
    }

    #[test]
    fn test_quantize_zeros() -> Result<()> {
        let tensor = Tensor::zeros(&[4, 4], candle_core::DType::F32, &Device::Cpu)?;
        let quantized = quantize_2d(&tensor)?;

        assert!(quantized.data.is_empty());
        assert_eq!(quantized.scale, 1.0);

        Ok(())
    }

    #[test]
    fn test_quantization_calibrator() -> Result<()> {
        let mut calibrator = QuantizationCalibrator::new();

        let t1_data: Vec<f32> = vec![1.0, 2.0, 3.0];
        let t1 = Tensor::from_slice(&t1_data, &[3], &Device::Cpu)?;
        calibrator.observe("layer1", &t1)?;

        let t2_data: Vec<f32> = vec![5.0, 6.0];
        let t2 = Tensor::from_slice(&t2_data, &[2], &Device::Cpu)?;
        calibrator.observe("layer1", &t2)?;

        let scales = calibrator.compute_scales();

        let layer1_scale = scales.get("layer1").unwrap();
        assert!(*layer1_scale > 0.0);

        let t3_data: Vec<f32> = vec![10.0];
        let t3 = Tensor::from_slice(&t3_data, &[1], &Device::Cpu)?;
        calibrator.observe("layer2", &t3)?;

        assert_eq!(calibrator.max_values.len(), 2);

        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FP8Format {
    E4M3,
    E5M2,
}

impl FP8Format {
    pub fn max_value(&self) -> f32 {
        match self {
            FP8Format::E4M3 => 240.0,
            FP8Format::E5M2 => 57344.0,
        }
    }

    pub fn exponent_bits(&self) -> u32 {
        match self {
            FP8Format::E4M3 => 4,
            FP8Format::E5M2 => 5,
        }
    }

    pub fn mantissa_bits(&self) -> u32 {
        match self {
            FP8Format::E4M3 => 3,
            FP8Format::E5M2 => 2,
        }
    }
}

pub struct FP8Tensor {
    pub data: Vec<u8>,
    pub scale: f32,
    pub format: FP8Format,
    pub shape: Vec<usize>,
}

fn float_to_fp8(value: f32, format: FP8Format) -> u8 {
    if value == 0.0 {
        return 0;
    }

    let sign = if value < 0.0 { 1u8 } else { 0u8 };
    let abs_value = value.abs();

    let (exp_bits, mantissa_bits) = match format {
        FP8Format::E4M3 => (4i32, 3i32),
        FP8Format::E5M2 => (5i32, 2i32),
    };

    let max_exp = (1 << exp_bits) - 1;
    let bias = max_exp / 2;

    let log2 = abs_value.log2();
    let mut exp = log2.floor() as i32 + bias;
    let mantissa = abs_value * 2.0f32.powf(log2.fract() - mantissa_bits as f32);

    if exp <= 0 {
        exp = 0;
    } else if exp >= max_exp {
        exp = max_exp;
    }

    let mantissa_int = ((mantissa - 1.0) * (1i32 << mantissa_bits) as f32).round() as u8;

    (sign << (exp_bits + mantissa_bits)) | ((exp as u8) << mantissa_bits) | mantissa_int
}

fn fp8_to_float(byte: u8, format: FP8Format) -> f32 {
    let (exp_bits, mantissa_bits) = match format {
        FP8Format::E4M3 => (4i32, 3i32),
        FP8Format::E5M2 => (5i32, 2i32),
    };

    let max_exp = (1 << exp_bits) - 1;
    let bias = max_exp / 2;

    let sign = (byte >> (exp_bits + mantissa_bits)) & 1;
    let exp = ((byte >> mantissa_bits) & ((1 << exp_bits) - 1)) as i32;
    let mantissa = byte & ((1 << mantissa_bits) - 1);

    if exp == max_exp {
        return f32::INFINITY;
    }

    let significand = 1.0 + (mantissa as f32) / (1i32 << mantissa_bits) as f32;
    let value = significand * 2.0f32.powi(exp - bias);

    if sign == 1 { -value } else { value }
}

pub fn quantize_fp8(tensor: &Tensor, format: FP8Format) -> Result<FP8Tensor> {
    let shape = tensor.dims().to_vec();
    let flat = tensor.flatten_all()?;
    let data_fp32: Vec<f32> = flat.to_vec1()?;

    let max_abs = data_fp32.iter().map(|v| v.abs()).fold(0.0f32, f32::max);

    if max_abs == 0.0 {
        return Ok(FP8Tensor {
            data: vec![],
            scale: 1.0,
            format,
            shape,
        });
    }

    let scale = max_abs / format.max_value();
    let data_fp8: Vec<u8> = data_fp32
        .iter()
        .map(|v| float_to_fp8(v / scale, format))
        .collect();

    Ok(FP8Tensor {
        data: data_fp8,
        scale,
        format,
        shape,
    })
}

pub fn dequantize_fp8(quant: &FP8Tensor) -> Result<Tensor> {
    if quant.data.is_empty() {
        return Tensor::zeros(&quant.shape[..], candle_core::DType::F32, &Device::Cpu);
    }

    let data_fp32: Vec<f32> = quant
        .data
        .iter()
        .map(|&v| fp8_to_float(v, quant.format) * quant.scale)
        .collect();

    Tensor::from_slice(&data_fp32, &quant.shape[..], &Device::Cpu)
}

pub struct FP8Calibrator {
    max_values: HashMap<String, f32>,
}

impl FP8Calibrator {
    pub fn new() -> Self {
        Self {
            max_values: HashMap::new(),
        }
    }

    pub fn observe(&mut self, name: &str, tensor: &Tensor) -> Result<()> {
        let data: Vec<f32> = tensor.flatten_all()?.to_vec1()?;
        let max_abs = data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);

        let current_max = self.max_values.get(name).copied().unwrap_or(0.0);
        if max_abs > current_max {
            self.max_values.insert(name.to_string(), max_abs);
        }
        Ok(())
    }

    pub fn compute_scales(&self, format: FP8Format) -> HashMap<String, f32> {
        self.max_values
            .iter()
            .map(|(k, v)| (k.clone(), v / format.max_value()))
            .collect()
    }
}

impl Default for FP8Calibrator {
    fn default() -> Self {
        Self::new()
    }
}
