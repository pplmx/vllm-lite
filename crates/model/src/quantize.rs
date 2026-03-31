use candle_core::{Device, Module, Result, Tensor};

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
}
