use candle_core::{Device, Result, Tensor};

pub struct QuantizedTensor {
    pub data: Vec<i8>,
    pub scale: f32,
    pub shape: Vec<usize>,
}

pub fn quantize(tensor: &Tensor) -> Result<QuantizedTensor> {
    let shape = tensor.dims().to_vec();
    let data_fp32: Vec<Vec<Vec<f32>>> = tensor.to_vec3()?;

    let max_abs = data_fp32
        .iter()
        .flat_map(|b: &Vec<Vec<f32>>| b.iter().flat_map(|h: &Vec<f32>| h.iter()))
        .map(|v: &f32| v.abs())
        .fold(0.0f32, |a: f32, b: f32| a.max(b));

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
        .flat_map(|b: &Vec<Vec<f32>>| b.iter().flat_map(|h: &Vec<f32>| h.iter()))
        .map(|v: &f32| (v / scale).round() as i8)
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
