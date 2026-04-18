//! Normalization utilities.
//!
//! Provides unified RMSNorm and LayerNorm implementations.

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::LayerNorm;

pub trait NormLayer: Send + Sync {
    fn forward(&self, x: &Tensor) -> Result<Tensor>;
    fn hidden_size(&self) -> usize;
}

#[derive(Clone, Debug)]
pub struct RmsNormConfig {
    pub hidden_size: usize,
    pub eps: f64,
}

pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    pub fn new(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims();

        if dims.len() == 3 {
            let (batch, seq, hidden) = x.dims3()?;
            let x_flat = x.reshape((batch * seq, hidden))?;
            let weight_2d = self.weight.reshape((1, hidden))?;

            let variance = x_flat.sqr()?.mean_keepdim(1)?;
            let x_normed = x_flat.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
            let x = x_normed.broadcast_mul(&weight_2d)?;

            x.reshape((batch, seq, hidden))
        } else {
            let hidden = *dims.last().ok_or_else(|| candle_core::Error::msg("Empty tensor"))?;
            let weight_2d = self.weight.reshape((1, hidden))?;

            let variance = x.sqr()?.mean_keepdim(1)?;
            let x_normed = x.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
            x_normed.broadcast_mul(&weight_2d)
        }
    }
}

impl NormLayer for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward(x)
    }

    fn hidden_size(&self) -> usize {
        self.weight.dims().last().copied().unwrap_or(0)
    }
}

pub struct LnLayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl LnLayerNorm {
    pub fn new(weight: Tensor, bias: Tensor, eps: f64) -> Self {
        Self { weight, bias, eps }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims();

        if dims.len() == 3 {
            let (batch, seq, hidden) = x.dims3()?;
            let x = x.reshape((batch * seq, hidden))?;
            let norm = LayerNorm::new(self.weight.clone(), self.bias.clone(), self.eps);
            let x = norm.forward(&x)?;
            x.reshape((batch, seq, hidden))
        } else {
            let norm = LayerNorm::new(self.weight.clone(), self.bias.clone(), self.eps);
            norm.forward(x)
        }
    }
}

impl NormLayer for LnLayerNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward(x)
    }

    fn hidden_size(&self) -> usize {
        self.weight.dims().last().copied().unwrap_or(0)
    }
}

pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let dims = x.dims();

    if dims.len() == 3 {
        let (batch, seq, hidden) = x.dims3()?;
        let x_flat = x.reshape((batch * seq, hidden))?;
        let weight_2d = weight.reshape((1, hidden))?;

        let variance = x_flat.sqr()?.mean(1)?;
        let x_normed = x_flat.broadcast_div(&(variance + eps)?.sqrt()?)?;
        let x = x_normed.broadcast_mul(&weight_2d)?;

        x.reshape((batch, seq, hidden))
    } else if dims.len() == 2 {
        let hidden = dims[1];

        let x_sq = x.sqr()?;
        let mean_sq = x_sq.mean_keepdim(1)?;
        let rms = (mean_sq + eps)?.sqrt()?;

        let normalized = x.broadcast_div(&rms)?;
        let weight_broadcast = weight.reshape((1, hidden))?;
        normalized.broadcast_mul(&weight_broadcast)
    } else {
        let hidden = *dims.last().ok_or_else(|| candle_core::Error::msg("Empty tensor"))?;
        let weight_2d = weight.reshape((1, hidden))?;

        let variance = x.sqr()?.mean(1)?;
        let x_normed = x.broadcast_div(&(variance + eps)?.sqrt()?)?;
        x_normed.broadcast_mul(&weight_2d)
    }
}

pub fn layer_norm(x: &Tensor, weight: &Tensor, bias: &Tensor, eps: f64) -> Result<Tensor> {
    let dims = x.dims();

    if dims.len() == 3 {
        let (batch, seq, hidden) = x.dims3()?;
        let x = x.reshape((batch * seq, hidden))?;
        let norm = LayerNorm::new(weight.clone(), bias.clone(), eps);
        let x = norm.forward(&x)?;
        x.reshape((batch, seq, hidden))
    } else {
        let norm = LayerNorm::new(weight.clone(), bias.clone(), eps);
        norm.forward(x)
    }
}

pub fn rms_norm_layer(hidden_size: usize, eps: f64, device: &Device) -> Result<LnLayerNorm> {
    let weight = Tensor::ones(hidden_size, DType::F32, device)?;
    let bias = Tensor::zeros(hidden_size, DType::F32, device)?;
    Ok(LnLayerNorm::new(weight, bias, eps))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    #[test]
    fn test_rms_norm_shape_preserved() {
        let weight = Tensor::ones((32,), DType::F32, &Device::Cpu).unwrap();
        let rms = RmsNorm::new(weight, 1e-6);

        let input = Tensor::ones((2, 10, 32), DType::F32, &Device::Cpu).unwrap();
        let output = rms.forward(&input).unwrap();

        assert_eq!(output.dims(), &[2, 10, 32]);
    }

    #[test]
    fn test_layer_norm_shape_preserved() {
        let weight = Tensor::ones((32,), DType::F32, &Device::Cpu).unwrap();
        let bias = Tensor::zeros((32,), DType::F32, &Device::Cpu).unwrap();
        let ln = LnLayerNorm::new(weight, bias, 1e-6);

        let input = Tensor::ones((2, 10, 32), DType::F32, &Device::Cpu).unwrap();
        let output = ln.forward(&input).unwrap();

        assert_eq!(output.dims(), &[2, 10, 32]);
    }

    #[test]
    fn test_rms_norm_2d() {
        let device = Device::Cpu;
        let x = Tensor::new(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device).unwrap();
        let weight = Tensor::new(&[1.0, 1.0, 1.0], &device).unwrap();

        let result = rms_norm(&x, &weight, 1e-5).unwrap();
        assert_eq!(result.dims(), &[2, 3]);
    }

    #[test]
    fn test_rms_norm_3d() {
        let device = Device::Cpu;
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[1, 2, 2], &device).unwrap();
        let weight = Tensor::new(&[1.0f32, 1.0], &device).unwrap();

        let result = rms_norm(&x, &weight, 1e-5).unwrap();
        assert_eq!(result.dims(), &[1, 2, 2]);
    }
}
