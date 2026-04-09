//! Normalization utilities.
//!
//! Provides unified RMSNorm and LayerNorm implementations.

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{LayerNorm, Linear};

/// RMSNorm configuration
#[derive(Clone, Debug)]
pub struct RmsNormConfig {
    pub hidden_size: usize,
    pub eps: f64,
}

/// Unified RMSNorm function.
///
/// Handles both 2D [batch * seq, hidden] and 3D [batch, seq, hidden] tensors.
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let dims = x.dims();

    if dims.len() == 3 {
        // 3D tensor: [batch, seq, hidden]
        let (batch, seq, hidden) = x.dims3()?;
        let x_flat = x.reshape((batch * seq, hidden))?;
        let weight_2d = weight.reshape((1, hidden))?;

        let variance = x_flat.sqr()?.mean(1)?;
        let x_normed = x_flat.broadcast_div(&(variance + eps)?.sqrt()?)?;
        let x = x_normed.broadcast_mul(&weight_2d)?;

        x.reshape((batch, seq, hidden))
    } else if dims.len() == 2 {
        // 2D tensor: [batch, hidden] - compute RMS across hidden dimension
        let hidden = dims[1];

        // Compute RMS: sqrt(mean(x^2)) for each row, then normalize
        let x_sq = x.sqr()?;
        let mean_sq = x_sq.mean_keepdim(1)?; // [batch, 1]
        let rms = (mean_sq + eps)?.sqrt()?; // [batch, 1]

        // Normalize: x / rms * weight
        let normalized = x.broadcast_div(&rms)?; // [batch, hidden]
        let weight_broadcast = weight.reshape((1, hidden))?;
        normalized.broadcast_mul(&weight_broadcast)
    } else {
        // Fallback: use standard approach
        let hidden = *dims
            .last()
            .ok_or_else(|| candle_core::Error::msg("Empty tensor"))?;
        let weight_2d = weight.reshape((1, hidden))?;

        let variance = x.sqr()?.mean(1)?;
        let x_normed = x.broadcast_div(&(variance + eps)?.sqrt()?)?;
        x_normed.broadcast_mul(&weight_2d)
    }
}

/// LayerNorm function (standard, not RMS)
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

/// Create a new RMSNorm layer with proper initialization
pub fn rms_norm_layer(hidden_size: usize, eps: f64, device: &Device) -> Result<LayerNorm> {
    let weight = Tensor::ones(hidden_size, DType::F32, device)?;
    let bias = Tensor::zeros(hidden_size, DType::F32, device)?;
    Ok(LayerNorm::new(weight, bias, eps))
}

/// Wrap a Linear layer as RMSNorm weight (for compatibility)
pub fn linear_as_rms_norm(linear: &Linear, eps: f64) -> Result<LayerNorm> {
    let weight = linear.weight().clone();
    let hidden_size = weight
        .dim(0)
        .map_err(|e| candle_core::Error::msg(e.to_string()))?;
    let bias = Tensor::zeros(hidden_size, weight.dtype(), weight.device())?;
    Ok(LayerNorm::new(weight, bias, eps))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

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
