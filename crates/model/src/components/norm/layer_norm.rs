#![allow(clippy::module_name_repetitions)]
//! `LayerNorm` implementation with weight and bias.

use candle_core::{Module, Result, Tensor};
use candle_nn::LayerNorm;

#[derive(Debug)]
/// `LnLayerNorm`. See the type definition for fields and behavior.
pub struct LnLayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl LnLayerNorm {
    #[must_use]
    pub const fn new(weight: Tensor, bias: Tensor, eps: f64) -> Self {
        Self { weight, bias, eps }
    }

    /// Run the layer forward pass over the input.
    /// # Errors
    ///
    /// Returns `Err` if any tensor operation fails (shape mismatch, out-of-memory, dtype incompatibility, or kernel error).
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

impl Module for LnLayerNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        Self::forward(self, xs)
    }
}

/// Run the operation (see signature for params and return type).
/// # Errors
///
/// Returns `Err` if the operation fails.
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

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    #[test]
    fn test_layer_norm_shape_preserved() {
        let weight = Tensor::ones((32,), DType::F32, &candle_core::Device::Cpu).unwrap();
        let bias = Tensor::zeros((32,), DType::F32, &candle_core::Device::Cpu).unwrap();
        let ln = LnLayerNorm::new(weight, bias, 1e-6);

        let input = Tensor::ones((2, 10, 32), DType::F32, &candle_core::Device::Cpu).unwrap();
        let output = ln.forward(&input).unwrap();

        assert_eq!(output.dims(), &[2, 10, 32]);
    }
}
