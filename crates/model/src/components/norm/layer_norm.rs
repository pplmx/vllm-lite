//! LayerNorm implementation with weight and bias.

use candle_core::{Module, Result, Tensor};
use candle_nn::LayerNorm;

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
