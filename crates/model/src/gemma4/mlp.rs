#![allow(dead_code)]

use candle_core::{Module, Result, Tensor};
use candle_nn::Linear;

pub struct GeGLU {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl GeGLU {
    pub fn new(
        hidden_size: usize,
        intermediate_size: usize,
        vb: candle_nn::VarBuilder,
    ) -> Result<Self> {
        let gate_proj = candle_nn::linear(hidden_size, intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = candle_nn::linear(hidden_size, intermediate_size, vb.pp("up_proj"))?;
        let down_proj = candle_nn::linear(intermediate_size, hidden_size, vb.pp("down_proj"))?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;

        let activated = gate.gelu()?;
        let activated = activated.broadcast_mul(&up)?;

        self.down_proj.forward(&activated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    #[test]
    fn test_gelu_forward_single_token() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let mlp = GeGLU::new(256, 512, candle_nn::VarBuilder::zeros(DType::F32, &device))?;

        let x = Tensor::ones((1, 256), DType::F32, &device)?;
        let output = mlp.forward(&x)?;

        assert_eq!(output.dims(), &[1, 256]);
        Ok(())
    }

    #[test]
    fn test_gelu_forward_batch() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let mlp = GeGLU::new(256, 512, candle_nn::VarBuilder::zeros(DType::F32, &device))?;

        let x = Tensor::ones((4, 256), DType::F32, &device)?;
        let output = mlp.forward(&x)?;

        assert_eq!(output.dims(), &[4, 256]);
        Ok(())
    }

    #[test]
    fn test_gelu_output_shape() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let mlp = GeGLU::new(128, 256, candle_nn::VarBuilder::zeros(DType::F32, &device))?;

        let x = Tensor::zeros((2, 128), DType::F32, &device)?;
        let output = mlp.forward(&x)?;

        assert_eq!(output.dims(), &[2, 128]);
        Ok(())
    }
}
