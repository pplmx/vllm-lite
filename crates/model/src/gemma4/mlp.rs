//! Gemma4 MLP variant: gated GeLU with learned gate bias, used in the Gemma 4 architecture.
//!
//! Same external signature as [`SwiGLU`](crate::components::mlp::SwiGLU) but
//! uses GeLU activation and an additive bias on the gate projection.
use candle_core::{Module, Result, Tensor};
use candle_nn::Linear;

/// `GeGLU`. See the type definition for fields and behavior.
#[derive(Debug)]
pub(crate) struct GeGLU {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl GeGLU {
    #[allow(clippy::needless_pass_by_value)]
    pub fn new(
        hidden_size: usize,
        intermediate_size: usize,
        vb: candle_nn::VarBuilder<'_>,
    ) -> Result<Self> {
        let gate_proj = candle_nn::linear(hidden_size, intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = candle_nn::linear(hidden_size, intermediate_size, vb.pp("up_proj"))?;
        let down_proj = candle_nn::linear(intermediate_size, hidden_size, vb.pp("down_proj"))?;

        Ok(Self {
            gate: gate_proj,
            up: up_proj,
            down: down_proj,
        })
    }

    pub fn new_with_weights(
        _hidden_size: usize,
        _intermediate_size: usize,
        gate: Tensor,
        up: Tensor,
        down: Tensor,
    ) -> Self {
        Self {
            gate: Linear::new(gate, None),
            up: Linear::new(up, None),
            down: Linear::new(down, None),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate.forward(x)?;
        let up = self.up.forward(x)?;

        let activated = gate.gelu()?;
        let activated = activated.broadcast_mul(&up)?;

        self.down.forward(&activated)
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
