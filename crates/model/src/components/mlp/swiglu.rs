use candle_core::{Module, Result, Tensor};
use candle_nn::Linear;

pub fn swiglu_forward(
    x: &Tensor,
    gate_proj: &Linear,
    up_proj: &Linear,
    down_proj: &Linear,
) -> Result<Tensor> {
    let gate = gate_proj.forward(x)?;
    let up = up_proj.forward(x)?;

    let silu = gate.silu()?;
    let activated = silu.broadcast_mul(&up)?;

    down_proj.forward(&activated)
}

pub struct SwiGLU {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl SwiGLU {
    pub fn new(
        hidden_size: usize,
        intermediate_size: usize,
        vb: Option<candle_nn::VarBuilder>,
    ) -> Result<Self> {
        let vb = vb.unwrap_or_else(|| {
            candle_nn::VarBuilder::zeros(candle_core::DType::F32, &candle_core::Device::Cpu)
        });

        let gate_proj = candle_nn::linear(hidden_size, intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = candle_nn::linear(hidden_size, intermediate_size, vb.pp("up_proj"))?;
        let down_proj = candle_nn::linear(intermediate_size, hidden_size, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    pub fn new_with_weights(
        _hidden_size: usize,
        _intermediate_size: usize,
        gate_weight: Tensor,
        up_weight: Tensor,
        down_weight: Tensor,
    ) -> Result<Self> {
        let gate_proj = Linear::new(gate_weight, None);
        let up_proj = Linear::new(up_weight, None);
        let down_proj = Linear::new(down_weight, None);

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        swiglu_forward(x, &self.gate_proj, &self.up_proj, &self.down_proj)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    #[test]
    fn test_swiglu_forward_single_token() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let mlp = SwiGLU::new(256, 512, None)?;

        let x = Tensor::ones((1, 256), DType::F32, &device)?;
        let output = mlp.forward(&x)?;

        assert_eq!(output.dims(), &[1, 256]);
        Ok(())
    }

    #[test]
    fn test_swiglu_forward_batch() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let mlp = SwiGLU::new(256, 512, None)?;

        let x = Tensor::ones((4, 256), DType::F32, &device)?;
        let output = mlp.forward(&x)?;

        assert_eq!(output.dims(), &[4, 256]);
        Ok(())
    }

    #[test]
    fn test_swiglu_output_shape() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let mlp = SwiGLU::new(128, 256, None)?;

        let x = Tensor::zeros((2, 128), DType::F32, &device)?;
        let output = mlp.forward(&x)?;

        assert_eq!(output.dims(), &[2, 128]);
        Ok(())
    }
}
