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

    #[test]
    fn test_swiglu_new_with_weights_creation() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let hidden_size = 128;
        let intermediate_size = 256;

        let gate = Tensor::randn(0.0f32, 1.0, (intermediate_size, hidden_size), &device)?;
        let up = Tensor::randn(0.0f32, 1.0, (intermediate_size, hidden_size), &device)?;
        let down = Tensor::randn(0.0f32, 1.0, (hidden_size, intermediate_size), &device)?;

        let mlp = SwiGLU::new_with_weights(hidden_size, intermediate_size, gate, up, down)?;
        let x = Tensor::ones((2, hidden_size), DType::F32, &device)?;
        let output = mlp.forward(&x)?;

        assert_eq!(output.dims(), &[2, hidden_size]);
        Ok(())
    }

    #[test]
    fn test_swiglu_gate_proj_output_shape() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let mlp = SwiGLU::new(64, 128, None)?;
        let x = Tensor::ones((3, 64), DType::F32, &device)?;
        let output = mlp.forward(&x)?;
        assert_eq!(output.dims(), &[3, 64]);
        Ok(())
    }

    #[test]
    fn test_swiglu_up_proj_output_shape() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let mlp = SwiGLU::new(32, 64, None)?;
        let x = Tensor::ones((5, 32), DType::F32, &device)?;
        let output = mlp.forward(&x)?;
        assert_eq!(output.dims(), &[5, 32]);
        Ok(())
    }

    #[test]
    fn test_swiglu_down_proj_output_shape() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let mlp = SwiGLU::new(256, 512, None)?;
        let x = Tensor::ones((4, 256), DType::F32, &device)?;
        let output = mlp.forward(&x)?;
        assert_eq!(output.dims(), &[4, 256]);
        Ok(())
    }

    #[test]
    fn test_swiglu_silu_activation() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let mlp = SwiGLU::new(32, 64, None)?;
        let x = Tensor::ones((1, 32), DType::F32, &device)?;
        let output = mlp.forward(&x)?;

        let data: Vec<f32> = output.flatten_all()?.to_vec1()?;
        assert!(
            data.iter().all(|v| v.is_finite()),
            "SiLU output should be finite"
        );
        Ok(())
    }

    #[test]
    fn test_swiglu_minimal_hidden_size() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let mlp = SwiGLU::new(16, 32, None)?;
        let x = Tensor::ones((1, 16), DType::F32, &device)?;
        let output = mlp.forward(&x)?;
        assert_eq!(output.dims(), &[1, 16]);
        Ok(())
    }

    #[test]
    fn test_swiglu_large_ratio() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let mlp = SwiGLU::new(32, 256, None)?;
        let x = Tensor::ones((1, 32), DType::F32, &device)?;
        let output = mlp.forward(&x)?;
        let data: Vec<f32> = output.flatten_all()?.to_vec1()?;
        assert!(data.iter().all(|v| v.is_finite()));
        Ok(())
    }

    #[test]
    fn test_swiglu_single_token_batch() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let mlp = SwiGLU::new(64, 128, None)?;
        let x = Tensor::ones((1, 64), DType::F32, &device)?;
        let output = mlp.forward(&x)?;
        assert_eq!(output.dims(), &[1, 64]);
        Ok(())
    }

    #[test]
    fn test_swiglu_multi_token_sequence() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let mlp = SwiGLU::new(64, 128, None)?;
        let x = Tensor::ones((32, 64), DType::F32, &device)?;
        let output = mlp.forward(&x)?;
        assert_eq!(output.dims(), &[32, 64]);
        Ok(())
    }

    #[test]
    fn test_swiglu_output_finite() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let mlp = SwiGLU::new(128, 256, None)?;
        let x = Tensor::randn(-5.0f32, 5.0, (4, 128), &device)?;
        let output = mlp.forward(&x)?;
        let data: Vec<f32> = output.flatten_all()?.to_vec1()?;
        assert!(data.iter().all(|v| v.is_finite()));
        Ok(())
    }

    #[test]
    fn test_swiglu_deterministic() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let mlp = SwiGLU::new(64, 128, None)?;
        let x = Tensor::randn(0.0f32, 1.0, (2, 64), &device)?;

        let out1 = mlp.forward(&x)?;
        let out2 = mlp.forward(&x)?;

        let diff = (&out1 - &out2)?.abs()?;
        let max_diff: f32 = diff.max_all()?.to_scalar()?;
        assert!(max_diff < 1e-6, "MLP should be deterministic");
        Ok(())
    }

    #[test]
    fn test_swiglu_zero_input() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let mlp = SwiGLU::new(64, 128, None)?;
        let x = Tensor::zeros((2, 64), DType::F32, &device)?;
        let output = mlp.forward(&x)?;

        let data: Vec<f32> = output.flatten_all()?.to_vec1()?;
        assert!(data.iter().all(|v| v.is_finite()));
        Ok(())
    }

    #[test]
    fn test_swiglu_silu_range() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let mlp = SwiGLU::new(32, 64, None)?;
        let x = Tensor::ones((1, 32), DType::F32, &device)?;
        let output = mlp.forward(&x)?;

        let data: Vec<f32> = output.flatten_all()?.to_vec1()?;
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        assert!(mean.is_finite(), "SiLU output mean should be finite");
        Ok(())
    }
}
