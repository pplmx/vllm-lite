use candle_core::{Module, Result, Tensor};
use candle_nn::Linear;

pub struct SwiGLU {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl SwiGLU {
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

        let silu = gate.silu()?;
        let activated = silu.broadcast_mul(&up)?;

        self.down_proj.forward(&activated)
    }
}
