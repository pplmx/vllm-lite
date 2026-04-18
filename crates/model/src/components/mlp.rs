//! MLP layer trait for component abstraction.

use candle_core::{Module, Result, Tensor};
use candle_nn::Linear;

pub trait MlpLayer: Send + Sync {
    fn forward(&self, x: &Tensor) -> Result<Tensor>;
}

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
