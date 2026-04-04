//! Sparse MoE layer for Mixtral.

use candle_core::{Result, Tensor};

pub struct MixtralSparseMoe {
    _gate: Option<Tensor>,
    _num_experts: usize,
    _top_k: usize,
}

impl MixtralSparseMoe {
    pub fn new(num_experts: usize, top_k: usize) -> Self {
        Self {
            _gate: None,
            _num_experts: num_experts,
            _top_k: top_k,
        }
    }

    pub fn forward(&self, _hidden_states: &Tensor) -> Result<Tensor> {
        todo!("MixtralSparseMoe forward not implemented")
    }
}
