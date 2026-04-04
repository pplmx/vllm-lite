//! Sparse MoE layer for Mixtral.

use candle_core::{Module, Result, Tensor};
use candle_nn::Linear;

pub struct MixtralSparseMoe {
    experts: Vec<Expert>,
    gate: Linear,
    num_experts: usize,
    top_k: usize,
}

struct Expert {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Expert {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        crate::components::swiglu_forward(x, &self.gate_proj, &self.up_proj, &self.down_proj)
    }
}

impl MixtralSparseMoe {
    pub fn new(
        hidden_size: usize,
        num_experts: usize,
        expert_intermediate_size: usize,
        top_k: usize,
        vb: candle_nn::VarBuilder,
    ) -> Result<Self> {
        let mut experts = Vec::new();
        for i in 0..num_experts {
            let vb = vb.pp(format!("expert_{}", i));
            let gate_proj =
                candle_nn::linear(hidden_size, expert_intermediate_size, vb.pp("gate_proj"))?;
            let up_proj =
                candle_nn::linear(hidden_size, expert_intermediate_size, vb.pp("up_proj"))?;
            let down_proj =
                candle_nn::linear(expert_intermediate_size, hidden_size, vb.pp("down_proj"))?;
            experts.push(Expert {
                gate_proj,
                up_proj,
                down_proj,
            });
        }

        let gate = candle_nn::linear(hidden_size, num_experts, vb.pp("gate"))?;

        Ok(Self {
            experts,
            gate,
            num_experts,
            top_k,
        })
    }

    pub fn new_with_weights(
        _hidden_size: usize,
        num_experts: usize,
        _expert_intermediate_size: usize,
        top_k: usize,
        gate_weight: Tensor,
        expert_weights: Vec<(Tensor, Tensor, Tensor)>,
    ) -> Result<Self> {
        if expert_weights.len() != num_experts {
            return Err(candle_core::Error::msg(format!(
                "Expected {} expert weights, got {}",
                num_experts,
                expert_weights.len()
            )));
        }

        let mut experts = Vec::new();
        for (gate_w, up_w, down_w) in expert_weights {
            let gate_proj = Linear::new(gate_w, None);
            let up_proj = Linear::new(up_w, None);
            let down_proj = Linear::new(down_w, None);
            experts.push(Expert {
                gate_proj,
                up_proj,
                down_proj,
            });
        }

        let gate = Linear::new(gate_weight, None);

        Ok(Self {
            experts,
            gate,
            num_experts,
            top_k,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, seq, hidden) = x.dims3()?;

        let gate_logits = self.gate.forward(x)?;

        let mut outputs = Vec::new();

        for b in 0..batch {
            for s in 0..seq {
                let token_logits_slice = gate_logits.narrow(0, b, 1)?.narrow(1, s, 1)?;
                let token_logits = token_logits_slice.reshape((self.num_experts,))?;

                let mut top_experts: Vec<(f32, usize)> = Vec::new();
                for e in 0..self.num_experts {
                    let weight = token_logits
                        .narrow(0, e, 1)?
                        .squeeze(0)?
                        .to_scalar::<f32>()?;
                    top_experts.push((weight, e));
                }
                top_experts
                    .sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                top_experts.truncate(self.top_k);

                let token_hidden_slice = x.narrow(0, b, 1)?.narrow(1, s, 1)?;
                let token_hidden = token_hidden_slice.reshape((hidden,))?.unsqueeze(0)?;
                let mut weighted_sum = Tensor::zeros((hidden,), x.dtype(), x.device())?;

                for (weight, expert_idx) in top_experts {
                    let expert_out = self.experts[expert_idx].forward(&token_hidden)?;
                    let weight_tensor =
                        Tensor::full(weight, expert_out.shape(), expert_out.device())?;
                    let weighted = expert_out.mul(&weight_tensor)?;
                    weighted_sum = weighted_sum.add(&weighted.squeeze(0)?)?;
                }

                outputs.push(weighted_sum);
            }
        }

        let output = Tensor::stack(&outputs, 0)?.reshape((batch, seq, hidden))?;
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    #[test]
    fn test_sparse_moe_creation() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let moe = MixtralSparseMoe::new(
            4096,
            8,
            14336,
            2,
            candle_nn::VarBuilder::zeros(DType::F32, &device),
        )?;

        assert_eq!(moe.num_experts, 8);
        assert_eq!(moe.top_k, 2);
        Ok(())
    }

    #[test]
    fn test_sparse_moe_forward_shape() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let moe = MixtralSparseMoe::new(
            256,
            4,
            512,
            2,
            candle_nn::VarBuilder::zeros(DType::F32, &device),
        )?;

        let x = Tensor::ones((2, 3, 256), DType::F32, &device)?;
        let output = moe.forward(&x)?;

        assert_eq!(output.dims(), &[2, 3, 256]);
        Ok(())
    }
}
