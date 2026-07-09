#![allow(clippy::module_name_repetitions)]
//! Sparse `MoE` layer for Mixtral.

use candle_core::{D, Module, Result, Tensor};
use candle_nn::Linear;
use tracing::trace;

#[derive(Debug)]
/// `MixtralSparseMoe`. See the type definition for fields and behavior.
pub struct MixtralSparseMoe {
    experts: Vec<Expert>,
    gate: Linear,
    num_experts: usize,
    top_k: usize,
}

#[derive(Debug)]
struct Expert {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl Expert {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        crate::components::mlp::swiglu::swiglu_forward(x, &self.gate, &self.up, &self.down)
    }
}

struct ExpertRoute {
    token_idx: u32,
    weight: f32,
}

/// Softmax gate logits and select top-k experts per token.
fn compute_topk_routing(
    gate_logits: &Tensor,
    num_tokens: usize,
    num_experts: usize,
    top_k: usize,
) -> Result<Vec<Vec<ExpertRoute>>> {
    let routing_weights = candle_nn::ops::softmax(gate_logits, 2)?;
    let flat_weights = routing_weights
        .contiguous()?
        .reshape((num_tokens, num_experts))?;

    let sorted_indices = flat_weights.arg_sort_last_dim(false)?;
    let top_expert_idx = sorted_indices.narrow(1, 0, top_k)?.contiguous()?;
    let top_weights = flat_weights.gather(&top_expert_idx, 1)?;
    let weight_sum = top_weights.sum_keepdim(1)?;
    let top_weights = top_weights.broadcast_div(&weight_sum)?;

    let expert_indices: Vec<Vec<u32>> = top_expert_idx.to_vec2()?;
    let expert_weights: Vec<Vec<f32>> = top_weights.to_vec2()?;

    let mut routes_by_expert: Vec<Vec<ExpertRoute>> =
        (0..num_experts).map(|_| Vec::new()).collect();
    for (token_idx, (indices, weights)) in
        expert_indices.into_iter().zip(expert_weights).enumerate()
    {
        for (expert_idx, weight) in indices.into_iter().zip(weights) {
            // invariant: expert_idx is bounded by the configured expert count
            // (small architectural constant); usize truncation is not reachable.
            #[allow(clippy::cast_possible_truncation)]
            let route = ExpertRoute {
                token_idx: u32::try_from(token_idx).unwrap_or(0),
                weight,
            };
            routes_by_expert[expert_idx as usize].push(route);
        }
    }

    Ok(routes_by_expert)
}

impl MixtralSparseMoe {
    /// Construct a new instance from the given configuration.
    /// # Errors
    ///
    /// Returns `Err` if any required tensor allocation or weight loading fails.
    #[allow(clippy::needless_pass_by_value)]
    pub fn new(
        hidden_size: usize,
        num_experts: usize,
        expert_intermediate_size: usize,
        top_k: usize,
        vb: candle_nn::VarBuilder<'_>,
    ) -> Result<Self> {
        let mut experts = Vec::new();
        for i in 0..num_experts {
            let vb = vb.pp(format!("expert_{i}"));
            let gate_proj =
                candle_nn::linear(hidden_size, expert_intermediate_size, vb.pp("gate_proj"))?;
            let up_proj =
                candle_nn::linear(hidden_size, expert_intermediate_size, vb.pp("up_proj"))?;
            let down_proj =
                candle_nn::linear(expert_intermediate_size, hidden_size, vb.pp("down_proj"))?;
            experts.push(Expert {
                gate: gate_proj,
                up: up_proj,
                down: down_proj,
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

    /// Construct a new instance from already-loaded weight tensors.
    /// # Errors
    ///
    /// Returns `Err` if any required tensor allocation or weight loading fails.
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
                gate: gate_proj,
                up: up_proj,
                down: down_proj,
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

    /// Run the layer forward pass over the input.
    /// # Errors
    ///
    /// Returns `Err` if any tensor operation fails (shape mismatch, out-of-memory, dtype incompatibility, or kernel error).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let squeeze_seq = x.dims().len() == 2;
        let x = if squeeze_seq {
            x.unsqueeze(1)?
        } else {
            x.clone()
        };

        let (batch, seq, hidden) = x.dims3()?;
        let num_tokens = batch * seq;

        trace!(
            batch_size = batch,
            seq_len = seq,
            num_experts = self.num_experts,
            top_k = self.top_k,
            "MoE forward"
        );

        let gate_logits = self.gate.forward(&x)?;
        let routes_by_expert =
            compute_topk_routing(&gate_logits, num_tokens, self.num_experts, self.top_k)?;

        let flat_x = x.reshape((num_tokens, hidden))?;
        let mut output = Tensor::zeros((num_tokens, hidden), x.dtype(), x.device())?;

        for (expert_idx, routes) in routes_by_expert.into_iter().enumerate() {
            if routes.is_empty() {
                continue;
            }

            let n = routes.len();
            let token_indices: Vec<u32> = routes.iter().map(|r| r.token_idx).collect();
            let route_weights: Vec<f32> = routes.iter().map(|r| r.weight).collect();

            let idx_tensor = Tensor::new(token_indices.as_slice(), x.device())?;
            let expert_input = flat_x.index_select(&idx_tensor, 0)?;
            let expert_out = self.experts[expert_idx].forward(&expert_input)?;

            let weights = Tensor::new(route_weights.as_slice(), x.device())?.reshape((n, 1))?;
            let weighted = expert_out.broadcast_mul(&weights)?.contiguous()?;
            let scatter_idx = idx_tensor
                .unsqueeze(D::Minus1)?
                .broadcast_as(weighted.shape())?
                .contiguous()?;
            output = output.scatter_add(&scatter_idx, &weighted, 0)?;
        }

        let output = output.reshape((batch, seq, hidden))?;
        if squeeze_seq {
            output.squeeze(1)
        } else {
            Ok(output)
        }
    }

    #[cfg(test)]
    fn forward_naive(&self, x: &Tensor) -> Result<Tensor> {
        let squeeze_seq = x.dims().len() == 2;
        let x = if squeeze_seq {
            x.unsqueeze(1)?
        } else {
            x.clone()
        };

        let (batch, seq, hidden) = x.dims3()?;
        let gate_logits = self.gate.forward(&x)?;

        let mut outputs = Vec::new();

        for b in 0..batch {
            for s in 0..seq {
                let token_logits_slice = gate_logits.narrow(0, b, 1)?.narrow(1, s, 1)?;
                let token_logits = token_logits_slice.reshape((self.num_experts,))?;
                let routing_weights = candle_nn::ops::softmax(&token_logits, 0)?;

                let mut top_experts: Vec<(f32, usize)> = Vec::new();
                for e in 0..self.num_experts {
                    let weight = routing_weights
                        .narrow(0, e, 1)?
                        .squeeze(0)?
                        .to_scalar::<f32>()?;
                    top_experts.push((weight, e));
                }
                top_experts
                    .sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                top_experts.truncate(self.top_k);

                let weight_sum: f32 = top_experts.iter().map(|(w, _)| *w).sum();
                if weight_sum > 0.0 {
                    for entry in &mut top_experts {
                        entry.0 /= weight_sum;
                    }
                }

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
        if squeeze_seq {
            output.squeeze(1)
        } else {
            Ok(output)
        }
    }
}

// Unit tests are extracted to `tests.rs` (sibling) to keep this
// MoE module under the 800-line soft cap. They cover shape
// preservation (prefill `[B, T, H]` and decode `[B, H]`) and
// vectorized-vs-naive numerical equivalence for both prefill and
// decode inputs.
#[cfg(test)]
mod tests;
