//! full: full.

#![allow(non_snake_case)]
//! Full (MRoPE + paged GQA) attention block for Qwen3.5 hybrid layers.

use std::collections::HashMap;

use crate::components::SwiGLU;
use crate::components::positional::MRoPE;
use crate::paged_tensor::PagedKvCache;
use crate::qwen3_5::attention35::Attention35WithRoPE;
use candle_core::{Module, Result as CandleResult, Tensor};
use candle_nn::{LayerNorm, Linear, VarBuilder};

/// FullAttentionBlock35: full attention block35.
pub struct FullAttentionBlock35 {
    input_ln: LayerNorm,
    self_attn: Attention35WithRoPE,
    mlp: SwiGLU,
    post_attn_ln: LayerNorm,
    gate: Option<Linear>,
}

impl FullAttentionBlock35 {
    /// new: new.
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
        eps: f64,
        rope: MRoPE,
        vb: VarBuilder,
    ) -> CandleResult<Self> {
        let input_ln = candle_nn::layer_norm(hidden_size, eps, vb.clone())?;
        let self_attn = Attention35WithRoPE::new(
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            rope,
            vb.clone(),
        )?;
        let mlp = SwiGLU::new(hidden_size, intermediate_size, Some(vb.clone()))?;
        let post_attn_ln = candle_nn::layer_norm(hidden_size, eps, vb)?;

        Ok(Self {
            input_ln,
            self_attn,
            mlp,
            post_attn_ln,
            gate: None,
        })
    }

    /// with_attn_gate: with attn gate.
    pub fn with_attn_gate(mut self, gate: Linear) -> Self {
        self.gate = Some(gate);
        self
    }

    /// forward: forward.
    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        self.forward_with_attn(x, |x| self.self_attn.forward(x))
    }

    /// forward_prefill: forward prefill.
    pub fn forward_prefill(
        &self,
        x: &Tensor,
        kv_cache: &mut PagedKvCache,
        layer_idx: usize,
        block_ids: &[usize],
        positions: &[usize],
    ) -> CandleResult<Tensor> {
        self.forward_with_attn(x, |x| {
            self.self_attn
                .forward_prefill(x, kv_cache, layer_idx, block_ids, positions)
        })
    }

    /// forward_decode: forward decode.
    pub fn forward_decode(
        &self,
        x: &Tensor,
        kv_cache: &mut PagedKvCache,
        layer_idx: usize,
        block_ids: &[usize],
        num_computed_tokens: usize,
        positions: &[usize],
    ) -> CandleResult<Tensor> {
        self.forward_with_attn(x, |x| {
            self.self_attn.forward_decode(
                x,
                kv_cache,
                layer_idx,
                block_ids,
                num_computed_tokens,
                positions,
            )
        })
    }

    fn forward_with_attn<F>(&self, x: &Tensor, attn_fn: F) -> CandleResult<Tensor>
    where
        F: FnOnce(&Tensor) -> CandleResult<Tensor>,
    {
        let residual = x.clone();
        let x = self.input_ln.forward(x)?;

        let mut attn_out = attn_fn(&x)?;

        if let Some(ref g) = self.gate {
            let g_val = g.forward(&residual)?;
            attn_out = attn_out.broadcast_mul(&g_val)?;
        }

        let x = (x + attn_out)?;
        let x = (x + residual)?;

        let residual = x.clone();
        let x = self.post_attn_ln.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        x + residual
    }
}

impl FullAttentionBlock35 {
    /// from_weights: from weights.
    pub fn from_weights(
        prefix: &str,
        weights: &HashMap<String, Tensor>,
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
        eps: f64,
        rope: MRoPE,
    ) -> CandleResult<Self> {
        let input_ln_key = format!("{}.input_layernorm.weight", prefix);
        let input_ln_w = weights
            .get(&input_ln_key)
            .cloned()
            .ok_or_else(|| candle_core::Error::msg(format!("Missing {}", input_ln_key)))?;
        let input_ln_bias = Tensor::zeros(
            input_ln_w.dim(0).unwrap_or(hidden_size),
            input_ln_w.dtype(),
            input_ln_w.device(),
        )?;
        let input_ln = LayerNorm::new(input_ln_w, input_ln_bias, eps);

        let self_attn = Attention35WithRoPE::from_weights(
            prefix,
            weights,
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            rope,
        )?;

        let gate_proj_key = format!("{}.mlp.gate_proj.weight", prefix);
        let up_proj_key = format!("{}.mlp.up_proj.weight", prefix);
        let down_proj_key = format!("{}.mlp.down_proj.weight", prefix);

        let gate_proj_w = weights
            .get(&gate_proj_key)
            .cloned()
            .ok_or_else(|| candle_core::Error::msg(format!("Missing {}", gate_proj_key)))?;
        let up_proj_w = weights
            .get(&up_proj_key)
            .cloned()
            .ok_or_else(|| candle_core::Error::msg(format!("Missing {}", up_proj_key)))?;
        let down_proj_w = weights
            .get(&down_proj_key)
            .cloned()
            .ok_or_else(|| candle_core::Error::msg(format!("Missing {}", down_proj_key)))?;

        let mlp = SwiGLU::new_with_weights(
            hidden_size,
            intermediate_size,
            gate_proj_w,
            up_proj_w,
            down_proj_w,
        )?;

        let post_ln_key = format!("{}.post_attention_layernorm.weight", prefix);
        let post_ln_w = weights
            .get(&post_ln_key)
            .cloned()
            .ok_or_else(|| candle_core::Error::msg(format!("Missing {}", post_ln_key)))?;
        let post_ln_bias = Tensor::zeros(
            post_ln_w.dim(0).unwrap_or(hidden_size),
            post_ln_w.dtype(),
            post_ln_w.device(),
        )?;
        let post_attn_ln = LayerNorm::new(post_ln_w, post_ln_bias, eps);

        Ok(Self {
            input_ln,
            self_attn,
            mlp,
            post_attn_ln,
            gate: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;

    #[test]
    fn test_full_attention_block_residual_connection() {
        let device = Device::Cpu;
        let rope = MRoPE::new(32, 10000.0, vec![10, 10, 12], 0.25);
        let block = FullAttentionBlock35::new(
            128,
            2,
            2,
            32,
            256,
            1e-6,
            rope,
            VarBuilder::zeros(DType::F32, &device),
        )
        .unwrap();

        let x = Tensor::zeros((1, 2, 128), DType::F32, &device).unwrap();
        let out = block.forward(&x).unwrap();

        assert_eq!(out.dims(), &[1, 2, 128]);
    }
}
