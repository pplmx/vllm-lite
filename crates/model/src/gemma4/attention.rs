//! Gemma4 Attention implementation.

#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]

use crate::config::architecture::{LayerType, RoPEConfig};
use crate::gemma4::rope::Gemma4RoPE;
use candle_core::{Module, Result, Tensor};
use candle_nn::Linear;

pub struct Gemma4Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    sliding_window: usize,
    layer_type: LayerType,
    rope: Option<Gemma4RoPE>,
}

impl Gemma4Attention {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        sliding_window: usize,
        layer_type: LayerType,
        rope_config: &RoPEConfig,
        vb: candle_nn::VarBuilder,
    ) -> Result<Self> {
        let q_proj = candle_nn::linear(hidden_size, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = candle_nn::linear(hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = candle_nn::linear(hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = candle_nn::linear(num_heads * head_dim, hidden_size, vb.pp("o_proj"))?;

        let rope = Gemma4RoPE::new(rope_config, head_dim);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            head_dim,
            sliding_window,
            layer_type,
            rope: Some(rope),
        })
    }

    pub fn forward(&self, x: &Tensor, positions: &[usize]) -> Result<Tensor> {
        match self.layer_type {
            LayerType::FullAttention => self.forward_full(x, positions),
            LayerType::SlidingAttention => self.forward_sliding(x, positions),
        }
    }

    fn forward_full(&self, x: &Tensor, positions: &[usize]) -> Result<Tensor> {
        self.gqa_attention(x, positions)
    }

    fn forward_sliding(&self, x: &Tensor, positions: &[usize]) -> Result<Tensor> {
        self.gqa_attention(x, positions)
    }

    fn gqa_attention(&self, x: &Tensor, positions: &[usize]) -> Result<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;

        let q = self
            .q_proj
            .forward(x)?
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?;
        let k =
            self.k_proj
                .forward(x)?
                .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?;
        let v =
            self.v_proj
                .forward(x)?
                .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?;

        let k = self.expand_kv(&k, self.num_heads)?;
        let v = self.expand_kv(&v, self.num_heads)?;

        let positions_i64: Vec<i64> = positions.iter().map(|&p| p as i64).collect();
        if let Some(ref rope) = self.rope {
            let (q_rope, k_rope) = rope.apply(&q, &k, &positions_i64)?;
            let q = q_rope;
            let k = k_rope;
            self.compute_attention(&q, &k, &v, seq_len, batch)
        } else {
            self.compute_attention(&q, &k, &v, seq_len, batch)
        }
    }

    fn compute_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        seq_len: usize,
        batch: usize,
    ) -> Result<Tensor> {
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        let qk = Tensor::matmul(&q, &k.transpose(2, 3)?)?;

        let scale = 1.0f32 / (self.head_dim as f32).sqrt();
        let scale_tensor = Tensor::new(&[scale], q.device())?.broadcast_as(qk.dims())?;
        let qk = qk.mul(&scale_tensor)?;

        let attn_weights = candle_nn::ops::softmax(&qk, 3)?;
        let attn_output = Tensor::matmul(&attn_weights, &v)?;

        let attn_output = attn_output.transpose(1, 2)?;
        let attn_output = attn_output.reshape((batch, seq_len, self.num_heads * self.head_dim))?;

        self.o_proj.forward(&attn_output)
    }

    fn expand_kv(&self, kv: &Tensor, num_q_heads: usize) -> Result<Tensor> {
        if num_q_heads == self.num_kv_heads {
            return Ok(kv.clone());
        }

        let repeat_factor = num_q_heads / self.num_kv_heads;
        let (batch, seq, heads, dim) = kv.dims4()?;

        let kv = kv.reshape((batch, seq, heads, 1, dim))?;
        let expanded = kv.broadcast_as((batch, seq, heads, repeat_factor, dim))?;
        let expanded = expanded.reshape((batch, seq, heads * repeat_factor, dim))?;

        Ok(expanded)
    }
}

impl Default for Gemma4Attention {
    fn default() -> Self {
        Self {
            q_proj: Linear::new(
                Tensor::zeros((1, 1), candle_core::DType::F32, &candle_core::Device::Cpu).unwrap(),
                None,
            ),
            k_proj: Linear::new(
                Tensor::zeros((1, 1), candle_core::DType::F32, &candle_core::Device::Cpu).unwrap(),
                None,
            ),
            v_proj: Linear::new(
                Tensor::zeros((1, 1), candle_core::DType::F32, &candle_core::Device::Cpu).unwrap(),
                None,
            ),
            o_proj: Linear::new(
                Tensor::zeros((1, 1), candle_core::DType::F32, &candle_core::Device::Cpu).unwrap(),
                None,
            ),
            num_heads: 0,
            num_kv_heads: 0,
            head_dim: 0,
            sliding_window: 0,
            layer_type: LayerType::FullAttention,
            rope: None,
        }
    }
}
