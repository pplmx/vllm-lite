//! Pure tensor kernels for Gemma4 attention: `Q/K/V` projection, `RoPE`
//! application, KV-head expansion, and the two attention-compute paths
//! (paged vs non-paged). Forward entry points live in [`super::forward`].

use candle_core::{Module, Result, Tensor};

use super::Gemma4Attention;
use crate::components::attention::paged_gqa::{compute_gqa_attention, project_attention_output};
use crate::config::architecture::LayerType;

impl Gemma4Attention {
    pub(super) fn project_qkv(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;
        Ok((q, k, v))
    }

    pub(super) fn apply_rope(
        &self,
        q: &Tensor,
        k: &Tensor,
        positions: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        let positions_i64: Vec<i64> = positions.iter().map(|&p| p as i64).collect();
        if let Some(ref rope) = self.rope {
            let q = q.transpose(1, 2)?;
            let k = k.transpose(1, 2)?;
            let (q, k) = rope.apply(&q, &k, &positions_i64)?;
            Ok((q.transpose(1, 2)?, k.transpose(1, 2)?))
        } else {
            Ok((q.clone(), k.clone()))
        }
    }

    pub(super) fn expand_kv(&self, kv: &Tensor, num_q_heads: usize) -> Result<Tensor> {
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

    /// Attention with q/k/v in `[batch, num_heads, seq, head_dim]` layout (paged KV path).
    pub(super) fn compute_paged_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        query_positions: &[usize],
    ) -> Result<Tensor> {
        let batch_size = q.dims()[0];
        let seq_len = q.dims()[2];
        let kv_seq = k.dims()[2];

        let mask = if matches!(self.layer_type, LayerType::SlidingAttention) {
            Some(self.sliding_causal_mask(seq_len, kv_seq, query_positions, q.device())?)
        } else {
            None
        };

        let attn_output = compute_gqa_attention(q, k, v, self.head_dim, mask.as_ref())?;
        project_attention_output(
            &attn_output,
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
            &self.o_proj,
        )
    }

    pub(super) fn gqa_attention(
        &self,
        x: &Tensor,
        positions: &[usize],
        apply_sliding_mask: bool,
    ) -> Result<Tensor> {
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
            let q = q.transpose(1, 2)?;
            let k = k.transpose(1, 2)?;
            let (q_rope, k_rope) = rope.apply(&q, &k, &positions_i64)?;
            let q = q_rope.transpose(1, 2)?;
            let k = k_rope.transpose(1, 2)?;
            self.compute_attention(&q, &k, &v, seq_len, batch, positions, apply_sliding_mask)
        } else {
            self.compute_attention(&q, &k, &v, seq_len, batch, positions, apply_sliding_mask)
        }
    }

    pub(super) fn compute_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        seq_len: usize,
        batch: usize,
        query_positions: &[usize],
        apply_sliding_mask: bool,
    ) -> Result<Tensor> {
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        let mut qk = Tensor::matmul(&q, &k.transpose(2, 3)?)?;

        if apply_sliding_mask {
            let kv_seq = k.dims()[2];
            let mask = self.sliding_causal_mask(seq_len, kv_seq, query_positions, q.device())?;
            qk = qk.broadcast_add(&mask)?;
        }

        let scale = 1.0f32 / (self.head_dim as f32).sqrt();
        let scale_tensor = Tensor::new(&[scale], q.device())?.broadcast_as(qk.dims())?;
        let qk = qk.mul(&scale_tensor)?;

        let attn_weights = candle_nn::ops::softmax(&qk, 3)?;
        let attn_output = Tensor::matmul(&attn_weights, &v)?;

        let attn_output = attn_output.transpose(1, 2)?;
        let attn_output = attn_output.reshape((batch, seq_len, self.num_heads * self.head_dim))?;

        self.o_proj.forward(&attn_output)
    }
}
