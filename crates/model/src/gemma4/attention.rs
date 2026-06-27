//! Gemma4 Attention implementation.

#![allow(clippy::too_many_arguments)]

use crate::components::attention::paged_gqa::{
    compute_gqa_attention, project_attention_output, read_decode_kv, write_prefill_kv,
};
use crate::config::architecture::{LayerType, RoPEConfig};
use crate::gemma4::rope::Gemma4RoPE;
use crate::paged_tensor::PagedKvCache;
use candle_core::{Device, Module, Result, Tensor};
use candle_nn::Linear;
use tracing::trace;

/// Gemma4Attention: gemma4 attention.
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

    pub fn new_from_weights(
        _hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        sliding_window: usize,
        layer_type: LayerType,
        rope_config: &RoPEConfig,
        q_w: Tensor,
        k_w: Tensor,
        v_w: Tensor,
        o_w: Tensor,
    ) -> Result<Self> {
        let rope = Gemma4RoPE::new(rope_config, head_dim);
        Ok(Self {
            q_proj: Linear::new(q_w, None),
            k_proj: Linear::new(k_w, None),
            v_proj: Linear::new(v_w, None),
            o_proj: Linear::new(o_w, None),
            num_heads,
            num_kv_heads,
            head_dim,
            sliding_window,
            layer_type,
            rope: Some(rope),
        })
    }

    fn key_position(
        &self,
        key_idx: usize,
        kv_seq: usize,
        q_seq: usize,
        positions: &[usize],
    ) -> usize {
        if kv_seq == q_seq {
            positions.get(key_idx).copied().unwrap_or(key_idx)
        } else {
            key_idx
        }
    }

    fn sliding_causal_mask(
        &self,
        q_seq: usize,
        kv_seq: usize,
        query_positions: &[usize],
        device: &Device,
    ) -> Result<Tensor> {
        let mut mask_data = vec![0f32; q_seq * kv_seq];
        for qi in 0..q_seq {
            let q_pos = query_positions.get(qi).copied().unwrap_or(qi);
            for kj in 0..kv_seq {
                let k_pos = self.key_position(kj, kv_seq, q_seq, query_positions);
                let in_window = q_pos.saturating_sub(k_pos) < self.sliding_window;
                let causal = k_pos <= q_pos;
                if !(causal && in_window) {
                    mask_data[qi * kv_seq + kj] = f32::NEG_INFINITY;
                }
            }
        }
        Tensor::from_slice(&mask_data, (1, 1, q_seq, kv_seq), device)
    }

    fn project_qkv(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;
        Ok((q, k, v))
    }

    fn apply_rope(&self, q: &Tensor, k: &Tensor, positions: &[usize]) -> Result<(Tensor, Tensor)> {
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

    pub fn forward_prefill(
        &self,
        x: &Tensor,
        kv_cache: &mut PagedKvCache,
        layer_idx: usize,
        block_ids: &[usize],
        positions: &[usize],
    ) -> Result<Tensor> {
        let (batch_size, seq_len, _) = x.dims3()?;

        let (q, k, v) = self.project_qkv(x)?;
        let q = q.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?;
        let k = k.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?;
        let v = v.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?;

        let (q, k) = self.apply_rope(&q, &k, positions)?;
        let k_expanded = self.expand_kv(&k, self.num_heads)?;
        let v_expanded = self.expand_kv(&v, self.num_heads)?;

        let q = q.transpose(1, 2)?.contiguous()?;
        let k_expanded = k_expanded.transpose(1, 2)?.contiguous()?;
        let v_expanded = v_expanded.transpose(1, 2)?.contiguous()?;

        write_prefill_kv(
            kv_cache,
            layer_idx,
            block_ids,
            seq_len,
            &k_expanded,
            &v_expanded,
        )?;

        self.compute_paged_attention(&q, &k_expanded, &v_expanded, positions)
    }

    pub fn forward_decode(
        &self,
        x: &Tensor,
        kv_cache: &mut PagedKvCache,
        layer_idx: usize,
        block_ids: &[usize],
        num_computed_tokens: usize,
        positions: &[usize],
    ) -> Result<Tensor> {
        let batch_size = x.dims()[0];
        let _seq_len = num_computed_tokens + 1;

        let (q, k, v) = self.project_qkv(x)?;
        let q = q.reshape((batch_size, 1, self.num_heads, self.head_dim))?;
        let k = k.reshape((batch_size, 1, self.num_kv_heads, self.head_dim))?;
        let v = v.reshape((batch_size, 1, self.num_kv_heads, self.head_dim))?;

        let (q, k) = self.apply_rope(&q, &k, positions)?;
        let k_expanded = self.expand_kv(&k, self.num_heads)?;
        let v_expanded = self.expand_kv(&v, self.num_heads)?;

        let q = q.transpose(1, 2)?;
        let k_for_cache = k_expanded.transpose(1, 2)?.squeeze(0)?.contiguous()?;
        let v_for_cache = v_expanded.transpose(1, 2)?.squeeze(0)?.contiguous()?;

        let (full_k, full_v) = read_decode_kv(
            kv_cache,
            layer_idx,
            block_ids,
            num_computed_tokens,
            &k_for_cache,
            &v_for_cache,
        )?;

        self.compute_paged_attention(&q, &full_k, &full_v, positions)
    }

    /// Attention with q/k/v in `[batch, num_heads, seq, head_dim]` layout (paged KV path).
    fn compute_paged_attention(
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

    pub fn forward(&self, x: &Tensor, positions: &[usize]) -> Result<Tensor> {
        let (batch_size, seq_len, _) = x.dims3().unwrap_or((1, 1, 0));
        trace!(
            batch_size,
            seq_len,
            num_heads = self.num_heads,
            num_kv_heads = self.num_kv_heads,
            "Gemma4Attention forward"
        );

        match self.layer_type {
            LayerType::FullAttention => self.forward_full(x, positions),
            LayerType::SlidingAttention => self.forward_sliding(x, positions),
        }
    }

    fn forward_full(&self, x: &Tensor, positions: &[usize]) -> Result<Tensor> {
        self.gqa_attention(x, positions, false)
    }

    fn forward_sliding(&self, x: &Tensor, positions: &[usize]) -> Result<Tensor> {
        self.gqa_attention(x, positions, true)
    }

    fn gqa_attention(
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

    fn compute_attention(
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
        // 1x1 F32 CPU tensor allocation cannot realistically fail (4 bytes).
        // Using `.expect` with a descriptive message instead of `.unwrap()`
        // to satisfy ERR-03 (documented production .expect() sites).
        Self {
            q_proj: Linear::new(
                Tensor::zeros((1, 1), candle_core::DType::F32, &candle_core::Device::Cpu).expect(
                    "failed to allocate 1x1 F32 CPU tensor for Gemma4Attention::default q_proj",
                ),
                None,
            ),
            k_proj: Linear::new(
                Tensor::zeros((1, 1), candle_core::DType::F32, &candle_core::Device::Cpu).expect(
                    "failed to allocate 1x1 F32 CPU tensor for Gemma4Attention::default k_proj",
                ),
                None,
            ),
            v_proj: Linear::new(
                Tensor::zeros((1, 1), candle_core::DType::F32, &candle_core::Device::Cpu).expect(
                    "failed to allocate 1x1 F32 CPU tensor for Gemma4Attention::default v_proj",
                ),
                None,
            ),
            o_proj: Linear::new(
                Tensor::zeros((1, 1), candle_core::DType::F32, &candle_core::Device::Cpu).expect(
                    "failed to allocate 1x1 F32 CPU tensor for Gemma4Attention::default o_proj",
                ),
                None,
            ),
            num_heads: 0,
            num_kv_heads: 0,
            head_dim: 0,
            sliding_window: 512,
            layer_type: LayerType::FullAttention,
            rope: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::architecture::RoPEConfig;
    use candle_core::DType;

    fn tiny_sliding_attention(sliding_window: usize) -> Result<Gemma4Attention> {
        let device = Device::Cpu;
        let hidden = 32;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 8;
        let rope_config = RoPEConfig {
            rope_theta: 10000.0,
            partial_rotary_factor: 1.0,
        };
        Gemma4Attention::new(
            hidden,
            num_heads,
            num_kv_heads,
            head_dim,
            sliding_window,
            LayerType::SlidingAttention,
            &rope_config,
            candle_nn::VarBuilder::zeros(DType::F32, &device),
        )
    }

    #[test]
    fn test_sliding_causal_mask_blocks_out_of_window() -> Result<()> {
        let attn = tiny_sliding_attention(2)?;
        let device = Device::Cpu;
        let seq_len = 4;
        let positions: Vec<usize> = (0..seq_len).collect();
        let mask = attn.sliding_causal_mask(seq_len, seq_len, &positions, &device)?;
        let data: Vec<f32> = mask.flatten_all()?.to_vec1()?;

        // Query at position 3 should not attend to key at position 0 (distance 3 > window 2).
        let idx = 3 * seq_len;
        assert!(
            data[idx].is_infinite() && data[idx].is_sign_negative(),
            "expected -inf mask for out-of-window key, got {}",
            data[idx]
        );

        // Query at position 3 should attend to key at position 2 (distance 1 <= window 2).
        let idx = 3 * seq_len + 2;
        assert_eq!(data[idx], 0.0, "in-window causal pair should be unmasked");
        Ok(())
    }

    #[test]
    fn test_sliding_mask_matches_paged_path() -> Result<()> {
        let attn = tiny_sliding_attention(3)?;
        let device = Device::Cpu;
        let seq_len = 5;
        let hidden = 32;
        let x = Tensor::randn(0.0f32, 1.0, (1, seq_len, hidden), &device)?;
        let positions: Vec<usize> = (0..seq_len).collect();

        let non_paged = attn.forward(&x, &positions)?;

        let (q, k, v) = attn.project_qkv(&x)?;
        let q = q.reshape((1, seq_len, attn.num_heads, attn.head_dim))?;
        let k = k.reshape((1, seq_len, attn.num_kv_heads, attn.head_dim))?;
        let v = v.reshape((1, seq_len, attn.num_kv_heads, attn.head_dim))?;
        let (q, k) = attn.apply_rope(&q, &k, &positions)?;
        let k = attn.expand_kv(&k, attn.num_heads)?;
        let v = attn.expand_kv(&v, attn.num_heads)?;
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;
        let paged = attn.compute_paged_attention(&q, &k, &v, &positions)?;

        let diff = (&non_paged - &paged)?.abs()?;
        let max_diff: f32 = diff.max_all()?.to_scalar()?;
        assert!(
            max_diff < 1e-5,
            "non-paged sliding path should match paged attention, max_diff={max_diff}"
        );
        Ok(())
    }
}
