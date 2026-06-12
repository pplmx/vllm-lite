//! Gemma4 Attention implementation.

#![allow(clippy::too_many_arguments)]

use crate::config::architecture::{LayerType, RoPEConfig};
use crate::gemma4::rope::Gemma4RoPE;
use crate::paged_tensor::PagedKvCache;
use candle_core::{Module, Result, Tensor};
use candle_nn::Linear;
use tracing::trace;

pub struct Gemma4Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    layer_type: LayerType,
    rope: Option<Gemma4RoPE>,
}

impl Gemma4Attention {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
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
            layer_type,
            rope: Some(rope),
        })
    }

    pub fn new_from_weights(
        _hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
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
            layer_type,
            rope: Some(rope),
        })
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

        let mut block_groups: std::collections::BTreeMap<usize, Vec<usize>> =
            std::collections::BTreeMap::new();
        for (token_idx, &block_id) in block_ids.iter().take(seq_len).enumerate() {
            block_groups.entry(block_id).or_default().push(token_idx);
        }

        for (block_id, token_indices) in &block_groups {
            if token_indices.is_empty() {
                continue;
            }
            let indices: Vec<u32> = token_indices.iter().map(|&i| i as u32).collect();
            let indices_tensor = Tensor::new(indices.as_slice(), k.device())?;
            let k_block = k_expanded.index_select(&indices_tensor, 2)?.contiguous()?;
            let v_block = v_expanded.index_select(&indices_tensor, 2)?.contiguous()?;
            let k_block = k_block.transpose(1, 2)?.contiguous()?;
            let v_block = v_block.transpose(1, 2)?.contiguous()?;
            kv_cache.write_kv_batch(layer_idx, *block_id, 0, &k_block, &v_block)?;
        }

        self.compute_paged_attention(&q, &k_expanded, &v_expanded)
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

        let (cached_k, cached_v) = kv_cache.read_kv(layer_idx, block_ids, num_computed_tokens)?;
        let cached_k = cached_k.transpose(0, 1)?.contiguous()?;
        let cached_v = cached_v.transpose(0, 1)?.contiguous()?;

        let full_k = Tensor::cat(&[&cached_k, &k_for_cache], 1)?.contiguous()?;
        let full_v = Tensor::cat(&[&cached_v, &v_for_cache], 1)?.contiguous()?;

        if !block_ids.is_empty() {
            let block_size = kv_cache.block_size();
            let token_offset = num_computed_tokens % block_size;
            let block_id = num_computed_tokens / block_size;
            let k_for_write = k_for_cache.permute((1, 0, 2))?.contiguous()?;
            let v_for_write = v_for_cache.permute((1, 0, 2))?.contiguous()?;
            kv_cache.write_kv(
                layer_idx,
                block_id,
                token_offset,
                &k_for_write,
                &v_for_write,
            )?;
        }

        let full_k = full_k.unsqueeze(0)?.contiguous()?;
        let full_v = full_v.unsqueeze(0)?.contiguous()?;
        self.compute_paged_attention(&q, &full_k, &full_v)
    }

    /// Attention with q/k/v in `[batch, num_heads, seq, head_dim]` layout (paged KV path).
    fn compute_paged_attention(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let batch_size = q.dims()[0];
        let seq_len = q.dims()[2];

        let qk = Tensor::matmul(q, &k.transpose(2, 3)?)?;
        let scale = 1.0f32 / (self.head_dim as f32).sqrt();
        let scale_tensor = Tensor::new(&[scale], q.device())?.broadcast_as(qk.dims())?;
        let qk = qk.mul(&scale_tensor)?;
        let attn_weights = candle_nn::ops::softmax(&qk, 3)?;
        let attn_output = Tensor::matmul(&attn_weights, v)?;

        let attn_output = attn_output.transpose(1, 2)?;
        let attn_output =
            attn_output.reshape((batch_size, seq_len, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&attn_output)
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
            let q = q.transpose(1, 2)?;
            let k = k.transpose(1, 2)?;
            let (q_rope, k_rope) = rope.apply(&q, &k, &positions_i64)?;
            let q = q_rope.transpose(1, 2)?;
            let k = k_rope.transpose(1, 2)?;
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
            layer_type: LayerType::FullAttention,
            rope: None,
        }
    }
}
