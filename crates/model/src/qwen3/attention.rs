#![allow(clippy::too_many_arguments)]

use super::rope::apply_rope;
pub use crate::components::AttentionConfig;
use crate::components::{expand_kv, paged_attention, tiled_attention};
use crate::kv_cache::PagedKvCache;
use candle_core::{Module, Result, Tensor};
use candle_nn::{LayerNorm, Linear};

pub struct GqaAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    theta: f32,
    config: AttentionConfig,
    q_norm: Option<LayerNorm>,
    k_norm: Option<LayerNorm>,
}

impl GqaAttention {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        theta: f32,
        vb: Option<candle_nn::VarBuilder>,
        config: AttentionConfig,
        has_qk_norm: bool,
    ) -> Result<Self> {
        let vb = vb.unwrap_or_else(|| {
            candle_nn::VarBuilder::zeros(candle_core::DType::F32, &candle_core::Device::Cpu)
        });

        let q_proj = candle_nn::linear(hidden_size, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = candle_nn::linear(hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = candle_nn::linear(hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = candle_nn::linear(num_heads * head_dim, hidden_size, vb.pp("o_proj"))?;

        let q_norm = if has_qk_norm {
            Some(candle_nn::layer_norm(head_dim, 1e-6, vb.pp("q_norm"))?)
        } else {
            None
        };
        let k_norm = if has_qk_norm {
            Some(candle_nn::layer_norm(head_dim, 1e-6, vb.pp("k_norm"))?)
        } else {
            None
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            head_dim,
            theta,
            config,
            q_norm,
            k_norm,
        })
    }

    pub fn new_with_weights(
        _hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        theta: f32,
        q_weight: Tensor,
        k_weight: Tensor,
        v_weight: Tensor,
        o_weight: Tensor,
        config: AttentionConfig,
        has_qk_norm: bool,
        q_norm_weight: Option<Tensor>,
        k_norm_weight: Option<Tensor>,
    ) -> Result<Self> {
        let q_proj = Linear::new(q_weight, None);
        let k_proj = Linear::new(k_weight, None);
        let v_proj = Linear::new(v_weight, None);
        let o_proj = Linear::new(o_weight, None);

        let q_norm = if has_qk_norm {
            let q_norm_weight =
                q_norm_weight.ok_or_else(|| candle_core::Error::msg("Missing q_norm weight"))?;
            let q_norm_bias =
                Tensor::zeros(head_dim, q_norm_weight.dtype(), q_norm_weight.device())?;
            Some(LayerNorm::new(q_norm_weight, q_norm_bias, 1e-6))
        } else {
            None
        };
        let k_norm = if has_qk_norm {
            let k_norm_weight =
                k_norm_weight.ok_or_else(|| candle_core::Error::msg("Missing k_norm weight"))?;
            let k_norm_bias =
                Tensor::zeros(head_dim, k_norm_weight.dtype(), k_norm_weight.device())?;
            Some(LayerNorm::new(k_norm_weight, k_norm_bias, 1e-6))
        } else {
            None
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            head_dim,
            theta,
            config,
            q_norm,
            k_norm,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let batch_size = x.dims()[0];
        let seq_len = x.dims()[1];

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?;
        let k = k.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?;
        let v = v.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?;

        let q = self.apply_q_norm(q, batch_size, seq_len)?;
        let k = self.apply_k_norm(k, batch_size, seq_len)?;

        let k = self.expand_kv(&k, self.num_heads, self.num_kv_heads)?;
        let v = self.expand_kv(&v, self.num_heads, self.num_kv_heads)?;

        let k = k.transpose(2, 3)?;
        let qk = Tensor::matmul(&q, &k)?;

        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let qk = qk.mul(&Tensor::new(&[scale], q.device())?.broadcast_as(qk.dims())?)?;
        let attn_weights = candle_nn::ops::softmax(&qk, 3)?;

        let attn_output = Tensor::matmul(&attn_weights, &v)?;

        let attn_output =
            attn_output.reshape((batch_size, seq_len, self.num_heads * self.head_dim))?;

        let o = self.o_proj.forward(&attn_output)?;
        Ok(o)
    }

    pub fn expand_kv(
        &self,
        kv: &Tensor,
        num_q_heads: usize,
        num_kv_heads: usize,
    ) -> Result<Tensor> {
        expand_kv(kv, num_q_heads, num_kv_heads)
    }

    fn apply_q_norm(&self, q: Tensor, batch_size: usize, seq_len: usize) -> Result<Tensor> {
        if let Some(ref q_norm) = self.q_norm {
            let q = q.transpose(1, 2)?; // [batch, num_heads, seq, head_dim]
            let q = q.reshape((batch_size * self.num_heads * seq_len, self.head_dim))?;
            let q = q_norm.forward(&q)?;
            let q = q.reshape((batch_size, self.num_heads, seq_len, self.head_dim))?;
            let q = q.transpose(1, 2)?; // [batch, seq, num_heads, head_dim]
            Ok(q)
        } else {
            Ok(q)
        }
    }

    fn apply_k_norm(&self, k: Tensor, batch_size: usize, seq_len: usize) -> Result<Tensor> {
        if let Some(ref k_norm) = self.k_norm {
            let k = k.transpose(1, 2)?; // [batch, num_kv_heads, seq, head_dim]
            let k = k.reshape((batch_size * self.num_kv_heads * seq_len, self.head_dim))?;
            let k = k_norm.forward(&k)?;
            let k = k.reshape((batch_size, self.num_kv_heads, seq_len, self.head_dim))?;
            let k = k.transpose(1, 2)?; // [batch, seq, num_kv_heads, head_dim]
            Ok(k)
        } else {
            Ok(k)
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
        let batch_size = x.dims()[0];
        let seq_len = x.dims()[1];
        let tile_size = self.config.tile_size.unwrap_or(16);

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?;
        let k = k.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?;
        let v = v
            .reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let q = self.apply_q_norm(q, batch_size, seq_len)?;
        let k = self.apply_k_norm(k, batch_size, seq_len)?;

        let position_ids: Vec<i64> = positions.iter().map(|&p| p as i64).collect();
        let q = apply_rope(&q, &position_ids, self.theta)?;
        let k = apply_rope(&k, &position_ids, self.theta)?;

        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;

        let mut block_groups: std::collections::BTreeMap<usize, Vec<usize>> =
            std::collections::BTreeMap::new();
        for (token_idx, &block_id) in block_ids.iter().take(seq_len).enumerate() {
            block_groups.entry(block_id).or_default().push(token_idx);
        }

        let k_t = k.transpose(1, 2)?.contiguous()?;
        let v_t = v.transpose(1, 2)?.contiguous()?;

        for (block_id, token_indices) in &block_groups {
            if token_indices.is_empty() {
                continue;
            }

            let indices: Vec<u32> = token_indices.iter().map(|&i| i as u32).collect();
            let indices_tensor = Tensor::new(indices.as_slice(), k.device())?;

            let k_block = k_t.index_select(&indices_tensor, 1)?;
            let v_block = v_t.index_select(&indices_tensor, 1)?;

            kv_cache.write_kv_batch(layer_idx, *block_id, 0, &k_block, &v_block)?;
        }

        // expand_kv expects [batch, seq, heads, dim]
        // k_t and v_t are already in correct shape from lines 231-232
        let k_expanded = self.expand_kv(&k_t, self.num_heads, self.num_kv_heads)?;
        let v_expanded = self.expand_kv(&v_t, self.num_heads, self.num_kv_heads)?;

        // paged_attention expects [batch, heads, seq, dim]
        // expand_kv outputs [batch, seq, heads, dim], so transpose
        let k_expanded = k_expanded.transpose(1, 2)?.contiguous()?;
        let v_expanded = v_expanded.transpose(1, 2)?.contiguous()?;

        if seq_len > tile_size {
            self.tiled_attention(&q, &k_expanded, &v_expanded, seq_len)
        } else {
            self.paged_attention(&q, &k_expanded, &v_expanded, seq_len)
        }
    }

    pub fn forward_decode(
        &self,
        x: &Tensor,
        kv_cache: &PagedKvCache,
        layer_idx: usize,
        block_ids: &[usize],
        num_computed_tokens: usize,
        positions: &[usize],
    ) -> Result<Tensor> {
        let batch_size = x.dims()[0];
        let seq_len = num_computed_tokens + 1;
        let tile_size = self.config.tile_size.unwrap_or(16);

        let q = self.q_proj.forward(x)?;
        let q = q.reshape((batch_size, 1, self.num_heads, self.head_dim))?;

        let q = self.apply_q_norm(q, batch_size, 1)?;
        let q = q.transpose(1, 2)?;

        let position_ids: Vec<i64> = positions.iter().map(|&p| p as i64).collect();
        let q = apply_rope(&q, &position_ids, self.theta)?;

        let (k, v) = kv_cache.read_kv(layer_idx, block_ids, num_computed_tokens)?;

        let k = k.transpose(0, 1)?.transpose(1, 2)?;
        let v = v.transpose(0, 1)?.transpose(1, 2)?;

        let k = apply_rope(&k, &position_ids, self.theta)?;

        // k/v from read_kv after transposes: [head_dim, num_kv_heads, seq]
        // Need to reshape to [batch=1, seq, num_kv_heads, head_dim] for expand_kv
        let k = k.transpose(0, 2)?; // [head_dim, num_kv_heads, seq] -> [seq, num_kv_heads, head_dim]
        let v = v.transpose(0, 2)?;
        let k = k.unsqueeze(0)?; // Add batch dimension: [1, seq, num_kv_heads, head_dim]
        let v = v.unsqueeze(0)?;
        let k_expanded = self.expand_kv(&k, self.num_heads, self.num_kv_heads)?;
        let v_expanded = self.expand_kv(&v, self.num_heads, self.num_kv_heads)?;
        let k_expanded = k_expanded.squeeze(0)?; // Remove batch dimension for attention
        let v_expanded = v_expanded.squeeze(0)?;

        if seq_len > tile_size {
            self.tiled_attention(&q, &k_expanded, &v_expanded, seq_len)
        } else {
            self.paged_attention(&q, &k_expanded, &v_expanded, seq_len)
        }
    }

    fn paged_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        _seq_len: usize,
    ) -> Result<Tensor> {
        let attn_output = paged_attention(q, k, v, self.num_heads, self.head_dim)?;
        let o = self.o_proj.forward(&attn_output)?;
        Ok(o)
    }

    fn tiled_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        _seq_len: usize,
    ) -> Result<Tensor> {
        let tile_size = self.config.tile_size.unwrap_or(16);
        let attn_output = tiled_attention(q, k, v, self.num_heads, tile_size)?;
        let o = self.o_proj.forward(&attn_output)?;
        Ok(o)
    }
}
