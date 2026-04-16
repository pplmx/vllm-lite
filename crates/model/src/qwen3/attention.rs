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

        // q, k, v are [batch, seq, heads, dim]
        // For matmul, need [batch, heads, seq, dim]
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        // Now q, k, v are [batch, heads, seq, dim]
        // For q @ k^T, need k as [batch, heads, dim, seq]
        let q = q.contiguous()?;
        let k_t = k.transpose(2, 3)?.contiguous()?;
        let qk = Tensor::matmul(&q, &k_t)?;

        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let qk = qk.mul(&Tensor::new(&[scale], q.device())?.broadcast_as(qk.dims())?)?;
        let attn_weights = candle_nn::ops::softmax(&qk, 3)?.contiguous()?;

        // attn_weights: [batch, heads, seq_q, seq_kv]
        // v: [batch, heads, seq_kv, dim]
        // attn_output: [batch, heads, seq_q, dim]
        let attn_output = Tensor::matmul(&attn_weights, &v.contiguous()?)?;

        // attn_output: [batch, heads, seq_q, dim]
        // Transpose to [batch, seq_q, heads, dim]
        let attn_output = attn_output.transpose(1, 2)?;

        // Reshape to [batch, seq_q, num_heads * head_dim]
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
            eprintln!("DEBUG apply_k_norm: k.dims={:?}, batch_size={}, num_kv_heads={}, seq_len={}, head_dim={}",
                     k.dims(), batch_size, self.num_kv_heads, seq_len, self.head_dim);
            let k = k.transpose(1, 2)?; // [batch, num_kv_heads, seq, head_dim]
            eprintln!("DEBUG apply_k_norm: after transpose k.dims={:?}", k.dims());
            let reshape_size = batch_size * self.num_kv_heads * seq_len;
            eprintln!(
                "DEBUG apply_k_norm: reshaping to ({}, {})",
                reshape_size, self.head_dim
            );
            let k = k.reshape((reshape_size, self.head_dim))?;
            eprintln!("DEBUG apply_k_norm: after reshape k.dims={:?}", k.dims());
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

        eprintln!("DEBUG prefill L{}: q dims={:?} (expect {}), k dims={:?} (expect {}), num_heads={}, head_dim={}",
                 layer_idx, q.dims(), batch_size * seq_len * self.num_heads * self.head_dim,
                 k.dims(), batch_size * seq_len * self.num_kv_heads * self.head_dim,
                 self.num_heads, self.head_dim);

        let q = q.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?;
        let k = k.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?;
        let v = v.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?;

        let q = self.apply_q_norm(q, batch_size, seq_len)?;
        let k = self.apply_k_norm(k, batch_size, seq_len)?;

        let position_ids: Vec<i64> = positions.iter().map(|&p| p as i64).collect();
        let q = apply_rope(&q, &position_ids, self.theta)?;
        let k = apply_rope(&k, &position_ids, self.theta)?;

        // Expand k/v from num_kv_heads to num_heads for storage and attention
        let k_expanded = self.expand_kv(&k, self.num_heads, self.num_kv_heads)?;
        let v_expanded = self.expand_kv(&v, self.num_heads, self.num_kv_heads)?;

        let q = q.transpose(1, 2)?.contiguous()?;
        let k_expanded = k_expanded.transpose(1, 2)?.contiguous()?;
        let v_expanded = v_expanded.transpose(1, 2)?.contiguous()?;

        // Write expanded k/v to cache
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

            // After transpose, k_expanded is [batch, num_heads, seq, head_dim]
            // Select tokens from dim 2 (sequence dimension)
            // Result: [batch, num_heads, selected_seq, head_dim]
            let k_block = k_expanded.index_select(&indices_tensor, 2)?.contiguous()?;
            let v_block = v_expanded.index_select(&indices_tensor, 2)?.contiguous()?;

            // Transpose to [batch, seq, num_heads, head_dim] for write_kv_batch
            let k_block = k_block.transpose(1, 2)?.contiguous()?;
            let v_block = v_block.transpose(1, 2)?.contiguous()?;

            kv_cache.write_kv_batch(layer_idx, *block_id, 0, &k_block, &v_block)?;
        }

        // k_expanded and v_expanded are already [batch, num_heads, seq, head_dim] from transpose above

        if seq_len > tile_size {
            self.tiled_attention(&q, &k_expanded, &v_expanded, seq_len)
        } else {
            self.paged_attention(&q, &k_expanded, &v_expanded, seq_len)
        }
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
        let seq_len = num_computed_tokens + 1;
        let tile_size = self.config.tile_size.unwrap_or(16);

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        eprintln!("DEBUG decode: q_proj output dims={:?}, k_proj dims={:?}, x dims={:?}, num_heads={}, num_kv_heads={}, head_dim={}",
                 q.dims(), k.dims(), x.dims(), self.num_heads, self.num_kv_heads, self.head_dim);

        let q = q.reshape((batch_size, 1, self.num_heads, self.head_dim))?;
        let k = k.reshape((batch_size, 1, self.num_kv_heads, self.head_dim))?;
        let v = v.reshape((batch_size, 1, self.num_kv_heads, self.head_dim))?;

        let q = self.apply_q_norm(q, batch_size, 1)?;
        let k = self.apply_k_norm(k, batch_size, 1)?;

        let position_ids: Vec<i64> = positions.iter().map(|&p| p as i64).collect();
        let q = apply_rope(&q, &position_ids, self.theta)?;
        let k = apply_rope(&k, &position_ids, self.theta)?;

        let q = q.transpose(1, 2)?;

        // For GQA: expand k/v from num_kv_heads to num_heads before writing to cache
        let k_expanded = self.expand_kv(&k, self.num_heads, self.num_kv_heads)?;
        let v_expanded = self.expand_kv(&v, self.num_heads, self.num_kv_heads)?;

        // k_expanded/v_expanded is [batch=1, seq=1, num_heads, head_dim]
        // After transpose: [batch=1, num_heads, seq=1, head_dim] -> squeeze batch: [num_heads, 1, head_dim]
        let k_for_cache = k_expanded.transpose(1, 2)?.squeeze(0)?.contiguous()?; // [num_heads, 1, head_dim]
        let v_for_cache = v_expanded.transpose(1, 2)?.squeeze(0)?.contiguous()?;

        // Read cached k/v (which is stored with num_heads for the expanded version)
        let (cached_k, cached_v) = kv_cache.read_kv(layer_idx, block_ids, num_computed_tokens)?;

        // read_kv returns [seq_len, num_heads, head_dim], need [num_heads, seq_len, head_dim]
        let cached_k = cached_k.transpose(0, 1)?.contiguous()?;
        let cached_v = cached_v.transpose(0, 1)?.contiguous()?;

        // Concatenate cached and new k/v along seq dimension
        // cached_k/v is [num_heads, seq, head_dim]
        // k_for_cache/v_for_cache is [num_heads, 1, head_dim]
        let full_k = Tensor::cat(&[&cached_k, &k_for_cache], 1)?.contiguous()?; // [num_heads, seq+1, head_dim]
        let full_v = Tensor::cat(&[&cached_v, &v_for_cache], 1)?.contiguous()?;

        // Write new k/v to cache (with expanded num_heads)
        // k_for_cache/v_for_cache is [num_heads, 1, head_dim]
        // write_kv expects [1, num_heads, head_dim]
        // token_offset should be within current block, not absolute position
        if !block_ids.is_empty() {
            let block_size = kv_cache.block_size();
            let token_offset = num_computed_tokens % block_size;
            let block_id = num_computed_tokens / block_size;

            // k_for_cache: [num_heads, 1, head_dim] -> [1, num_heads, head_dim]
            eprintln!("DEBUG write L{}: k_for_cache dims={:?}, self.num_heads={}, self.head_dim={}, num_computed={}",
                     layer_idx, k_for_cache.dims(), self.num_heads, self.head_dim, num_computed_tokens);
            let k_for_write = k_for_cache.permute((1, 0, 2))?.contiguous()?; // [1, num_heads, head_dim]
            eprintln!(
                "DEBUG write L{}: k_for_write dims={:?} (expect [1, {}, {}])",
                layer_idx,
                k_for_write.dims(),
                self.num_heads,
                self.head_dim
            );
            let v_for_write = v_for_cache.permute((1, 0, 2))?.contiguous()?;
            kv_cache.write_kv(
                layer_idx,
                block_id,
                token_offset,
                &k_for_write,
                &v_for_write,
            )?;
        }

        // Expand cached k/v for attention (they should already be expanded from storage)
        // full_k/v already has num_heads, just need to unsqueeze and transpose
        let full_k_unsqueezed = full_k.unsqueeze(0)?.contiguous()?; // [1, num_heads, seq+1, head_dim]
        let full_v_unsqueezed = full_v.unsqueeze(0)?.contiguous()?;

        // q is already [batch, num_heads, seq_q, head_dim] after transpose at line 327
        let q_for_attn = &q;

        if seq_len > tile_size {
            eprintln!(
                "DEBUG forward_decode L{}: using tiled_attention, seq_len={}, tile_size={}",
                layer_idx, seq_len, tile_size
            );
            self.tiled_attention(&q_for_attn, &full_k_unsqueezed, &full_v_unsqueezed, seq_len)
        } else {
            self.paged_attention(&q_for_attn, &full_k_unsqueezed, &full_v_unsqueezed, seq_len)
        }
    }

    fn paged_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        seq_len: usize,
    ) -> Result<Tensor> {
        let attn_output = paged_attention(q, k, v, self.num_heads, self.head_dim)?;
        eprintln!(
            "DEBUG paged_attention result: attn_output.dims={:?}, o_proj weight dims={:?}",
            attn_output.dims(),
            self.o_proj.weight().dims()
        );
        let o = self.o_proj.forward(&attn_output)?;
        eprintln!("DEBUG after o_proj: o.dims={:?}", o.dims());
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
