#![allow(clippy::too_many_arguments)]

use crate::kv_cache::PagedKvCache;
use candle_core::{Device, Module, Result, Tensor};
use candle_nn::{LayerNorm, Linear};

#[derive(Debug, Clone)]
pub struct AttentionConfig {
    pub tile_size: Option<usize>,
    pub use_fused: bool,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            tile_size: None,
            use_fused: true,
        }
    }
}

pub struct GqaAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
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
            let q_norm_bias = Tensor::zeros(head_dim, q_norm_weight.dtype(), &Device::Cpu)?;
            Some(LayerNorm::new(q_norm_weight, q_norm_bias, 1e-6))
        } else {
            None
        };
        let k_norm = if has_qk_norm {
            let k_norm_weight =
                k_norm_weight.ok_or_else(|| candle_core::Error::msg("Missing k_norm weight"))?;
            let k_norm_bias = Tensor::zeros(head_dim, k_norm_weight.dtype(), &Device::Cpu)?;
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
        if num_q_heads == num_kv_heads {
            return Ok(kv.clone());
        }

        let repeat_factor = num_q_heads / num_kv_heads;
        let (batch, seq, heads, dim) = kv.dims4()?;

        // Input: [batch, seq, num_kv_heads, head_dim]
        // Output: [batch, seq, num_q_heads, head_dim]
        // Use reshape + broadcast_as to expand the heads dimension
        let kv = kv.reshape((batch, seq, heads, 1, dim))?; // [batch, seq, heads, 1, dim]
        let expanded = kv.broadcast_as((batch, seq, heads, repeat_factor, dim))?; // [batch, seq, heads, repeat, dim]
        let expanded = expanded.reshape((batch, seq, heads * repeat_factor, dim))?;

        Ok(expanded)
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

        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;

        // Group tokens by block and write in batches
        let mut block_tokens: std::collections::HashMap<usize, Vec<usize>> =
            std::collections::HashMap::new();
        for (token_idx, &block_id) in block_ids.iter().take(seq_len).enumerate() {
            block_tokens.entry(block_id).or_default().push(token_idx);
        }

        for (&block_id, token_indices) in &block_tokens {
            // Get all tokens for this block
            let indices_vec: Vec<u32> = token_indices.iter().map(|&i| i as u32).collect();
            let indices_tensor = Tensor::new(indices_vec.as_slice(), k.device())?;
            let k_block = k.index_select(&indices_tensor, 2)?;
            let v_block = v.index_select(&indices_tensor, 2)?;

            let k_block = k_block.transpose(0, 1)?; // [num_kv_heads, tokens, head_dim]
            let v_block = v_block.transpose(0, 1)?;

            // Write all tokens in this block at once
            for (offset, &_token_idx) in token_indices.iter().enumerate() {
                let k_slice = k_block.narrow(1, offset, 1)?.squeeze(1)?;
                let v_slice = v_block.narrow(1, offset, 1)?.squeeze(1)?;
                let k_slice = k_slice.reshape((1, self.num_kv_heads, self.head_dim))?;
                let v_slice = v_slice.reshape((1, self.num_kv_heads, self.head_dim))?;
                kv_cache.write_kv(layer_idx, block_id, offset, &k_slice, &v_slice)?;
            }
        }

        let k_expanded = self.expand_kv(&k.transpose(1, 2)?, self.num_heads, self.num_kv_heads)?;
        let v_expanded = self.expand_kv(&v.transpose(1, 2)?, self.num_heads, self.num_kv_heads)?;
        let k_expanded = k_expanded.transpose(1, 2)?;
        let v_expanded = v_expanded.transpose(1, 2)?;

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
    ) -> Result<Tensor> {
        let batch_size = x.dims()[0];
        let seq_len = num_computed_tokens + 1;
        let tile_size = self.config.tile_size.unwrap_or(16);

        let q = self.q_proj.forward(x)?;
        let q = q.reshape((batch_size, 1, self.num_heads, self.head_dim))?;

        let q = self.apply_q_norm(q, batch_size, 1)?;
        let q = q.transpose(1, 2)?;

        let (k, v) = kv_cache.read_kv(layer_idx, block_ids, num_computed_tokens)?;

        let k = k.transpose(0, 1)?.transpose(1, 2)?;
        let v = v.transpose(0, 1)?.transpose(1, 2)?;

        let k_expanded = self.expand_kv(&k, self.num_heads, self.num_kv_heads)?;
        let v_expanded = self.expand_kv(&v, self.num_heads, self.num_kv_heads)?;
        let k_expanded = k_expanded.transpose(1, 2)?;
        let v_expanded = v_expanded.transpose(1, 2)?;

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
        seq_len: usize,
    ) -> Result<Tensor> {
        let qk = Tensor::matmul(q, &k.transpose(2, 3)?)?;

        let mask = self.causal_mask(seq_len, q.device())?;
        let qk = (&qk + &mask)?;

        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let qk = qk.mul(&Tensor::new(&[scale], q.device())?)?;
        let attn_weights = candle_nn::ops::softmax(&qk, 3)?;

        let attn_output = Tensor::matmul(&attn_weights, v)?;

        let attn_output = attn_output.transpose(1, 2)?;
        let attn_output = attn_output.reshape((q.dims()[0], 1, self.num_heads * self.head_dim))?;

        let o = self.o_proj.forward(&attn_output)?;
        Ok(o)
    }

    fn causal_mask(&self, seq_len: usize, device: &Device) -> Result<Tensor> {
        let row_indices =
            Tensor::arange(0u32, seq_len as u32, device)?.reshape((1, 1, seq_len, 1))?;
        let col_indices =
            Tensor::arange(0u32, seq_len as u32, device)?.reshape((1, 1, 1, seq_len))?;
        let mask = row_indices.ge(&col_indices)?;
        let zero = Tensor::new(0.0f32, device)?;
        let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?;
        let mask = mask.where_cond(&zero, &neg_inf)?;
        Ok(mask)
    }

    fn tiled_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        seq_len: usize,
    ) -> Result<Tensor> {
        let tile_size = self.config.tile_size.unwrap_or(16);

        let num_tiles = seq_len.div_ceil(tile_size);
        let mut output_parts = Vec::new();

        for tile_idx in 0..num_tiles {
            let start = tile_idx * tile_size;
            let end = (start + tile_size).min(seq_len);
            let tile_len = end - start;

            let k_tile = k.narrow(2, start, tile_len)?;
            let v_tile = v.narrow(2, start, tile_len)?;

            let qk = Tensor::matmul(q, &k_tile.transpose(2, 3)?)?;

            let mask = self.causal_mask_tile(q.dims()[0], self.num_heads, tile_len, q.device())?;
            let qk = (&qk + &mask)?;

            let scale = 1.0 / (self.head_dim as f32).sqrt();
            let qk = qk.mul(&Tensor::new(&[scale], q.device())?)?;
            let attn = candle_nn::ops::softmax(&qk, 3)?;

            let out = Tensor::matmul(&attn, &v_tile)?;

            output_parts.push(out);
        }

        Tensor::cat(&output_parts, 2)
    }

    fn causal_mask_tile(
        &self,
        batch_size: usize,
        _num_heads: usize,
        tile_len: usize,
        device: &Device,
    ) -> Result<Tensor> {
        let mask: Vec<f32> = (0..tile_len)
            .flat_map(|i| (0..tile_len).map(move |j| if i >= j { 0.0 } else { f32::NEG_INFINITY }))
            .collect();
        Tensor::from_slice(&mask, (batch_size, 1, tile_len, tile_len), device)
    }
}
