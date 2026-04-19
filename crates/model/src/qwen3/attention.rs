#![allow(clippy::too_many_arguments)]

pub use crate::components::AttentionConfig;
use crate::components::attention::GqaAttention as SharedGqaAttention;
use crate::components::positional::apply_rope;
use crate::kv_cache::PagedKvCache;
use candle_core::{Result, Tensor};

pub struct Qwen3Attention {
    inner: SharedGqaAttention,
    theta: f32,
}

impl Qwen3Attention {
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
        let inner = SharedGqaAttention::new(
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            vb,
            config,
            has_qk_norm,
        )?;
        Ok(Self { inner, theta })
    }

    pub fn new_with_weights(
        hidden_size: usize,
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
        let inner = SharedGqaAttention::new_with_weights(
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            q_weight,
            k_weight,
            v_weight,
            o_weight,
            config,
            has_qk_norm,
            q_norm_weight,
            k_norm_weight,
        )?;
        Ok(Self { inner, theta })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.inner.forward(x)
    }

    fn apply_qk_norm(
        &self,
        q: Tensor,
        k: Tensor,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<(Tensor, Tensor)> {
        let num_heads = self.inner.num_heads();
        let num_kv_heads = self.inner.num_kv_heads();
        let head_dim = self.inner.head_dim();

        let q = if self.inner.has_q_norm() {
            let q = q.transpose(1, 2)?;
            let reshape_size = batch_size * num_heads * seq_len;
            let q = q.reshape((reshape_size, head_dim))?;
            let q = self.inner.apply_q_norm_impl_flattened(q)?;
            let q = q.reshape((batch_size, num_heads, seq_len, head_dim))?;
            q.transpose(1, 2)?
        } else {
            q
        };

        let k = if self.inner.has_k_norm() {
            let k = k.transpose(1, 2)?;
            let reshape_size = batch_size * num_kv_heads * seq_len;
            let k = k.reshape((reshape_size, head_dim))?;
            let k = self.inner.apply_k_norm_impl_flattened(k)?;
            let k = k.reshape((batch_size, num_kv_heads, seq_len, head_dim))?;
            k.transpose(1, 2)?
        } else {
            k
        };

        Ok((q, k))
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
        let num_heads = self.inner.num_heads();
        let num_kv_heads = self.inner.num_kv_heads();
        let head_dim = self.inner.head_dim();

        let (q, k, v) = self.inner.project_qkv(x)?;

        let q = q.reshape((batch_size, seq_len, num_heads, head_dim))?;
        let k = k.reshape((batch_size, seq_len, num_kv_heads, head_dim))?;
        let v = v.reshape((batch_size, seq_len, num_kv_heads, head_dim))?;

        let (q, k) = self.apply_qk_norm(q, k, batch_size, seq_len)?;

        let position_ids: Vec<i64> = positions.iter().map(|&p| p as i64).collect();
        let q = apply_rope(&q, &position_ids, self.theta)?;
        let k = apply_rope(&k, &position_ids, self.theta)?;

        let k_expanded = self.inner.expand_kv(&k, num_heads, num_kv_heads)?;
        let v_expanded = self.inner.expand_kv(&v, num_heads, num_kv_heads)?;

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

        let tile_size = self.inner.config().tile_size.unwrap_or(16);
        if seq_len > tile_size {
            self.inner.tiled_attention_fn(&q, &k_expanded, &v_expanded)
        } else {
            self.inner.paged_attention_fn(&q, &k_expanded, &v_expanded)
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
        let num_heads = self.inner.num_heads();
        let num_kv_heads = self.inner.num_kv_heads();
        let head_dim = self.inner.head_dim();
        let tile_size = self.inner.config().tile_size.unwrap_or(16);

        let (q, k, v) = self.inner.project_qkv(x)?;

        let q = q.reshape((batch_size, 1, num_heads, head_dim))?;
        let k = k.reshape((batch_size, 1, num_kv_heads, head_dim))?;
        let v = v.reshape((batch_size, 1, num_kv_heads, head_dim))?;

        let (q, k) = self.apply_qk_norm(q, k, batch_size, 1)?;

        let position_ids: Vec<i64> = positions.iter().map(|&p| p as i64).collect();
        let q = apply_rope(&q, &position_ids, self.theta)?;
        let k = apply_rope(&k, &position_ids, self.theta)?;

        let k_expanded = self.inner.expand_kv(&k, num_heads, num_kv_heads)?;
        let v_expanded = self.inner.expand_kv(&v, num_heads, num_kv_heads)?;

        let q = q.transpose(1, 2)?;

        let k_transposed = k_expanded.transpose(1, 2)?;
        let k_for_cache = k_transposed.squeeze(0)?.contiguous()?;
        let v_transposed = v_expanded.transpose(1, 2)?;
        let v_for_cache = v_transposed.squeeze(0)?.contiguous()?;

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

        let full_k_unsqueezed = full_k.unsqueeze(0)?.contiguous()?;
        let full_v_unsqueezed = full_v.unsqueeze(0)?.contiguous()?;

        if seq_len > tile_size {
            self.inner
                .tiled_attention_fn(&q, &full_k_unsqueezed, &full_v_unsqueezed)
        } else {
            self.inner
                .paged_attention_fn(&q, &full_k_unsqueezed, &full_v_unsqueezed)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qwen3_attention_forward_output_shape() {
        let device = candle_core::Device::Cpu;
        let num_heads = 8;
        let num_kv_heads = 4;
        let head_dim = 64;
        let hidden_size = num_heads * head_dim;
        let batch_size = 1;
        let seq_len = 4;

        let attention = Qwen3Attention::new(
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            10000.0,
            None,
            AttentionConfig::default(),
            false,
        )
        .unwrap();

        let x = Tensor::randn(0.0f32, 1.0, (batch_size, seq_len, hidden_size), &device).unwrap();
        let output = attention.forward(&x).unwrap();

        assert_eq!(output.dims(), &[batch_size, seq_len, hidden_size]);
    }

    #[test]
    fn test_qwen3_attention_with_qk_norm() {
        let device = candle_core::Device::Cpu;
        let num_heads = 8;
        let num_kv_heads = 4;
        let head_dim = 64;
        let hidden_size = num_heads * head_dim;
        let batch_size = 1;
        let seq_len = 4;

        let attention = Qwen3Attention::new(
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            10000.0,
            None,
            AttentionConfig::default(),
            true,
        )
        .unwrap();

        let x = Tensor::randn(0.0f32, 1.0, (batch_size, seq_len, hidden_size), &device).unwrap();
        let output = attention.forward(&x).unwrap();

        assert_eq!(output.dims(), &[batch_size, seq_len, hidden_size]);
    }

    #[test]
    fn test_qwen3_attention_decode_single_token() {
        let device = candle_core::Device::Cpu;
        let num_heads = 8;
        let num_kv_heads = 4;
        let head_dim = 64;
        let hidden_size = num_heads * head_dim;

        let attention = Qwen3Attention::new(
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            10000.0,
            None,
            AttentionConfig::default(),
            false,
        )
        .unwrap();

        let x = Tensor::ones((1, hidden_size), candle_core::DType::F32, &device).unwrap();
        let mut kv_cache =
            crate::kv_cache::PagedKvCache::new(1, num_heads, head_dim, 8, device.clone(), false)
                .unwrap();

        let block_ids: Vec<usize> = vec![0];
        let positions = vec![0];

        let result = attention
            .forward_decode(&x, &mut kv_cache, 0, &block_ids, 0, &positions)
            .unwrap();

        assert_eq!(result.dims(), &[1, 1, hidden_size]);
    }

    #[test]
    fn test_qwen3_attention_decode_with_kv_cache() {
        let device = candle_core::Device::Cpu;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 64;
        let hidden_size = num_heads * head_dim;

        let attention = Qwen3Attention::new(
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            10000.0,
            None,
            AttentionConfig::default(),
            false,
        )
        .unwrap();

        let mut kv_cache =
            crate::kv_cache::PagedKvCache::new(1, num_heads, head_dim, 16, device.clone(), false)
                .unwrap();

        for step in 0..8 {
            let x = Tensor::ones((1, hidden_size), candle_core::DType::F32, &device).unwrap();
            let block_id = step / 8;
            let block_ids: Vec<usize> = vec![block_id];
            let positions = vec![step];

            let result = attention
                .forward_decode(&x, &mut kv_cache, 0, &block_ids, step, &positions)
                .unwrap();

            assert_eq!(result.dims(), &[1, 1, hidden_size], "step={}", step);
        }
    }
}
