//! `Gemma4Attention::forward` + the paged prefill/decode paths.
//!
//! Five public entry points:
//!
//! - [`Self::forward`] — dispatcher that picks `forward_full` or `forward_sliding`
//!   based on `self.layer_type`
//! - [`Self::forward_full`] / [`Self::forward_sliding`] — non-paged paths,
//!   delegate to `gqa_attention`
//! - [`Self::forward_prefill`] / [`Self::forward_decode`] — paged paths that
//!   read/write KV through `PagedKvCache`

use candle_core::{Result, Tensor};
use tracing::trace;

use super::Gemma4Attention;
use crate::components::attention::paged_gqa::{read_decode_kv, write_prefill_kv};
use crate::config::architecture::LayerType;
use crate::paged_tensor::PagedKvCache;

impl Gemma4Attention {
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
            positions,
            seq_len,
            &k_expanded,
            &v_expanded,
        )?;

        self.compute_paged_attention(&q, &k_expanded, &v_expanded, positions)
    }

    /// Chunked-prefill continuation: attend over the cached prefix plus new tokens.
    /// # Errors
    ///
    /// Returns `Err` if projection, RoPE, KV read/write, or attention fails.
    pub fn forward_prefill_continue(
        &self,
        x: &Tensor,
        kv_cache: &mut PagedKvCache,
        layer_idx: usize,
        block_ids: &[usize],
        positions: &[usize],
        num_computed_tokens: usize,
    ) -> Result<Tensor> {
        use crate::components::attention::paged_gqa::{
            compute_gqa_attention, prefill_continue_causal_mask, project_attention_output,
        };

        let (batch_size, seq_len, _) = x.dims3()?;

        let (q, k, v) = self.project_qkv(x)?;
        let q = q.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?;
        let k = k.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?;
        let v = v.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?;

        let (q, k) = self.apply_rope(&q, &k, positions)?;
        let k_expanded = self.expand_kv(&k, self.num_heads)?;
        let v_expanded = self.expand_kv(&v, self.num_heads)?;

        let q = q.transpose(1, 2)?.contiguous()?;
        let k_new = k_expanded.transpose(1, 2)?.contiguous()?;
        let v_new = v_expanded.transpose(1, 2)?.contiguous()?;

        let (cached_k, cached_v) =
            kv_cache.read_kv(layer_idx, block_ids, num_computed_tokens)?;
        let cached_k = cached_k
            .transpose(0, 1)?
            .unsqueeze(0)?
            .contiguous()?;
        let cached_v = cached_v
            .transpose(0, 1)?
            .unsqueeze(0)?
            .contiguous()?;

        write_prefill_kv(
            kv_cache,
            layer_idx,
            block_ids,
            positions,
            seq_len,
            &k_new,
            &v_new,
        )?;

        let full_k = Tensor::cat(&[&cached_k, &k_new], 2)?.contiguous()?;
        let full_v = Tensor::cat(&[&cached_v, &v_new], 2)?.contiguous()?;

        let mask = match self.layer_type {
            LayerType::SlidingAttention => Some(self.sliding_causal_mask(
                seq_len,
                num_computed_tokens + seq_len,
                positions,
                q.device(),
            )?),
            LayerType::FullAttention => Some(prefill_continue_causal_mask(
                seq_len,
                num_computed_tokens,
                q.device(),
            )?),
        };

        let attn_output =
            compute_gqa_attention(&q, &full_k, &full_v, self.head_dim, mask.as_ref())?;
        project_attention_output(
            &attn_output,
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
            &self.o_proj,
        )
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
        let _ = num_computed_tokens + 1;

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
}
