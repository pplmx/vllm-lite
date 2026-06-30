//! Shared paged KV cache operations and GQA attention matmul for production architectures.

// invariant: tensor-dimension casts (head_dim/token_idx -> f32/u32) are
// bounded by model architecture constants; precision loss / truncation is
// intentional.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use std::collections::BTreeMap;

use candle_core::{Device, Module, Result, Tensor};
use candle_nn::Linear;

use super::causal_mask;
use crate::paged_tensor::PagedKvCache;

/// Write expanded K/V tensors to the paged KV cache during prefill.
///
/// # Errors
///
/// Returns `Err` if the operation fails.
/// `k` and `v` must be `[batch, num_heads, seq_len, head_dim]`.
pub fn write_prefill_kv(
    kv_cache: &mut PagedKvCache,
    layer_idx: usize,
    block_ids: &[usize],
    seq_len: usize,
    k: &Tensor,
    v: &Tensor,
) -> Result<()> {
    let mut block_groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (token_idx, &block_id) in block_ids.iter().take(seq_len).enumerate() {
        block_groups.entry(block_id).or_default().push(token_idx);
    }

    for (block_id, token_indices) in &block_groups {
        if token_indices.is_empty() {
            continue;
        }

        let indices: Vec<u32> = token_indices.iter().map(|&i| i as u32).collect();
        let indices_tensor = Tensor::new(indices.as_slice(), k.device())?;

        let k_block = k.index_select(&indices_tensor, 2)?.contiguous()?;
        let v_block = v.index_select(&indices_tensor, 2)?.contiguous()?;
        let k_block = k_block.transpose(1, 2)?.contiguous()?;
        let v_block = v_block.transpose(1, 2)?.contiguous()?;

        kv_cache.write_kv_batch(layer_idx, *block_id, 0, &k_block, &v_block)?;
    }

    Ok(())
}

/// Read cached KV, append the current token, and write the new token into the cache.
///
/// # Errors
///
/// Returns `Err` if reading or parsing the source fails.
/// Returns full K/V in `[batch, num_heads, kv_seq, head_dim]` layout.
pub fn read_decode_kv(
    kv_cache: &mut PagedKvCache,
    layer_idx: usize,
    block_ids: &[usize],
    num_computed_tokens: usize,
    k_for_cache: &Tensor,
    v_for_cache: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let (cached_k, cached_v) = kv_cache.read_kv(layer_idx, block_ids, num_computed_tokens)?;
    let cached_k = cached_k.transpose(0, 1)?.contiguous()?;
    let cached_v = cached_v.transpose(0, 1)?.contiguous()?;

    let full_k = Tensor::cat(&[&cached_k, k_for_cache], 1)?.contiguous()?;
    let full_v = Tensor::cat(&[&cached_v, v_for_cache], 1)?.contiguous()?;

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
    Ok((full_k, full_v))
}

/// # Errors
///
/// Returns `Err` if the operation fails.
/// Causal mask for square prefill (`q_seq == kv_seq > 1`); none for decode or non-square paths.
pub fn prefill_causal_mask(q_seq: usize, kv_seq: usize, device: &Device) -> Result<Option<Tensor>> {
    if q_seq == kv_seq && q_seq > 1 {
        Ok(Some(causal_mask(q_seq, device)?))
    } else {
        Ok(None)
    }
}

/// Scaled dot-product GQA attention with optional broadcast mask.
///
/// # Errors
///
/// Returns `Err` if the operation fails.
/// `q`, `k`, `v` are `[batch, num_heads, seq, head_dim]`.
pub fn compute_gqa_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    head_dim: usize,
    mask: Option<&Tensor>,
) -> Result<Tensor> {
    let mut qk = Tensor::matmul(q, &k.transpose(2, 3)?)?;
    if let Some(mask) = mask {
        qk = qk.broadcast_add(mask)?;
    }
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    qk = qk.mul(&Tensor::new(&[scale], q.device())?.broadcast_as(qk.dims())?)?;
    let attn_weights = candle_nn::ops::softmax(&qk, 3)?;
    Tensor::matmul(&attn_weights, v)
}

/// # Errors
///
/// Returns `Err` if the operation fails.
/// Reshape head-first attention output and apply the output projection.
pub fn project_attention_output(
    attn_output: &Tensor,
    batch_size: usize,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    o_proj: &Linear,
) -> Result<Tensor> {
    let attn_output = attn_output.transpose(1, 2)?;
    let attn_output = attn_output.reshape((batch_size, seq_len, num_heads * head_dim))?;
    o_proj.forward(&attn_output)
}

/// Plugin for applying rotary embeddings to Q/K before paged KV write.
pub trait QkRotaryEmb: Send + Sync {
    /// Runs the operation.
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    fn apply_qk(&self, q: &Tensor, k: &Tensor, positions: &[usize]) -> Result<(Tensor, Tensor)>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::attention::expand_kv;

    #[test]
    fn test_prefill_causal_mask_square_only() {
        let device = Device::Cpu;
        assert!(prefill_causal_mask(4, 4, &device).unwrap().is_some());
        assert!(prefill_causal_mask(1, 5, &device).unwrap().is_none());
    }

    #[test]
    fn test_write_and_read_decode_kv_roundtrip() -> Result<()> {
        let device = Device::Cpu;
        let num_heads = 2;
        let head_dim = 8;
        let mut kv_cache = PagedKvCache::new(1, num_heads, head_dim, 16, device.clone(), false)?;

        let seq_len = 3usize;
        let k = Tensor::randn(0.0f32, 1.0, (1, num_heads, seq_len, head_dim), &device)?;
        let v = Tensor::randn(0.0f32, 1.0, (1, num_heads, seq_len, head_dim), &device)?;
        let block_ids: Vec<usize> = (0..seq_len).map(|i| i / 16).collect();

        write_prefill_kv(&mut kv_cache, 0, &block_ids, seq_len, &k, &v)?;

        let k_new = Tensor::randn(0.0f32, 1.0, (num_heads, 1, head_dim), &device)?;
        let v_new = Tensor::randn(0.0f32, 1.0, (num_heads, 1, head_dim), &device)?;
        let (full_k, full_v) = read_decode_kv(&mut kv_cache, 0, &[0], seq_len, &k_new, &v_new)?;

        assert_eq!(full_k.dims(), &[1, num_heads, seq_len + 1, head_dim]);
        assert_eq!(full_v.dims(), &[1, num_heads, seq_len + 1, head_dim]);
        Ok(())
    }

    #[test]
    fn test_compute_gqa_attention_with_causal_mask() -> Result<()> {
        let device = Device::Cpu;
        let num_heads = 2;
        let head_dim = 8;
        let seq_len = 4;
        let q = Tensor::randn(0.0f32, 1.0, (1, num_heads, seq_len, head_dim), &device)?;
        let k = q.clone();
        let v = Tensor::randn(0.0f32, 1.0, (1, num_heads, seq_len, head_dim), &device)?;
        let mask = prefill_causal_mask(seq_len, seq_len, &device)?.unwrap();
        let out = compute_gqa_attention(&q, &k, &v, head_dim, Some(&mask))?;
        assert_eq!(out.dims(), &[1, num_heads, seq_len, head_dim]);
        Ok(())
    }

    #[test]
    fn test_expand_kv_used_by_paged_path() -> Result<()> {
        let device = Device::Cpu;
        let k = Tensor::randn(0.0f32, 1.0, (1, 4, 2, 8), &device)?;
        let expanded = expand_kv(&k, 4, 2)?;
        assert_eq!(expanded.dims(), &[1, 4, 4, 8]);
        Ok(())
    }
}
