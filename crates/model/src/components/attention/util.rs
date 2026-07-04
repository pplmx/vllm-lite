//! Attention utilities.
//!
//! Helper functions and small types used by the attention submodules
//! (`gqa`, `mla`, `flash`, `paged_gqa`, etc.). Kept separate from `mod.rs`
//! so `mod.rs` stays a thin re-export surface.

#![allow(clippy::too_many_arguments)]
// invariant: tensor-dimension casts (head_dim/seq_len -> f32/u32) are bounded
// by model architecture constants; precision loss / truncation is intentional.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]

use candle_core::{Result, Tensor};

/// Configuration for Attention. Constructed via the `builder()` associated function or by deserializing from JSON / TOML. Pass-by-value to construction APIs.
#[derive(Debug, Clone, Default)]
pub struct AttentionConfig {
    pub tile_size: Option<usize>,
    pub use_fused: bool,
}

impl AttentionConfig {
    #[must_use]
    pub const fn new(tile_size: Option<usize>, use_fused: bool) -> Self {
        Self {
            tile_size,
            use_fused,
        }
    }

    /// Returns a builder for configuring this type with the documented field defaults.
    /// Use `with_*(...)` to override individual fields, then `build()` to produce the type.
    #[allow(dead_code)]
    pub(crate) fn builder() -> AttentionConfigBuilder {
        AttentionConfigBuilder::default()
    }
}

/// Builder for [`AttentionConfig`].
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub(crate) struct AttentionConfigBuilder {
    inner: AttentionConfig,
}

impl AttentionConfigBuilder {
    #[allow(dead_code)]
    pub const fn with_tile_size(mut self, v: Option<usize>) -> Self {
        self.inner.tile_size = v;
        self
    }
    #[allow(dead_code)]
    pub const fn with_use_fused(mut self, v: bool) -> Self {
        self.inner.use_fused = v;
        self
    }
    /// build: build the [`AttentionConfig`].
    #[allow(dead_code)]
    pub const fn build(self) -> AttentionConfig {
        self.inner
    }
}

/// Expand a grouped-query-attention KV tensor along the head axis so it
/// has the same number of heads as the query tensor.
///
/// When `num_q_heads == num_kv_heads` the input is returned unchanged. For
/// GQA/MQA ratios the KV is broadcast along axis 2 (heads).
/// # Errors
///
/// Returns `Err` if the input tensor is not 4-dimensional, has the wrong
/// number of heads, or any underlying candle op fails.
pub fn expand_kv(kv: &Tensor, num_q_heads: usize, num_kv_heads: usize) -> Result<Tensor> {
    if num_q_heads == num_kv_heads {
        return Ok(kv.clone());
    }

    let dims = kv.dims();
    if dims.len() != 4 {
        return Err(candle_core::Error::msg(format!(
            "KV tensor must have exactly 4 dimensions [batch, seq, heads, dim], got {dims:?}"
        )));
    }

    let _ = dims[0];
    let _ = dims[1];
    let heads = dims[2];
    let _ = dims[3];

    if heads != num_kv_heads {
        return Err(candle_core::Error::msg(format!(
            "KV tensor has {heads} heads but expected {num_kv_heads}"
        )));
    }

    if !num_q_heads.is_multiple_of(num_kv_heads) {
        let repeat_factor = num_q_heads.div_ceil(num_kv_heads);
        let kv_repeated = kv.repeat(&[1, 1, repeat_factor, 1])?;
        return kv_repeated.narrow(2, 0, num_q_heads);
    }

    let repeat_factor = num_q_heads / num_kv_heads;
    kv.repeat(&[1, 1, repeat_factor, 1])
}

/// Build a `[1, 1, seq_len, seq_len]` causal attention mask.
///
/// Entries are `0.0` where the key position `j <= query position i`
/// (visible) and `-inf` otherwise (masked). Add this mask to the
/// pre-softmax `Q @ Kᵀ` tensor to enforce causality.
/// # Errors
///
/// Returns `Err` if `arange` or `broadcast_as` fails (e.g. device OOM).
pub fn causal_mask(seq_len: usize, device: &candle_core::Device) -> Result<Tensor> {
    let row_indices = Tensor::arange(0u32, seq_len as u32, device)?.reshape((1, 1, seq_len, 1))?;
    let col_indices = Tensor::arange(0u32, seq_len as u32, device)?.reshape((1, 1, 1, seq_len))?;
    let row_indices = row_indices.broadcast_as((1, 1, seq_len, seq_len))?;
    let col_indices = col_indices.broadcast_as((1, 1, seq_len, seq_len))?;
    let mask = row_indices.ge(&col_indices)?;
    let zero = Tensor::new(0.0f32, device)?.broadcast_as((1, 1, seq_len, seq_len))?;
    let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?.broadcast_as((1, 1, seq_len, seq_len))?;
    let mask = mask.where_cond(&zero, &neg_inf)?;
    Ok(mask)
}

/// Build a `[batch_size, 1, tile_len, key_len]` causal mask for one
/// query tile in [`tiled_attention`].
///
/// Unlike [`causal_mask`] which uses full row indices, this materialises
/// the mask for query indices `start..start+tile_len` and global column
/// indices `0..key_len` — necessary because the tiled path processes
/// rows in shifted windows.
/// # Errors
///
/// Returns `Err` if `from_slice` fails (shape mismatch or device error).
pub fn causal_mask_tile(
    batch_size: usize,
    start: usize,
    tile_len: usize,
    key_len: usize,
    device: &candle_core::Device,
) -> Result<Tensor> {
    let mask: Vec<f32> = (0..tile_len)
        .flat_map(|i| {
            let global_q = start + i;
            (0..key_len).map(move |j| {
                if j <= global_q {
                    0.0
                } else {
                    f32::NEG_INFINITY
                }
            })
        })
        .collect();
    Tensor::from_slice(&mask, (batch_size, 1, tile_len, key_len), device)
}

#[allow(clippy::too_many_arguments)]
/// Compute scaled dot-product causal attention for already-materialised
/// `q`, `k`, `v` tensors (i.e. the non-paged reference path used by unit
/// tests and small-batch inference).
///
/// Returns the output reshaped to `[batch, seq, num_heads * head_dim]`.
/// # Errors
///
/// Returns `Err` if any matmul, softmax, or reshape fails.
pub fn paged_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    num_heads: usize,
    head_dim: usize,
) -> Result<Tensor> {
    let batch_size = q.dims()[0];
    let seq_len = q.dims()[2];

    let qk = Tensor::matmul(q, &k.transpose(2, 3)?.contiguous()?)?;
    let mask = causal_mask(seq_len, q.device())?;
    let mask = mask.broadcast_as(qk.dims())?;
    let qk = (&qk + &mask)?;

    // H-11 #2: replaced `qk.mul(broadcast(scalar_tensor))` with `qk.affine(scale, 0.0)`.
    // Eliminates per-call scalar allocation and O(B*H*S*S) broadcast materialization.
    let scale = 1.0 / (head_dim as f32).sqrt();
    let qk = qk.affine(f64::from(scale), 0.0)?;
    let attn_weights = candle_nn::ops::softmax(&qk, 3)?.contiguous()?;

    let attn_output = Tensor::matmul(&attn_weights, v)?;
    let attn_output = attn_output.transpose(1, 2)?;
    let actual_seq_len = attn_output.dims()[1];
    let attn_output = attn_output.reshape((batch_size, actual_seq_len, num_heads * head_dim))?;
    Ok(attn_output)
}

#[allow(clippy::too_many_arguments)]
/// Compute causal attention one query tile at a time, then concatenate
/// the per-tile outputs along the sequence axis.
///
/// Used as the "fallback" path when the fused attention kernel is
/// unavailable; reduces peak memory at the cost of more kernel launches.
/// # Errors
///
/// Returns `Err` if any tile's matmul/softmax fails or the final
/// `concat` cannot stack the per-tile outputs.
pub fn tiled_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    num_heads: usize,
    tile_size: usize,
) -> Result<Tensor> {
    let batch_size = q.dims()[0];
    let seq_len = q.dims()[2];
    let head_dim = q.dims()[3];
    let num_tiles = seq_len.div_ceil(tile_size);
    let mut output_parts = Vec::new();

    for tile_idx in 0..num_tiles {
        let start = tile_idx * tile_size;
        if start >= seq_len {
            break;
        }
        let end = (start + tile_size).min(seq_len);
        let tile_len = end - start;

        let q_tile = q.narrow(2, start, tile_len)?;
        let k_tile = k.narrow(2, 0, end)?;
        let v_tile = v.narrow(2, 0, end)?;

        let qk = Tensor::matmul(&q_tile, &k_tile.transpose(2, 3)?.contiguous()?)?;

        let mask = causal_mask_tile(q.dims()[0], start, tile_len, end, q.device())?;
        let mask = mask.broadcast_as(qk.dims())?;
        let qk = (&qk + &mask)?;

        // H-11 #2: replaced `qk.mul(broadcast(scalar_tensor))` with `qk.affine(scale, 0.0)`.
        // Per-tile savings compound: tiled forward at seq_len=2048 / tile_size=16
        // saves ~128 O(B*H*T*T) broadcast materializations.
        let scale = 1.0 / (q.dims()[3] as f32).sqrt();
        let qk = qk.affine(f64::from(scale), 0.0)?;
        let attn = candle_nn::ops::softmax(&qk, 3)?.contiguous()?;

        let out = Tensor::matmul(&attn, &v_tile)?;
        output_parts.push(out);
    }

    let attn_output = Tensor::cat(&output_parts, 2)?;
    let attn_output = attn_output.transpose(1, 2)?.contiguous()?;
    let attn_output = attn_output.reshape((batch_size, seq_len, num_heads * head_dim))?;
    Ok(attn_output)
}

#[cfg(test)]
mod tests {
    use super::*;

    const DEVICE: &candle_core::Device = &candle_core::Device::Cpu;

    #[test]
    fn test_paged_attention_output_shape() {
        let batch_size = 2;
        let seq_len = 4;
        let num_heads = 8;
        let head_dim = 64;

        let q = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, num_heads, seq_len, head_dim),
            DEVICE,
        )
        .unwrap();
        let k = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, num_heads, seq_len, head_dim),
            DEVICE,
        )
        .unwrap();
        let v = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, num_heads, seq_len, head_dim),
            DEVICE,
        )
        .unwrap();

        let output = paged_attention(&q, &k, &v, num_heads, head_dim).unwrap();

        assert_eq!(output.dims(), &[batch_size, seq_len, num_heads * head_dim]);
    }

    #[test]
    fn test_tiled_attention_output_shape_matches_paged_attention() {
        let batch_size = 1;
        let seq_len = 20;
        let num_heads = 4;
        let head_dim = 32;
        let tile_size = 8;

        let q = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, num_heads, seq_len, head_dim),
            DEVICE,
        )
        .unwrap();
        let k = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, num_heads, seq_len, head_dim),
            DEVICE,
        )
        .unwrap();
        let v = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, num_heads, seq_len, head_dim),
            DEVICE,
        )
        .unwrap();

        let paged_output = paged_attention(&q, &k, &v, num_heads, head_dim).unwrap();
        let tiled_output = tiled_attention(&q, &k, &v, num_heads, tile_size).unwrap();

        let expected = [batch_size, seq_len, num_heads * head_dim];
        assert_eq!(
            paged_output.dims(),
            &expected[..],
            "paged_attention output shape mismatch"
        );
        assert_eq!(
            tiled_output.dims(),
            &expected[..],
            "tiled_attention output shape mismatch"
        );
    }

    #[test]
    fn test_tiled_attention_single_tile() {
        let batch_size = 1;
        let seq_len = 8;
        let num_heads = 4;
        let head_dim = 32;
        let tile_size = 16;

        let q = Tensor::ones(
            (batch_size, num_heads, seq_len, head_dim),
            candle_core::DType::F32,
            DEVICE,
        )
        .unwrap();
        let k = Tensor::ones(
            (batch_size, num_heads, seq_len, head_dim),
            candle_core::DType::F32,
            DEVICE,
        )
        .unwrap();
        let v = Tensor::ones(
            (batch_size, num_heads, seq_len, head_dim),
            candle_core::DType::F32,
            DEVICE,
        )
        .unwrap();

        let output = tiled_attention(&q, &k, &v, num_heads, tile_size).unwrap();

        assert_eq!(output.dims(), &[batch_size, seq_len, num_heads * head_dim]);
    }

    #[test]
    fn test_expand_kv_gqa_basic() {
        let batch_size = 1;
        let seq_len = 4;
        let num_kv_heads = 2;
        let num_q_heads = 14;
        let head_dim = 64;

        let kv = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, seq_len, num_kv_heads, head_dim),
            DEVICE,
        )
        .unwrap();

        let expanded = expand_kv(&kv, num_q_heads, num_kv_heads).unwrap();

        assert_eq!(
            expanded.dims(),
            &[batch_size, seq_len, num_q_heads, head_dim]
        );
    }

    #[test]
    fn test_expand_kv_no_expansion_needed() {
        let batch_size = 1;
        let seq_len = 4;
        let num_heads = 8;
        let head_dim = 64;

        let kv = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, seq_len, num_heads, head_dim),
            DEVICE,
        )
        .unwrap();

        let expanded = expand_kv(&kv, num_heads, num_heads).unwrap();

        assert_eq!(expanded.dims(), kv.dims());
    }

    #[test]
    fn test_expand_kv_invalid_head_count() {
        let batch_size = 1;
        let seq_len = 4;
        let wrong_kv_heads = 4;
        let expected_kv_heads = 2;
        let num_q_heads = 14;
        let head_dim = 64;

        let kv = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, seq_len, wrong_kv_heads, head_dim),
            DEVICE,
        )
        .unwrap();

        let result = expand_kv(&kv, num_q_heads, expected_kv_heads);
        assert!(result.is_err());
    }

    #[test]
    fn test_causal_mask_shape() {
        let seq_len = 16;
        let mask = causal_mask(seq_len, DEVICE).unwrap();

        assert_eq!(mask.dims(), &[1, 1, seq_len, seq_len]);
    }

    #[test]
    fn test_causal_mask_causality() {
        let seq_len = 4;
        let mask = causal_mask(seq_len, DEVICE).unwrap();
        let mask_data: Vec<f32> = mask.flatten_all().unwrap().to_vec1().unwrap();

        for i in 0..seq_len {
            for j in 0..seq_len {
                let idx = i * seq_len + j;
                if j <= i {
                    assert!(
                        mask_data[idx].abs() < 1e-6,
                        "Position ({i}, {j}) should be 0"
                    );
                } else {
                    assert!(
                        mask_data[idx] == f32::NEG_INFINITY
                            || mask_data[idx].is_infinite() && mask_data[idx] < 0.0,
                        "Position ({i}, {j}) should be -inf, got {}",
                        mask_data[idx]
                    );
                }
            }
        }
    }

    #[test]
    fn test_expand_kv_exact_division() {
        let batch_size = 2;
        let seq_len = 4;
        let num_kv_heads = 2;
        let num_q_heads = 16;
        let head_dim = 64;

        let kv = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, seq_len, num_kv_heads, head_dim),
            DEVICE,
        )
        .unwrap();

        let expanded = expand_kv(&kv, num_q_heads, num_kv_heads).unwrap();

        assert_eq!(
            expanded.dims(),
            &[batch_size, seq_len, num_q_heads, head_dim]
        );
    }

    #[test]
    fn test_paged_attention_single_token_decode() {
        let batch_size = 1;
        let seq_q = 1;
        let seq_kv = 8;
        let num_heads = 16;
        let head_dim = 128;

        let q = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, num_heads, seq_q, head_dim),
            DEVICE,
        )
        .unwrap();
        let k = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, num_heads, seq_kv, head_dim),
            DEVICE,
        )
        .unwrap();
        let v = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, num_heads, seq_kv, head_dim),
            DEVICE,
        )
        .unwrap();

        let output = paged_attention(&q, &k, &v, num_heads, head_dim).unwrap();

        assert_eq!(output.dims(), &[batch_size, seq_q, num_heads * head_dim]);
    }
}
