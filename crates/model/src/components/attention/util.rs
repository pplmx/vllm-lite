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

// Unit tests are extracted to `tests.rs` to keep this file under the
// 800-line soft cap. See `tests.rs` for the test surface
// (paged_attention / tiled_attention output shapes, expand_kv GQA
// expansion paths and error path, causal_mask shape and causality).
#[cfg(test)]
mod tests;
