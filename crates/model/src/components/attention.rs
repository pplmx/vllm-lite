#![allow(clippy::too_many_arguments)]

use candle_core::{Result, Tensor};

#[derive(Debug, Clone, Default)]
pub struct AttentionConfig {
    pub tile_size: Option<usize>,
    pub use_fused: bool,
}

impl AttentionConfig {
    pub fn new(tile_size: Option<usize>, use_fused: bool) -> Self {
        Self {
            tile_size,
            use_fused,
        }
    }
}

/// Expand KV heads to match Q heads count for GQA
///
/// For GQA, we repeat KV heads to match the number of Q heads.
/// Example: num_q_heads=14, num_kv_heads=2 => repeat_factor=7
pub fn expand_kv(kv: &Tensor, num_q_heads: usize, num_kv_heads: usize) -> Result<Tensor> {
    if num_q_heads == num_kv_heads {
        // Standard MHA - no expansion needed
        return Ok(kv.clone());
    }

    let dims = kv.dims();
    if dims.len() != 4 {
        return Err(candle_core::Error::msg(format!(
            "KV tensor must have exactly 4 dimensions [batch, seq, heads, dim], got {:?}",
            dims
        )));
    }

    let _batch = dims[0];
    let _seq = dims[1];
    let heads = dims[2];
    let _dim = dims[3];

    if heads != num_kv_heads {
        return Err(candle_core::Error::msg(format!(
            "KV tensor has {} heads but expected {}",
            heads, num_kv_heads
        )));
    }

    // Check if num_q_heads is divisible by num_kv_heads
    if num_q_heads % num_kv_heads != 0 {
        // Edge case: use ceil division and slice
        let repeat_factor = (num_q_heads + num_kv_heads - 1) / num_kv_heads;
        let kv_repeated = kv.repeat(&[1, 1, repeat_factor, 1])?;
        // Slice to exact num_q_heads
        return kv_repeated.narrow(2, 0, num_q_heads);
    }

    let repeat_factor = num_q_heads / num_kv_heads;
    kv.repeat(&[1, 1, repeat_factor, 1])
}

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

#[allow(clippy::too_many_arguments)]
pub fn paged_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    num_heads: usize,
    head_dim: usize,
) -> Result<Tensor> {
    let batch_size = q.dims()[0];
    let seq_len = q.dims()[2];

    let qk = Tensor::matmul(q, &k.transpose(2, 3)?)?;
    let mask = causal_mask(seq_len, q.device())?;
    let mask = mask.broadcast_as(qk.dims())?;
    let qk = (&qk + &mask)?;

    let scale = 1.0 / (head_dim as f32).sqrt();
    let scale_tensor = Tensor::new(&[scale], q.device())?.broadcast_as(qk.dims())?;
    let qk = qk.mul(&scale_tensor)?;
    let attn_weights = candle_nn::ops::softmax(&qk, 3)?;

    let attn_output = Tensor::matmul(&attn_weights, v)?;
    let attn_output = attn_output.transpose(1, 2)?;
    let actual_seq_len = attn_output.dims()[1];
    let attn_output = attn_output.reshape((batch_size, actual_seq_len, num_heads * head_dim))?;
    Ok(attn_output)
}

#[allow(clippy::too_many_arguments)]
pub fn tiled_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    _num_heads: usize,
    tile_size: usize,
) -> Result<Tensor> {
    let seq_len = q.dims()[2];
    let num_tiles = seq_len.div_ceil(tile_size);
    let mut output_parts = Vec::new();

    for tile_idx in 0..num_tiles {
        let start = tile_idx * tile_size;
        let end = (start + tile_size).min(seq_len);
        let tile_len = end - start;

        let q_tile = q.narrow(2, start, tile_len)?;
        let k_tile = k.narrow(2, 0, end)?;
        let v_tile = v.narrow(2, 0, end)?;

        let qk = Tensor::matmul(&q_tile, &k_tile.transpose(2, 3)?)?;

        let mask = causal_mask_tile(q.dims()[0], start, tile_len, end, q.device())?;
        let mask = mask.broadcast_as(qk.dims())?;
        let qk = (&qk + &mask)?;

        let scale = 1.0 / (q.dims()[3] as f32).sqrt();
        let scale_tensor = Tensor::new(&[scale], q.device())?.broadcast_as(qk.dims())?;
        let qk = qk.mul(&scale_tensor)?;
        let attn = candle_nn::ops::softmax(&qk, 3)?;

        let out = Tensor::matmul(&attn, &v_tile)?;
        output_parts.push(out);
    }

    Tensor::cat(&output_parts, 2)
}

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
