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
        let repeat_factor = num_q_heads.div_ceil(num_kv_heads);
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
    let q_dims = q.dims();
    let k_dims = k.dims();
    eprintln!(
        "DEBUG paged_attention: q.dims={:?}, k.dims={:?}, num_heads={}, head_dim={}",
        q_dims, k_dims, num_heads, head_dim
    );

    let qk = Tensor::matmul(q, &k.transpose(2, 3)?.contiguous()?)?;
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

        let scale = 1.0 / (q.dims()[3] as f32).sqrt();
        let scale_tensor = Tensor::new(&[scale], q.device())?.broadcast_as(qk.dims())?;
        let qk = qk.mul(&scale_tensor)?;
        let attn = candle_nn::ops::softmax(&qk, 3)?;

        let out = Tensor::matmul(&attn, &v_tile)?;
        output_parts.push(out);
    }

    let attn_output = Tensor::cat(&output_parts, 2)?;
    let attn_output = attn_output.transpose(1, 2)?.contiguous()?;
    let attn_output = attn_output.reshape((batch_size, seq_len, num_heads * head_dim))?;
    Ok(attn_output)
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
    fn test_tiled_attention_exact_multiple_of_tile_size() {
        let batch_size = 1;
        let seq_len = 32;
        let num_heads = 4;
        let head_dim = 32;
        let tile_size = 16;

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
    fn test_expand_kv_invalid_dimensions() {
        let wrong_dims = Tensor::randn(0.0f32, 1.0, (1, 64), DEVICE).unwrap();

        let result = expand_kv(&wrong_dims, 8, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_expand_kv_repeat_factor() {
        let batch_size = 1;
        let seq_len = 1;
        let num_kv_heads = 1;
        let num_q_heads = 8;
        let head_dim = 128;

        let kv = Tensor::ones(
            (batch_size, seq_len, num_kv_heads, head_dim),
            candle_core::DType::F32,
            DEVICE,
        )
        .unwrap();

        let expanded = expand_kv(&kv, num_q_heads, num_kv_heads).unwrap();

        assert_eq!(
            expanded.dims(),
            &[batch_size, seq_len, num_q_heads, head_dim]
        );

        let expanded_data: Vec<f32> = expanded.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(expanded_data.len(), num_q_heads * head_dim);
        assert!(expanded_data.iter().all(|&v| (v - 1.0).abs() < 1e-6));
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
                    assert_eq!(mask_data[idx], 0.0, "Position ({}, {}) should be 0", i, j);
                } else {
                    assert_eq!(
                        mask_data[idx],
                        f32::NEG_INFINITY,
                        "Position ({}, {}) should be -inf",
                        i,
                        j
                    );
                }
            }
        }
    }

    #[test]
    fn test_causal_mask_tile_shape() {
        let batch_size = 2;
        let start = 4;
        let tile_len = 3;
        let key_len = 8;

        let mask = causal_mask_tile(batch_size, start, tile_len, key_len, DEVICE).unwrap();

        assert_eq!(mask.dims(), &[batch_size, 1, tile_len, key_len]);
    }

    #[test]
    fn test_causal_mask_tile_causality() {
        let batch_size = 1;
        let start = 2;
        let tile_len = 4;
        let key_len = 10;

        let mask = causal_mask_tile(batch_size, start, tile_len, key_len, DEVICE).unwrap();
        let mask_data: Vec<f32> = mask.flatten_all().unwrap().to_vec1().unwrap();

        for local_i in 0..tile_len {
            let global_i = start + local_i;
            for j in 0..key_len {
                let idx = local_i * key_len + j;
                if j <= global_i {
                    assert_eq!(
                        mask_data[idx], 0.0,
                        "local_i={}, global_i={}, j={} should be 0",
                        local_i, global_i, j
                    );
                } else {
                    assert_eq!(
                        mask_data[idx],
                        f32::NEG_INFINITY,
                        "local_i={}, global_i={}, j={} should be -inf",
                        local_i,
                        global_i,
                        j
                    );
                }
            }
        }
    }

    #[test]
    fn test_paged_attention_and_tiled_attention_produce_similar_output() {
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

        assert_eq!(paged_output.dims(), tiled_output.dims());

        let paged_data: Vec<f32> = paged_output.flatten_all().unwrap().to_vec1().unwrap();
        let tiled_data: Vec<f32> = tiled_output.flatten_all().unwrap().to_vec1().unwrap();

        let diff: f32 = paged_data
            .iter()
            .zip(tiled_data.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / paged_data.len() as f32;

        assert!(diff < 0.1, "Outputs differ too much: avg diff = {}", diff);
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

    #[test]
    fn test_paged_attention_decode_with_longer_kv() {
        let batch_size = 1;
        let num_heads = 8;
        let head_dim = 64;

        for kv_len in [1, 8, 16, 32, 64] {
            let q =
                Tensor::randn(0.0f32, 1.0, (batch_size, num_heads, 1, head_dim), DEVICE).unwrap();
            let k = Tensor::randn(
                0.0f32,
                1.0,
                (batch_size, num_heads, kv_len, head_dim),
                DEVICE,
            )
            .unwrap();
            let v = Tensor::randn(
                0.0f32,
                1.0,
                (batch_size, num_heads, kv_len, head_dim),
                DEVICE,
            )
            .unwrap();

            let output = paged_attention(&q, &k, &v, num_heads, head_dim).unwrap();

            assert_eq!(
                output.dims(),
                &[1, 1, num_heads * head_dim],
                "kv_len={}",
                kv_len
            );
        }
    }

    #[test]
    fn test_tiled_attention_decode_single_token() {
        let batch_size = 1;
        let seq_q = 1;
        let seq_kv = 32;
        let num_heads = 8;
        let head_dim = 64;
        let tile_size = 16;

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

        let output = tiled_attention(&q, &k, &v, num_heads, tile_size).unwrap();

        assert_eq!(output.dims(), &[batch_size, seq_q, num_heads * head_dim]);
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
    fn test_expand_kv_non_exact_division() {
        let batch_size = 1;
        let seq_len = 2;
        let num_kv_heads = 3;
        let num_q_heads = 10;
        let head_dim = 32;

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
    fn test_expand_kv_kv_equals_q() {
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
    fn test_expand_kv_large_ratio() {
        let batch_size = 1;
        let seq_len = 1;
        let num_kv_heads = 1;
        let num_q_heads = 32;
        let head_dim = 128;

        let kv = Tensor::ones(
            (batch_size, seq_len, num_kv_heads, head_dim),
            candle_core::DType::F32,
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
    fn test_paged_attention_and_tiled_attention_single_token() {
        let batch_size = 1;
        let num_heads = 8;
        let head_dim = 64;
        let tile_size = 16;

        let q = Tensor::randn(0.0f32, 1.0, (batch_size, num_heads, 1, head_dim), DEVICE).unwrap();
        let k = Tensor::randn(0.0f32, 1.0, (batch_size, num_heads, 16, head_dim), DEVICE).unwrap();
        let v = Tensor::randn(0.0f32, 1.0, (batch_size, num_heads, 16, head_dim), DEVICE).unwrap();

        let paged_output = paged_attention(&q, &k, &v, num_heads, head_dim).unwrap();
        let tiled_output = tiled_attention(&q, &k, &v, num_heads, tile_size).unwrap();

        assert_eq!(paged_output.dims(), tiled_output.dims());
        assert_eq!(paged_output.dims(), &[1, 1, num_heads * head_dim]);
    }

    #[test]
    fn test_causal_mask_decode_mode() {
        let seq_len = 1;
        let mask = causal_mask(seq_len, DEVICE).unwrap();

        assert_eq!(mask.dims(), &[1, 1, seq_len, seq_len]);
        let mask_data: Vec<f32> = mask.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(mask_data.len(), 1);
        assert_eq!(mask_data[0], 0.0);
    }
}
