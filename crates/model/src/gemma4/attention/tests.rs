//! Unit tests for `Gemma4Attention`.
//!
//! Locks in two contracts for the sliding-window attention path:
//!
//! 1. **Sliding-window mask correctness**: a query at position `q`
//!    must mask out keys whose distance from `q` exceeds the
//!    sliding window (`-inf` in the additive mask), but allow
//!    in-window positions through (mask value ~0).
//! 2. **Paged-vs-non-paged equivalence**: the `forward(x, positions)`
//!    shortcut (which uses the paged attention path internally) must
//!    produce the same output as the explicit
//!    `project_qkv → apply_rope → expand_kv → compute_paged_attention`
//!    pipeline, to within `1e-5` max abs diff.
//!
//! All tests run on `Device::Cpu` with `DType::F32`.
use super::*;
use crate::config::architecture::RoPEConfig;
use candle_core::DType;

fn tiny_sliding_attention(sliding_window: usize) -> Result<Gemma4Attention> {
    let device = Device::Cpu;
    let hidden = 32;
    let num_heads = 4;
    let num_kv_heads = 2;
    let head_dim = 8;
    let rope_config = RoPEConfig {
        rope_theta: 10000.0,
        partial_rotary_factor: 1.0,
    };
    Gemma4Attention::new(
        hidden,
        num_heads,
        num_kv_heads,
        head_dim,
        sliding_window,
        LayerType::SlidingAttention,
        &rope_config,
        candle_nn::VarBuilder::zeros(DType::F32, &device),
    )
}

#[test]
fn test_sliding_causal_mask_blocks_out_of_window() -> Result<()> {
    let attn = tiny_sliding_attention(2)?;
    let device = Device::Cpu;
    let seq_len = 4;
    let positions: Vec<usize> = (0..seq_len).collect();
    let mask = attn.sliding_causal_mask(seq_len, seq_len, &positions, &device)?;
    let data: Vec<f32> = mask.flatten_all()?.to_vec1()?;

    // Query at position 3 should not attend to key at position 0 (distance 3 > window 2).
    let idx = 3 * seq_len;
    assert!(
        data[idx].is_infinite() && data[idx].is_sign_negative(),
        "expected -inf mask for out-of-window key, got {}",
        data[idx]
    );

    // Query at position 3 should attend to key at position 2 (distance 1 <= window 2).
    let idx = 3 * seq_len + 2;
    assert!(
        data[idx].abs() < 1e-6,
        "in-window causal pair should be unmasked"
    );
    Ok(())
}

#[test]
fn test_sliding_mask_matches_paged_path() -> Result<()> {
    let attn = tiny_sliding_attention(3)?;
    let device = Device::Cpu;
    let seq_len = 5;
    let hidden = 32;
    let x = Tensor::randn(0.0f32, 1.0, (1, seq_len, hidden), &device)?;
    let positions: Vec<usize> = (0..seq_len).collect();

    let non_paged = attn.forward(&x, &positions)?;

    let (q, k, v) = attn.project_qkv(&x)?;
    let q = q.reshape((1, seq_len, attn.num_heads, attn.head_dim))?;
    let k = k.reshape((1, seq_len, attn.num_kv_heads, attn.head_dim))?;
    let v = v.reshape((1, seq_len, attn.num_kv_heads, attn.head_dim))?;
    let (q, k) = attn.apply_rope(&q, &k, &positions)?;
    let k = attn.expand_kv(&k, attn.num_heads)?;
    let v = attn.expand_kv(&v, attn.num_heads)?;
    let q = q.transpose(1, 2)?.contiguous()?;
    let k = k.transpose(1, 2)?.contiguous()?;
    let v = v.transpose(1, 2)?.contiguous()?;
    let paged = attn.compute_paged_attention(&q, &k, &v, &positions)?;

    let diff = (&non_paged - &paged)?.abs()?;
    let max_diff: f32 = diff.max_all()?.to_scalar()?;
    assert!(
        max_diff < 1e-5,
        "non-paged sliding path should match paged attention, max_diff={max_diff}"
    );
    Ok(())
}
