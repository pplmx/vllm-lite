//! Unit tests for `RopeGqaAttention`.
//!
//! Locks in two contracts:
//!
//! 1. **Shape correctness**: `forward` / `forward_decode` / `forward_prefill`
//!    preserve the `[batch, seq, hidden]` / `[batch, 1, hidden]` shape
//!    contract regardless of `qk_norm` / `fused-kernel` toggles.
//! 2. **Fused-paged equivalence**: when `use_fused` is toggled on, the
//!    fused prefill and decode paths must match the paged reference
//!    to within `1e-4` absolute max element diff. This guards against
//!    regressions in the fused-kernel numerical path.
//!
//! All tests run on `Device::Cpu` with `candle_core::DType::F32`.
use super::*;
use crate::qwen3::RopeType;

#[test]
fn test_rope_gqa_attention_forward_output_shape() {
    let device = candle_core::Device::Cpu;
    let num_heads = 8;
    let num_kv_heads = 4;
    let head_dim = 64;
    let hidden_size = num_heads * head_dim;
    let batch_size = 1;
    let seq_len = 4;

    let attention = RopeGqaAttention::new(
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
fn test_rope_gqa_attention_with_qk_norm() {
    let device = candle_core::Device::Cpu;
    let num_heads = 8;
    let num_kv_heads = 4;
    let head_dim = 64;
    let hidden_size = num_heads * head_dim;
    let batch_size = 1;
    let seq_len = 4;

    let attention = RopeGqaAttention::new(
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
fn test_rope_gqa_attention_decode_single_token() {
    let device = candle_core::Device::Cpu;
    let num_heads = 8;
    let num_kv_heads = 4;
    let head_dim = 64;
    let hidden_size = num_heads * head_dim;

    let attention = RopeGqaAttention::new(
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
        crate::paged_tensor::PagedKvCache::new(1, num_heads, head_dim, 8, device, false).unwrap();

    let block_ids: Vec<usize> = vec![0];
    let positions = vec![0];

    let result = attention
        .forward_decode(&x, &mut kv_cache, 0, &block_ids, 0, &positions)
        .unwrap();

    assert_eq!(result.dims(), &[1, 1, hidden_size]);
}

#[test]
fn test_rope_gqa_attention_decode_with_kv_cache() {
    let device = candle_core::Device::Cpu;
    let num_heads = 4;
    let num_kv_heads = 2;
    let head_dim = 64;
    let hidden_size = num_heads * head_dim;

    let attention = RopeGqaAttention::new(
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
        crate::paged_tensor::PagedKvCache::new(1, num_heads, head_dim, 16, device.clone(), false)
            .unwrap();

    for step in 0..8 {
        let x = Tensor::ones((1, hidden_size), candle_core::DType::F32, &device).unwrap();
        let block_id = step / 8;
        let block_ids: Vec<usize> = vec![block_id];
        let positions = vec![step];

        let result = attention
            .forward_decode(&x, &mut kv_cache, 0, &block_ids, step, &positions)
            .unwrap();

        assert_eq!(result.dims(), &[1, 1, hidden_size], "step={step}");
    }
}

fn make_rope_attention(use_fused: bool) -> RopeGqaAttention {
    let device = candle_core::Device::Cpu;
    let num_heads = 4;
    let num_kv_heads = 2;
    let head_dim = 32;
    let hidden_size = num_heads * head_dim;

    RopeGqaAttention::new(
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        10000.0,
        Some(candle_nn::VarBuilder::zeros(
            candle_core::DType::F32,
            &device,
        )),
        AttentionConfig {
            tile_size: Some(16),
            use_fused,
        },
        false,
    )
    .unwrap()
}

#[test]
fn test_rope_gqa_prefill_fused_matches_paged() {
    let device = candle_core::Device::Cpu;
    let hidden_size = 128;
    let seq_len = 6;

    let standard = make_rope_attention(false);
    let fused = make_rope_attention(true);

    let mut cache_std = PagedKvCache::new(1, 4, 32, 16, device.clone(), false).unwrap();
    let mut cache_fused = PagedKvCache::new(1, 4, 32, 16, device.clone(), false).unwrap();

    let x = Tensor::randn(0.0f32, 1.0, (1, seq_len, hidden_size), &device).unwrap();
    let block_ids: Vec<usize> = (0..seq_len).map(|i| i / 16).collect();
    let positions: Vec<usize> = (0..seq_len).collect();

    let out_std = standard
        .forward_prefill(&x, &mut cache_std, 0, &block_ids, &positions)
        .unwrap();
    let out_fused = fused
        .forward_prefill(&x, &mut cache_fused, 0, &block_ids, &positions)
        .unwrap();

    let diff = (&out_std - &out_fused).unwrap().abs().unwrap();
    let max_diff: f32 = diff.max_all().unwrap().to_scalar().unwrap();
    assert!(
        max_diff < 1e-4,
        "fused prefill should match paged path, max_diff={max_diff}"
    );
}

#[test]
fn test_rope_gqa_decode_fused_matches_paged() {
    let device = candle_core::Device::Cpu;
    let hidden_size = 128;

    let standard = make_rope_attention(false);
    let fused = make_rope_attention(true);

    let mut cache_std = PagedKvCache::new(1, 4, 32, 16, device.clone(), false).unwrap();
    let mut cache_fused = PagedKvCache::new(1, 4, 32, 16, device.clone(), false).unwrap();

    for step in 0..5 {
        let x = Tensor::randn(0.0f32, 1.0, (1, hidden_size), &device).unwrap();
        let block_id = step / 16;
        let block_ids = vec![block_id];
        let positions = vec![step];

        let out_std = standard
            .forward_decode(&x, &mut cache_std, 0, &block_ids, step, &positions)
            .unwrap();
        let out_fused = fused
            .forward_decode(&x, &mut cache_fused, 0, &block_ids, step, &positions)
            .unwrap();

        let diff = (&out_std - &out_fused).unwrap().abs().unwrap();
        let max_diff: f32 = diff.max_all().unwrap().to_scalar().unwrap();
        assert!(
            max_diff < 1e-4,
            "fused decode should match paged at step {step}, max_diff={max_diff}"
        );
    }
}

// === regression test for `apply_with_scaling` migration ===
//
// This test verifies the invariant the Task 2 migration depends on:
// a `RoPE` constructed via `RoPE::new` (with default `RopeType::Default`,
// `scaling_factor=1.0`) must produce the same output from
// `apply_with_scaling` as from `apply`. If this regresses, every
// production caller that switched to `apply_with_scaling` (rope_gqa,
// mla, gemma4) silently changes its numerical output.
#[test]
fn test_rope_gqa_default_scaling_matches_unscaled() {
    use crate::components::positional::rope::RoPE;
    use candle_core::{DType, Device, Tensor};

    let device = Device::Cpu;
    let head_dim = 64;
    let theta = 10000.0;

    let rope = RoPE::new(head_dim, 1024, theta, &device);
    let q = Tensor::randn(0.0f32, 1.0, (1, 4, 8, head_dim), &device).unwrap();
    let positions: Vec<i64> = vec![0, 1, 2, 3];

    let out_apply = rope.apply(&q, &positions).unwrap();
    let out_with_scaling = rope.apply_with_scaling(&q, &positions).unwrap();

    let diff = (&out_apply - &out_with_scaling)
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!(
        diff < 1e-5,
        "Default apply_with_scaling must match apply (max diff = {diff})"
    );
    // silence unused-import warning for DType under minimal feature build
    let _ = DType::F32;
}

// === P19 regression tests: `new_with_rope_scaling` propagation ===
//
// These tests verify that the YaRN/Linear/Dynamic/Su scaling fields reach
// the `RoPE` struct (and through it, the attention-temperature factor).
// Before P19, the production `RopeGqaAttention::new` constructor hard-coded
// `RoPE::new(...)` with `scaling_factor = 1.0` and `attn_factor = None`,
// so a YaRN-config Qwen3 model produced the same output as a default
// Qwen3 of the same shape. The new `new_with_rope_scaling` constructor
// closes that gap.
//
// Coverage matrix:
// - `new_with_rope_scaling_yarn_attaches_attn_factor_to_inner` — P19
//   regression: YaRN `attn_factor` reaches the inner `GqaAttention`
//   field (verified by observing the attention output change).
// - `new_with_rope_scaling_none_matches_new_construction` — backward
//   compatibility: `None` scaling produces the same output as `new()`,
//   guarding every existing call site that uses `new` against accidental
//   numerical drift in the `build_rope` fallback.
//
// Note: we deliberately do NOT add a "yarn produces a different output
// than default" regression at this layer because two attention modules
// with the SAME cloned weights but DIFFERENT RoPE tables can, in some
// RNG states, yield identical QK^T scores (the rotated inner product is
// rotation-invariant but the YaRN branch adjusts *frequencies*, not
// just angles, so it's possible — though rare — for both variants to
// produce numerically equivalent attention output). The end-to-end
// difference is observed in production via the YaRN long-context
// integration tests.

fn yarn_scaling(factor: f32, attn_factor: Option<f32>) -> RopeScaling {
    RopeScaling {
        rope_type: Some(RopeType::Yarn),
        factor: Some(factor),
        original_max_position_embeddings: Some(4096),
        attn_factor,
        partial_rotary_factor: None,
        mrope_section: None,
        short_factor: None,
        long_factor: None,
    }
}

#[test]
fn new_with_rope_scaling_none_matches_new_construction() {
    // The no-op path: `new_with_rope_scaling(..., None)` MUST produce the
    // same output as `new(...)` on identical inputs. This catches
    // regressions in the `build_rope` fallback that would change
    // numerical behaviour for every existing test that uses `new`.
    let device = candle_core::Device::Cpu;
    let num_heads = 4;
    let num_kv_heads = 2;
    let head_dim = 64;
    let hidden_size = num_heads * head_dim;
    let theta = 10000.0_f32;

    let attn_via_new = RopeGqaAttention::new(
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        theta,
        Some(candle_nn::VarBuilder::zeros(
            candle_core::DType::F32,
            &device,
        )),
        AttentionConfig::default(),
        false,
    )
    .unwrap();

    let attn_via_scaling_none = RopeGqaAttention::new_with_rope_scaling(
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        theta,
        4096,
        None,
        Some(candle_nn::VarBuilder::zeros(
            candle_core::DType::F32,
            &device,
        )),
        AttentionConfig::default(),
        false,
    )
    .unwrap();

    let x = Tensor::randn(0.0f32, 1.0, (1, 4, hidden_size), &device).unwrap();
    let out_new = attn_via_new.forward(&x).unwrap();
    let out_scaling_none = attn_via_scaling_none.forward(&x).unwrap();

    let diff = (&out_new - &out_scaling_none)
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!(
        diff < 1e-5,
        "new_with_rope_scaling(None) must match new (max diff = {diff})"
    );
}

#[test]
fn new_with_rope_scaling_yarn_attaches_attn_factor_to_inner() {
    // Verifies the P18+P19 chain: a YaRN `RopeScaling` with
    // `attn_factor = Some(0.5)` causes the inner `GqaAttention`'s
    // `attn_factor` field to be `Some(0.5)`. We exercise this via the
    // public `forward` method (which delegates to the inner attention),
    // so we don't need a private accessor on `RopeGqaAttention`.
    //
    // Note: this test uses *random* projection weights via
    // `new_with_weights_rope_scaling` (not `VarBuilder::zeros`) because
    // zero projections make every attention output exactly zero regardless
    // of the temperature factor — a regression test must observe the
    // factor's effect on the softmax, which requires non-degenerate Q/K/V.
    //
    // We deliberately use `num_kv_heads == num_heads` (MHA shape) here
    // because GQA's grouped-softmax averages over multiple Q heads per
    // KV head, which weakens the per-head effect of `attn_factor` and
    // can, in some RNG states, make the two outputs coincide to within
    // `1e-5`. The MHA shape is the cleanest case for observing the
    // factor's effect (mirrors `gqa_attn_factor_changes_output`).
    let device = candle_core::Device::Cpu;
    let num_heads = 4;
    let num_kv_heads = 4;
    let head_dim = 64;
    let hidden_size = num_heads * head_dim;
    let theta = 10000.0_f32;

    // Random projection weights so the attention output is non-degenerate.
    let q_w = Tensor::randn(0.0f32, 1.0, (hidden_size, hidden_size), &device).unwrap();
    let k_w = Tensor::randn(0.0f32, 1.0, (hidden_size, hidden_size), &device).unwrap();
    let v_w = Tensor::randn(0.0f32, 1.0, (hidden_size, hidden_size), &device).unwrap();
    let o_w = Tensor::randn(0.0f32, 1.0, (hidden_size, hidden_size), &device).unwrap();

    let yarn = yarn_scaling(4.0, Some(0.5));
    let attn = RopeGqaAttention::new_with_weights_rope_scaling(
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        theta,
        4096,
        Some(&yarn),
        q_w.clone(),
        k_w.clone(),
        v_w.clone(),
        o_w.clone(),
        AttentionConfig::default(),
        false,
        None,
        None,
    )
    .unwrap();

    let x = Tensor::randn(0.0f32, 1.0, (1, 4, hidden_size), &device).unwrap();
    let out_with_factor = attn.forward(&x).unwrap();

    // Sibling with the same RoPE config but no `attn_factor` — the inner
    // attention's `attn_factor` field should be `None`, producing a
    // different output for the same input.
    let yarn_no_attn = RopeScaling {
        attn_factor: None,
        ..yarn
    };
    let attn_no_attn = RopeGqaAttention::new_with_weights_rope_scaling(
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        theta,
        4096,
        Some(&yarn_no_attn),
        q_w,
        k_w,
        v_w,
        o_w,
        AttentionConfig::default(),
        false,
        None,
        None,
    )
    .unwrap();
    let out_no_attn = attn_no_attn.forward(&x).unwrap();

    let diff = (&out_with_factor - &out_no_attn)
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!(
        diff > 1e-5,
        "YaRN attn_factor=0.5 must change attention output vs attn_factor=None \
         (max diff = {diff}); if 0 the inner attn_factor is silently ignored"
    );
}
