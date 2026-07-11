//! Unit tests for the `rope` (rotary positional embedding) module.
//!
//! Two free-function paths and one struct path are exercised:
//!
//! 1. **`apply_rope` (free fn, 3 tests)**: shape preservation
//!    (`[B, H, S, D]` → same shape), positional sensitivity
//!    (different positions produce different outputs), determinism
//!    (same input → same output).
//! 2. **`precompute_rope_cache` (3 tests)**: cache length matches
//!    `max_seq_len * (head_dim / 2)`, edge cases for length=1 and
//!    length=10.
//! 3. **`RoPE::new` + `RoPE::apply` + `RoPE::forward` (8 tests)**:
//!    construction records `theta` / `head_dim` / `scaling_factor`,
//!    shape preservation for both q and k outputs, rotation actually
//!    modifies the tensor (not identity), and numerical robustness
//!    for large positions (8 KiB) — every output element must
//!    remain finite.
//!
//! All tests run on `Device::Cpu` with `DType::F32`.
use super::*;
use candle_core::{DType, Device, Tensor};

#[test]
fn test_apply_rope_returns_same_shape() -> Result<()> {
    let device = Device::Cpu;
    let query = Tensor::ones((2, 4, 2, 32), DType::F32, &device)?;
    let positions: Vec<i64> = vec![0, 1, 2, 3];

    let result = apply_rope(&query, &positions, 10000.0)?;
    assert_eq!(result.dims(), query.dims());

    Ok(())
}

#[test]
fn test_apply_rope_different_positions() -> Result<()> {
    let device = Device::Cpu;
    let query = Tensor::ones((1, 2, 2, 4), DType::F32, &device)?;

    let pos0: Vec<i64> = vec![0, 1];
    let out0 = apply_rope(&query, &pos0, 10000.0)?;

    let pos1: Vec<i64> = vec![10, 11];
    let out1 = apply_rope(&query, &pos1, 10000.0)?;

    let diff = (out0 - out1)?.abs()?.sum_all()?;
    assert!(
        diff.to_scalar::<f32>()? > 1e-5,
        "RoPE should produce different outputs for different positions"
    );

    Ok(())
}

#[test]
fn test_apply_rope_deterministic() -> Result<()> {
    let device = Device::Cpu;
    let query = Tensor::ones((1, 2, 2, 4), DType::F32, &device)?;
    let positions: Vec<i64> = vec![0, 1];

    let out1 = apply_rope(&query, &positions, 10000.0)?;
    let out2 = apply_rope(&query, &positions, 10000.0)?;

    let diff = (out1 - out2)?.abs()?.sum_all()?;
    assert!(
        diff.to_scalar::<f32>()?.abs() < 1e-6,
        "RoPE should be deterministic"
    );

    Ok(())
}

#[test]
fn test_precompute_rope_cache_length() {
    let cache = precompute_rope_cache(10, 64, 10000.0);
    assert_eq!(cache.len(), 10 * 32);
}

#[test]
fn test_precompute_rope_cache_first_position() {
    let cache = precompute_rope_cache(1, 64, 10000.0);
    assert_eq!(cache.len(), 32);
}

#[test]
fn test_precompute_rope_cache_values() {
    let cache = precompute_rope_cache(10, 64, 10000.0);
    assert_eq!(cache.len(), 320);
}

#[test]
fn test_rope_creation() {
    let device = Device::Cpu;
    let rope = RoPE::new(128, 2048, 10000.0, &device);
    assert!((rope.theta - 10_000.0).abs() < 1e-6);
    assert_eq!(rope.head_dim, 128);
    assert!((rope.scaling_factor - 1.0).abs() < 1e-6);
}

#[test]
fn test_rope_apply() -> Result<()> {
    let device = Device::Cpu;
    let rope = RoPE::new(64, 1024, 10000.0, &device);
    let query = Tensor::ones((1, 4, 8, 64), DType::F32, &device)?;
    let positions: Vec<i64> = vec![0, 1, 2, 3];

    let result = rope.apply(&query, &positions)?;
    assert_eq!(result.dims(), query.dims());

    Ok(())
}

#[test]
fn test_rope_forward_q_shape_preserved() -> Result<()> {
    let device = Device::Cpu;
    let rope = RoPE::new(64, 1024, 10000.0, &device);

    let q = Tensor::randn(0.0f32, 1.0, (2, 4, 8, 64), &device)?;
    let k = Tensor::randn(0.0f32, 1.0, (2, 4, 8, 64), &device)?;

    let (q_out, _) = rope.forward(&q, &k, 0)?;
    assert_eq!(q_out.dims(), q.dims());
    Ok(())
}

#[test]
fn test_rope_forward_k_shape_preserved() -> Result<()> {
    let device = Device::Cpu;
    let rope = RoPE::new(64, 1024, 10000.0, &device);

    let q = Tensor::randn(0.0f32, 1.0, (2, 4, 8, 64), &device)?;
    let k = Tensor::randn(0.0f32, 1.0, (2, 4, 8, 64), &device)?;

    let (_, k_out) = rope.forward(&q, &k, 0)?;
    assert_eq!(k_out.dims(), k.dims());
    Ok(())
}

#[test]
fn test_rope_rotation_applied() -> Result<()> {
    let device = Device::Cpu;
    let rope = RoPE::new(64, 1024, 10000.0, &device);

    let q = Tensor::ones((1, 2, 8, 64), DType::F32, &device)?;
    let k = Tensor::ones((1, 2, 8, 64), DType::F32, &device)?;

    let (q_out, _) = rope.forward(&q, &k, 0)?;

    let diff = (&q_out - &q)?.abs()?;
    let sum_diff = diff.sum_all()?.to_scalar::<f32>()?;
    assert!(sum_diff > 1e-6, "RoPE should modify the tensor");
    Ok(())
}

#[test]
fn test_rope_minimal_head_dim() -> Result<()> {
    let device = Device::Cpu;
    let rope = RoPE::new(64, 512, 10000.0, &device);

    let q = Tensor::randn(0.0f32, 1.0, (1, 1, 1, 64), &device)?;
    let k = Tensor::randn(0.0f32, 1.0, (1, 1, 1, 64), &device)?;

    let (q_out, _) = rope.forward(&q, &k, 0)?;
    assert_eq!(q_out.dims(), q.dims());
    Ok(())
}

#[test]
fn test_rope_large_position() -> Result<()> {
    let device = Device::Cpu;
    let rope = RoPE::new(64, 1024, 10000.0, &device);

    let q = Tensor::randn(0.0f32, 1.0, (1, 1, 1, 64), &device)?;
    let k = Tensor::randn(0.0f32, 1.0, (1, 1, 1, 64), &device)?;

    let (q_out, _) = rope.forward(&q, &k, 8192)?;
    let flat = q_out.flatten_all()?;
    for i in 0..flat.elem_count() {
        let val: f32 = flat.get(i)?.to_scalar()?;
        assert!(val.is_finite(), "Value at index {i} is not finite: {val}");
    }
    Ok(())
}

// === Phase 15: long-context scaling (RopeType-aware apply_with_scaling) ===

use crate::qwen3::config::{RopeScaling, RopeType};

fn scaled_rope(rope_type: RopeType, scaling_factor: f32) -> RoPE {
    RoPE {
        theta: 10000.0,
        head_dim: 64,
        max_position: 1024,
        scaling_factor,
        device: Device::Cpu,
        rope_type,
        attn_factor: None,
        original_max_position: None,
    }
}

#[test]
fn test_apply_with_scaling_default_matches_unscaled() -> Result<()> {
    // Default rope_type + scaling_factor=1.0 should produce the same
    // output as the plain apply path (no scaling).
    let device = Device::Cpu;
    let q = Tensor::randn(0.0f32, 1.0, (1, 4, 4, 64), &device)?;
    let positions: Vec<i64> = vec![0, 1, 2, 3];

    let rope_default = scaled_rope(RopeType::Default, 1.0);
    let rope_unscaled = RoPE::new(64, 1024, 10000.0, &device);

    let out_default = rope_default.apply_with_scaling(&q, &positions)?;
    let out_unscaled = rope_unscaled.apply(&q, &positions)?;

    let diff = (&out_default - &out_unscaled)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    assert!(
        diff < 1e-5,
        "Default scaling should match unscaled apply (max diff = {diff})"
    );
    Ok(())
}

#[test]
fn test_apply_with_scaling_linear_modifies_output() -> Result<()> {
    // Linear scaling with factor > 1 must produce a different output
    // from the unscaled path.
    let device = Device::Cpu;
    let q = Tensor::randn(0.0f32, 1.0, (1, 4, 4, 64), &device)?;
    let positions: Vec<i64> = vec![4, 5, 6, 7];

    let rope_unscaled = RoPE::new(64, 1024, 10000.0, &device);
    let rope_linear = scaled_rope(RopeType::Linear, 2.0);

    let out_unscaled = rope_unscaled.apply(&q, &positions)?;
    let out_linear = rope_linear.apply_with_scaling(&q, &positions)?;

    let diff = (&out_unscaled - &out_linear)?
        .abs()?
        .sum_all()?
        .to_scalar::<f32>()?;
    assert!(
        diff > 1e-3,
        "Linear scaling (factor=2) should noticeably change the output (sum diff = {diff})"
    );
    Ok(())
}

#[test]
fn test_apply_with_scaling_yarn_modifies_output() -> Result<()> {
    // YaRN scaling must produce a different output from the unscaled
    // path (NTK-aware theta adjustment). Note that YaRN's effects are
    // smaller than Linear's at the same factor — high-frequency dims
    // barely move.
    let device = Device::Cpu;
    let q = Tensor::randn(0.0f32, 1.0, (1, 4, 4, 64), &device)?;
    let positions: Vec<i64> = vec![10, 11, 12, 13];

    let rope_unscaled = RoPE::new(64, 1024, 10000.0, &device);
    let rope_yarn = scaled_rope(RopeType::Yarn, 4.0);

    let out_unscaled = rope_unscaled.apply(&q, &positions)?;
    let out_yarn = rope_yarn.apply_with_scaling(&q, &positions)?;

    let diff = (&out_unscaled - &out_yarn)?
        .abs()?
        .sum_all()?
        .to_scalar::<f32>()?;
    assert!(
        diff > 1e-6,
        "YaRN scaling (factor=4) should change the output (sum diff = {diff})"
    );
    Ok(())
}

#[test]
fn test_apply_with_scaling_factor_one_is_noop() -> Result<()> {
    // For any rope_type, scaling_factor == 1.0 should produce the same
    // output as the unscaled path.
    let device = Device::Cpu;
    let q = Tensor::randn(0.0f32, 1.0, (1, 4, 4, 64), &device)?;
    let positions: Vec<i64> = vec![5, 6, 7, 8];

    let rope_unscaled = RoPE::new(64, 1024, 10000.0, &device);

    for &kind in &[RopeType::Linear, RopeType::Yarn] {
        let rope_scaled = scaled_rope(kind, 1.0);
        let out_unscaled = rope_unscaled.apply(&q, &positions)?;
        let out_scaled = rope_scaled.apply_with_scaling(&q, &positions)?;
        let diff = (&out_unscaled - &out_scaled)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(
            diff < 1e-5,
            "factor=1.0 must be a no-op for {kind:?} (max diff = {diff})"
        );
    }
    Ok(())
}

#[test]
fn test_rope_scaling_context_from_rope_scaling_extracts_all_fields() {
    // RopeScalingContext::from(&RopeScaling) must extract every field
    // the YAML/JSON config exposes.
    let scaling = RopeScaling {
        rope_type: Some(RopeType::Yarn),
        factor: Some(8.0),
        original_max_position_embeddings: Some(4096),
        attn_factor: Some(0.2),
        partial_rotary_factor: None,
        mrope_section: None,
        short_factor: None,
        long_factor: None,
    };
    let ctx = RopeScalingContext::from(&scaling);
    assert_eq!(ctx.rope_type, RopeType::Yarn);
    assert!((ctx.scaling_factor - 8.0).abs() < 1e-6);
    assert!((ctx.attn_factor.unwrap() - 0.2).abs() < 1e-6);
    assert_eq!(ctx.original_max_position, Some(4096));
}

#[test]
fn test_new_with_config_extracts_yarn_fields() {
    // new_with_config must populate rope_type / attn_factor /
    // original_max_position from a config that declares them.
    use crate::qwen3::config::{Qwen3Config, TextConfig};
    use serde_json::json;

    let cfg: Qwen3Config = serde_json::from_value(json!({
        "rope_theta": 1000000.0,
        "max_position_embeddings": 32768,
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "rope_scaling": {
            "rope_type": "yarn",
            "factor": 4.0,
            "attn_factor": 0.1,
            "original_max_position_embeddings": 8192
        }
    }))
    .expect("config deserializes");

    // Sanity-check the deserialized fields before constructing RoPE.
    let scaling = cfg.rope_scaling().expect("rope_scaling present");
    assert_eq!(scaling.rope_type, Some(RopeType::Yarn));
    let _ = TextConfig::default(); // keep `text_config` referenced so clippy doesn't complain about unused

    let rope = RoPE::new_with_config(&cfg);
    assert_eq!(rope.rope_type, RopeType::Yarn);
    assert!((rope.scaling_factor - 4.0).abs() < 1e-6);
    assert!((rope.attn_factor.unwrap() - 0.1).abs() < 1e-6);
    assert_eq!(rope.original_max_position, Some(8192));
}

#[test]
fn test_forward_with_scaling_matches_apply_with_scaling() -> Result<()> {
    // The struct's `forward_with_scaling` helper must produce the same
    // Q / K outputs as two separate `apply_with_scaling` calls.
    let device = Device::Cpu;
    let rope = scaled_rope(RopeType::Yarn, 4.0);

    let q = Tensor::randn(0.0f32, 1.0, (1, 2, 4, 64), &device)?;
    let k = Tensor::randn(0.0f32, 1.0, (1, 2, 4, 64), &device)?;

    let (q_out, k_out) = rope.forward_with_scaling(&q, &k, 0)?;

    let positions: Vec<i64> = (0..q.dim(1)? as i64).collect();
    let q_ref = rope.apply_with_scaling(&q, &positions)?;
    let k_ref = rope.apply_with_scaling(&k, &positions)?;

    let q_diff = (&q_out - &q_ref)?.abs()?.max_all()?.to_scalar::<f32>()?;
    let k_diff = (&k_out - &k_ref)?.abs()?.max_all()?.to_scalar::<f32>()?;
    assert!(q_diff < 1e-5, "forward Q != apply Q (diff = {q_diff})");
    assert!(k_diff < 1e-5, "forward K != apply K (diff = {k_diff})");
    Ok(())
}

// === Phase 16: Dynamic NTK scaling ===

fn dynamic_rope(scaling_factor: f32, orig_max: usize) -> RoPE {
    RoPE {
        theta: 10000.0,
        head_dim: 64,
        max_position: 1024,
        scaling_factor,
        device: Device::Cpu,
        rope_type: RopeType::Dynamic,
        attn_factor: None,
        original_max_position: Some(orig_max),
    }
}

#[test]
fn test_dynamic_scaling_matches_default_below_orig_max() -> Result<()> {
    // Dynamic at cur_seq_len <= orig_max should fall back to Default inv_freq.
    // We use seq_len=4 with positions in [0, 4) so derived_seq_len=4 < orig_max=64.
    let device = Device::Cpu;
    let q = Tensor::randn(0.0f32, 1.0, (1, 4, 4, 64), &device)?;
    let positions: Vec<i64> = (0..4).collect();

    let rope_default = RoPE::new(64, 1024, 10000.0, &device);
    let rope_dynamic = dynamic_rope(4.0, 64);

    let out_default = rope_default.apply_with_scaling(&q, &positions)?;
    let out_dynamic = rope_dynamic.apply_with_scaling(&q, &positions)?;

    let diff = (&out_default - &out_dynamic)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    assert!(
        diff < 1e-5,
        "Dynamic at cur<=orig_max must match Default (max diff = {diff})"
    );
    Ok(())
}

#[test]
fn test_dynamic_scaling_differs_above_orig_max() -> Result<()> {
    // Dynamic at cur_seq_len > orig_max should differ from Default.
    // positions start at 200 so derived_seq_len=204 > orig_max=64.
    let device = Device::Cpu;
    let q = Tensor::randn(0.0f32, 1.0, (1, 4, 4, 64), &device)?;
    let positions: Vec<i64> = vec![200, 201, 202, 203];

    let rope_default = RoPE::new(64, 1024, 10000.0, &device);
    let rope_dynamic = dynamic_rope(4.0, 64);

    let out_default = rope_default.apply_with_scaling(&q, &positions)?;
    let out_dynamic = rope_dynamic.apply_with_scaling(&q, &positions)?;

    let diff = (&out_default - &out_dynamic)?
        .abs()?
        .sum_all()?
        .to_scalar::<f32>()?;
    assert!(
        diff > 1e-3,
        "Dynamic at cur>orig_max must differ from Default (sum diff = {diff})"
    );
    Ok(())
}

#[test]
fn test_dynamic_scaling_at_orig_max_boundary() -> Result<()> {
    // At cur_seq_len == orig_max, Dynamic must match Default (boundary).
    // positions [60, 61, 62, 63] → derived_seq_len = 64 == orig_max.
    let device = Device::Cpu;
    let q = Tensor::randn(0.0f32, 1.0, (1, 4, 4, 64), &device)?;
    let positions: Vec<i64> = vec![60, 61, 62, 63];

    let rope_default = RoPE::new(64, 1024, 10000.0, &device);
    let rope_dynamic = dynamic_rope(4.0, 64);

    let out_default = rope_default.apply_with_scaling(&q, &positions)?;
    let out_dynamic = rope_dynamic.apply_with_scaling(&q, &positions)?;

    let diff = (&out_default - &out_dynamic)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    assert!(
        diff < 1e-5,
        "Dynamic at boundary cur==orig_max must match Default (max diff = {diff})"
    );
    Ok(())
}

#[test]
fn test_derive_seq_len_handles_empty_positions() {
    use super::derive_seq_len;
    assert_eq!(derive_seq_len(&[]), 0);
    assert_eq!(derive_seq_len(&[0]), 1);
    assert_eq!(derive_seq_len(&[0, 1, 2, 3]), 4);
    assert_eq!(derive_seq_len(&[5]), 6); // non-contiguous: max + 1
}
