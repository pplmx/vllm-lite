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
fn test_precompute_rope_cache_correct_formula() {
    // Verify the RoPE formula: angle = pos * theta^(-2i/d).
    // At position 0, every angle must be 0 → cos=1, sin=0.
    let cache = precompute_rope_cache(4, 64, 10000.0);
    for i in 0..32 {
        let (cos, sin) = cache[i]; // first seq position (pos=0)
        assert!(
            (cos - 1.0).abs() < 1e-6,
            "cos(angle) should be 1.0 at pos=0, dim={i}"
        );
        assert!(
            sin.abs() < 1e-6,
            "sin(angle) should be 0.0 at pos=0, dim={i}"
        );
    }

    // At position 1, dim 0: angle = 1.0 * theta^0 = 1.0
    let (cos0, sin0) = cache[32]; // pos=1, i=0
    assert!(
        (cos0 - 1.0_f32.cos()).abs() < 1e-6 && (sin0 - 1.0_f32.sin()).abs() < 1e-6,
        "pos=1, dim=0: expected (cos(1.0), sin(1.0)), got ({cos0}, {sin0})"
    );

    // Cross-check against compute_inv_freq_for_head_dim + manual angle.
    let head_dim = 64;
    let theta: f32 = 10000.0;
    let inv_freq: Vec<f32> = (0..head_dim / 2)
        .map(|i| theta.powf(-2.0 * (i as f32) / (head_dim as f32)))
        .collect();
    for pos in 0..4 {
        for i in 0..head_dim / 2 {
            let angle = (pos as f32) * inv_freq[i];
            let (cos, sin) = cache[pos * head_dim / 2 + i];
            assert!(
                (cos - angle.cos()).abs() < 1e-5 && (sin - angle.sin()).abs() < 1e-5,
                "pos={pos}, dim={i}: expected (cos({angle}), sin({angle})), got ({cos}, {sin})"
            );
        }
    }
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

// === long-context scaling (RopeType-aware apply_with_scaling) ===

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
        "rope_theta": 1_000_000.0,
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

/// Regression test: `forward` must honour scaling configuration.
/// Pre-fix, `forward` was hardcoded to `apply_rope` (non-scaling), so a
/// scaled RoPE (e.g. YaRN) passed to `forward` silently dropped the
/// long-context correction. Post-fix, `forward` delegates to
/// `apply_rope_with_scaling`, so the output should match
/// `forward_with_scaling` for a scaled RoPE.
#[test]
fn test_forward_honours_scaling() -> Result<()> {
    let device = Device::Cpu;
    let rope = scaled_rope(RopeType::Yarn, 4.0);

    let q = Tensor::randn(0.0f32, 1.0, (1, 4, 4, 64), &device)?;
    let k = Tensor::randn(0.0f32, 1.0, (1, 4, 4, 64), &device)?;

    let (q_fwd, k_fwd) = rope.forward(&q, &k, 0)?;
    let (q_scaled, k_scaled) = rope.forward_with_scaling(&q, &k, 0)?;

    let q_diff = (&q_fwd - &q_scaled)?.abs()?.max_all()?.to_scalar::<f32>()?;
    let k_diff = (&k_fwd - &k_scaled)?.abs()?.max_all()?.to_scalar::<f32>()?;
    assert!(
        q_diff < 1e-5,
        "forward (Yarn) should match forward_with_scaling (q diff = {q_diff})"
    );
    assert!(
        k_diff < 1e-5,
        "forward (Yarn) should match forward_with_scaling (k diff = {k_diff})"
    );
    Ok(())
}

/// Regression test: `apply` must honour scaling configuration, just like
/// `forward` (pre-fix both silently dropped scaling).
#[test]
fn test_apply_honours_scaling() -> Result<()> {
    let device = Device::Cpu;
    let rope = scaled_rope(RopeType::Yarn, 4.0);

    let q = Tensor::randn(0.0f32, 1.0, (1, 4, 4, 64), &device)?;
    let positions: Vec<i64> = vec![0, 1, 2, 3];

    let out_apply = rope.apply(&q, &positions)?;
    let out_scaled = rope.apply_with_scaling(&q, &positions)?;

    let diff = (&out_apply - &out_scaled)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    assert!(
        diff < 1e-5,
        "apply (Yarn) should match apply_with_scaling (diff = {diff})"
    );
    Ok(())
}

// === Dynamic NTK scaling ===

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

// === Phase 16: Su RoPE per-dim scaling ===

#[test]
fn test_su_with_identity_factors_matches_default() -> Result<()> {
    // Su with short_factor == long_factor == ones must match Default.
    let device = Device::Cpu;
    let q = Tensor::randn(0.0f32, 1.0, (1, 4, 4, 64), &device)?;
    let positions: Vec<i64> = (0..4).collect();

    let rope_default = RoPE::new(64, 4096, 10000.0, &device);
    let ctx = RopeScalingContext {
        rope_type: RopeType::Su,
        scaling_factor: 1.0,
        attn_factor: None,
        original_max_position: Some(32),
        short_factor: Some(vec![1.0; 32]),
        long_factor: Some(vec![1.0; 32]),
    };
    let out_default = rope_default.apply_with_scaling(&q, &positions)?;
    let out_su = apply_rope_with_scaling(&q, &positions, 10000.0, ctx)?;

    let diff = (&out_default - &out_su)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    assert!(
        diff < 1e-5,
        "Su with identity factors must match Default (max diff = {diff})"
    );
    Ok(())
}

#[test]
fn test_su_short_factor_modifies_high_freq_dims() -> Result<()> {
    // Su with a non-identity short_factor should produce different output.
    let device = Device::Cpu;
    let q = Tensor::randn(0.0f32, 1.0, (1, 4, 4, 64), &device)?;
    let positions: Vec<i64> = (0..4).collect();

    let rope_default = RoPE::new(64, 4096, 10000.0, &device);
    let mut short_factor = vec![1.0; 32];
    short_factor[0] = 2.0; // boost high-freq dim 0
    let ctx = RopeScalingContext {
        rope_type: RopeType::Su,
        scaling_factor: 1.0,
        attn_factor: None,
        original_max_position: Some(32),
        short_factor: Some(short_factor),
        long_factor: Some(vec![1.0; 32]),
    };
    let out_default = rope_default.apply_with_scaling(&q, &positions)?;
    let out_su = apply_rope_with_scaling(&q, &positions, 10000.0, ctx)?;

    let diff = (&out_default - &out_su)?
        .abs()?
        .sum_all()?
        .to_scalar::<f32>()?;
    assert!(
        diff > 1e-6,
        "Su with non-identity short_factor must differ from Default (sum diff = {diff})"
    );
    Ok(())
}

#[test]
fn test_su_long_factor_modifies_low_freq_dims() -> Result<()> {
    // Su with a non-identity long_factor should produce different output.
    let device = Device::Cpu;
    let q = Tensor::randn(0.0f32, 1.0, (1, 4, 4, 64), &device)?;
    let positions: Vec<i64> = (0..4).collect();

    let rope_default = RoPE::new(64, 4096, 10000.0, &device);
    let mut long_factor = vec![1.0; 32];
    long_factor[31] = 4.0; // boost low-freq dim 31
    let ctx = RopeScalingContext {
        rope_type: RopeType::Su,
        scaling_factor: 1.0,
        attn_factor: None,
        original_max_position: Some(32),
        short_factor: Some(vec![1.0; 32]),
        long_factor: Some(long_factor),
    };
    let out_default = rope_default.apply_with_scaling(&q, &positions)?;
    let out_su = apply_rope_with_scaling(&q, &positions, 10000.0, ctx)?;

    let diff = (&out_default - &out_su)?
        .abs()?
        .sum_all()?
        .to_scalar::<f32>()?;
    assert!(
        diff > 1e-6,
        "Su with non-identity long_factor must differ from Default (sum diff = {diff})"
    );
    Ok(())
}

#[test]
fn test_su_scaling_context_from_rope_scaling_extracts_factors() {
    let scaling = RopeScaling {
        rope_type: Some(RopeType::Su),
        factor: Some(1.0),
        original_max_position_embeddings: Some(4096),
        attn_factor: None,
        partial_rotary_factor: None,
        mrope_section: None,
        short_factor: Some(vec![1.0, 1.5, 2.0]),
        long_factor: Some(vec![4.0, 5.0, 6.0]),
    };
    let ctx = RopeScalingContext::from(&scaling);
    assert_eq!(
        ctx.short_factor.as_deref(),
        Some([1.0_f32, 1.5, 2.0].as_slice())
    );
    assert_eq!(
        ctx.long_factor.as_deref(),
        Some([4.0_f32, 5.0, 6.0].as_slice())
    );
}

#[test]
fn test_su_missing_orig_max_falls_back_to_default() -> Result<()> {
    // Without original_max_position_embeddings, Su cannot compute the
    // boundary. Verify it falls back to Default (silently).
    let device = Device::Cpu;
    let q = Tensor::ones((1, 2, 2, 64), DType::F32, &device)?;
    let positions: Vec<i64> = vec![0, 1];
    let ctx = RopeScalingContext {
        rope_type: RopeType::Su,
        scaling_factor: 1.0,
        attn_factor: None,
        original_max_position: None, // <-- missing
        short_factor: Some(vec![1.0; 32]),
        long_factor: Some(vec![2.0; 32]),
    };
    let rope_default = RoPE::new(64, 4096, 10000.0, &device);
    let out_default = rope_default.apply_with_scaling(&q, &positions)?;
    let out_su = apply_rope_with_scaling(&q, &positions, 10000.0, ctx)?;
    let diff = (&out_default - &out_su)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    assert!(
        diff < 1e-5,
        "Su without original_max_position must fall back to Default (max diff = {diff})"
    );
    Ok(())
}

// === absolute formula cross-checks for long-context scaling ===
//
// The differential tests above only assert that scaling *changes* the
// output. These tests cross-validate each algorithm's *numerical* formula
// against an independent derivation straight from the defining math
// (using `compute_inv_freq_for_head_dim` as the trusted inv_freq table).
// They guard against regressions in the theta-adjustment / scale /
// boundary logic — a wrong long-context formula silently corrupts
// extended-context output.

/// `YaARN` / `NTK`: `theta' = theta * scale^(d/(d-2))`, then the standard
/// `inv_freq[i] = theta'^(-2i/d)`. `compute_inv_freq_for_head_dim` is the
/// trusted reference for the `inv_freq` table; the check validates the
/// theta-adjustment exponent and multiplication.
#[test]
fn test_compute_inv_freq_yarn_impl_matches_theta_adjustment() {
    let head_dim = 64;
    let theta = 10_000.0_f32;
    let scale = 4.0_f32;
    let d = head_dim as f32;

    // Independent reference: theta' = theta * scale^(d/(d-2)).
    let expected_theta = theta * scale.powf(d / (d - 2.0));
    let expected = compute_inv_freq_for_head_dim(head_dim, expected_theta);
    let actual = compute_inv_freq_yarn_impl(head_dim, theta, scale);

    assert_eq!(
        actual.len(),
        expected.len(),
        "inv_freq length must be head_dim/2"
    );
    for (i, (a, e)) in actual.iter().zip(&expected).enumerate() {
        assert!(
            (a - e).abs() < 1e-5,
            "dim {i}: YaRN inv_freq {a} != expected {e} (theta'={expected_theta})"
        );
    }

    // scale > 1 => theta' > theta => inv_freq must shrink at every dim
    // *except* i=0, where inv_freq = theta^0 == theta'^0 == 1.0 by
    // definition (the base frequency of the first dim is always 1.0).
    let default_inv = compute_inv_freq_for_head_dim(head_dim, theta);
    for (i, (a, dflt)) in actual.iter().zip(&default_inv).enumerate() {
        if i == 0 {
            assert!(
                (a - 1.0).abs() < 1e-6 && (dflt - 1.0).abs() < 1e-6,
                "dim 0 must be theta^0 == 1.0 ({a}, {dflt})"
            );
            continue;
        }
        assert!(
            a < dflt,
            "scale>1 should lower inv_freq at dim {i} ({a} vs default {dflt})"
        );
    }

    // scale == 1.0 is a no-op (1.0^exp == 1.0).
    let noop = compute_inv_freq_yarn_impl(head_dim, theta, 1.0);
    for (a, e) in noop.iter().zip(&default_inv) {
        assert!((a - e).abs() < 1e-6, "scale=1.0 must be a no-op");
    }
}

/// Linear: `inv_freq'[i] = inv_freq[i] / scaling_factor`.
#[test]
fn test_compute_inv_freq_linear_divides_by_scale() -> Result<()> {
    let device = Device::Cpu;
    let head_dim = 64;
    let q = Tensor::ones((1, 1, 1, head_dim), DType::F32, &device)?;
    let theta = 10_000.0_f32;
    let scale = 2.0_f32;

    let default_inv = compute_inv_freq_for_head_dim(head_dim, theta);
    let expected: Vec<f32> = default_inv.iter().map(|f| f / scale).collect();
    let actual = compute_inv_freq_linear(&q, theta, scale);

    assert_eq!(actual.len(), expected.len());
    for (i, (a, e)) in actual.iter().zip(&expected).enumerate() {
        assert!(
            (a - e).abs() < 1e-6,
            "dim {i}: linear inv_freq {a} != expected {e}"
        );
    }

    // factor == 1.0 returns the default table unchanged.
    let noop = compute_inv_freq_linear(&q, theta, 1.0);
    for (a, e) in noop.iter().zip(&default_inv) {
        assert!((a - e).abs() < 1e-6, "factor=1.0 must be a no-op");
    }
    Ok(())
}

/// Dynamic NTK: `scale = factor * (cur/orig) - (factor - 1)`, then it
/// delegates to the `YaARN` impl. Validates the dynamic scale formula and
/// the boundary fallback to the default table when `cur <= orig_max`.
#[test]
fn test_compute_inv_freq_dynamic_delegates_to_yarn_impl() -> Result<()> {
    let device = Device::Cpu;
    let head_dim = 64;
    let q = Tensor::ones((1, 1, 1, head_dim), DType::F32, &device)?;
    let theta = 10_000.0_f32;
    let factor = 4.0_f32;
    let orig_max = 1024;
    let cur_seq_len = 2048; // > orig_max

    // Independent dynamic scale: factor*(cur/orig)-(factor-1).
    let dynamic_scale = factor.mul_add(cur_seq_len as f32 / orig_max as f32, -(factor - 1.0));
    assert!(dynamic_scale > 1.0, "cur > orig must yield scale > 1.0");
    let expected = compute_inv_freq_yarn_impl(head_dim, theta, dynamic_scale);
    let actual = compute_inv_freq_dynamic(&q, theta, factor, Some(orig_max), cur_seq_len);

    assert_eq!(actual.len(), expected.len());
    for (i, (a, e)) in actual.iter().zip(&expected).enumerate() {
        assert!(
            (a - e).abs() < 1e-5,
            "dim {i}: dynamic {a} != YaARN impl {e} (scale={dynamic_scale})"
        );
    }

    // Boundary: cur <= orig_max must fall back to the default table.
    let fallback = compute_inv_freq_dynamic(&q, theta, factor, Some(orig_max), orig_max);
    let default_inv = compute_inv_freq_for_head_dim(head_dim, theta);
    for (a, e) in fallback.iter().zip(&default_inv) {
        assert!(
            (a - e).abs() < 1e-6,
            "cur <= orig must use the default inv_freq"
        );
    }
    Ok(())
}

/// Su: `boundary` = first dim whose base wavelength `2π/inv_freq[i]` exceeds
/// `orig_max`; dims `< boundary` use `short_factor`, dims `>= boundary` use
/// `long_factor`, each as `inv_freq[i] / factor`. Validates the boundary
/// computation and the per-dim factor application.
#[test]
fn test_compute_inv_freq_su_applies_boundary_and_factors() {
    let device = Device::Cpu;
    let head_dim = 64;
    let half_dim = head_dim / 2;
    let theta = 10_000.0_f32;
    let orig_max = 8192;

    let short_factor: Vec<f32> = (0..half_dim)
        .map(|i| 0.01f32.mul_add(i as f32, 1.0))
        .collect();
    let long_factor: Vec<f32> = (0..half_dim)
        .map(|i| 0.02f32.mul_add(i as f32, 2.0))
        .collect();
    let ctx = RopeScalingContext {
        rope_type: RopeType::Su,
        scaling_factor: 1.0,
        attn_factor: None,
        original_max_position: Some(orig_max),
        short_factor: Some(short_factor.clone()),
        long_factor: Some(long_factor.clone()),
    };

    let q = Tensor::ones((1, 1, 1, head_dim), DType::F32, &device).unwrap();
    let base_inv = compute_inv_freq_for_head_dim(head_dim, theta);

    // Independent boundary: first i where 2π/inv_freq[i] > orig_max.
    let boundary = (0..half_dim)
        .find(|&i| 2.0 * std::f32::consts::PI / base_inv[i] > orig_max as f32)
        .unwrap_or(half_dim);

    let expected: Vec<f32> = (0..half_dim)
        .map(|i| {
            let factor = if i < boundary {
                short_factor[i]
            } else {
                long_factor[i]
            };
            base_inv[i] / factor
        })
        .collect();

    let actual = compute_inv_freq_su(&q, theta, &ctx);
    assert_eq!(actual.len(), expected.len());
    for (i, (a, e)) in actual.iter().zip(&expected).enumerate() {
        assert!(
            (a - e).abs() < 1e-6,
            "dim {i}: Su inv_freq {a} != expected {e} (boundary={boundary}, i<boundary uses short)"
        );
    }
}
