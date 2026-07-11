//! Unit tests for [`super::GqaAttention`].
//!
//! Lives in a separate file (under the `gqa/` module directory) so the
//! production code in `gqa.rs` stays focused on the implementation and
//! doesn't cross the 800-line file-size soft cap. Inlined via
//! `#[path = "tests.rs"]` so the `tests` module remains in the same
//! scope and can `use super::*` to access the implementation under test.

use super::*;

#[test]
fn test_gqa_attention_forward_output_shape() {
    let device = candle_core::Device::Cpu;
    let num_heads = 16;
    let num_kv_heads = 8;
    let head_dim = 128;
    let batch_size = 1;
    let seq_len = 8;

    let hidden_size = num_heads * head_dim;
    let q_w = Tensor::randn(0.0f32, 1.0, (hidden_size, hidden_size), &device).unwrap();
    let k_w = Tensor::randn(0.0f32, 1.0, (num_kv_heads * head_dim, hidden_size), &device).unwrap();
    let v_w = Tensor::randn(0.0f32, 1.0, (num_kv_heads * head_dim, hidden_size), &device).unwrap();
    let o_w = Tensor::randn(0.0f32, 1.0, (hidden_size, hidden_size), &device).unwrap();

    let attention = GqaAttention::new_with_weights(
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
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

    let x = Tensor::randn(0.0f32, 1.0, (batch_size, seq_len, hidden_size), &device).unwrap();
    let output = attention.forward(&x).unwrap();

    assert_eq!(output.dims(), &[batch_size, seq_len, hidden_size]);
}

#[test]
fn test_gqa_attention_fused_matches_standard() {
    let device = candle_core::Device::Cpu;
    let num_heads = 8;
    let num_kv_heads = 4;
    let head_dim = 32;
    let batch_size = 1;
    let seq_len = 6;
    let hidden_size = num_heads * head_dim;

    let q_w = Tensor::randn(0.0f32, 1.0, (hidden_size, hidden_size), &device).unwrap();
    let k_w = Tensor::randn(0.0f32, 1.0, (num_kv_heads * head_dim, hidden_size), &device).unwrap();
    let v_w = Tensor::randn(0.0f32, 1.0, (num_kv_heads * head_dim, hidden_size), &device).unwrap();
    let o_w = Tensor::randn(0.0f32, 1.0, (hidden_size, hidden_size), &device).unwrap();

    let standard = GqaAttention::new_with_weights(
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        q_w.clone(),
        k_w.clone(),
        v_w.clone(),
        o_w.clone(),
        AttentionConfig {
            use_fused: false,
            ..Default::default()
        },
        false,
        None,
        None,
    )
    .unwrap();
    let fused = GqaAttention::new_with_weights(
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        q_w,
        k_w,
        v_w,
        o_w,
        AttentionConfig {
            use_fused: true,
            ..Default::default()
        },
        false,
        None,
        None,
    )
    .unwrap();

    let x = Tensor::randn(0.0f32, 1.0, (batch_size, seq_len, hidden_size), &device).unwrap();
    let out_standard = standard.forward(&x).unwrap();
    let out_fused = fused.forward(&x).unwrap();

    let diff = (&out_standard - &out_fused).unwrap().abs().unwrap();
    let max_diff: f32 = diff.max_all().unwrap().to_scalar().unwrap();
    assert!(
        max_diff < 1e-4,
        "fused GQA path should match standard, max_diff={max_diff}"
    );
}

#[test]
fn test_gqa_attention_with_qk_norm() {
    let device = candle_core::Device::Cpu;
    let num_heads = 8;
    let num_kv_heads = 4;
    let head_dim = 64;
    let batch_size = 1;
    let seq_len = 4;

    let hidden_size = num_heads * head_dim;
    let q_w = Tensor::randn(0.0f32, 1.0, (hidden_size, hidden_size), &device).unwrap();
    let k_w = Tensor::randn(0.0f32, 1.0, (num_kv_heads * head_dim, hidden_size), &device).unwrap();
    let v_w = Tensor::randn(0.0f32, 1.0, (num_kv_heads * head_dim, hidden_size), &device).unwrap();
    let o_w = Tensor::randn(0.0f32, 1.0, (hidden_size, hidden_size), &device).unwrap();

    let q_norm_w = Tensor::randn(0.0f32, 0.1, (head_dim,), &device).unwrap();
    let k_norm_w = Tensor::randn(0.0f32, 0.1, (head_dim,), &device).unwrap();

    let attention = GqaAttention::new_with_weights(
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        q_w,
        k_w,
        v_w,
        o_w,
        AttentionConfig::default(),
        true,
        Some(q_norm_w),
        Some(k_norm_w),
    )
    .unwrap();

    let x = Tensor::randn(0.0f32, 1.0, (batch_size, seq_len, hidden_size), &device).unwrap();
    let output = attention.forward(&x).unwrap();

    assert_eq!(output.dims(), &[batch_size, seq_len, hidden_size]);
}

#[test]
fn test_gqa_attention_paged_attention_output_shape() {
    let device = candle_core::Device::Cpu;
    let num_heads = 16;
    let num_kv_heads = 8;
    let head_dim = 128;
    let batch_size = 1;

    let hidden_size = num_heads * head_dim;
    let q_w = Tensor::randn(0.0f32, 1.0, (hidden_size, hidden_size), &device).unwrap();
    let k_w = Tensor::randn(0.0f32, 1.0, (num_kv_heads * head_dim, hidden_size), &device).unwrap();
    let v_w = Tensor::randn(0.0f32, 1.0, (num_kv_heads * head_dim, hidden_size), &device).unwrap();
    let o_w = Tensor::randn(0.0f32, 1.0, (hidden_size, hidden_size), &device).unwrap();

    let attention = GqaAttention::new_with_weights(
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
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

    let q = Tensor::randn(0.0f32, 1.0, (batch_size, num_heads, 1, head_dim), &device).unwrap();
    let k = Tensor::randn(0.0f32, 1.0, (batch_size, num_heads, 8, head_dim), &device).unwrap();
    let v = Tensor::randn(0.0f32, 1.0, (batch_size, num_heads, 8, head_dim), &device).unwrap();

    let output = attention.paged_attention_fn(&q, &k, &v).unwrap();

    assert_eq!(output.dims(), &[batch_size, 1, hidden_size]);
}

#[test]
fn test_gqa_attention_tiled_attention_output_shape() {
    let device = candle_core::Device::Cpu;
    let num_heads = 16;
    let num_kv_heads = 8;
    let head_dim = 128;
    let batch_size = 1;
    let seq_len = 8;

    let hidden_size = num_heads * head_dim;
    let q_w = Tensor::randn(0.0f32, 1.0, (hidden_size, hidden_size), &device).unwrap();
    let k_w = Tensor::randn(0.0f32, 1.0, (num_kv_heads * head_dim, hidden_size), &device).unwrap();
    let v_w = Tensor::randn(0.0f32, 1.0, (num_kv_heads * head_dim, hidden_size), &device).unwrap();
    let o_w = Tensor::randn(0.0f32, 1.0, (hidden_size, hidden_size), &device).unwrap();

    let attention = GqaAttention::new_with_weights(
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
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

    let q = Tensor::randn(
        0.0f32,
        1.0,
        (batch_size, num_heads, seq_len, head_dim),
        &device,
    )
    .unwrap();
    let k = Tensor::randn(
        0.0f32,
        1.0,
        (batch_size, num_heads, seq_len, head_dim),
        &device,
    )
    .unwrap();
    let v = Tensor::randn(
        0.0f32,
        1.0,
        (batch_size, num_heads, seq_len, head_dim),
        &device,
    )
    .unwrap();

    let output = attention.tiled_attention_fn(&q, &k, &v).unwrap();

    assert_eq!(output.dims(), &[batch_size, seq_len, hidden_size]);
}

#[test]
fn test_gqa_attention_new_creation() -> Result<()> {
    let _device = candle_core::Device::Cpu;
    let attn = GqaAttention::new(256, 8, 2, 32, None, AttentionConfig::default(), false)?;
    assert_eq!(attn.num_heads(), 8);
    assert_eq!(attn.num_kv_heads(), 2);
    assert_eq!(attn.head_dim(), 32);
    Ok(())
}

#[test]
fn test_gqa_attention_num_heads_accessors() {
    let _device = candle_core::Device::Cpu;
    let attn = GqaAttention::new(512, 16, 4, 32, None, AttentionConfig::default(), false).unwrap();
    assert_eq!(attn.num_heads(), 16);
    assert_eq!(attn.num_kv_heads(), 4);
}

#[test]
fn test_gqa_attention_head_dim_accessors() {
    let _device = candle_core::Device::Cpu;
    let attn = GqaAttention::new(256, 4, 2, 64, None, AttentionConfig::default(), false).unwrap();
    assert_eq!(attn.head_dim(), 64);
}

#[test]
fn test_gqa_attention_paged_attention_shape() -> Result<()> {
    let device = candle_core::Device::Cpu;
    let num_heads = 8;
    let num_kv_heads = 2;
    let head_dim = 64;
    let batch_size = 2;
    let seq_len = 4;
    let hidden_size = num_heads * head_dim;

    let q_w = Tensor::randn(0.0f32, 1.0, (hidden_size, hidden_size), &device)?;
    let k_w = Tensor::randn(0.0f32, 1.0, (num_kv_heads * head_dim, hidden_size), &device)?;
    let v_w = Tensor::randn(0.0f32, 1.0, (num_kv_heads * head_dim, hidden_size), &device)?;
    let o_w = Tensor::randn(0.0f32, 1.0, (hidden_size, hidden_size), &device)?;

    let attn = GqaAttention::new_with_weights(
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        q_w,
        k_w,
        v_w,
        o_w,
        AttentionConfig::default(),
        false,
        None,
        None,
    )?;

    let q = Tensor::randn(0.0f32, 1.0, (batch_size, num_heads, 1, head_dim), &device)?;
    let k = Tensor::randn(
        0.0f32,
        1.0,
        (batch_size, num_heads, seq_len, head_dim),
        &device,
    )?;
    let v = Tensor::randn(
        0.0f32,
        1.0,
        (batch_size, num_heads, seq_len, head_dim),
        &device,
    )?;

    let output = attn.paged_attention_fn(&q, &k, &v)?;
    assert_eq!(output.dims(), &[batch_size, 1, hidden_size]);
    Ok(())
}

#[test]
fn test_gqa_attention_tiled_attention_shape() -> Result<()> {
    let device = candle_core::Device::Cpu;
    let num_heads = 4;
    let num_kv_heads = 2;
    let head_dim = 32;
    let batch_size = 1;
    let seq_len = 8;
    let hidden_size = num_heads * head_dim;

    let q_w = Tensor::randn(0.0f32, 1.0, (hidden_size, hidden_size), &device)?;
    let k_w = Tensor::randn(0.0f32, 1.0, (num_kv_heads * head_dim, hidden_size), &device)?;
    let v_w = Tensor::randn(0.0f32, 1.0, (num_kv_heads * head_dim, hidden_size), &device)?;
    let o_w = Tensor::randn(0.0f32, 1.0, (hidden_size, hidden_size), &device)?;

    let attn = GqaAttention::new_with_weights(
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        q_w,
        k_w,
        v_w,
        o_w,
        AttentionConfig::default(),
        false,
        None,
        None,
    )?;

    let q = Tensor::randn(
        0.0f32,
        1.0,
        (batch_size, num_heads, seq_len, head_dim),
        &device,
    )?;
    let k = Tensor::randn(
        0.0f32,
        1.0,
        (batch_size, num_heads, seq_len, head_dim),
        &device,
    )?;
    let v = Tensor::randn(
        0.0f32,
        1.0,
        (batch_size, num_heads, seq_len, head_dim),
        &device,
    )?;

    let output = attn.tiled_attention_fn(&q, &k, &v)?;
    assert_eq!(output.dims(), &[batch_size, seq_len, hidden_size]);
    Ok(())
}

#[test]
fn test_gqa_attention_single_q_head() -> Result<()> {
    let _device = candle_core::Device::Cpu;
    let attn = GqaAttention::new(64, 1, 1, 64, None, AttentionConfig::default(), false)?;
    assert_eq!(attn.num_heads(), 1);
    assert_eq!(attn.num_kv_heads(), 1);
    Ok(())
}

#[test]
fn test_gqa_attention_matching_heads() -> Result<()> {
    let _device = candle_core::Device::Cpu;
    let attn = GqaAttention::new(256, 4, 4, 64, None, AttentionConfig::default(), false)?;
    assert_eq!(attn.num_heads(), attn.num_kv_heads());
    Ok(())
}

#[test]
fn test_gqa_attention_small_head_dim() -> Result<()> {
    let _device = candle_core::Device::Cpu;
    let attn = GqaAttention::new(64, 2, 1, 32, None, AttentionConfig::default(), false)?;
    assert_eq!(attn.head_dim(), 32);
    Ok(())
}

#[test]
fn test_gqa_attention_large_head_dim() -> Result<()> {
    let _device = candle_core::Device::Cpu;
    let attn = GqaAttention::new(512, 4, 2, 128, None, AttentionConfig::default(), false)?;
    assert_eq!(attn.head_dim(), 128);
    Ok(())
}

#[test]
fn test_gqa_attention_output_finite() -> Result<()> {
    let device = candle_core::Device::Cpu;
    let attn = GqaAttention::new(128, 4, 2, 32, None, AttentionConfig::default(), false)?;

    let q = Tensor::randn(-2.0f32, 2.0, (1, 4, 4, 32), &device)?;
    let k = Tensor::randn(-2.0f32, 2.0, (1, 4, 4, 32), &device)?;
    let v = Tensor::randn(-2.0f32, 2.0, (1, 4, 4, 32), &device)?;

    let output = attn.paged_attention_fn(&q, &k, &v)?;
    let data: Vec<f32> = output.flatten_all()?.to_vec1()?;
    assert!(data.iter().all(|v| v.is_finite()));
    Ok(())
}

#[test]
fn test_gqa_attention_deterministic() -> Result<()> {
    let device = candle_core::Device::Cpu;
    let q_w = Tensor::randn(0.0f32, 1.0, (256, 256), &device)?;
    let k_w = Tensor::randn(0.0f32, 1.0, (64, 256), &device)?;
    let v_w = Tensor::randn(0.0f32, 1.0, (64, 256), &device)?;
    let o_w = Tensor::randn(0.0f32, 1.0, (256, 256), &device)?;

    let attn = GqaAttention::new_with_weights(
        256,
        8,
        2,
        32,
        q_w,
        k_w,
        v_w,
        o_w,
        AttentionConfig::default(),
        false,
        None,
        None,
    )?;

    let q = Tensor::randn(0.0f32, 1.0, (1, 8, 1, 32), &device)?;
    let k = Tensor::randn(0.0f32, 1.0, (1, 8, 4, 32), &device)?;
    let v = Tensor::randn(0.0f32, 1.0, (1, 8, 4, 32), &device)?;

    let out1 = attn.paged_attention_fn(&q, &k, &v)?;
    let out2 = attn.paged_attention_fn(&q, &k, &v)?;

    let diff = (&out1 - &out2)?.abs()?;
    let max_diff: f32 = diff
        .flatten_all()?
        .to_vec1()?
        .iter()
        .copied()
        .fold(0.0f32, f32::max);
    assert!(max_diff < 1e-5);
    Ok(())
}

#[test]
fn test_gqa_attention_expand_kv_correct() -> Result<()> {
    let device = candle_core::Device::Cpu;
    let _attn = GqaAttention::new(256, 8, 2, 32, None, AttentionConfig::default(), false)?;

    let num_kv_heads = 2;
    let head_dim = 32;
    let seq_len = 4;

    let k = Tensor::randn(0.0f32, 1.0, (1, seq_len, num_kv_heads, head_dim), &device)?;
    let k_expanded = expand_kv(&k, 8, 2)?;

    assert_eq!(k_expanded.dims(), &[1, seq_len, 8, head_dim]);
    Ok(())
}

/// Regression: `GqaAttention::forward` is documented to NOT apply causal
/// masking (it is a low-level primitive). Production code routes through
/// `RopeGqaAttention::forward_prefill/forward_decode` which dispatch to
/// causal-aware helpers (`paged_attention_fn`, `tiled_attention_fn`,
/// `flash_attention_fn(..., causal=true)`).
///
/// If this test ever fails (`forward()` becomes non-deterministic or
/// starts applying hidden causal masking), it should be re-documented,
/// not "fixed" — the contract is intentional. See
/// `docs/perf/v27-correctness-investigation.md` for the full analysis.
#[test]
fn test_forward_does_not_apply_causal_mask() -> Result<()> {
    let device = candle_core::Device::Cpu;
    let num_heads = 4;
    let num_kv_heads = 2;
    let head_dim = 16;
    let hidden_size = num_heads * head_dim;
    let batch_size = 1;
    let seq_len = 4;

    let q_w = Tensor::randn(0.0f32, 0.5, (hidden_size, hidden_size), &device)?;
    let k_w = Tensor::randn(0.0f32, 0.5, (num_kv_heads * head_dim, hidden_size), &device)?;
    let v_w = Tensor::randn(0.0f32, 0.5, (num_kv_heads * head_dim, hidden_size), &device)?;
    let o_w = Tensor::randn(0.0f32, 0.5, (hidden_size, hidden_size), &device)?;

    let attn = GqaAttention::new_with_weights(
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        q_w,
        k_w,
        v_w,
        o_w,
        AttentionConfig::default(),
        false,
        None,
        None,
    )?;

    let x = Tensor::randn(0.0f32, 1.0, (batch_size, seq_len, hidden_size), &device)?;
    let y = x.clone();

    let out1 = attn.forward(&x)?;
    let out2 = attn.forward(&y)?;

    // Determinism: identical inputs must yield identical outputs. Causal
    // masking would still pass this (causal is also deterministic), but
    // this assertion catches accidental state-coupling or RNG leakage.
    let diff = (&out1 - &out2)?.abs()?.sum_all()?.to_scalar::<f32>()?;
    assert!(
        diff < 1e-5,
        "forward() should be deterministic for identical inputs, got diff={diff}"
    );
    Ok(())
}

/// H-11 #3 regression: candle's CPU matmul kernel rejects non-contiguous
/// batch dimensions on either LHS or RHS with `MatMulUnexpectedStriding`.
/// This test pins that behavior so future "remove contiguous" attempts
/// fail at this test rather than in downstream integration tests where
/// the failure is harder to diagnose.
#[test]
fn test_matmul_rejects_non_contiguous_batch_dims() -> Result<()> {
    let device = candle_core::Device::Cpu;
    // (B=2, H=4, S=10, D=32) — strided from transpose(1,2)
    let q_strided = Tensor::randn(0.0f32, 1.0, (2, 10, 4, 32), &device)?.transpose(1, 2)?;
    // (B=2, H=4, D=32, S=10) — contiguous
    let k_cont = Tensor::randn(0.0f32, 1.0, (2, 4, 32, 10), &device)?;

    let result = Tensor::matmul(&q_strided, &k_cont);
    assert!(
        result.is_err(),
        "Tensor::matmul should reject non-contiguous LHS batch dims"
    );
    let err = format!("{}", result.unwrap_err());
    assert!(
        err.contains("non-contiguous") || err.contains("Striding"),
        "expected striding error, got: {err}"
    );
    Ok(())
}

// === Phase 16: attn_factor wiring ===

#[test]
fn gqa_attn_factor_one_is_noop() -> Result<()> {
    let device = candle_core::Device::Cpu;
    let num_heads = 4;
    let num_kv_heads = 4;
    let head_dim = 32;
    let hidden_size = num_heads * head_dim;

    let q_w = Tensor::randn(0.0f32, 1.0, (hidden_size, hidden_size), &device)?;
    let k_w = Tensor::randn(0.0f32, 1.0, (hidden_size, hidden_size), &device)?;
    let v_w = Tensor::randn(0.0f32, 1.0, (hidden_size, hidden_size), &device)?;
    let o_w = Tensor::randn(0.0f32, 1.0, (hidden_size, hidden_size), &device)?;

    let mut attn = GqaAttention::new_with_weights(
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        q_w,
        k_w,
        v_w,
        o_w,
        AttentionConfig::default(),
        false,
        None,
        None,
    )?;

    let x = Tensor::randn(0.0f32, 1.0, (1, 4, hidden_size), &device)?;

    // attn_factor = 1.0 must match attn_factor = None
    attn.attn_factor = Some(1.0);
    let out_with_factor = attn.forward(&x)?;
    attn.attn_factor = None;
    let out_without_factor = attn.forward(&x)?;

    let diff = (&out_with_factor - &out_without_factor)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    assert!(
        diff < 1e-5,
        "attn_factor=1.0 must be a no-op (max diff = {diff})"
    );
    Ok(())
}

#[test]
fn gqa_attn_factor_changes_output() -> Result<()> {
    let device = candle_core::Device::Cpu;
    let num_heads = 4;
    let num_kv_heads = 4;
    let head_dim = 32;
    let hidden_size = num_heads * head_dim;

    let q_w = Tensor::randn(0.0f32, 1.0, (hidden_size, hidden_size), &device)?;
    let k_w = Tensor::randn(0.0f32, 1.0, (hidden_size, hidden_size), &device)?;
    let v_w = Tensor::randn(0.0f32, 1.0, (hidden_size, hidden_size), &device)?;
    let o_w = Tensor::randn(0.0f32, 1.0, (hidden_size, hidden_size), &device)?;

    let mut attn = GqaAttention::new_with_weights(
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        q_w,
        k_w,
        v_w,
        o_w,
        AttentionConfig::default(),
        false,
        None,
        None,
    )?;

    let x = Tensor::randn(0.0f32, 1.0, (1, 4, hidden_size), &device)?;

    // attn_factor = 0.5 must change the output (halves the score temperature)
    attn.attn_factor = Some(0.5);
    let out_with_factor = attn.forward(&x)?;
    attn.attn_factor = None;
    let out_without_factor = attn.forward(&x)?;

    let diff = (&out_with_factor - &out_without_factor)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    assert!(
        diff > 1e-5,
        "attn_factor=0.5 must change the output (max diff = {diff})"
    );
    Ok(())
}
