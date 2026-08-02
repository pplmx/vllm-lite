//! Unit tests for the attention helper functions (`expand_kv`,
//! `causal_mask`, `paged_attention`, `tiled_attention`).
//!
//! Extracted from `util.rs` to keep the implementation file under the
//! project's 800-line soft cap. Exercises:
//!
//! - `paged_attention` output shape (basic + single-token decode)
//! - `tiled_attention` output shape parity vs. `paged_attention`
//! - `tiled_attention` single-tile path (`tile_size > seq_len`)
//! - `expand_kv` GQA expansion (basic, no-expansion, invalid head count,
//!   exact division)
//! - `causal_mask` shape and causal values (0 below diagonal,
//!   -inf above)

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

/// Regression (RIL ISS-010 / TASK-010): GQA expansion must be **blocked**
/// (repeat-interleave), not tiled. Query head `h` attends to KV head
/// `h / group_size`, so the expanded head order must be
/// `[K0, K0, ..., K1, K1, ...]`. Pre-fix `expand_kv` used `Tensor::repeat`,
/// which tiles (`[K0, K1, K0, K1, …]`) and pairs query head `h` with the
/// WRONG KV head `h % num_kv_heads`.
#[test]
fn test_expand_kv_blocked_grouping() {
    // Two KV heads with distinct constant values: head 0 = 0.0, head 1 = 1.0.
    let kv = Tensor::from_vec(vec![0.0f32, 1.0], (1, 1, 2, 1), DEVICE).unwrap();
    // 8 query heads / 2 KV heads => group_size 4.
    let expanded = expand_kv(&kv, 8, 2).unwrap();
    assert_eq!(expanded.dims(), &[1, 1, 8, 1]);
    let vals: Vec<f32> = expanded.flatten_all().unwrap().to_vec1().unwrap();
    assert_eq!(
        vals,
        vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        "expansion must be blocked [K0x4, K1x4]; got {vals:?} (tiled => RIL ISS-010)"
    );
}

/// Regression (RIL ISS-010 / TASK-010): end-to-end GQA grouping through
/// `paged_attention`. With a single key position the softmax weight is
/// exactly 1.0, so each query head's output equals its paired VALUE head
/// verbatim — directly revealing the grouping. Query heads {0,1} (group 0)
/// must output V0; query heads {2,3} (group 1) must output V1.
#[test]
fn test_gqa_grouping_end_to_end_single_token() {
    let head_dim = 2;
    // KV in [batch, seq, heads, dim]; V0 = [1,0], V1 = [0,1].
    let v = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], (1, 1, 2, head_dim), DEVICE).unwrap();
    let k = Tensor::zeros((1, 1, 2, head_dim), candle_core::DType::F32, DEVICE).unwrap();
    // Expand to 4 query heads (group_size 2), then to [batch, heads, seq, dim].
    let v_exp = expand_kv(&v, 4, 2).unwrap().transpose(1, 2).unwrap();
    let k_exp = expand_kv(&k, 4, 2).unwrap().transpose(1, 2).unwrap();
    let q = Tensor::zeros((1, 4, 1, head_dim), candle_core::DType::F32, DEVICE).unwrap();

    let out = paged_attention(&q, &k_exp, &v_exp, 4, head_dim).unwrap();
    assert_eq!(out.dims(), &[1, 1, 4 * head_dim]);
    let vals: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
    // Blocked grouping => heads [V0, V0, V1, V1] = [1,0, 1,0, 0,1, 0,1].
    // (Tiled grouping would give [V0, V1, V0, V1] = [1,0, 0,1, 1,0, 0,1].)
    assert_eq!(
        vals,
        vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
        "query heads {{0,1}} must attend to KV head 0 and {{2,3}} to KV head 1; \
         got {vals:?} (wrong grouping => RIL ISS-010)"
    );
}

/// Independent, obviously-correct GQA attention reference: explicit
/// per-head grouping (query head `h` attends to KV head `h / group`),
/// scaled dot-product, causal mask, stable softmax. Used to validate the
/// production `expand_kv` + `paged_attention` path against the mathematical
/// definition (RIL ISS-010 / DEC-007: pin VALUES against a reference, not
/// just shape or path-parity — parity tests missed the grouping bug because
/// both paths shared the same buggy `expand_kv`).
#[allow(clippy::too_many_arguments)]
fn naive_gqa_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    num_q_heads: usize,
    num_kv_heads: usize,
    seq: usize,
    dim: usize,
) -> Vec<f32> {
    let group = num_q_heads / num_kv_heads;
    let scale = 1.0 / (dim as f32).sqrt();
    // q: [h_q, s, d]; k/v: [h_kv, s, d]; output: [h_q, s, d] (head-first).
    let qat = |h: usize, i: usize, x: usize| q[(h * seq + i) * dim + x];
    let kat = |h: usize, j: usize, x: usize| k[(h * seq + j) * dim + x];
    let vat = |h: usize, j: usize, x: usize| v[(h * seq + j) * dim + x];
    let mut out = vec![0.0f32; num_q_heads * seq * dim];
    for h in 0..num_q_heads {
        let kvh = h / group;
        for i in 0..seq {
            let mut scores: Vec<f32> = (0..seq)
                .map(|j| {
                    if j <= i {
                        (0..dim).map(|x| qat(h, i, x) * kat(kvh, j, x)).sum::<f32>() * scale
                    } else {
                        f32::NEG_INFINITY
                    }
                })
                .collect();
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = scores.iter().map(|s| (s - max).exp()).sum();
            for sc in &mut scores {
                *sc = (*sc - max).exp() / sum;
            }
            for x in 0..dim {
                out[(h * seq + i) * dim + x] = (0..seq).map(|j| scores[j] * vat(kvh, j, x)).sum();
            }
        }
    }
    out
}

/// Regression (RIL ISS-010 / DEC-007): the production GQA path
/// (`expand_kv` → transpose → `paged_attention`) must match an independent
/// naive reference that groups query head `h` to KV head `h / group`.
/// Pre-fix `expand_kv` tiled instead of blocked, so query heads past the
/// first group attended to the wrong KV head and this diverges.
#[test]
fn test_gqa_attention_matches_naive_reference() {
    // b=1, num_q_heads=4, num_kv_heads=2 (group=2), seq=3, dim=4.
    let (num_q_heads, num_kv_heads, seq, dim) = (4usize, 2, 3, 4);
    // Deterministic distinct values.
    let q_hsd: Vec<f32> = (0..num_q_heads * seq * dim)
        .map(|i| ((i * 7 + 3) % 11) as f32 - 5.0)
        .collect();
    let k_hsd: Vec<f32> = (0..num_kv_heads * seq * dim)
        .map(|i| ((i * 5 + 1) % 9) as f32 - 4.0)
        .collect();
    let v_hsd: Vec<f32> = (0..num_kv_heads * seq * dim)
        .map(|i| ((i * 3 + 2) % 13) as f32 - 6.0)
        .collect();

    // Reference (head-first inputs).
    let reference =
        naive_gqa_attention(&q_hsd, &k_hsd, &v_hsd, num_q_heads, num_kv_heads, seq, dim);

    // Production path: expand_kv works on [b, s, h, d]; paged_attention on
    // [b, h, s, d]. Mirror gqa/forward.rs: transpose to [b,s,h,d], expand,
    // transpose back to [b,h,s,d].
    let to_bs_hd = |t: &[f32], heads: usize| -> Tensor {
        Tensor::from_slice(t, (1, heads, seq, dim), DEVICE)
            .unwrap()
            .transpose(1, 2)
            .unwrap()
            .contiguous()
            .unwrap()
    };
    let k_bs_hd = to_bs_hd(&k_hsd, num_kv_heads);
    let v_bs_hd = to_bs_hd(&v_hsd, num_kv_heads);
    let k_exp = expand_kv(&k_bs_hd, num_q_heads, num_kv_heads)
        .unwrap()
        .transpose(1, 2)
        .unwrap()
        .contiguous()
        .unwrap();
    let v_exp = expand_kv(&v_bs_hd, num_q_heads, num_kv_heads)
        .unwrap()
        .transpose(1, 2)
        .unwrap()
        .contiguous()
        .unwrap();
    let q_bhsd = Tensor::from_slice(&q_hsd, (1, num_q_heads, seq, dim), DEVICE).unwrap();

    let out = paged_attention(&q_bhsd, &k_exp, &v_exp, num_q_heads, dim).unwrap();
    // paged_attention returns [b, s, h*d]; reshape to [h, s, d] for comparison.
    let out = out
        .reshape((1, seq, num_q_heads, dim))
        .unwrap()
        .transpose(1, 2)
        .unwrap()
        .contiguous()
        .unwrap();
    let got: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();

    assert_eq!(got.len(), reference.len());
    for (i, (g, r)) in got.iter().zip(reference.iter()).enumerate() {
        assert!(
            (g - r).abs() < 1e-4,
            "GQA attention diverges from naive reference at idx {i}: got {g}, want {r} \
             (wrong head grouping => RIL ISS-010)"
        );
    }
}
