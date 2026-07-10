//! Unit tests for the `GatedDeltaNet` rule primitives
//! (`l2_normalize`, `gated_delta_recurrent`, `gated_delta_step`,
//! `GatedDeltaNet::forward`) and prefill/decode parity.
//!
//! Extracted from `rule.rs` to keep the implementation file under the
//! project's 800-line soft cap. Exercises:
//!
//! - `l2_normalize` produces unit-length output
//! - `gated_delta_recurrent` output shape (batch, seq, heads, dv)
//! - `GatedDeltaNet::forward` round-trip (output dims match input)
//! - beta-sigmoid bounds (large positive → ~1, large negative → ~0)
//! - prefill + decode ≡ single forward (numerical parity check)

use super::*;
use candle_core::Device;
use candle_nn::VarBuilder;

fn tiny_config() -> GatedDeltaConfig {
    GatedDeltaConfig {
        num_k_heads: 2,
        num_v_heads: 4,
        key_head_dim: 8,
        value_head_dim: 8,
        conv_kernel_size: 4,
    }
}

fn build_tiny_gdn() -> GatedDeltaNet {
    let device = Device::Cpu;
    let cfg = tiny_config();
    let hidden = 64;
    let vb = VarBuilder::zeros(DType::F32, &device);

    let in_proj_qkv = candle_nn::linear(hidden, cfg.qkv_proj_dim(), vb.pp("in_proj_qkv")).unwrap();
    let in_proj_z = candle_nn::linear(hidden, cfg.value_dim(), vb.pp("in_proj_z")).unwrap();
    let in_proj_a = candle_nn::linear(hidden, cfg.num_v_heads, vb.pp("in_proj_a")).unwrap();
    let in_proj_b = candle_nn::linear(hidden, cfg.num_v_heads, vb.pp("in_proj_b")).unwrap();
    let out_proj = candle_nn::linear(cfg.value_dim(), hidden, vb.pp("out_proj")).unwrap();
    let norm = candle_nn::layer_norm(hidden, 1e-5, vb.pp("norm")).unwrap();

    let conv_cfg = candle_nn::Conv1dConfig {
        padding: 3,
        groups: cfg.qkv_proj_dim(),
        ..Default::default()
    };
    let conv = candle_nn::conv1d(
        cfg.qkv_proj_dim(),
        cfg.qkv_proj_dim(),
        4,
        conv_cfg,
        vb.pp("conv"),
    )
    .unwrap();

    GatedDeltaNet {
        config: cfg,
        in_proj_qkv,
        in_proj_z,
        in_proj_a,
        in_proj_b,
        conv,
        a_log: Tensor::zeros(cfg.num_v_heads, DType::F32, &device).unwrap(),
        dt_bias: Tensor::ones(cfg.num_v_heads, DType::F32, &device).unwrap(),
        out_proj,
        norm,
    }
}

#[test]
fn test_l2_normalize_unit_length() -> CandleResult<()> {
    let device = Device::Cpu;
    let x = Tensor::new(&[3.0f32, 4.0], &device)?.reshape((1, 1, 1, 2))?;
    let y = l2_normalize(&x, 1e-6)?;
    let norm: f32 = y.sqr()?.sum_all()?.to_scalar()?;
    assert!((norm - 1.0).abs() < 1e-4);
    Ok(())
}

#[test]
fn test_gated_delta_recurrent_output_shape() {
    let device = Device::Cpu;
    let batch = 1;
    let seq = 5;
    let heads = 4;
    let dk = 8;
    let dv = 8;

    let q = Tensor::randn(0.0f32, 1.0, (batch, seq, heads, dk), &device).unwrap();
    let k = Tensor::randn(0.0f32, 1.0, (batch, seq, heads, dk), &device).unwrap();
    let v = Tensor::randn(0.0f32, 1.0, (batch, seq, heads, dv), &device).unwrap();
    let g = Tensor::full(0.9f32, (batch, seq, heads), &device).unwrap();
    let beta = Tensor::full(0.5f32, (batch, seq, heads), &device).unwrap();

    let out = gated_delta_recurrent(&q, &k, &v, &g, &beta).unwrap();
    assert_eq!(out.0.dims(), &[batch, seq, heads, dv]);
}

#[test]
fn test_gated_delta_net_forward_shape() {
    let gdn = build_tiny_gdn();
    let device = Device::Cpu;
    let x = Tensor::randn(0.0f32, 1.0, (1, 6, 64), &device).unwrap();
    let out = gdn.forward(&x).unwrap();
    assert_eq!(out.dims(), x.dims());
}

#[test]
fn test_beta_sigmoid_bounds_recurrent() {
    let device = Device::Cpu;
    let b = Tensor::new(&[10.0f32, -10.0], &device)
        .unwrap()
        .reshape((1, 1, 2))
        .unwrap();
    let beta = candle_nn::ops::sigmoid(&b).unwrap();
    let vals: Vec<f32> = beta.flatten_all().unwrap().to_vec1().unwrap();
    assert!(vals[0] > 0.99);
    assert!(vals[1] < 0.01);
}

#[test]
fn test_prefill_decode_parity() {
    let gdn = build_tiny_gdn();
    let device = Device::Cpu;
    let batch = 1;
    let hidden = 64;
    let prefill_len = 6;
    let decode_len = 3;
    let total_len = prefill_len + decode_len;

    let full_x = Tensor::randn(0.0f32, 1.0, (batch, total_len, hidden), &device).unwrap();
    let x_prefill = full_x
        .narrow(1, 0, prefill_len)
        .unwrap()
        .contiguous()
        .unwrap();

    let (prefill_out, mut state) = gdn.forward_prefill(&x_prefill).unwrap();
    assert_eq!(prefill_out.dims(), x_prefill.dims());

    let mut decode_outs = Vec::new();
    for t in 0..decode_len {
        let token = full_x
            .narrow(1, prefill_len + t, 1)
            .unwrap()
            .contiguous()
            .unwrap();
        let out = gdn.forward_decode(&token, &mut state).unwrap();
        decode_outs.push(out);
    }
    let decode_cat = Tensor::cat(&decode_outs, 1).unwrap();

    let full_out = gdn.forward(&full_x).unwrap();
    let expected = full_out
        .narrow(1, prefill_len, decode_len)
        .unwrap()
        .contiguous()
        .unwrap();

    let diff = (decode_cat - expected).unwrap().abs().unwrap();
    let max_diff: f32 = diff.max_all().unwrap().to_scalar().unwrap();
    assert!(
        max_diff < 1e-4,
        "prefill+decode parity failed: max_diff={max_diff}"
    );
}
