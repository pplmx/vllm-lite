//! Gated DeltaNet (GDN) recurrent linear attention for Qwen3.5 hybrid layers.

use crate::components::ssm::softplus;
use candle_core::{DType, Module, Result as CandleResult, Tensor};
use candle_nn::{Conv1d, LayerNorm, Linear};

#[derive(Debug, Clone, Copy)]
pub struct GatedDeltaConfig {
    pub num_k_heads: usize,
    pub num_v_heads: usize,
    pub key_head_dim: usize,
    pub value_head_dim: usize,
}

impl GatedDeltaConfig {
    pub fn key_dim(&self) -> usize {
        self.num_k_heads * self.key_head_dim
    }

    pub fn value_dim(&self) -> usize {
        self.num_v_heads * self.value_head_dim
    }

    pub fn qkv_proj_dim(&self) -> usize {
        2 * self.key_dim() + self.value_dim()
    }
}

pub struct GatedDeltaNet {
    pub config: GatedDeltaConfig,
    in_proj_qkv: Linear,
    in_proj_z: Linear,
    in_proj_a: Linear,
    in_proj_b: Linear,
    conv: Conv1d,
    a_log: Tensor,
    dt_bias: Tensor,
    out_proj: Linear,
    norm: LayerNorm,
}

pub fn l2_normalize(xs: &Tensor, eps: f32) -> CandleResult<Tensor> {
    let last = xs.dims().len().saturating_sub(1);
    let sq = xs.sqr()?;
    let norm = sq.sum_keepdim(last)?;
    let eps_t = Tensor::new(eps, xs.device())?.broadcast_as(norm.dims())?;
    let norm = (norm + eps_t)?.sqrt()?;
    xs.broadcast_div(&norm)
}

fn repeat_kv_heads(kv: &Tensor, num_v_heads: usize) -> CandleResult<Tensor> {
    let num_k_heads = kv.dims()[2];
    if num_k_heads == num_v_heads {
        return Ok(kv.clone());
    }
    if num_v_heads % num_k_heads != 0 {
        return Err(candle_core::Error::msg(format!(
            "num_v_heads {num_v_heads} must be a multiple of num_k_heads {num_k_heads}"
        )));
    }
    let repeat = num_v_heads / num_k_heads;
    kv.repeat(&[1, 1, repeat, 1])
}

fn apply_causal_conv(
    conv: &Conv1d,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
) -> CandleResult<(Tensor, Tensor, Tensor)> {
    let seq_len = q.dims()[1];
    let q_flat = q.flatten(2, 3)?;
    let k_flat = k.flatten(2, 3)?;
    let v_flat = v.flatten(2, 3)?;
    let mixed = Tensor::cat(&[&q_flat, &k_flat, &v_flat], 2)?;
    let mixed = mixed.transpose(1, 2)?;
    let mixed = conv.forward(&mixed)?;
    let mixed = mixed.transpose(1, 2)?;
    let mixed = candle_nn::ops::silu(&mixed)?;

    let out_len = mixed.dims()[1];
    let start = out_len.saturating_sub(seq_len);
    let mixed = mixed.narrow(1, start, seq_len)?;

    let q_dim = q_flat.dims()[2];
    let k_dim = k_flat.dims()[2];
    let v_dim = v_flat.dims()[2];

    let q_out = mixed.narrow(2, 0, q_dim)?.reshape(q.dims())?;
    let k_out = mixed.narrow(2, q_dim, k_dim)?.reshape(k.dims())?;
    let v_out = mixed
        .narrow(2, q_dim + k_dim, v_dim)?
        .reshape(v.dims())?;

    Ok((q_out, k_out, v_out))
}

fn split_qkv(qkv: &Tensor, config: &GatedDeltaConfig) -> CandleResult<(Tensor, Tensor, Tensor)> {
    let (batch, seq_len, _) = qkv.dims3()?;
    let q = qkv.narrow(2, 0, config.key_dim())?;
    let k = qkv.narrow(2, config.key_dim(), config.key_dim())?;
    let v = qkv.narrow(2, 2 * config.key_dim(), config.value_dim())?;

    let q = q.reshape((batch, seq_len, config.num_k_heads, config.key_head_dim))?;
    let k = k.reshape((batch, seq_len, config.num_k_heads, config.key_head_dim))?;
    let v = v.reshape((batch, seq_len, config.num_v_heads, config.value_head_dim))?;
    Ok((q, k, v))
}

fn compute_decay(g_alpha: &Tensor, a_log: &Tensor, dt_bias: &Tensor) -> CandleResult<Tensor> {
    let num_heads = a_log.dims()[0];
    let a_log = a_log.reshape((1, 1, num_heads))?;
    let dt_bias = dt_bias.reshape((1, 1, num_heads))?;
    let neg_a_exp = a_log.exp()?.neg()?;
    let softplus_a = softplus(&g_alpha.broadcast_add(&dt_bias)?)?;
    let alpha = neg_a_exp.broadcast_mul(&softplus_a)?;
    alpha.exp()
}

/// Recurrent gated delta rule over a sequence (prefill-style sequential scan).
pub fn gated_delta_recurrent(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
) -> CandleResult<Tensor> {
    let (batch, seq_len, num_heads, value_head_dim) = v.dims4()?;
    let key_head_dim = q.dims()[3];
    let scale = 1.0f32 / (key_head_dim as f32).sqrt();

    let mut state =
        Tensor::zeros((batch, num_heads, key_head_dim, value_head_dim), DType::F32, q.device())?;
    let mut outputs = Vec::with_capacity(seq_len);

    for t in 0..seq_len {
        let q_t = q.narrow(1, t, 1)?.squeeze(1)?;
        let k_t = k.narrow(1, t, 1)?.squeeze(1)?;
        let v_t = v.narrow(1, t, 1)?.squeeze(1)?;
        let g_t = g.narrow(1, t, 1)?.squeeze(1)?;
        let beta_t = beta.narrow(1, t, 1)?.squeeze(1)?;

        let g_4d = g_t.reshape((batch, num_heads, 1, 1))?;
        state = state.broadcast_mul(&g_4d)?;

        let k_exp = k_t.unsqueeze(3)?;
        let kv_mem = state.broadcast_mul(&k_exp)?.sum(2)?;

        let delta = (v_t - kv_mem)?.broadcast_mul(&beta_t.reshape((batch, num_heads, 1))?)?;

        let k_outer = k_t.unsqueeze(3)?;
        let delta_outer = delta.unsqueeze(2)?;
        state = (state + k_outer.broadcast_mul(&delta_outer)?)?;

        let q_exp = q_t.unsqueeze(3)?;
        let out_t = state.broadcast_mul(&q_exp)?.sum(2)?;
        let out_t = out_t.mul(&Tensor::new(scale, q.device())?.broadcast_as(out_t.dims())?)?;
        outputs.push(out_t.unsqueeze(1)?);
    }

    Tensor::cat(&outputs, 1)
}

impl GatedDeltaNet {
    #[allow(clippy::too_many_arguments)]
    pub fn from_components(
        config: GatedDeltaConfig,
        in_proj_qkv: Linear,
        in_proj_z: Linear,
        in_proj_a: Linear,
        in_proj_b: Linear,
        conv: Conv1d,
        a_log: Tensor,
        dt_bias: Tensor,
        out_proj: Linear,
        norm: LayerNorm,
    ) -> Self {
        Self {
            config,
            in_proj_qkv,
            in_proj_z,
            in_proj_a,
            in_proj_b,
            conv,
            a_log,
            dt_bias,
            out_proj,
            norm,
        }
    }

    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let residual = x.clone();
        let (batch, seq_len, _hidden) = x.dims3()?;

        if seq_len < 4 {
            let qkv = self.in_proj_qkv.forward(x)?;
            let gated = candle_nn::ops::silu(&qkv)?;
            let output = self.out_proj.forward(&gated)?;
            return output.add(&residual);
        }

        let qkv = self.in_proj_qkv.forward(x)?;
        let z = self.in_proj_z.forward(x)?;
        let a = self.in_proj_a.forward(x)?;
        let b = self.in_proj_b.forward(x)?;

        let (mut q, mut k, mut v) = split_qkv(&qkv, &self.config)?;
        (q, k, v) = apply_causal_conv(&self.conv, &q, &k, &v)?;

        k = repeat_kv_heads(&k, self.config.num_v_heads)?;
        q = repeat_kv_heads(&q, self.config.num_v_heads)?;

        q = l2_normalize(&q, 1e-6)?;
        k = l2_normalize(&k, 1e-6)?;

        let g = compute_decay(&a, &self.a_log, &self.dt_bias)?;
        let beta = candle_nn::ops::sigmoid(&b)?;

        let core_out = gated_delta_recurrent(&q, &k, &v, &g, &beta)?;

        let core_flat = core_out.reshape((batch, seq_len, self.config.value_dim()))?;
        let z_gate = candle_nn::ops::silu(&z)?;
        let gated = core_flat.broadcast_mul(&z_gate)?;

        let output = self.out_proj.forward(&gated)?;
        let output = output.add(&residual)?;
        self.norm.forward(&output)
    }

    pub fn a_log(&self) -> &Tensor {
        &self.a_log
    }

    pub fn dt_bias(&self) -> &Tensor {
        &self.dt_bias
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::{VarBuilder, conv1d};

    fn tiny_config() -> GatedDeltaConfig {
        GatedDeltaConfig {
            num_k_heads: 2,
            num_v_heads: 4,
            key_head_dim: 8,
            value_head_dim: 8,
        }
    }

    fn build_tiny_gdn() -> GatedDeltaNet {
        let device = Device::Cpu;
        let cfg = tiny_config();
        let hidden = 64;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let in_proj_qkv =
            candle_nn::linear(hidden, cfg.qkv_proj_dim(), vb.pp("in_proj_qkv")).unwrap();
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
        let conv = conv1d(
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
        assert_eq!(out.dims(), &[batch, seq, heads, dv]);
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
}
