//! Pure tensor helpers for the Gated `DeltaNet` rule: L2 normalise, head
//! repetition, and qkv splitting/joining. No parameter state, no
//! recurrence — these are the smallest reusable primitives.

use super::GatedDeltaConfig;
use candle_core::{Result as CandleResult, Tensor};

/// `l2_normalize`: l2 normalize.
pub fn l2_normalize(xs: &Tensor, eps: f32) -> CandleResult<Tensor> {
    let last = xs.dims().len().saturating_sub(1);
    let sq = xs.sqr()?;
    let norm = sq.sum_keepdim(last)?;
    let eps_t = Tensor::new(eps, xs.device())?.broadcast_as(norm.dims())?;
    let norm = (norm + eps_t)?.sqrt()?;
    xs.broadcast_div(&norm)
}

#[allow(clippy::similar_names)]
pub(super) fn repeat_kv_heads(kv: &Tensor, num_v_heads: usize) -> CandleResult<Tensor> {
    let num_k_heads = kv.dims()[2];
    if num_k_heads == num_v_heads {
        return Ok(kv.clone());
    }
    if !num_v_heads.is_multiple_of(num_k_heads) {
        return Err(candle_core::Error::msg(format!(
            "num_v_heads {num_v_heads} must be a multiple of num_k_heads {num_k_heads}"
        )));
    }
    let repeat = num_v_heads / num_k_heads;
    kv.repeat(&[1, 1, repeat, 1])
}

pub(super) fn mixed_qkv_flat(q: &Tensor, k: &Tensor, v: &Tensor) -> CandleResult<Tensor> {
    let q_flat = q.flatten(2, 3)?;
    let k_flat = k.flatten(2, 3)?;
    let v_flat = v.flatten(2, 3)?;
    Tensor::cat(&[&q_flat, &k_flat, &v_flat], 2)
}

pub(super) fn split_mixed_qkv(
    mixed: &Tensor,
    q_shape: &[usize],
    k_shape: &[usize],
    v_shape: &[usize],
) -> CandleResult<(Tensor, Tensor, Tensor)> {
    let q_dim = q_shape[2] * q_shape[3];
    let k_dim = k_shape[2] * k_shape[3];
    let v_dim = v_shape[2] * v_shape[3];
    let q_out = mixed.narrow(2, 0, q_dim)?.reshape(q_shape)?;
    let k_out = mixed.narrow(2, q_dim, k_dim)?.reshape(k_shape)?;
    let v_out = mixed.narrow(2, q_dim + k_dim, v_dim)?.reshape(v_shape)?;
    Ok((q_out, k_out, v_out))
}

pub(super) fn split_qkv(
    qkv: &Tensor,
    config: &GatedDeltaConfig,
) -> CandleResult<(Tensor, Tensor, Tensor)> {
    let (batch, seq_len, _) = qkv.dims3()?;
    let q = qkv.narrow(2, 0, config.key_dim())?;
    let k = qkv.narrow(2, config.key_dim(), config.key_dim())?;
    let v = qkv.narrow(2, 2 * config.key_dim(), config.value_dim())?;

    let q = q.reshape((batch, seq_len, config.num_k_heads, config.key_head_dim))?;
    let k = k.reshape((batch, seq_len, config.num_k_heads, config.key_head_dim))?;
    let v = v.reshape((batch, seq_len, config.num_v_heads, config.value_head_dim))?;
    Ok((q, k, v))
}
