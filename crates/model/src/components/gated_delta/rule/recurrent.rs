//! Gated delta single-step recurrence + sequential scan over a sequence.
//!
//! - [`compute_decay`] produces the per-head decay factor `α` from the
//!   `g` gate, learned `a_log`, and `dt_bias`.
//! - [`gated_delta_step`] is one recurrent step on `(B, H, D_v)` tensors.
//! - [`gated_delta_recurrent`] / [`gated_delta_recurrent_with_state`]
//!   scan the step over a sequence dimension (prefill-style).

use crate::components::ssm::softplus;
use candle_core::{DType, Result as CandleResult, Tensor};

pub(super) fn compute_decay(
    g_alpha: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
) -> CandleResult<Tensor> {
    let num_heads = a_log.dims()[0];
    let a_log = a_log.reshape((1, 1, num_heads))?;
    let dt_bias = dt_bias.reshape((1, 1, num_heads))?;
    let neg_a_exp = a_log.exp()?.neg()?;
    let softplus_a = softplus(&g_alpha.broadcast_add(&dt_bias)?)?;
    let alpha = neg_a_exp.broadcast_mul(&softplus_a)?;
    alpha.exp()
}

/// # Errors
///
/// Returns `Err` if the operation fails.
/// Single recurrent GDN step. `q/k/v/g/beta` are `(batch, num_heads, ...)`.
pub fn gated_delta_step(
    q_t: &Tensor,
    k_t: &Tensor,
    v_t: &Tensor,
    g_t: &Tensor,
    beta_t: &Tensor,
    state: &Tensor,
) -> CandleResult<(Tensor, Tensor)> {
    let (batch, num_heads, _value_head_dim) = v_t.dims3()?;
    let key_head_dim = q_t.dims()[2];
    let scale = 1.0f32 / (key_head_dim as f32).sqrt();

    let g_4d = g_t.reshape((batch, num_heads, 1, 1))?;
    let mut next_state = state.broadcast_mul(&g_4d)?;

    let k_exp = k_t.unsqueeze(3)?;
    let kv_mem = next_state.broadcast_mul(&k_exp)?.sum(2)?;
    let delta = (v_t - kv_mem)?.broadcast_mul(&beta_t.reshape((batch, num_heads, 1))?)?;

    let k_outer = k_t.unsqueeze(3)?;
    let delta_outer = delta.unsqueeze(2)?;
    next_state = (next_state + k_outer.broadcast_mul(&delta_outer)?)?;

    let q_exp = q_t.unsqueeze(3)?;
    let out_t = next_state.broadcast_mul(&q_exp)?.sum(2)?;
    let out_t = out_t.mul(&Tensor::new(scale, q_t.device())?.broadcast_as(out_t.dims())?)?;
    Ok((out_t, next_state))
}

/// # Errors
///
/// Returns `Err` if the operation fails.
/// Recurrent gated delta rule over a sequence (prefill-style sequential scan).
pub fn gated_delta_recurrent(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
) -> CandleResult<(Tensor, Tensor)> {
    gated_delta_recurrent_with_state(q, k, v, g, beta, None)
}

/// Run the operation (see signature for params and return type).
/// # Errors
///
/// Returns `Err` if the operation fails.
#[allow(clippy::many_single_char_names)]
pub fn gated_delta_recurrent_with_state(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    initial_state: Option<&Tensor>,
) -> CandleResult<(Tensor, Tensor)> {
    let (batch, seq_len, num_heads, value_head_dim) = v.dims4()?;
    let key_head_dim = q.dims()[3];

    let mut state = match initial_state {
        Some(s) => s.clone(),
        None => Tensor::zeros(
            (batch, num_heads, key_head_dim, value_head_dim),
            DType::F32,
            q.device(),
        )?,
    };
    let mut outputs = Vec::with_capacity(seq_len);

    for t in 0..seq_len {
        let q_t = q.narrow(1, t, 1)?.squeeze(1)?;
        let k_t = k.narrow(1, t, 1)?.squeeze(1)?;
        let v_t = v.narrow(1, t, 1)?.squeeze(1)?;
        let g_t = g.narrow(1, t, 1)?.squeeze(1)?;
        let beta_t = beta.narrow(1, t, 1)?.squeeze(1)?;
        let (out_t, next_state) = gated_delta_step(&q_t, &k_t, &v_t, &g_t, &beta_t, &state)?;
        outputs.push(out_t.unsqueeze(1)?);
        state = next_state;
    }

    Ok((Tensor::cat(&outputs, 1)?, state))
}
