//! Causal convolution helpers for the Gated `DeltaNet` rule.
//!
//! Two execution modes share the same `Conv1d` weight:
//! - [`apply_causal_conv`] runs the convolution on a full prefill
//!   window and slices the output to the input sequence length.
//! - [`apply_causal_conv_incremental`] runs the convolution on the
//!   concatenation of the cached state plus the new frame, returning
//!   only the last token's output plus the updated state.
//! - [`update_conv_state_from_prefill`] builds the cache that the
//!   decode path will feed back in.

use super::GatedDeltaConfig;
use super::kernels::{mixed_qkv_flat, split_mixed_qkv};
use candle_core::{Module, Result as CandleResult, Tensor};
use candle_nn::Conv1d;

pub(super) fn apply_causal_conv(
    conv: &Conv1d,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
) -> CandleResult<(Tensor, Tensor, Tensor)> {
    let seq_len = q.dims()[1];
    let mixed = mixed_qkv_flat(q, k, v)?;
    let mixed = mixed.transpose(1, 2)?;
    let mixed = conv.forward(&mixed)?;
    let mixed = mixed.transpose(1, 2)?;
    let mixed = candle_nn::ops::silu(&mixed)?;

    let out_len = mixed.dims()[1];
    let start = out_len.saturating_sub(seq_len);
    let mixed = mixed.narrow(1, start, seq_len)?;

    split_mixed_qkv(&mixed, q.dims(), k.dims(), v.dims())
}

pub(super) fn apply_causal_conv_incremental(
    conv: &Conv1d,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    conv_state: &Tensor,
) -> CandleResult<(Tensor, Tensor, Tensor, Tensor)> {
    let mixed = mixed_qkv_flat(q, k, v)?;
    let frame = mixed.transpose(1, 2)?;
    let window = Tensor::cat(&[conv_state, &frame], 2)?;
    let conv_out = conv.forward(&window)?;
    let last = conv_out.narrow(2, window.dims()[2] - 1, 1)?;
    let activated = candle_nn::ops::silu(&last.transpose(1, 2)?)?;
    let (q_out, k_out, v_out) = split_mixed_qkv(&activated, q.dims(), k.dims(), v.dims())?;
    let state_width = conv_state.dims()[2];
    let new_state = window.narrow(2, window.dims()[2] - state_width, state_width)?;
    Ok((q_out, k_out, v_out, new_state))
}

pub(super) fn update_conv_state_from_prefill(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    config: &GatedDeltaConfig,
) -> CandleResult<Tensor> {
    let mixed = mixed_qkv_flat(q, k, v)?;
    let mixed = mixed.transpose(1, 2)?;
    let seq_len = mixed.dims()[2];
    let width = config.conv_state_width();
    if seq_len >= width {
        Ok(mixed.narrow(2, seq_len - width, width)?)
    } else {
        let pad = Tensor::zeros(
            (mixed.dims()[0], mixed.dims()[1], width - seq_len),
            mixed.dtype(),
            mixed.device(),
        )?;
        let padded = Tensor::cat(&[&pad, &mixed], 2)?;
        Ok(padded)
    }
}
