//! `GatedDeltaNet::forward` paths: prefill (process the full prompt and
//! build the recurrent + conv state) and decode (process one new token
//! against the cached state).
//!
//! Both paths share the same kernel helpers in [`super::kernels`],
//! [`super::conv`], and [`super::recurrent`]. The `forward` entry point
//! dispatches to `forward_prefill` and discards the state for tests /
//! callers that don't need incremental decoding.

use super::GatedDeltaNet;
use super::conv::{
    apply_causal_conv, apply_causal_conv_incremental, update_conv_state_from_prefill,
};
use super::kernels::{l2_normalize, repeat_kv_heads, split_qkv};
use super::recurrent::{compute_decay, gated_delta_recurrent, gated_delta_step};
use crate::components::gated_delta::state::GatedDeltaState;
use candle_core::{Module, Result as CandleResult, Tensor};

impl GatedDeltaNet {
    /// Run the layer forward pass over the input.
    /// # Errors
    ///
    /// Returns `Err` if any tensor operation fails (shape mismatch, out-of-memory, dtype incompatibility, or kernel error).
    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        self.forward_prefill(x).map(|(out, _)| out)
    }

    /// Run the prefill path: process the full prompt and cache its KV.
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    #[allow(clippy::many_single_char_names)]
    pub fn forward_prefill(&self, x: &Tensor) -> CandleResult<(Tensor, GatedDeltaState)> {
        let residual = x.clone();
        let (batch, seq_len, _hidden) = x.dims3()?;

        if seq_len < 4 {
            let qkv = self.in_proj_qkv.forward(x)?;
            let gated = candle_nn::ops::silu(&qkv)?;
            let output = self.out_proj.forward(&gated)?;
            let output = output.add(&residual)?;
            let state = GatedDeltaState::new(batch, &self.config, x.device())?;
            return Ok((output, state));
        }

        let qkv = self.in_proj_qkv.forward(x)?;
        let z = self.in_proj_z.forward(x)?;
        let a = self.in_proj_a.forward(x)?;
        let b = self.in_proj_b.forward(x)?;

        let (q_raw, k_raw, v_raw) = split_qkv(&qkv, &self.config)?;
        let conv = update_conv_state_from_prefill(&q_raw, &k_raw, &v_raw, &self.config)?;

        let (mut q, mut k, v) = apply_causal_conv(&self.conv, &q_raw, &k_raw, &v_raw)?;
        k = repeat_kv_heads(&k, self.config.num_v_heads)?;
        q = repeat_kv_heads(&q, self.config.num_v_heads)?;
        q = l2_normalize(&q, 1e-6)?;
        k = l2_normalize(&k, 1e-6)?;

        let g = compute_decay(&a, &self.a_log, &self.dt_bias)?;
        let beta = candle_nn::ops::sigmoid(&b)?;

        let (core_out, recurrent) = gated_delta_recurrent(&q, &k, &v, &g, &beta)?;
        let output = self.finalize_output(&core_out, &z, &residual, batch, seq_len)?;

        Ok((output, GatedDeltaState { recurrent, conv }))
    }

    /// Run the decode path: process one new token against cached KV.
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    #[allow(clippy::many_single_char_names)]
    pub fn forward_decode(&self, x: &Tensor, state: &mut GatedDeltaState) -> CandleResult<Tensor> {
        let residual = x.clone();
        let (batch, seq_len, _hidden) = x.dims3()?;
        if seq_len != 1 {
            return Err(candle_core::Error::msg(format!(
                "GDN decode expects seq_len=1, got {seq_len}"
            )));
        }

        if self.config.conv_state_width() == 0 {
            return self.forward(x);
        }

        let qkv = self.in_proj_qkv.forward(x)?;
        let z = self.in_proj_z.forward(x)?;
        let a = self.in_proj_a.forward(x)?;
        let b = self.in_proj_b.forward(x)?;

        let (q_raw, k_raw, v_raw) = split_qkv(&qkv, &self.config)?;
        let (q, k, v, conv) =
            apply_causal_conv_incremental(&self.conv, &q_raw, &k_raw, &v_raw, &state.conv)?;

        let mut q = repeat_kv_heads(&q, self.config.num_v_heads)?;
        let mut k = repeat_kv_heads(&k, self.config.num_v_heads)?;
        q = l2_normalize(&q, 1e-6)?;
        k = l2_normalize(&k, 1e-6)?;

        let g = compute_decay(&a, &self.a_log, &self.dt_bias)?;
        let beta = candle_nn::ops::sigmoid(&b)?;

        let q_t = q.squeeze(1)?;
        let k_t = k.squeeze(1)?;
        let v_t = v.squeeze(1)?;
        let g_t = g.squeeze(1)?;
        let beta_t = beta.squeeze(1)?;

        let (core_t, recurrent) =
            gated_delta_step(&q_t, &k_t, &v_t, &g_t, &beta_t, &state.recurrent)?;
        state.recurrent = recurrent;
        state.conv = conv;

        let core_out = core_t.unsqueeze(1)?;
        self.finalize_output(&core_out, &z, &residual, batch, 1)
    }

    fn finalize_output(
        &self,
        core_out: &Tensor,
        z: &Tensor,
        residual: &Tensor,
        batch: usize,
        seq_len: usize,
    ) -> CandleResult<Tensor> {
        let core_flat = core_out.reshape((batch, seq_len, self.config.value_dim()))?;
        let z_gate = candle_nn::ops::silu(z)?;
        let gated = core_flat.broadcast_mul(&z_gate)?;
        let output = self.out_proj.forward(&gated)?;
        let output = output.add(residual)?;
        self.norm.forward(&output)
    }
}
