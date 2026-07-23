//! Gated `DeltaNet` recurrent rule + supporting kernels: causal convolution, qkv splitting, head repetition, L2 normalisation, and the gated delta step / recurrent scan.
//!
//! The `gated_delta_recurrent` function is the public entry point; it
//! composes the helper kernels into a single recurrent call. Used by
//! Qwen3.5 hybrid layers as the SSM component.
//!
//! Module layout:
//!
//! - [`self`] (`mod.rs`) — `GatedDeltaNet` struct + construction + getters
//! - [`kernels`] — pure tensor helpers (l2 normalise, qkv split, kv head repeat)
//! - [`conv`] — causal convolution (prefill + incremental) + state update
//! - [`recurrent`] — gated-delta single step + recurrent scan
//! - [`forward`] — `GatedDeltaNet::forward` / `forward_prefill` / `forward_decode`
//!
//! Tests live in the sibling `tests.rs` and exercise the public surface
//! (`l2_normalize` unit length, `gated_delta_recurrent` output shape,
//! `GatedDeltaNet::forward` shape, beta sigmoid bounds, and
//! prefill+decode numerical parity vs. single forward).

// invariant: tensor-dimension casts (head_dim -> f32) are bounded by model
// architecture constants; precision loss is intentional.
#![allow(clippy::cast_precision_loss)]

mod conv;
mod forward;
mod kernels;
mod recurrent;

use candle_core::Tensor;
use candle_nn::{Conv1d, LayerNorm, Linear};

use super::state::GatedDeltaConfig;

// Re-imported into the rule module's scope under `#[cfg(test)]` so the
// sibling `tests.rs` can `use super::*` to pick them up (the original
// rule.rs file inlined everything; the split would otherwise hide them
// from the tests module).
#[cfg(test)]
use candle_core::{DType, Result as CandleResult};

#[derive(Debug)]
/// `GatedDeltaNet`. See the type definition for fields and behavior.
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

impl GatedDeltaNet {
    /// Construct a `GatedDeltaNet` from pre-built components (used when
    /// loading weights from a non-HF checkpoint or custom loader).
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub const fn from_components(
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

    /// Access the `a_log` (state-transition matrix) tensor.
    #[must_use]
    pub const fn a_log(&self) -> &Tensor {
        &self.a_log
    }

    /// Access the `dt_bias` (step-size bias) tensor.
    #[must_use]
    pub const fn dt_bias(&self) -> &Tensor {
        &self.dt_bias
    }
}

// Public re-exports: keep the `gated_delta::` API stable.
pub use self::recurrent::{
    gated_delta_recurrent, gated_delta_recurrent_with_state, gated_delta_step,
};
// Internal re-export so `tests.rs`'s `use super::*;` can find it.
#[allow(unused_imports)]
use self::kernels::l2_normalize;

// Unit tests are extracted to `tests.rs` to keep this file under the
// 800-line soft cap. See `tests.rs` for the test surface
// (`l2_normalize` unit length, `gated_delta_recurrent` output shape,
// `GatedDeltaNet::forward` shape, beta sigmoid bounds, and
// prefill+decode numerical parity vs. single forward).
#[cfg(test)]
mod tests;
