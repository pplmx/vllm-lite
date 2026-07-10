//! Gemma4 Attention implementation.
//!
//! Module layout:
//!
//! - [`self`] (`mod.rs`) — `Gemma4Attention` struct + `new` + `new_from_weights` + `Default` impl
//! - [`mask`] — sliding-window causal mask helpers
//! - [`kernels`] — projection / `RoPE` / `expand_kv` / attention compute kernels
//! - [`forward`] — `forward` + `forward_full` + `forward_sliding` + `forward_prefill` + `forward_decode`
//!
//! Tests live in the sibling `tests.rs` and exercise the sliding-window
//! mask contract + non-paged vs paged numerical equivalence for sliding
//! attention.

#![allow(clippy::too_many_arguments)]
// invariant: tensor-dimension casts (position/seq_len -> f32) are bounded by
// sequence length and head_dim, both small model-architecture constants;
// precision loss / truncation is intentional.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap
)]

mod forward;
mod kernels;
mod mask;

use candle_core::{Result, Tensor};
use candle_nn::Linear;

use crate::config::architecture::{LayerType, RoPEConfig};
use crate::gemma4::rope::Gemma4RoPE;

// Re-imported into the attention module's scope under `#[cfg(test)]` so
// the sibling `tests.rs` can `use super::*;` to pick up `Device` (the
// original `attention.rs` inlined everything; the split would otherwise
// hide it from the tests module).
#[cfg(test)]
use candle_core::Device;

/// `Gemma4Attention`. See the type definition for fields and behavior.
#[derive(Debug)]
pub(crate) struct Gemma4Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    sliding_window: usize,
    layer_type: LayerType,
    rope: Option<Gemma4RoPE>,
}

impl Gemma4Attention {
    #[allow(clippy::needless_pass_by_value)]
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        sliding_window: usize,
        layer_type: LayerType,
        rope_config: &RoPEConfig,
        vb: candle_nn::VarBuilder<'_>,
    ) -> Result<Self> {
        let q_proj = candle_nn::linear(hidden_size, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = candle_nn::linear(hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = candle_nn::linear(hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = candle_nn::linear(num_heads * head_dim, hidden_size, vb.pp("o_proj"))?;

        let rope = Gemma4RoPE::new(rope_config, head_dim);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            head_dim,
            sliding_window,
            layer_type,
            rope: Some(rope),
        })
    }

    pub fn new_from_weights(
        _hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        sliding_window: usize,
        layer_type: LayerType,
        rope_config: &RoPEConfig,
        q_w: Tensor,
        k_w: Tensor,
        v_w: Tensor,
        o_w: Tensor,
    ) -> Self {
        let rope = Gemma4RoPE::new(rope_config, head_dim);
        Self {
            q_proj: Linear::new(q_w, None),
            k_proj: Linear::new(k_w, None),
            v_proj: Linear::new(v_w, None),
            o_proj: Linear::new(o_w, None),
            num_heads,
            num_kv_heads,
            head_dim,
            sliding_window,
            layer_type,
            rope: Some(rope),
        }
    }
}

impl Default for Gemma4Attention {
    fn default() -> Self {
        // invariant: 1x1 F32 CPU tensor allocation cannot realistically fail (4 bytes).
        Self {
            q_proj: Linear::new(
                Tensor::zeros((1, 1), candle_core::DType::F32, &candle_core::Device::Cpu).expect(
                    "failed to allocate 1x1 F32 CPU tensor for Gemma4Attention::default q_proj",
                ),
                None,
            ),
            k_proj: Linear::new(
                Tensor::zeros((1, 1), candle_core::DType::F32, &candle_core::Device::Cpu).expect(
                    "failed to allocate 1x1 F32 CPU tensor for Gemma4Attention::default k_proj",
                ),
                None,
            ),
            v_proj: Linear::new(
                Tensor::zeros((1, 1), candle_core::DType::F32, &candle_core::Device::Cpu).expect(
                    "failed to allocate 1x1 F32 CPU tensor for Gemma4Attention::default v_proj",
                ),
                None,
            ),
            o_proj: Linear::new(
                Tensor::zeros((1, 1), candle_core::DType::F32, &candle_core::Device::Cpu).expect(
                    "failed to allocate 1x1 F32 CPU tensor for Gemma4Attention::default o_proj",
                ),
                None,
            ),
            num_heads: 0,
            num_kv_heads: 0,
            head_dim: 0,
            sliding_window: 512,
            layer_type: LayerType::FullAttention,
            rope: None,
        }
    }
}

// Unit tests are extracted to `tests.rs` (sibling) to keep this
// attention module under the 800-line soft cap. They cover the
// sliding-window mask contract (out-of-window keys → -inf) and the
// non-paged-vs-paged numerical equivalence for sliding attention.
#[cfg(test)]
mod tests;
