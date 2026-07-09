//! Rotary Position Embedding (RoPE): precompute sin/cos cache and apply rotation to query/key tensors.
//!
//! The cache shape is `(max_seq_len, head_dim/2)`; `apply_rope` mutates
//! the input tensor in-place when possible. `MRoPE` (multi-modal `RoPE`
//! for Qwen3.5-VL) lives in `mrope.rs` alongside this module.
#![allow(clippy::module_name_repetitions)]
// invariant: rope positional-index casts (position/seq_len -> f32) are bounded
// by sequence length and head_dim, both small model-architecture constants;
// precision loss / truncation is intentional.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap
)]

use crate::qwen3::config::Qwen3Config;
use candle_core::{Result, Tensor};

/// `RoPE`. See the type definition for fields and behavior.
#[derive(Clone, Debug)]
#[allow(dead_code)] // audited 2026-06-26 (Wave 1): pub(crate) fields never read externally; struct only used in self-tests
pub struct RoPE {
    pub(crate) theta: f32,
    pub(crate) head_dim: usize,
    pub(crate) max_position: usize,
    pub(crate) scaling_factor: f32,
    pub(crate) device: candle_core::Device,
}

impl RoPE {
    #[must_use]
    pub fn new(
        head_dim: usize,
        max_position: usize,
        theta: f32,
        device: &candle_core::Device,
    ) -> Self {
        Self {
            theta,
            head_dim,
            max_position,
            scaling_factor: 1.0,
            device: device.clone(),
        }
    }

    #[must_use]
    pub fn new_with_config(config: &Qwen3Config) -> Self {
        use candle_core::Device;
        let rope_scaling = config.rope_scaling();
        Self {
            theta: config.rope_theta(),
            head_dim: config.head_dim(),
            max_position: config.max_position_embeddings(),
            scaling_factor: rope_scaling.and_then(|r| r.factor).unwrap_or(1.0),
            device: Device::Cpu,
        }
    }

    #[must_use]
    pub const fn scaling_factor(&self) -> f32 {
        self.scaling_factor
    }

    /// Run the operation (see signature for params and return type).
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    pub fn apply(&self, x: &Tensor, positions: &[i64]) -> Result<Tensor> {
        apply_rope(x, positions, self.theta)
    }

    /// Run the layer forward pass over the input.
    /// # Errors
    ///
    /// Returns `Err` if any tensor operation fails (shape mismatch, out-of-memory, dtype incompatibility, or kernel error).
    pub fn forward(&self, q: &Tensor, k: &Tensor, position: i64) -> Result<(Tensor, Tensor)> {
        let positions: Vec<i64> = (0..q.dim(1)? as i64).map(|i| position + i).collect();
        let q_out = apply_rope(q, &positions, self.theta)?;
        let k_out = apply_rope(k, &positions, self.theta)?;
        Ok((q_out, k_out))
    }
}

/// Run the operation (see signature for params and return type).
/// # Errors
///
/// Returns `Err` if the operation fails.
pub fn apply_rope(query: &Tensor, positions: &[i64], theta: f32) -> Result<Tensor> {
    let (batch, seq_len, num_heads, head_dim) = query.dims4()?;

    let query = query.transpose(1, 2)?;

    let half_dim = head_dim / 2;

    let inv_freq: Vec<f32> = (0..half_dim)
        .map(|i| theta.powf(-2.0 * (i as f32) / (head_dim as f32)))
        .collect();

    let mut cos_matrix = Vec::with_capacity(seq_len * half_dim);
    let mut sin_matrix = Vec::with_capacity(seq_len * half_dim);

    for &pos in positions {
        let pos_f = pos as f32;
        for &freq in &inv_freq {
            let angle = pos_f * freq;
            cos_matrix.push(angle.cos());
            sin_matrix.push(angle.sin());
        }
    }

    let cos = Tensor::new(cos_matrix.as_slice(), query.device())?
        .reshape((1, seq_len, half_dim))?
        .broadcast_as((batch, num_heads, seq_len, half_dim))?;
    let sin = Tensor::new(sin_matrix.as_slice(), query.device())?
        .reshape((1, seq_len, half_dim))?
        .broadcast_as((batch, num_heads, seq_len, half_dim))?;

    let first_half = query.narrow(3, 0, half_dim)?;
    let second_half = query.narrow(3, half_dim, half_dim)?;

    let rotated_first = first_half
        .broadcast_mul(&cos)?
        .broadcast_add(&second_half.broadcast_mul(&sin)?)?;
    let rotated_second = second_half
        .broadcast_mul(&cos)?
        .broadcast_sub(&first_half.broadcast_mul(&sin)?)?;

    let result = Tensor::cat(&[&rotated_first, &rotated_second], 3)?;

    result.transpose(1, 2)
}

#[must_use]
pub fn precompute_rope_cache(seq_len: usize, head_dim: usize, theta: f32) -> Vec<(f32, f32)> {
    let mut cache = Vec::with_capacity(seq_len * head_dim / 2);
    for pos in 0..seq_len {
        for i in 0..head_dim / 2 {
            let freq = (pos as f32).powf(-2.0 * (i as f32) / (head_dim as f32)) * theta;
            cache.push((freq.cos(), freq.sin()));
        }
    }
    cache
}

// Unit tests are extracted to `tests.rs` (sibling) to keep this
// RoPE module under the 800-line soft cap. They cover the
// `apply_rope` / `precompute_rope_cache` free functions and the
// `RoPE` struct (`new`, `apply`, `forward`) — shape preservation,
// determinism, positional sensitivity, and numerical robustness
// at large positions.
#[cfg(test)]
mod tests;
