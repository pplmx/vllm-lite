//! Rotary Position Embedding (RoPE): precompute sin/cos cache and apply rotation to query/key tensors.
//!
//! The cache shape is `(max_seq_len, head_dim/2)`; `apply_rope` mutates
//! the input tensor in-place when possible. `MRoPE` (multi-modal `RoPE`
//! for Qwen3.5-VL) lives in `mrope.rs` alongside this module.
//!
//! ## Long-context scaling
//!
//! When constructed via `RoPE::new_with_config(&Qwen3Config)`, the
//! scaling fields in the config (`rope_scaling.rope_type`,
//! `rope_scaling.factor`, `rope_scaling.attn_factor`,
//! `rope_scaling.original_max_position_embeddings`) are captured into
//! the struct. Use [`RoPE::apply_with_scaling`] to honour them; the
//! plain [`RoPE::apply`] / free [`apply_rope`] remain scaling-free
//! for backward compatibility with callers that pass `theta` directly
//! (rope_gqa, mla, gemma4 attention modules).
//!
//! Supported algorithms (selected by `RopeType`):
//! - `Default` — no scaling (current behaviour, preserved).
//! - `Linear`  — position interpolation: angle = (pos / scale) * freq.
//! - `Yarn`    — NTK-aware theta adjustment: theta' = theta *
//!   scale^(d/(d-2)). High-frequency dims barely change; low-frequency
//!   dims compress to fit longer contexts. This is the "global NTK"
//!   approximation used by many open-source implementations; the
//!   attention-scaling half of YaRN (`attn_factor`) is **stored** on
//!   the struct so the attention layer can pick it up, but is not
//!   applied inside `apply_rope` (that lives in the attention kernel).
//! - `Dynamic`, `Su`, `Other` — fall through to Default for now;
//!   follow-up work can add bespoke algorithms.
#![allow(clippy::module_name_repetitions)]
// invariant: rope positional-index casts (position/seq_len -> f32) are bounded
// by sequence length and head_dim, both small model-architecture constants;
// precision loss / truncation is intentional.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap
)]

use crate::qwen3::config::{Qwen3Config, RopeScaling, RopeType};
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
    /// Which long-context algorithm to apply (default = no scaling).
    pub(crate) rope_type: RopeType,
    /// Attention-scaling factor for YaRN; consumed by the attention
    /// layer, not by `apply_rope` itself.
    pub(crate) attn_factor: Option<f32>,
    /// Original context length the scaling was tuned for (YaRN).
    pub(crate) original_max_position: Option<usize>,
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
            rope_type: RopeType::Default,
            attn_factor: None,
            original_max_position: None,
        }
    }

    /// Construct an `RoPE` populated with long-context scaling fields from
    /// the supplied [`RopeScalingContext`]. Use this when the upstream
    /// architecture config declares a non-default `rope_scaling` block
    /// (YaRN, Linear, Dynamic, Su). The plain [`RoPE::new`] remains
    /// available for backward compatibility with callers that don't have a
    /// scaling config handy.
    #[must_use]
    pub fn new_with_scaling(
        head_dim: usize,
        max_position: usize,
        theta: f32,
        device: &candle_core::Device,
        scaling: RopeScalingContext,
    ) -> Self {
        Self {
            theta,
            head_dim,
            max_position,
            scaling_factor: scaling.scaling_factor,
            device: device.clone(),
            rope_type: scaling.rope_type,
            attn_factor: scaling.attn_factor,
            original_max_position: scaling.original_max_position,
        }
    }

    /// Construct from a Qwen3 config, extracting `rope_scaling` fields
    /// (`rope_type`, `factor`, `attn_factor`, `original_max_position_embeddings`).
    #[must_use]
    #[allow(dead_code)] // test-only helper; reachable under cfg(test) only
    pub(crate) fn new_with_config(config: &Qwen3Config) -> Self {
        use candle_core::Device;
        let rope_scaling = config.rope_scaling();
        Self {
            theta: config.rope_theta(),
            head_dim: config.head_dim(),
            max_position: config.max_position_embeddings(),
            scaling_factor: rope_scaling.and_then(|r| r.factor).unwrap_or(1.0),
            device: Device::Cpu,
            rope_type: rope_scaling
                .and_then(|r| r.rope_type)
                .unwrap_or(RopeType::Default),
            attn_factor: rope_scaling.and_then(|r| r.attn_factor),
            original_max_position: rope_scaling.and_then(|r| r.original_max_position_embeddings),
        }
    }

    #[must_use]
    pub const fn scaling_factor(&self) -> f32 {
        self.scaling_factor
    }

    /// Run the operation (see signature for params and return type).
    ///
    /// **Does not apply any long-context scaling.** Use
    /// [`RoPE::apply_with_scaling`] when the config declares
    /// `rope_scaling.rope_type` other than `default`.
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    pub fn apply(&self, x: &Tensor, positions: &[i64]) -> Result<Tensor> {
        apply_rope(x, positions, self.theta)
    }

    /// Long-context-aware variant of [`RoPE::apply`].
    ///
    /// Selects the inverse-frequency formula based on `self.rope_type`:
    /// - `Default` / unset → same as `apply`.
    /// - `Linear` → position interpolation (`inv_freq / scaling_factor`).
    /// - `Yarn` → NTK-aware theta adjustment.
    /// - `Dynamic` → HF-style recomputed scale based on current seq len.
    /// - `Su` → per-dim factor arrays (`short_factor` / `long_factor`).
    /// - `Other` → falls through to Default.
    /// # Errors
    ///
    /// Returns `Err` if the candle operation fails.
    pub fn apply_with_scaling(&self, x: &Tensor, positions: &[i64]) -> Result<Tensor> {
        apply_rope_with_scaling(x, positions, self.theta, self.scaling_ctx())
    }

    /// Bundle the scaling-related fields for passing to the free function.
    const fn scaling_ctx(&self) -> RopeScalingContext {
        RopeScalingContext {
            rope_type: self.rope_type,
            scaling_factor: self.scaling_factor,
            attn_factor: self.attn_factor,
            original_max_position: self.original_max_position,
            // RoPE struct does not carry Su per-dim factors; callers
            // who need them must construct the context directly (e.g.
            // via `RopeScalingContext::from(&RopeScaling)`).
            short_factor: None,
            long_factor: None,
        }
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

    /// Long-context-aware variant of [`RoPE::forward`].
    /// # Errors
    ///
    /// Returns `Err` if any candle operation fails.
    pub fn forward_with_scaling(
        &self,
        q: &Tensor,
        k: &Tensor,
        position: i64,
    ) -> Result<(Tensor, Tensor)> {
        let positions: Vec<i64> = (0..q.dim(1)? as i64).map(|i| position + i).collect();
        let q_out = apply_rope_with_scaling(q, &positions, self.theta, self.scaling_ctx())?;
        let k_out = apply_rope_with_scaling(k, &positions, self.theta, self.scaling_ctx())?;
        Ok((q_out, k_out))
    }

    /// `attn_factor` accessor — read by attention layers to apply
    /// YaRN's attention-temperature scaling.
    #[must_use]
    pub const fn attn_factor(&self) -> Option<f32> {
        self.attn_factor
    }

    /// `original_max_position_embeddings` accessor — sometimes useful
    /// for attention kernels that need to know the trained context
    /// length to pick the right temperature schedule.
    #[must_use]
    pub const fn original_max_position(&self) -> Option<usize> {
        self.original_max_position
    }
}

/// Bundle of scaling parameters extracted from `RopeScaling`.
///
/// Cheap to clone (only the optional `Vec<f32>` Su factors may allocate);
/// intended for `apply_rope_with_scaling` callers that don't have a `RoPE`
/// struct handy. Not `Copy` because `Vec<f32>` is not `Copy`.
#[derive(Clone, Debug)]
pub struct RopeScalingContext {
    pub rope_type: RopeType,
    pub scaling_factor: f32,
    pub attn_factor: Option<f32>,
    pub original_max_position: Option<usize>,
    /// Su RoPE per-dim factor for high-frequency dims.
    pub short_factor: Option<Vec<f32>>,
    /// Su RoPE per-dim factor for low-frequency dims.
    pub long_factor: Option<Vec<f32>>,
}

impl Default for RopeScalingContext {
    fn default() -> Self {
        Self {
            rope_type: RopeType::Default,
            scaling_factor: 1.0,
            attn_factor: None,
            original_max_position: None,
            short_factor: None,
            long_factor: None,
        }
    }
}

impl From<&RopeScaling> for RopeScalingContext {
    fn from(r: &RopeScaling) -> Self {
        Self {
            rope_type: r.rope_type.unwrap_or(RopeType::Default),
            scaling_factor: r.factor.unwrap_or(1.0),
            attn_factor: r.attn_factor,
            original_max_position: r.original_max_position_embeddings,
            short_factor: r.short_factor.clone(),
            long_factor: r.long_factor.clone(),
        }
    }
}

/// Run the operation (see signature for params and return type).
/// # Errors
///
/// Returns `Err` if the operation fails.
pub fn apply_rope(query: &Tensor, positions: &[i64], theta: f32) -> Result<Tensor> {
    let inv_freq = compute_inv_freq_default(query, theta);
    apply_rope_with_inv_freq(query, positions, &inv_freq)
}

/// Long-context-aware variant of [`apply_rope`].
///
/// Selects the inverse-frequency formula based on `scaling.rope_type`:
/// - `Default` → no scaling (same as [`apply_rope`]).
/// - `Linear` → position interpolation (`inv_freq / scaling_factor`).
/// - `Yarn` → NTK-aware theta adjustment (`theta' = theta * scale^(d/(d-2))`).
/// - `Dynamic` → NTK-aware theta recomputed for the current seq len.
/// - `Su` → per-dim factor arrays (`short_factor` / `long_factor`).
/// - `Other` → fall through to Default.
///
/// # Errors
///
/// Returns `Err` if the operation fails.
pub fn apply_rope_with_scaling(
    query: &Tensor,
    positions: &[i64],
    theta: f32,
    scaling: RopeScalingContext,
) -> Result<Tensor> {
    let inv_freq = match scaling.rope_type {
        RopeType::Default => compute_inv_freq_default(query, theta),
        RopeType::Linear => compute_inv_freq_linear(query, theta, scaling.scaling_factor),
        RopeType::Yarn => compute_inv_freq_yarn(query, theta, scaling.scaling_factor),
        RopeType::Dynamic => {
            let cur_seq_len = derive_seq_len(positions);
            compute_inv_freq_dynamic(
                query,
                theta,
                scaling.scaling_factor,
                scaling.original_max_position,
                cur_seq_len,
            )
        }
        RopeType::Su => compute_inv_freq_su(query, theta, &scaling),
        RopeType::Other => compute_inv_freq_default(query, theta),
    };
    apply_rope_with_inv_freq(query, positions, &inv_freq)
}

/// Default inverse-frequency table for a given `theta`.
///
/// `inv_freq[i] = theta ^ (-2i / d)` for `i in 0..head_dim/2`.
/// This is the standard RoPE formula unchanged from `apply_rope`.
fn compute_inv_freq_default(query: &Tensor, theta: f32) -> Vec<f32> {
    let (_batch, _seq_len, _num_heads, head_dim) = query.dims4().expect("dims4");
    compute_inv_freq_for_head_dim(head_dim, theta)
}

/// Linear-scaling inverse-frequency table.
///
/// Linear position interpolation: divide the position by `scaling_factor`
/// before multiplying by the frequency. Equivalently, divide the
/// frequency by `scaling_factor`.
fn compute_inv_freq_linear(query: &Tensor, theta: f32, scaling_factor: f32) -> Vec<f32> {
    let (_batch, _seq_len, _num_heads, head_dim) = query.dims4().expect("dims4");
    let inv_freq = compute_inv_freq_for_head_dim(head_dim, theta);
    if scaling_factor == 1.0 {
        return inv_freq;
    }
    inv_freq.into_iter().map(|f| f / scaling_factor).collect()
}

/// YaRN-style NTK-aware inverse-frequency table.
///
/// Adjusts `theta` by a global factor `scale^(d/(d-2))` so that:
/// - high-frequency dims (small `i`) keep their original wavelength,
/// - low-frequency dims (large `i`) compress to fit longer contexts.
///
/// This is the "global NTK" approximation of YaRN — see the [YaRN
/// paper](https://arxiv.org/abs/2309.00071) §3.3. The attention-scaling
/// half of YaRN (`attn_factor`) is **not** applied here; that lives in
/// the attention kernel and is exposed via `RoPE::attn_factor()`.
fn compute_inv_freq_yarn(query: &Tensor, theta: f32, scaling_factor: f32) -> Vec<f32> {
    let (_, _, _, head_dim) = query.dims4().expect("dims4");
    if scaling_factor == 1.0 {
        return compute_inv_freq_for_head_dim(head_dim, theta);
    }
    compute_inv_freq_yarn_impl(head_dim, theta, scaling_factor)
}

/// Core YaRN NTK formula. `scaling_factor` must be > 1.0 (caller's
/// responsibility to check). Used by both YaRN (with the config-declared
/// scale) and Dynamic NTK (with a recomputed scale per forward).
fn compute_inv_freq_yarn_impl(head_dim: usize, theta: f32, scaling_factor: f32) -> Vec<f32> {
    // NTK-by-parts / global NTK correction: theta' = theta * scale^(d/(d-2))
    // For head_dim = 64, d/(d-2) = 64/62 ≈ 1.032; for head_dim = 128,
    // d/(d-2) = 128/126 ≈ 1.016. Higher head_dim → gentler correction.
    let d = head_dim as f32;
    let exponent = d / (d - 2.0);
    let new_theta = theta * scaling_factor.powf(exponent);
    compute_inv_freq_for_head_dim(head_dim, new_theta)
}

/// HF Transformers / YaRN-style Dynamic NTK scaling.
///
/// At each forward, recompute the scaling factor based on the current
/// sequence length:
///
/// ```text
/// scale = max(1, factor × (cur / orig_max) - (factor - 1))
/// ```
///
/// If `cur_seq_len <= orig_max`, fall back to the default inv_freq table
/// (no scaling). Otherwise apply the YaRN NTK formula with the
/// dynamic `scale`.
///
/// This matches the implementation in HF Transformers
/// (`transformers.modeling_qwen2.RotaryEmbedding._dynamic_frequency_update`)
/// and the open-source Qwen2.5 / Qwen3 long-context reference impls.
fn compute_inv_freq_dynamic(
    query: &Tensor,
    theta: f32,
    scaling_factor: f32,
    orig_max: Option<usize>,
    cur_seq_len: usize,
) -> Vec<f32> {
    let (_, _, _, head_dim) = query.dims4().expect("dims4");
    let Some(orig_max) = orig_max else {
        // Without orig_max, Dynamic cannot decide. Default to Default.
        return compute_inv_freq_for_head_dim(head_dim, theta);
    };
    if cur_seq_len <= orig_max || scaling_factor == 1.0 {
        return compute_inv_freq_for_head_dim(head_dim, theta);
    }
    let factor = scaling_factor;
    let dynamic_scale = factor * (cur_seq_len as f32 / orig_max as f32) - (factor - 1.0);
    // dynamic_scale is guaranteed >= 1.0 by the cur > orig_max branch
    compute_inv_freq_yarn_impl(head_dim, theta, dynamic_scale)
}

/// Su RoPE: per-dimension scaling using `short_factor` (high-frequency
/// dims) and `long_factor` (low-frequency dims).
///
/// Per Su et al. 2024 ("RoPE in any precision"). The algorithm:
///
/// 1. Compute the default inv_freq: `inv_freq[i] = 1 / theta^(2i/d)`.
/// 2. Compute `boundary` = smallest `i` such that the *base* wavelength
///    `2π / inv_freq[i]` exceeds `original_max_position_embeddings`.
/// 3. For each `i < boundary`: `inv_freq[i] /= short_factor[i]` (default 1.0).
/// 4. For each `i >= boundary`: `inv_freq[i] /= long_factor[i]` (default 1.0).
///
/// Boundary calculation ignores the actual factors — it depends only on
/// the base inv_freq and `orig_max`, matching the HF reference impl.
///
/// Falls back to Default when `original_max_position` is None.
fn compute_inv_freq_su(query: &Tensor, theta: f32, scaling: &RopeScalingContext) -> Vec<f32> {
    let (_, _, _, head_dim) = query.dims4().expect("dims4");
    let half_dim = head_dim / 2;

    let Some(orig_max) = scaling.original_max_position else {
        return compute_inv_freq_default(query, theta);
    };

    let base_inv_freq = compute_inv_freq_for_head_dim(head_dim, theta);
    let wavelength_at = |i: usize| 2.0 * std::f32::consts::PI / base_inv_freq[i];
    let boundary = (0..half_dim)
        .find(|&i| wavelength_at(i) > orig_max as f32)
        .unwrap_or(half_dim);

    let default_factor = 1.0_f32;
    (0..half_dim)
        .map(|i| {
            let factor = if i < boundary {
                scaling
                    .short_factor
                    .as_ref()
                    .and_then(|v| v.get(i).copied())
                    .unwrap_or(default_factor)
            } else {
                scaling
                    .long_factor
                    .as_ref()
                    .and_then(|v| v.get(i).copied())
                    .unwrap_or(default_factor)
            };
            base_inv_freq[i] / factor
        })
        .collect()
}

/// Derive the current sequence length from a positions slice.
///
/// Used by scaling algorithms that need to know how long the current
/// forward pass is (Dynamic NTK, Su RoPE).
///
/// Assumes positions are typically contiguous from 0; for non-contiguous
/// positions, returns `max(positions) + 1` (overestimate, but conservative
/// for both Dynamic and Su — they scale up at long contexts).
#[must_use]
pub(super) fn derive_seq_len(positions: &[i64]) -> usize {
    positions
        .iter()
        .copied()
        .max()
        .map_or(0, |m| (m + 1) as usize)
}

fn compute_inv_freq_for_head_dim(head_dim: usize, theta: f32) -> Vec<f32> {
    let half_dim = head_dim / 2;
    (0..half_dim)
        .map(|i| theta.powf(-2.0 * (i as f32) / (head_dim as f32)))
        .collect()
}

/// Inner implementation: rotate `query` by the given precomputed
/// `inv_freq` table. `query` is `[B, S, H, D]`, `inv_freq` is `D/2`.
fn apply_rope_with_inv_freq(query: &Tensor, positions: &[i64], inv_freq: &[f32]) -> Result<Tensor> {
    let (batch, seq_len, num_heads, head_dim) = query.dims4()?;

    let query = query.transpose(1, 2)?;

    let half_dim = head_dim / 2;
    debug_assert_eq!(inv_freq.len(), half_dim);

    let mut cos_matrix = Vec::with_capacity(seq_len * half_dim);
    let mut sin_matrix = Vec::with_capacity(seq_len * half_dim);

    for &pos in positions {
        let pos_f = pos as f32;
        for &freq in inv_freq {
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
// at large positions. The `apply_with_scaling` tests cover
// Default-vs-Linear-vs-Yarn-vs-Dynamic-vs-Su behaviour.
#[cfg(test)]
mod tests;
