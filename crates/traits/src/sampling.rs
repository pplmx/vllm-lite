//! Shared sampling primitives.
//!
//! [`argmax_logits`] is the single source of truth for greedy token
//! selection — both `vllm-core::sampling::greedy_sample` (flat `&[f32]`
//! caller) and `vllm-model::causal_lm::greedy_sample_token` (candle
//! `Tensor` caller) delegate here. Adding a new caller-side decoder
//! should use this directly rather than re-implementing the fold.
//!
//! Keeping the helper in `vllm-traits` (no upstream dependency on either
//! `core` or `model`) avoids both an upward `model → core` dependency
//! and code duplication.
//!
//! [`SamplingParams`] lives here too so that [`crate::types::Batch`]
//! can carry a per-sequence `Vec<SamplingParams>` without a cyclic
//! dependency on `vllm-core`. ARCH-02 (technical due diligence):
//! the HTTP layer accepted these params but the model layer always
//! sampled greedily; carrying them on `Batch` closes the seam.
//!
//! `clippy::module_name_repetitions` is allowed at the type level:
//! `vllm_traits::sampling::SamplingParams` reads naturally and
//! re-exporting through `vllm_core::types::SamplingParams` preserves
//! the existing public API path.
#![allow(clippy::module_name_repetitions)]

use serde::{Deserialize, Serialize};

use crate::types::TokenId;

/// Per-request sampling configuration.
///
/// Defaults are tuned for deterministic greedy decoding
/// (`temperature = 0`, `top_p = 1`, `repeat_penalty = 1`, `beam_width = 1`);
/// raise `temperature`, lower `top_p`, etc. for sampling.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SamplingParams {
    /// Sampling temperature. `0.0` selects greedy argmax; `1.0` is the
    /// un-scaled softmax; values `<1` sharpen, `>1` flatten.
    pub temperature: f32,
    /// Top-K truncation. `0` disables; otherwise keeps the K highest logits.
    pub top_k: usize,
    /// Nucleus sampling cutoff. `1.0` disables; otherwise keeps the smallest
    /// set of tokens whose cumulative probability ≥ `top_p`.
    pub top_p: f32,
    /// Repeat penalty applied to logits at positions already seen in this
    /// sequence. `1.0` disables.
    pub repeat_penalty: f32,
    /// Beam width. `1` ⇒ greedy; `>1` enables beam search.
    pub beam_width: usize,
    /// Length penalty applied during beam search ranking.
    pub length_penalty: f32,
    /// Reserved for the speculative-decoding fallback path. Currently unused
    /// by the default sampler.
    pub max_retries: u32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repeat_penalty: 1.0,
            beam_width: 1,
            length_penalty: 0.6,
            max_retries: 0,
        }
    }
}

impl SamplingParams {
    /// Returns a builder for configuring this type with the documented field defaults.
    /// Use `with_*(...)` to override individual fields, then `build()` to produce the type.
    #[must_use]
    pub fn builder() -> SamplingParamsBuilder {
        SamplingParamsBuilder::default()
    }
}

/// Builder for [`SamplingParams`].
#[derive(Debug, Clone, Default)]
pub struct SamplingParamsBuilder {
    inner: SamplingParams,
}

impl SamplingParamsBuilder {
    /// Set [`SamplingParams::temperature`].
    #[must_use]
    pub const fn with_temperature(mut self, v: f32) -> Self {
        self.inner.temperature = v;
        self
    }
    /// Set [`SamplingParams::top_k`].
    #[must_use]
    pub const fn with_top_k(mut self, v: usize) -> Self {
        self.inner.top_k = v;
        self
    }
    /// Set [`SamplingParams::top_p`].
    #[must_use]
    pub const fn with_top_p(mut self, v: f32) -> Self {
        self.inner.top_p = v;
        self
    }
    /// Set [`SamplingParams::repeat_penalty`].
    #[must_use]
    pub const fn with_repeat_penalty(mut self, v: f32) -> Self {
        self.inner.repeat_penalty = v;
        self
    }
    /// Set [`SamplingParams::beam_width`].
    #[must_use]
    pub const fn with_beam_width(mut self, v: usize) -> Self {
        self.inner.beam_width = v;
        self
    }
    /// Set [`SamplingParams::length_penalty`].
    #[must_use]
    pub const fn with_length_penalty(mut self, v: f32) -> Self {
        self.inner.length_penalty = v;
        self
    }
    /// Set [`SamplingParams::max_retries`].
    #[must_use]
    pub const fn with_max_retries(mut self, v: u32) -> Self {
        self.inner.max_retries = v;
        self
    }
    /// Finalize the builder into a [`SamplingParams`].
    #[must_use]
    pub const fn build(self) -> SamplingParams {
        self.inner
    }
}

/// Find the index of the largest element in `logits`, returned as a
/// `TokenId`.
///
/// This is the canonical greedy-sampling operation. Ties resolve to the
/// lowest index (the fold initialises `(0, NEG_INFINITY)` and only
/// updates on strict `>`, so the first occurrence of a value wins).
///
/// `TokenId` is `u32`, so any index beyond `u32::MAX` would overflow;
/// realistic vocabularies are well under that bound.
///
/// # Panics
///
/// This function never panics on a non-empty `logits` slice. An empty
/// slice returns `TokenId(0)` (the fold initial value).
#[must_use]
pub fn argmax_logits(logits: &[f32]) -> TokenId {
    let idx = logits
        .iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |(max_idx, max_val), (i, &val)| {
            if val > max_val {
                (i, val)
            } else {
                (max_idx, max_val)
            }
        })
        .0;
    // invariant: vocab indices are bounded by the model vocabulary size, well within u32 range.
    TokenId::try_from(idx).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_logits_returns_index_of_max() {
        assert_eq!(argmax_logits(&[1.0, 2.0, 3.0, 2.0, 1.0]), 2);
    }

    #[test]
    fn argmax_logits_handles_negative() {
        assert_eq!(argmax_logits(&[-3.0, -1.0, -2.0]), 1);
    }

    #[test]
    fn argmax_logits_breaks_ties_to_first() {
        assert_eq!(argmax_logits(&[1.0, 1.0, 1.0, 1.0]), 0);
    }

    #[test]
    fn argmax_logits_empty_returns_zero() {
        assert_eq!(argmax_logits(&[]), 0);
    }

    #[test]
    fn argmax_logits_single_element() {
        assert_eq!(argmax_logits(&[42.0]), 0);
    }
}
