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
//!
//! [`SampledToken`] (P36 v0.3 wire-type follow-up engine wire-through)
//! carries the sampled `token` alongside its `logprob` under the
//! post-temperature / post-top-k / post-top-p distribution, plus the
//! top-K tokens (id, logprob) when `SamplingParams::top_logprobs.is_some()`.
//! The sampler pair
//! ([`crate::sampling::sample_one_with_params`],
//! [`crate::sampling::sample_batch_with_params`]) returns
//! `SampledToken` so the HTTP layer can render OpenAI's
//! `choices[].logprobs` shape without re-running the softmax.
#![allow(clippy::module_name_repetitions)]

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::types::TokenId;

/// The result of one sample step (P36 v0.3 wire-type follow-up engine
/// wire-through).
///
/// `token` is the sampled `TokenId`. `logprob` is `ln(P(token))` under
/// the **post-filter** distribution (after `repeat_penalty` +
/// `presence_penalty` + `logit_bias` + temperature scaling + `top_k`
/// truncation + `top_p` nucleus cutoff). `logprob` is always populated
/// (OpenAI's contract: the response carries the logprob of the sampled
/// token even when `top_logprobs = 0`).
///
/// `top_logprobs` is empty when `SamplingParams::top_logprobs.is_none()`
/// or `Some(0)`. Otherwise it contains the top-N tokens by
/// post-filter probability, each alongside its `logprob`, sorted by
/// `logprob` descending. N is bounded by the request's
/// `top_logprobs` value (the sampled token appears in this slice iff
/// it is among the top-N — OpenAI does not insert the sampled token
/// at the head when it falls outside the top-N; instead it surfaces
/// only via the parent `token` / `logprob` fields).
///
/// `logprob` is `-∞` when the sampled token's post-filter logit is
/// `-∞` (e.g. masked by `top_k` truncation or driven to `-∞` by an
/// extreme negative `presence_penalty` — the sampler only ever picks
/// a token with finite logit in practice, but the contract guarantees
/// finiteness through the JSON serializer).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SampledToken {
    /// The sampled token ID.
    pub token: TokenId,
    /// `ln(P(token))` under the post-filter distribution. Always
    /// populated; `-∞` when the post-filter logit for `token` is `-∞`.
    pub logprob: f32,
    /// Top-N `(token, logprob)` pairs sorted by `logprob` descending.
    /// Empty when `SamplingParams::top_logprobs.is_none()` or
    /// `Some(0)`. Length ≤ `SamplingParams::top_logprobs.unwrap_or(0)`.
    pub top_logprobs: Vec<(TokenId, f32)>,
}

/// Per-request sampling configuration.
///
/// Defaults are tuned for deterministic greedy decoding
/// (`temperature = 0`, `top_p = 1`, `repeat_penalty = 1`, `presence_penalty = 0`,
/// `beam_width = 1`); raise `temperature`, lower `top_p`, etc. for sampling.
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
    /// Presence penalty (OpenAI `presence_penalty` semantic): a single
    /// additive bias subtracted from the logit of every *distinct* token
    /// already seen in this sequence, regardless of how many times it
    /// appeared. Positive values discourage repetition (encourage new
    /// topics); negative values *encourage* repetition. `0.0` disables.
    ///
    /// **Difference from `repeat_penalty`:** `repeat_penalty` is
    /// *frequency-style* — it divides the logit by `repeat_penalty`
    /// *once per occurrence*, so a token seen 3 times gets divided 3
    /// times. `presence_penalty` is *presence-style* — it subtracts
    /// `presence_penalty` from the logit once per *distinct* token, so
    /// a token seen 3 times still only gets the penalty subtracted once.
    /// This matches OpenAI's spec for `presence_penalty`: "Positive
    /// values penalize new tokens based on whether they appear in the
    /// prompt so far, increasing the model's likelihood to talk about
    /// new topics." See [`crate::sampling::apply_presence_penalty`]
    /// for the implementation.
    pub presence_penalty: f32,
    /// Per-token logit bias (OpenAI `logit_bias` semantic): an additive
    /// bias added to the logit of specific token IDs keyed by their ID.
    /// Positive values *increase* the probability of the biased tokens;
    /// negative values *decrease* it. `None` disables.
    ///
    /// Per OpenAI spec the bias values are constrained to the
    /// `[-100, 100]` range; the validator on the HTTP layer rejects
    /// out-of-range and non-finite values with `400`. Out-of-vocab
    /// token IDs (any ID `>= logits.len()`) are silently ignored
    /// at sampling time (matches OpenAI's server behaviour). The
    /// iteration order of the `HashMap` is non-deterministic, but
    /// because the bias is additive and independent per token, the
    /// *final logits* are deterministic regardless of iteration
    /// order — so determinism is preserved.
    ///
    /// **Difference from `presence_penalty`:** `presence_penalty` is
    /// *automatic* — every distinct seen token gets the same bias.
    /// `logit_bias` is *explicit* — the caller specifies exactly which
    /// token IDs to bias and by how much. See
    /// [`vllm_core::sampling::apply_logit_bias`] for the
    /// implementation.
    ///
    /// [`vllm_core::sampling::apply_logit_bias`]:
    ///     https://docs.rs/vllm-core/latest/vllm_core/sampling/fn.apply_logit_bias.html
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<HashMap<TokenId, f32>>,
    /// Beam width. `1` ⇒ greedy; `>1` enables beam search.
    pub beam_width: usize,
    /// Length penalty applied during beam search ranking.
    pub length_penalty: f32,
    /// Reserved for the speculative-decoding fallback path. Currently unused
    /// by the default sampler.
    pub max_retries: u32,
    /// Random seed for the sampling RNG (OpenAI `seed` semantic,
    /// P34 v0.2 wire-type follow-up engine wire-through). When
    /// `Some(seed)`, the sampler constructs a `StdRng::seed_from_u64`
    /// for each sampling step so the same seed + same model + same
    /// prompt produces the same output. When `None`, the sampler
    /// reads from the thread-local default RNG (the pre-P34
    /// behaviour).
    ///
    /// **Honouring is greedy-agnostic:** `temperature = 0` and
    /// `top_p = 1.0` paths bypass the RNG entirely (deterministic
    /// argmax) so the seed has no observable effect in those modes
    /// — same argmax regardless of seed. `seed = Some(0)` is a valid
    /// seed (NOT conflated with `None`).
    ///
    /// **Per-sequence independence:** `sample_batch_with_params`
    /// builds a fresh `StdRng` per call to `sample_one_with_params`,
    /// so each sequence's RNG state is independent even when they
    /// share the same `seed` (they each re-seed from the same u64,
    /// producing the same draws for the same logits — this is the
    /// correct behaviour for OpenAI's per-request determinism
    /// contract).
    ///
    /// OpenAI's `seed` field is `i64`; the HTTP layer does an
    /// `as u64` cast (wrapping negatives) so any i64 is accepted
    /// per spec.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// OpenAI `top_logprobs` count (P36 v0.3 wire-type follow-up
    /// engine wire-through): when `Some(n)`, the sampler computes the
    /// top-`n` most-likely tokens at each sampling step and attaches
    /// them to [`SampledToken::top_logprobs`] (sorted by logprob
    /// descending, length ≤ `n`). When `None` or `Some(0)`, no top-K
    /// computation runs — `SampledToken::top_logprobs` is empty and
    /// only the sampled token's logprob is populated.
    ///
    /// Per OpenAI spec the chat endpoint's valid range is `0..=20`
    /// and the legacy `/v1/completions` endpoint's range is `0..=5`;
    /// validation happens on the HTTP layer (see
    /// [`crate::sampling::validate_chat_logprobs`] /
    /// [`crate::sampling::validate_completion_logprobs`] siblings in
    /// `vllm_server::openai::sampling_validation`). The engine
    /// itself accepts any `u32` — out-of-range values are the HTTP
    /// layer's contract, not the sampler's.
    ///
    /// **Honoring is end-to-end when `Some(n)`:** the sampler runs a
    /// partial top-K selection on the post-filter logits and emits
    /// the `(token, logprob)` pairs alongside the sampled token.
    /// When `None`, the sampler skips the top-K selection entirely
    /// (no extra allocations, no second sort). This matches the
    /// legacy behaviour and keeps the default-path overhead at zero.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u32>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repeat_penalty: 1.0,
            presence_penalty: 0.0,
            logit_bias: None,
            beam_width: 1,
            length_penalty: 0.6,
            max_retries: 0,
            seed: None,
            top_logprobs: None,
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
    /// Set [`SamplingParams::presence_penalty`].
    ///
    /// Positive values discourage repetition; negative values encourage
    /// repetition. `0.0` disables. See the field doc-comment on
    /// [`SamplingParams::presence_penalty`] for the difference from
    /// `repeat_penalty`.
    #[must_use]
    pub const fn with_presence_penalty(mut self, v: f32) -> Self {
        self.inner.presence_penalty = v;
        self
    }
    /// Set [`SamplingParams::logit_bias`].
    ///
    /// Per-token additive bias map. Positive values *increase* the
    /// probability of the biased tokens; negative values *decrease*
    /// it. `None` disables. See the field doc-comment on
    /// [`SamplingParams::logit_bias`] for the difference from
    /// `presence_penalty` and the determinism guarantee.
    #[must_use]
    pub fn with_logit_bias(mut self, bias: Option<HashMap<TokenId, f32>>) -> Self {
        self.inner.logit_bias = bias;
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
    /// Set [`SamplingParams::seed`] to `Some(seed)`.
    ///
    /// When set, the sampler builds a `StdRng::seed_from_u64` for
    /// each sampling step so the same seed + same model + same
    /// prompt produces the same output (OpenAI `seed` semantic —
    /// P34 v0.2 wire-type follow-up engine wire-through). See the
    /// field doc-comment on [`SamplingParams::seed`] for the
    /// greedy-bypass + per-sequence independence guarantees.
    #[must_use]
    pub const fn with_seed(mut self, seed: u64) -> Self {
        self.inner.seed = Some(seed);
        self
    }
    /// Explicitly clear [`SamplingParams::seed`] (set to `None`).
    ///
    /// Useful in tests where the builder default is `None` but a
    /// downstream caller may have set a seed via a different code
    /// path. With the default builder default of `None`, this is
    /// rarely needed in production code.
    #[must_use]
    pub const fn with_seed_none(mut self) -> Self {
        self.inner.seed = None;
        self
    }
    /// Set [`SamplingParams::top_logprobs`] to `Some(n)`.
    ///
    /// When `n > 0`, the sampler computes the top-`n` most-likely
    /// tokens at each step and attaches them to
    /// [`SampledToken::top_logprobs`]. When `n == 0`, this is
    /// equivalent to `with_top_logprobs_none` — no top-K computation
    /// runs. See the field-level doc-comment for the contract.
    ///
    /// (P36 v0.3 wire-type follow-up engine wire-through.)
    #[must_use]
    pub const fn with_top_logprobs(mut self, n: u32) -> Self {
        self.inner.top_logprobs = Some(n);
        self
    }
    /// Explicitly clear [`SamplingParams::top_logprobs`] (set to `None`).
    ///
    /// Useful in tests where the builder default is `None` but a
    /// downstream caller may have set a value via a different code
    /// path. Mirrors [`SamplingParamsBuilder::with_seed_none`].
    #[must_use]
    pub const fn with_top_logprobs_none(mut self) -> Self {
        self.inner.top_logprobs = None;
        self
    }
    /// Finalize the builder into a [`SamplingParams`].
    #[must_use]
    pub fn build(self) -> SamplingParams {
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
