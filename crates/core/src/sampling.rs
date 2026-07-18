//! Token sampling: temperature scaling, top-k, top-p (nucleus), and the combined `sample` entry point.
//!
//! Designed to be called per sequence per decode step. Beam-search
//! state types live in [`crate::beam`]; no orchestration methods are
//! currently shipped.
//!
//! Greedy decoding delegates to [`vllm_traits::argmax_logits`] — the
//! local wrapper only adds the `tracing` instrumentation.
#![allow(unused_variables)]

use crate::types::TokenId;
use tracing::trace;
use vllm_traits::{SamplingParams, argmax_logits};

fn random_f32() -> f32 {
    rand::random::<f32>()
}

pub(crate) fn greedy_sample(logits: &[f32]) -> TokenId {
    trace!(vocab_size = logits.len(), "Greedy sampling");
    argmax_logits(logits)
}

pub(crate) fn temperature_sample(logits: &[f32], temperature: f32) -> TokenId {
    trace!(
        vocab_size = logits.len(),
        temperature = temperature,
        "Temperature sampling"
    );
    if temperature <= 0.0 || logits.is_empty() {
        return greedy_sample(logits);
    }

    let scaled: Vec<f32> = logits.iter().map(|x| x / temperature).collect();
    let max_val = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = scaled.iter().map(|x| (x - max_val).exp()).collect();
    let sum: f32 = exp.iter().sum();
    let probs: Vec<f32> = exp.iter().map(|x| x / sum).collect();

    let random_threshold = random_f32();
    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if random_threshold <= cumsum {
            return TokenId::try_from(i).unwrap_or(0);
        }
    }
    TokenId::try_from(probs.len() - 1).unwrap_or(0)
}

pub(crate) fn top_p_sample(logits: &[f32], top_p: f32) -> TokenId {
    trace!(vocab_size = logits.len(), top_p = top_p, "Top-p sampling");
    if top_p >= 1.0 || logits.is_empty() {
        return greedy_sample(logits);
    }

    let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or_else(|| a.1.is_nan().cmp(&b.1.is_nan()))
    });

    let max_val = indexed[0].1;
    let exp: Vec<f32> = indexed.iter().map(|(_, v)| (v - max_val).exp()).collect();
    let sum: f32 = exp.iter().sum();
    let mut probs: Vec<f32> = exp.iter().map(|x| x / sum).collect();

    let mut cumsum = 0.0;
    let mut cutoff = probs.len();
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if cumsum > top_p {
            cutoff = i + 1;
            break;
        }
    }

    probs.truncate(cutoff);
    let total: f32 = probs.iter().sum();
    for p in &mut probs {
        *p /= total;
    }

    let random_threshold = random_f32();
    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if random_threshold <= cumsum {
            return TokenId::try_from(indexed[i].0).unwrap_or(0);
        }
    }
    TokenId::try_from(indexed[probs.len() - 1].0).unwrap_or(0)
}

#[must_use]
pub fn top_k_sample(logits: &[f32], k: usize) -> TokenId {
    if k == 0 || logits.is_empty() {
        return greedy_sample(logits);
    }

    let top_k_limit = k.min(logits.len());

    let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();

    indexed.select_nth_unstable_by(top_k_limit - 1, |a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or_else(|| a.1.is_nan().cmp(&b.1.is_nan()))
    });
    let threshold = indexed[top_k_limit - 1].1;

    let masked: Vec<f32> = logits
        .iter()
        .map(|&v| if v >= threshold { v } else { f32::NEG_INFINITY })
        .collect();

    temperature_sample(&masked, 1.0)
}

/// Sample one token per row from a batch of per-sequence logits.
///
/// For each row the following pipeline is applied (short-circuiting where a
/// parameter makes later steps a no-op):
/// 1. **Repeat penalty** — divide logits at positions present in `seen_tokens`
///    by `repeat_penalty` (skipped when `repeat_penalty == 1.0` or `seen` is
///    empty).
/// 2. **Temperature** — divide logits by `temperature` when it differs from 1.0.
/// 3. **Top-K truncation** — zero out (set to `-inf`) all logits below the
///    `k`-th largest, when `top_k > 0`.
/// 4. **Top-P / temperature / greedy** — choose the final sampler based on the
///    remaining parameters: `top_p < 1.0` ⇒ nucleus, `temperature > 0.0`
///    ⇒ temperature sampling, otherwise greedy argmax.
///
/// `logits_list` and `seen_tokens` must have the same length. Returned vector
/// has length `logits_list.len()`.
#[must_use]
pub fn sample_batch(
    logits_list: &[Vec<f32>],
    temperature: f32,
    top_p: f32,
    top_k: usize,
    repeat_penalty: f32,
    seen_tokens: &[Vec<TokenId>],
) -> Vec<TokenId> {
    logits_list
        .iter()
        .zip(seen_tokens.iter())
        .map(|(logits, seen)| {
            let mut logits = logits.clone();

            if (repeat_penalty - 1.0).abs() > f32::EPSILON && !seen.is_empty() {
                apply_repeat_penalty(&mut logits, seen, repeat_penalty);
            }

            if temperature > 0.0 && (temperature - 1.0).abs() > f32::EPSILON {
                for l in &mut logits {
                    *l /= temperature;
                }
            }

            if top_k > 0 {
                let top_k_limit = top_k.min(logits.len());
                let mut indexed: Vec<(usize, f32)> =
                    logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
                indexed.select_nth_unstable_by(top_k_limit - 1, |a, b| {
                    b.1.partial_cmp(&a.1)
                        .unwrap_or_else(|| a.1.is_nan().cmp(&b.1.is_nan()))
                });
                let threshold = indexed[top_k_limit - 1].1;
                for l in &mut logits {
                    if *l < threshold {
                        *l = f32::NEG_INFINITY;
                    }
                }
            }

            if top_p < 1.0 {
                top_p_sample(&logits, top_p)
            } else if temperature > 0.0 {
                temperature_sample(&logits, temperature)
            } else {
                greedy_sample(&logits)
            }
        })
        .collect()
}

/// Sample one token per row from a batch of per-sequence logits, using a
/// per-sequence [`SamplingParams`] for the decision.
///
/// This is the engine-side entry point used after [`crate::scheduler`]
/// builds a batch: the model returns raw logits via
/// [`vllm_traits::ModelBackend::forward_logits`], and the engine hands
/// them back here together with the per-sequence sampling parameters
/// carried on the [`vllm_traits::Batch`]. Closing this loop is the
/// fix for ARCH-02 (technical due diligence) — previously the HTTP
/// layer accepted `temperature` / `top_p` / `top_k` and the model
/// layer always chose argmax, so the params silently had no effect.
///
/// `params_list`, `seen_tokens`, and `logits_list` must have the same
/// length. The returned `Vec<TokenId>` has length `logits_list.len()`.
///
/// Beam search (`beam_width > 1`) is not implemented here — callers
/// must intercept those requests before they reach this function.
#[must_use]
pub fn sample_batch_with_params(
    logits_list: &[Vec<f32>],
    params_list: &[SamplingParams],
    seen_tokens: &[Vec<TokenId>],
) -> Vec<TokenId> {
    logits_list
        .iter()
        .zip(params_list.iter())
        .zip(seen_tokens.iter())
        .map(|((logits, params), seen)| sample_one_with_params(logits, params, seen))
        .collect()
}

/// Apply the full sampling pipeline (repeat penalty → temperature →
/// top-k → top-p / temperature / greedy) using a single
/// [`SamplingParams`] for one sequence.
///
/// Public for tests; production callers should use
/// [`sample_batch_with_params`].
#[must_use]
pub fn sample_one_with_params(
    logits: &[f32],
    params: &SamplingParams,
    seen: &[TokenId],
) -> TokenId {
    let mut logits = logits.to_vec();

    if (params.repeat_penalty - 1.0).abs() > f32::EPSILON && !seen.is_empty() {
        apply_repeat_penalty(&mut logits, seen, params.repeat_penalty);
    }

    if params.presence_penalty.abs() > f32::EPSILON && !seen.is_empty() {
        apply_presence_penalty(&mut logits, seen, params.presence_penalty);
    }

    if params.temperature > 0.0 && (params.temperature - 1.0).abs() > f32::EPSILON {
        for l in &mut logits {
            *l /= params.temperature;
        }
    }

    if params.top_k > 0 {
        let top_k_limit = params.top_k.min(logits.len());
        let mut indexed: Vec<(usize, f32)> =
            logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed.select_nth_unstable_by(top_k_limit - 1, |a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or_else(|| a.1.is_nan().cmp(&b.1.is_nan()))
        });
        let threshold = indexed[top_k_limit - 1].1;
        for l in &mut logits {
            if *l < threshold {
                *l = f32::NEG_INFINITY;
            }
        }
    }

    if params.top_p < 1.0 {
        top_p_sample(&logits, params.top_p)
    } else if params.temperature > 0.0 {
        temperature_sample(&logits, params.temperature)
    } else {
        greedy_sample(&logits)
    }
}

/// Adjust the logit at each id present in `seen_tokens` by the inverse of
/// `penalty`, sign-aware so both positive and negative logits move in the
/// correct direction.
///
/// This is the standard "repeat penalty" trick used by llama.cpp, `HuggingFace`
/// `transformers`, and most open-source inference engines: tokens that have
/// already appeared are made less likely on subsequent steps (penalty > 1)
/// or *more* likely (penalty < 1, "boost" semantic). No-op when
/// `penalty == 1.0`, `seen_tokens` is empty, or `logits` is empty.
///
/// **Sign-aware implementation (P29 v0.3 wire-type follow-up):** the
/// helper handles positive and negative logits symmetrically so the
/// penalty direction (penalize vs boost) is consistent regardless of
/// the logit's sign:
///
/// - For `logit >= 0`: divide by `penalty`. Penalty > 1 reduces the
///   logit (penalize); penalty < 1 increases the logit (boost).
/// - For `logit < 0`: multiply by `penalty`. Penalty > 1 makes the
///   logit more negative (penalize); penalty < 1 makes the logit
///   less negative (boost).
///
/// The pre-P29 implementation used simple division for all logits,
/// which has a sign-flip bug for negative logits and `penalty < 1`
/// (dividing a negative logit by a value < 1.0 makes it *more*
/// negative — the opposite of the desired boost direction). The
/// chat handler's `max(1.0, 1.0 + frequency_penalty)` clamp existed
/// to work around this bug for negative `frequency_penalty` values;
/// the sign-aware refactor lets the handler forward the value
/// verbatim.
///
/// Out-of-range token ids are silently ignored. Duplicate entries in
/// `seen_tokens` are deduped via an internal `HashSet`.
pub fn apply_repeat_penalty(logits: &mut [f32], seen_tokens: &[TokenId], penalty: f32) {
    if (penalty - 1.0).abs() < f32::EPSILON || seen_tokens.is_empty() || logits.is_empty() {
        return;
    }

    let mut seen = std::collections::HashSet::new();
    for &token in seen_tokens {
        if let Ok(idx) = usize::try_from(token)
            && idx < logits.len()
            && seen.insert(token)
        {
            if logits[idx] >= 0.0 {
                logits[idx] /= penalty;
            } else {
                logits[idx] *= penalty;
            }
        }
    }
}

/// Subtract `penalty` from the logit of each *distinct* id present in
/// `seen_tokens`, regardless of how many times each id appeared
/// (OpenAI `presence_penalty` semantic).
///
/// **Difference from [`apply_repeat_penalty`]:** `apply_repeat_penalty`
/// is *frequency-style* — divides the logit by `penalty` once per
/// occurrence. `apply_presence_penalty` is *presence-style* —
/// subtracts `penalty` from the logit of each distinct seen token
/// exactly once, regardless of count. Per OpenAI's spec for
/// `presence_penalty`: "Positive values penalize new tokens based on
/// whether they appear in the prompt so far, increasing the model's
/// likelihood to talk about new topics."
///
/// Negative values *encourage* repetition by raising the logits of
/// seen tokens (because subtracting a negative is the same as
/// adding). No-op when `penalty == 0.0`, `seen_tokens` is empty, or
/// `logits` is empty.
///
/// Out-of-range token ids are silently ignored. Duplicate entries
/// in `seen_tokens` are deduped via an internal `HashSet` so the
/// penalty is applied exactly once per distinct id.
pub fn apply_presence_penalty(logits: &mut [f32], seen_tokens: &[TokenId], penalty: f32) {
    if penalty.abs() < f32::EPSILON || seen_tokens.is_empty() || logits.is_empty() {
        return;
    }

    let mut seen = std::collections::HashSet::new();
    for &token in seen_tokens {
        if let Ok(idx) = usize::try_from(token)
            && idx < logits.len()
            && seen.insert(token)
        {
            logits[idx] -= penalty;
        }
    }
}

// Unit tests are extracted to `tests.rs` and `prop_tests.rs` to keep
// this file under the 800-line soft cap. See those siblings for the
// test surface (greedy / temperature / top-k / top-p / sample_batch /
// repeat-penalty; plus proptest invariants for batch length
// preservation, greedy index bounds, batched-vs-per-row greedy, and
// repeat-penalty no-op at penalty=1.0).
#[cfg(test)]
mod prop_tests;
#[cfg(test)]
mod tests;
