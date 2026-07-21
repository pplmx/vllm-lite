//! Token sampling: temperature scaling, top-k, top-p (nucleus), and the combined `sample` entry point.
//!
//! Designed to be called per sequence per decode step. Beam-search
//! state types live in [`crate::beam`]; no orchestration methods are
//! currently shipped.
//!
//! Greedy decoding delegates to [`vllm_traits::argmax_logits`] — the
//! local wrapper only adds the `tracing` instrumentation.
//!
//! **RNG seeding (P34 v0.2 wire-type follow-up engine wire-through):**
//! the temperature / top-p samplers accept a precomputed
//! `random_threshold: f32` parameter so the caller (today:
//! [`sample_one_with_params`], tomorrow: any per-step orchestration
//! that wants per-sequence RNG independence) controls RNG selection.
//! `sample_one_with_params` reads `params.seed`: `Some(s)` builds a
//! fresh `StdRng::seed_from_u64(s)` and draws one `f32`; `None` reads
//! from the thread-local default RNG. Greedy paths bypass the RNG
//! entirely so `seed` has no observable effect when `temperature = 0`
//! or `top_p = 1.0`.
#![allow(unused_variables)]

use crate::types::TokenId;
use tracing::trace;
use vllm_traits::{SampledToken, SamplingParams, argmax_logits};

/// Read one `f32` from the thread-local default RNG. Used when
/// [`SamplingParams::seed`] is `None` (the pre-P34 default).
fn random_f32() -> f32 {
    rand::random::<f32>()
}

/// Read one `f32` from a freshly-seeded `StdRng` (OpenAI `seed`
/// semantic — P34 v0.2 wire-type follow-up engine wire-through).
///
/// `StdRng` is the same CSPRNG-quality generator `rand::random()`
/// uses internally under the default `rand` 0.10 feature set, so
/// the seeded and unseeded paths produce statistically-equivalent
/// random sequences — only the seed source differs.
fn random_f32_seeded(seed: u64) -> f32 {
    use rand::RngExt;
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    rng.random::<f32>()
}

/// Decide which RNG to use based on `params.seed` and return one
/// `f32` random threshold. This is the single point at which the
/// sampler touches an RNG, so adding a new RNG source in the future
/// only requires changing this helper.
fn sample_random_threshold(seed: Option<u64>) -> f32 {
    match seed {
        Some(s) => random_f32_seeded(s),
        None => random_f32(),
    }
}

/// Compute `log(softmax(logits)[token])` for a single token under
/// the post-filter distribution.
///
/// Numerically stable: subtracts `max(logits)` before `exp` and
/// `ln` so the intermediate `sum_exp` is bounded by `[0, vocab_size]`
/// even for very-negative or very-positive logits. Returns `-∞` when
/// the token's logit is `-∞` (e.g. masked by `top_k` truncation or
/// driven to `-∞` by an extreme negative `presence_penalty`) — the
/// JSON serializer will surface this as `null`, which OpenAI clients
/// interpret as "this token was excluded from the sampling distribution".
///
/// `logits` is the FINAL post-filter logits (the same distribution
/// the sampler draws from). Caller is responsible for applying
/// temperature scaling, `top_k` masking, `top_p` nucleus cutoff, and
/// the bias/penalty steps BEFORE this helper so the returned
/// probability reflects the actual sampling distribution.
#[allow(dead_code)] // exposed as part of the sampler API surface; currently called only by sample_one_with_params
fn logprob_of_token(logits: &[f32], token: TokenId) -> f32 {
    if logits.is_empty() {
        return 0.0;
    }
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max_val.is_finite() {
        // All logits are -inf (top_k zeroed everything). Return -inf
        // for any token; the sampler would only reach this in a
        // pathological edge case (e.g. all top-k values were NaN).
        return f32::NEG_INFINITY;
    }
    let sum_exp: f32 = logits.iter().map(|x| (x - max_val).exp()).sum();
    let log_sum_exp = max_val + sum_exp.ln();
    let idx = match usize::try_from(token) {
        Ok(i) => i.min(logits.len() - 1),
        Err(_) => return f32::NEG_INFINITY,
    };
    if !logits[idx].is_finite() {
        return f32::NEG_INFINITY;
    }
    logits[idx] - log_sum_exp
}

/// Compute the top-`n` `(token, logprob)` pairs under the
/// post-filter distribution, sorted by `logprob` descending.
///
/// Returns an empty `Vec` when `n == 0` or `logits` is empty.
/// Otherwise returns at most `n` pairs (fewer if the vocab is
/// smaller than `n`). Each entry is `(TokenId, logprob)` where
/// `logprob = log(softmax(logits)[token])` — computed via the same
/// numerically-stable helper as [`logprob_of_token`] so the two
/// helpers always agree on the logprob of any token that appears
/// in both the sampled token's slot and the top-K list.
///
/// Used by [`sample_one_with_params`] when
/// `SamplingParams::top_logprobs.is_some()` to populate
/// [`SampledToken::top_logprobs`].
#[allow(dead_code)] // exposed as part of the sampler API surface
fn top_logprobs_of(logits: &[f32], n: u32) -> Vec<(TokenId, f32)> {
    if n == 0 || logits.is_empty() {
        return Vec::new();
    }
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max_val.is_finite() {
        // All -inf — no token has finite probability; return empty.
        return Vec::new();
    }
    let sum_exp: f32 = logits.iter().map(|x| (x - max_val).exp()).sum();
    let log_sum_exp = max_val + sum_exp.ln();

    // Partial sort: keep only the top-n by raw logit. After this the
    // top-n entries by logit are also the top-n entries by
    // post-filter probability (softmax is monotonic), so the
    // logprob-of-each conversion is the only per-entry work left.
    let n_usize = (n as usize).min(logits.len());
    let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    // `select_nth_unstable_by` partitions so the n-th element is in
    // its final sorted position; everything before it is ≤ it and
    // everything after is ≥ it. We take the prefix `[..n_usize]`
    // and sort it descending for the final ordering.
    indexed.select_nth_unstable_by(n_usize - 1, |a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or_else(|| a.1.is_nan().cmp(&b.1.is_nan()))
    });
    indexed[..n_usize].sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or_else(|| a.1.is_nan().cmp(&b.1.is_nan()))
    });

    indexed[..n_usize]
        .iter()
        .map(|(i, v)| {
            let lp = if v.is_finite() {
                *v - log_sum_exp
            } else {
                f32::NEG_INFINITY
            };
            (TokenId::try_from(*i).unwrap_or(0), lp)
        })
        .collect()
}

/// Compute log-softmax of `logits` element-wise, returning one
/// log-probability per token. Tokens with `-inf` logit keep `-inf`;
/// an empty input returns an empty `Vec`; an input where every logit
/// is `-inf` (e.g. top_k masked everything) returns a `Vec` of
/// `-inf`s matching the input length.
///
/// Used by [`sample_one_with_params`] when no `top_p` cutoff is in
/// effect (the sampling distribution is the full renormalized
/// softmax over the post-filter logits).
fn log_softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max_val.is_finite() {
        return vec![f32::NEG_INFINITY; logits.len()];
    }
    let sum_exp: f32 = logits.iter().map(|x| (x - max_val).exp()).sum();
    let log_sum_exp = max_val + sum_exp.ln();
    logits
        .iter()
        .map(|&x| {
            if x.is_finite() {
                x - log_sum_exp
            } else {
                f32::NEG_INFINITY
            }
        })
        .collect()
}

/// Compute the log-probabilities under the post-`top_p` renormalized
/// distribution. Tokens excluded by the nucleus cutoff get `-inf`;
/// tokens in the cutoff get `ln(p_renormalized)`. Used by
/// [`sample_one_with_params`] when `params.top_p < 1.0` so the
/// sampled token's `logprob` reflects the *actual* sampling
/// distribution (nucleus-constrained) rather than the
/// un-renormalized softmax.
fn renormalized_top_p_logprobs(logits: &[f32], top_p: f32) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }

    // Sort by raw logit descending — softmax is monotonic, so the
    // top-K by logit is also the top-K by probability. Mirror the
    // existing `top_p_sample` cutoff rule: include position `i`
    // (zero-indexed) iff the cumulative probability up to and
    // including `i` is `≤ top_p`; the first index that pushes
    // cumsum over `top_p` becomes the exclusive upper bound.
    let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or_else(|| a.1.is_nan().cmp(&b.1.is_nan()))
    });

    let max_val = indexed[0].1;
    if !max_val.is_finite() {
        return vec![f32::NEG_INFINITY; logits.len()];
    }
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

    let mut logprobs = vec![f32::NEG_INFINITY; logits.len()];
    for (i, p) in indexed[..cutoff].iter().zip(probs.iter()) {
        let p_norm = if total > 0.0 { p / total } else { 0.0 };
        logprobs[i.0] = if p_norm > 0.0 {
            p_norm.ln()
        } else {
            f32::NEG_INFINITY
        };
    }
    logprobs
}

pub(crate) fn greedy_sample(logits: &[f32]) -> TokenId {
    trace!(vocab_size = logits.len(), "Greedy sampling");
    argmax_logits(logits)
}

pub(crate) fn temperature_sample(
    logits: &[f32],
    temperature: f32,
    random_threshold: f32,
) -> TokenId {
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

    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if random_threshold <= cumsum {
            return TokenId::try_from(i).unwrap_or(0);
        }
    }
    TokenId::try_from(probs.len() - 1).unwrap_or(0)
}

pub(crate) fn top_p_sample(logits: &[f32], top_p: f32, random_threshold: f32) -> TokenId {
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
pub fn top_k_sample(logits: &[f32], k: usize, random_threshold: f32) -> TokenId {
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

    temperature_sample(&masked, 1.0, random_threshold)
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
                top_p_sample(&logits, top_p, random_f32())
            } else if temperature > 0.0 {
                temperature_sample(&logits, temperature, random_f32())
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
/// **Per-sequence RNG independence (P34 v0.2 wire-type follow-up
/// engine wire-through):** each call to [`sample_one_with_params`]
/// reads ONE random threshold from either a fresh
/// `StdRng::seed_from_u64(params.seed)` (when `params.seed.is_some()`)
/// or the thread-local default RNG (when `params.seed.is_none()`).
/// Two sequences that share the same `params.seed` therefore draw the
/// SAME random threshold for the SAME logits — this is the correct
/// behaviour for OpenAI's per-request determinism contract (same
/// seed ⇒ same draws ⇒ same sampled token). The fresh-RNG-per-call
/// pattern ensures per-sequence independence for sequences with
/// DIFFERENT seeds: they don't share state.
///
/// `params_list`, `seen_tokens`, and `logits_list` must have the same
/// length. The returned `Vec<SampledToken>` has length
/// `logits_list.len()`. Each [`SampledToken`] carries the sampled
/// token alongside its `logprob` (and top-K logprobs when
/// `params.top_logprobs.is_some()`) under the post-filter
/// distribution (P36 v0.3 wire-type follow-up engine wire-through).
///
/// Beam search (`beam_width > 1`) is not implemented here — callers
/// must intercept those requests before they reach this function.
#[must_use]
pub fn sample_batch_with_params(
    logits_list: &[Vec<f32>],
    params_list: &[SamplingParams],
    seen_tokens: &[Vec<TokenId>],
) -> Vec<SampledToken> {
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
///
/// **RNG selection (P34 v0.2 wire-type follow-up engine wire-through):**
/// `params.seed` is consulted exactly once per call. `Some(s)` reads
/// from a fresh `StdRng::seed_from_u64(s)`; `None` reads from the
/// thread-local default RNG. Greedy paths (`temperature = 0`,
/// `top_p = 1.0`, `top_k = 0`) bypass the RNG entirely — `seed` has
/// no observable effect in those modes.
///
/// **Return type (P36 v0.3 wire-type follow-up engine wire-through):**
/// returns a [`SampledToken`] carrying the sampled `token` alongside
/// its `logprob` under the post-filter distribution and (when
/// `params.top_logprobs.is_some()`) the top-K `(token, logprob)`
/// pairs. The `logprob` and `top_logprobs` reflect the **actual
/// sampling distribution** — for `top_p < 1.0` this is the
/// nucleus-renormalized distribution, not the raw softmax.
#[must_use]
pub fn sample_one_with_params(
    logits: &[f32],
    params: &SamplingParams,
    seen: &[TokenId],
) -> SampledToken {
    let mut logits = logits.to_vec();

    if (params.repeat_penalty - 1.0).abs() > f32::EPSILON && !seen.is_empty() {
        apply_repeat_penalty(&mut logits, seen, params.repeat_penalty);
    }

    if params.presence_penalty.abs() > f32::EPSILON && !seen.is_empty() {
        apply_presence_penalty(&mut logits, seen, params.presence_penalty);
    }

    if let Some(ref bias) = params.logit_bias {
        apply_logit_bias(&mut logits, bias);
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

    // RNG selection: read ONE random threshold per call, then pass it
    // down. The greedy branch below never reads the threshold so the
    // RNG draw is wasted but cheap — and keeping it unconditional
    // means the seed determinism guarantee holds even when the user
    // toggles sampling params between calls (the RNG state never
    // changes based on temperature / top_p / top_k values).
    let random_threshold = sample_random_threshold(params.seed);

    let token = if params.top_p < 1.0 {
        top_p_sample(&logits, params.top_p, random_threshold)
    } else if params.temperature > 0.0 {
        temperature_sample(&logits, params.temperature, random_threshold)
    } else {
        greedy_sample(&logits)
    };

    // Compute the log-probabilities under the ACTUAL sampling
    // distribution (the same one the sampler drew `token` from).
    // For `top_p < 1.0` this is the nucleus-renormalized
    // distribution; otherwise it's the full log-softmax over the
    // post-filter (post-bias/penalty/temperature/top-k) logits.
    let sampling_logprobs = if params.top_p < 1.0 {
        renormalized_top_p_logprobs(&logits, params.top_p)
    } else {
        log_softmax(&logits)
    };

    let logprob = sampling_logprobs
        .get(
            usize::try_from(token)
                .unwrap_or(0)
                .min(sampling_logprobs.len().saturating_sub(1)),
        )
        .copied()
        .unwrap_or(f32::NEG_INFINITY);

    let top_logprobs = match params.top_logprobs {
        Some(n) if n > 0 => {
            // Re-rank the sampling-distribution log-probs to pick the
            // top-N. Uses the same partial-sort + suffix-sort idiom
            // as `top_logprobs_of` but operates on pre-computed
            // log-probs (so the values are exact copies of what
            // `logprob` already used — guaranteed consistency).
            let n_usize = (n as usize).min(sampling_logprobs.len());
            let mut indexed: Vec<(usize, f32)> = sampling_logprobs
                .iter()
                .enumerate()
                .map(|(i, &lp)| (i, lp))
                .collect();
            indexed.select_nth_unstable_by(n_usize - 1, |a, b| {
                b.1.partial_cmp(&a.1)
                    .unwrap_or_else(|| a.1.is_nan().cmp(&b.1.is_nan()))
            });
            indexed[..n_usize].sort_by(|a, b| {
                b.1.partial_cmp(&a.1)
                    .unwrap_or_else(|| a.1.is_nan().cmp(&b.1.is_nan()))
            });
            indexed[..n_usize]
                .iter()
                .map(|(i, lp)| (TokenId::try_from(*i).unwrap_or(0), *lp))
                .collect()
        }
        _ => Vec::new(),
    };

    SampledToken {
        token,
        logprob,
        top_logprobs,
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

/// Add the bias at each entry in `bias` to the logit of the
/// corresponding token ID (OpenAI `logit_bias` semantic).
///
/// Per OpenAI spec the bias is additive and per-token: positive
/// values *increase* the probability of the biased tokens; negative
/// values *decrease* it. Bias values are constrained to `[-100, 100]`
/// by the validator on the HTTP layer; values outside that range
/// are rejected with `400 invalid_request_error` so callers learn
/// about truly out-of-range values up front rather than producing
/// silently-extreme logits.
///
/// **Determinism:** `bias` is a `HashMap` whose iteration order is
/// non-deterministic, but because each bias is *additive and
/// independent per token*, the *final logits* are deterministic
/// regardless of iteration order. So this helper preserves the
/// determinism guarantee that `sample_one_with_params` requires
/// (no other sampling primitive relies on iteration order).
///
/// **Difference from [`apply_repeat_penalty`] / [`apply_presence_penalty`]:**
/// both `apply_repeat_penalty` and `apply_presence_penalty` are
/// *automatic* — they bias every seen token. `apply_logit_bias` is
/// *explicit* — the caller specifies exactly which token IDs to bias
/// and by how much.
///
/// No-op when `bias` is empty or `logits` is empty. Out-of-range
/// token ids (any ID `>= logits.len()`) are silently ignored
/// (matches OpenAI's server behaviour; a bias on a non-vocab token
/// is meaningless and would only consume compute).
pub fn apply_logit_bias(logits: &mut [f32], bias: &std::collections::HashMap<TokenId, f32>) {
    if bias.is_empty() || logits.is_empty() {
        return;
    }

    for (&token, &delta) in bias {
        if let Ok(idx) = usize::try_from(token)
            && idx < logits.len()
        {
            logits[idx] += delta;
        }
    }
}

/// P38 v0.x wire-type follow-up — engine wire-through helper for
/// `stop` sequences. Returns `true` iff any token sequence in `stops`
/// is a suffix of `generated_tokens`.
///
/// **Complexity:** O(N × M) where N = `generated_tokens.len()` and
/// M = sum(stop.len() for stop in stops). Both are tiny in practice:
/// - `generated_tokens.len()` ≤ `max_tokens` (typical ≤ 4096)
/// - `stops.len()` ≤ 4 (OpenAI spec upper bound, validated at HTTP layer)
/// - each `stop.len()` ≤ ~8 tokens (typical BPE-tokenized stop strings)
///
/// So the per-step cost is bounded by ~32 slice comparisons × max_tokens
/// positions — well under 100 ns per step on a modern CPU. The check runs
/// once per generated token in `step_regular`.
///
/// **Empty / oversized stops:** an empty stop (`vec![]`) or a stop
/// longer than `generated_tokens` is a no-op for that iteration. The
/// HTTP-layer `validate_stop_sequences` rejects empty-string stops
/// (which would tokenize to either zero or one token) so this function
/// doesn't need to handle "stop that can never match" specially.
///
/// Takes `&[u32]` rather than `&[vllm_traits::TokenId]` because
/// `TokenId` is `pub type TokenId = u32` and this helper lives in
/// `vllm_core::sampling` without a `vllm_traits` import. The
/// `SamplingParams::stop_token_sequences` field uses
/// `Vec<Vec<TokenId>>` for public-API ergonomics; the caller converts
/// via deref coercion at the call site.
#[must_use]
pub fn matches_stop_sequences(generated_tokens: &[u32], stops: &[Vec<u32>]) -> bool {
    for stop in stops {
        if stop.is_empty() || stop.len() > generated_tokens.len() {
            continue;
        }
        let start = generated_tokens.len() - stop.len();
        if &generated_tokens[start..] == stop.as_slice() {
            return true;
        }
    }
    false
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
