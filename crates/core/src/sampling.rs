#![allow(unused_variables)]

use crate::types::TokenId;
use tracing::trace;

fn random_f32() -> f32 {
    rand::random::<f32>()
}

pub(crate) fn greedy_sample(logits: &[f32]) -> TokenId {
    trace!(vocab_size = logits.len(), "Greedy sampling");
    logits
        .iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |(max_idx, max_val), (i, &val)| {
            if val > max_val {
                (i, val)
            } else {
                (max_idx, max_val)
            }
        })
        .0 as TokenId
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
            return i as TokenId;
        }
    }
    (probs.len() - 1) as TokenId
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
    probs.iter_mut().for_each(|p| *p /= total);

    let random_threshold = random_f32();
    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if random_threshold <= cumsum {
            return indexed[i].0 as TokenId;
        }
    }
    indexed[probs.len() - 1].0 as TokenId
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

            if repeat_penalty != 1.0 && !seen.is_empty() {
                apply_repeat_penalty(&mut logits, seen, repeat_penalty);
            }

            if temperature > 0.0 && temperature != 1.0 {
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

/// Divide the logit at each id present in `seen_tokens` by `penalty`.
///
/// This is the standard "repeat penalty" trick used by llama.cpp, HuggingFace
/// `transformers`, and most open-source inference engines: tokens that have
/// already appeared are made less likely on subsequent steps. No-op when
/// `penalty == 1.0`, `seen_tokens` is empty, or `logits` is empty.
///
/// Out-of-range token ids are silently ignored. Duplicate entries in
/// `seen_tokens` are deduped via an internal `HashSet`.
pub fn apply_repeat_penalty(logits: &mut [f32], seen_tokens: &[TokenId], penalty: f32) {
    if penalty == 1.0 || seen_tokens.is_empty() || logits.is_empty() {
        return;
    }

    let mut seen = std::collections::HashSet::new();
    for &token in seen_tokens {
        let idx = token as usize;
        if idx < logits.len() && seen.insert(token) {
            logits[idx] /= penalty;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_selects_max() {
        assert_eq!(greedy_sample(&[0.1, 0.5, 0.3]), 1);
    }

    #[test]
    fn test_greedy_first_on_tie() {
        assert_eq!(greedy_sample(&[0.5, 0.5, 0.3]), 0);
    }

    #[test]
    fn test_greedy_empty() {
        assert_eq!(greedy_sample(&[]), 0);
    }

    #[test]
    fn test_sample_batch() {
        let logits = vec![vec![0.1, 0.9], vec![0.8, 0.2]];
        let seen = vec![vec![], vec![]];
        assert_eq!(sample_batch(&logits, 0.0, 1.0, 0, 1.0, &seen), vec![1, 0]);
    }

    #[test]
    fn test_greedy_all_same_logit() {
        assert_eq!(greedy_sample(&[0.5, 0.5, 0.5]), 0);
    }

    #[test]
    fn test_greedy_negative_logits() {
        assert_eq!(greedy_sample(&[-1.0, -0.5, 0.0]), 2);
    }

    #[test]
    fn test_greedy_large_vocab() {
        let mut logits = vec![0.0; 10000];
        logits[9999] = 1.0;
        assert_eq!(greedy_sample(&logits), 9999);
    }

    #[test]
    fn test_temperature_one_unchanged() {
        let logits = &[0.1, 0.5, 0.3];
        let result = temperature_sample(logits, 1.0);
        assert!(result < 3);
    }

    #[test]
    fn test_temperature_zero_reverts_to_greedy() {
        let logits = &[0.1, 0.9, 0.3];
        let result = temperature_sample(logits, 0.0);
        assert_eq!(result, 1);
    }

    #[test]
    fn test_temperature_very_small() {
        let logits = &[0.1, 0.9, 0.3];
        let result = temperature_sample(logits, 0.01);
        assert!(result < 3);
    }

    #[test]
    fn test_temperature_very_large() {
        let logits = &[0.1, 0.9, 0.3];
        let _result = temperature_sample(logits, 10.0);
    }

    #[test]
    fn test_temperature_empty() {
        let result = temperature_sample(&[], 1.0);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_top_p_one_equals_greedy() {
        let logits = &[0.1, 0.9, 0.3];
        let result = top_p_sample(logits, 1.0);
        assert_eq!(result, 1);
    }

    #[test]
    fn test_top_p_small() {
        let logits = &[0.9, 0.05, 0.05];
        let result = top_p_sample(logits, 0.5);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_top_p_empty() {
        let result = top_p_sample(&[], 0.9);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_sample_batch_greedy() {
        let logits = vec![vec![0.1, 0.9], vec![0.8, 0.2], vec![0.3, 0.7]];
        let seen = vec![vec![], vec![], vec![]];
        let result = sample_batch(&logits, 0.0, 1.0, 0, 1.0, &seen);
        assert_eq!(result, vec![1, 0, 1]);
    }

    #[test]
    fn test_sample_batch_temperature_and_top_p() {
        let logits = vec![vec![0.1, 0.9], vec![0.5, 0.5]];
        let seen = vec![vec![], vec![]];
        let _result = sample_batch(&logits, 0.5, 0.8, 0, 1.0, &seen);
    }

    #[test]
    fn test_sample_batch_with_top_k() {
        let logits = vec![vec![0.1, 0.9, 0.3, 0.05, 0.05]];
        let seen = vec![vec![]];
        let result = sample_batch(&logits, 0.0, 1.0, 2, 1.0, &seen);
        assert!(result[0] == 1 || result[0] == 2);
    }

    #[test]
    fn test_sample_batch_with_repeat_penalty() {
        let logits = vec![vec![0.5, 0.5, 0.5]];
        let seen = vec![vec![1]];
        let result = sample_batch(&logits, 0.0, 1.0, 0, 2.0, &seen);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_repeat_penalty_basic() {
        let mut logits = vec![0.5, 0.5, 0.5];
        let seen = vec![1];
        apply_repeat_penalty(&mut logits, &seen, 2.0);
        assert!(logits[1] < 0.5);
        assert_eq!(logits[0], 0.5);
        assert_eq!(logits[2], 0.5);
    }

    #[test]
    fn test_repeat_penalty_no_effect_at_one() {
        let mut logits = vec![0.5, 0.5];
        let seen = vec![0];
        apply_repeat_penalty(&mut logits, &seen, 1.0);
        assert_eq!(logits[0], 0.5);
        assert_eq!(logits[1], 0.5);
    }

    #[test]
    fn test_top_k_only_top_k_selected() {
        let logits = vec![0.1, 0.9, 0.3, 0.05, 0.05];
        let result = top_k_sample(&logits, 2);
        assert!(result == 1 || result == 2);
    }

    #[test]
    fn test_top_k_zero_no_effect() {
        let logits = vec![0.1, 0.9, 0.3];
        let result = top_k_sample(&logits, 0);
        assert_eq!(result, 1);
    }
}

#[cfg(test)]
mod prop_tests {
    use super::*;
    use proptest::prelude::*;

    // Invariant 1: sample_batch output length equals input list length.
    // Invariant 2: each sampled token is in [0, vocab_size).
    // Invariant 3: greedy (T=0) on a uniform-distribution batch returns
    //              index 0 (greedy_sample picks first on ties / all-zero).

    proptest! {
        /// `sample_batch` always returns one token per input logit row,
        /// regardless of temperature, top-p, top-k, or repeat-penalty.
        #[test]
        fn prop_sample_batch_length_preserved(
            logits in proptest::collection::vec(
                proptest::collection::vec(-10.0f32..10.0, 1..20),
                1..30,
            ),
            temperature in 0.0f32..2.0,
            top_p in 0.0f32..1.0,
            top_k in 0usize..10,
            repeat_penalty in 0.5f32..2.0,
        ) {
            let seen: Vec<Vec<TokenId>> = vec![vec![]; logits.len()];
            let result = sample_batch(&logits, temperature, top_p, top_k, repeat_penalty, &seen);
            prop_assert_eq!(result.len(), logits.len());
        }

        /// Every token returned by `greedy_sample` is a valid index in
        /// `[0, logits.len())`. NaN-heavy inputs are common in fuzz; we
        /// restrict to finite logits so the assertion holds.
        #[test]
        fn prop_greedy_sample_in_bounds(
            logits in proptest::collection::vec(-5.0f32..5.0, 1..200),
        ) {
            let result = greedy_sample(&logits);
            prop_assert!(
                (result as usize) < logits.len(),
                "greedy_sample returned {} for vocab size {}",
                result,
                logits.len(),
            );
        }

        /// Greedy `sample_batch` (T=0) returns the same per-row result as
        /// calling `greedy_sample` on each row independently.
        #[test]
        fn prop_sample_batch_greedy_matches(
            logits in proptest::collection::vec(
                proptest::collection::vec(-5.0f32..5.0, 1..10),
                1..15,
            ),
        ) {
            let seen: Vec<Vec<TokenId>> = vec![vec![]; logits.len()];
            let batched = sample_batch(&logits, 0.0, 1.0, 0, 1.0, &seen);
            let per_row: Vec<TokenId> = logits.iter().map(|row| greedy_sample(row)).collect();
            prop_assert_eq!(batched, per_row);
        }

        /// `apply_repeat_penalty` with penalty=1.0 is a no-op (the early
        /// return path). Any input must round-trip unchanged.
        #[test]
        fn prop_repeat_penalty_one_is_noop(
            mut logits in proptest::collection::vec(-5.0f32..5.0, 1..50),
            seen in proptest::collection::vec(0u32..64, 0..20),
        ) {
            let original = logits.clone();
            apply_repeat_penalty(&mut logits, &seen, 1.0);
            prop_assert_eq!(logits, original);
        }
    }
}
