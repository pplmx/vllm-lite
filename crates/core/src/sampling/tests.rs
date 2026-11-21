//! Unit tests for the sampling primitives (`greedy_sample`,
//! `temperature_sample`, `top_k_sample`, `top_p_sample`,
//! `sample_batch`, `sample_batch_with_params`, `apply_repeat_penalty`).
//!
//! Extracted from `sampling.rs` to keep the implementation file under
//! the project's 800-line soft cap. Exercises:
//!
//! - `greedy_sample` edge cases (max, tie, empty, all-same, negative,
//!   large vocab)
//! - `temperature_sample` (T=1 unchanged, T=0 reverts to greedy,
//!   small/large T, empty input)
//! - `top_p_sample` (T=1 ≡ greedy, small p, empty)
//! - `top_k_sample` (top-k subset, k=0 ≡ greedy)
//! - `sample_batch` (basic, `T+top_p`, `top_k`, `repeat_penalty`)
//! - `sample_batch_with_params` (per-sequence params, matches
//!   legacy scalar `sample_batch`, empty params degrade to greedy)
//! - `apply_repeat_penalty` (basic penalty, no-op at 1.0)
//! - `apply_logit_bias` (basic bias, no-op on empty map, out-of-range
//!   keys silently ignored, integrates with `sample_one_with_params`)
//! - `seed` (P34 v0.2 wire-type follow-up engine wire-through):
//!   determinism guarantee, `seed = 0` is valid, `seed = None`
//!   falls back to thread RNG, per-sequence independence,
//!   greedy paths bypass the RNG
//! - Property-based tests (proptest) in the sibling `prop_tests`
//!   module: batch length preservation, greedy index bounds,
//!   batched greedy matches per-row greedy, repeat-penalty no-op.

use super::*;
use vllm_traits::SamplingParams;

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
    let result = temperature_sample(logits, 1.0, 0.5);
    assert!(result < 3);
}

#[test]
fn test_temperature_all_neg_inf_falls_back_to_greedy() {
    // Degenerate model output: every logit is `-inf`. There is no
    // valid distribution; the sampler must fall back to greedy
    // argmax (deterministic first index) instead of producing NaN
    // probabilities and returning an arbitrary tail index.
    let logits = vec![f32::NEG_INFINITY; 32];
    assert_eq!(temperature_sample(&logits, 0.8, 0.5), 0);
    // A NaN logit is also non-finite and must not poison the result.
    let nan_logits = vec![f32::NAN; 32];
    assert_eq!(temperature_sample(&nan_logits, 0.8, 0.5), 0);
}

#[test]
fn test_top_p_all_neg_inf_falls_back_to_greedy() {
    let logits = vec![f32::NEG_INFINITY; 32];
    assert_eq!(top_p_sample(&logits, 0.9, 0.5), 0);
    let nan_logits = vec![f32::NAN; 32];
    assert_eq!(top_p_sample(&nan_logits, 0.9, 0.5), 0);
}

#[test]
fn test_top_k_all_neg_inf_falls_back_to_greedy() {
    // top-k masking of a fully-degenerate row leaves every logit at
    // `-inf`; the downstream temperature/top-p samplers must degrade
    // to greedy rather than NaN math.
    let logits = vec![f32::NEG_INFINITY; 32];
    assert_eq!(top_k_sample(&logits, 5, 0.5), 0);
}

// a single NaN logit among FINITE logits must not
// produce an arbitrary (non-argmax) token — the pre-fix guard checked only
// `max_val` finiteness, but `f32::max` silently ignores NaN, so a NaN row
// left `max_val` finite while the softmax exp-sum became NaN and every
// `random_threshold <= cumsum` comparison was false, falling through to the
// tail index (len-1). Write the regression so the mixed-NaN input must pick
// the actual argmax, not the fallback tail.

#[test]
fn test_temperature_mixed_nan_picks_finite_argmax() {
    // Logits: index 1 is the clear max; index 0 is NaN. Pre-fix the NaN
    // poisoned the softmax -> fallback returned token 2 (the tail), not
    // the argmax at index 1.
    let logits = vec![f32::NAN, 1.0, 0.0];
    // Threshold 0.5: the argmax carries the probability mass, so the
    // cumulative sum crosses 0.5 at index 1 for any sane distribution.
    assert_eq!(temperature_sample(&logits, 0.8, 0.5), 1);

    // A different random threshold must still land in the finite set, never
    // the silent tail fallback for a NaN-adjacent row.
    let logits = vec![f32::NAN, 2.0, 1.0];
    assert_eq!(temperature_sample(&logits, 0.8, 0.001), 1);
    let logits = vec![2.0, f32::NAN, 1.0];
    assert_eq!(temperature_sample(&logits, 0.8, 0.001), 0);
    let logits = vec![f32::NAN, 0.0, 1.0];
    assert_eq!(temperature_sample(&logits, 0.8, 0.5), 2);

    // Positive infinity among finite logits must behave the same (the
    // infinite logit is the argmax and wins outright).
    let logits = vec![f32::NAN, f32::INFINITY, 0.0];
    assert_eq!(temperature_sample(&logits, 0.8, 0.5), 1);
}

#[test]
fn test_top_p_mixed_nan_picks_finite_argmax() {
    // Same mixed-NaN regression for the nucleus sampler: pre-fix the NaN
    // sorted last, the exp-sum collapsed to NaN, the cutoff loop never
    // triggered, and the fallback returned the NaN position (the last
    // sorted index) instead of the argmax.
    let logits = vec![f32::NAN, 1.0, 0.0];
    assert_eq!(top_p_sample(&logits, 0.9, 0.5), 1);

    let logits = vec![f32::NAN, 2.0, 1.0];
    assert_eq!(top_p_sample(&logits, 0.9, 0.001), 1);
    let logits = vec![2.0, f32::NAN, 1.0];
    assert_eq!(top_p_sample(&logits, 0.9, 0.001), 0);

    // `-inf` mixed with finite must keep degrading to greedy (the finite
    // argmax wins, never the -inf tail).
    let logits = vec![f32::NEG_INFINITY, 1.0, f32::NEG_INFINITY];
    assert_eq!(top_p_sample(&logits, 0.9, 0.5), 1);
}

// the logprob helpers shared the same mixed-NaN
// poisoning — a NaN slipped past the `max_val.is_finite()` guard and made
// `log_softmax` / `renormalized_top_p_logprobs` emit NaN logprobs for the
// finite entries (which breaks `SampledToken` JSON serialization).

/// `log_softmax` must produce the actual log-probabilities, not just a
/// correctly ordered vector.
///
/// The pre-existing test asserts only `out[1] > out[2]`, which any monotone
/// transformation satisfies: `cargo mutants` flipped both the `x - max_val`
/// subtraction and the final `x - log_sum_exp`, and turned the
/// `max_val + ln(sum)` into `-` and `*`, with all six variants surviving.
#[test]
fn log_softmax_matches_the_reference_values() {
    let out = log_softmax(&[0.0_f32, 0.0]);
    let ln2 = std::f32::consts::LN_2;
    assert!(
        (out[0] + ln2).abs() < 1e-6 && (out[1] + ln2).abs() < 1e-6,
        "log_softmax([0, 0]) must be -ln 2 everywhere; got {out:?}"
    );

    let out = log_softmax(&[1.0_f32, 0.0]);
    let log_denom = (1.0_f32 + (-1.0_f32).exp()).ln();
    assert!(
        (out[0] - (-log_denom)).abs() < 1e-6,
        "log p(x=1) must be -ln(1 + e^-1); got {out:?}"
    );
    assert!(
        (out[1] - (-1.0 - log_denom)).abs() < 1e-6,
        "log p(x=0) must be -1 - ln(1 + e^-1); got {out:?}"
    );

    let shifted = log_softmax(&[1001.0_f32, 1000.0]);
    assert!(
        (shifted[0] - out[0]).abs() < 1e-3 && (shifted[1] - out[1]).abs() < 1e-3,
        "adding a constant to every logit must not change log-softmax; got \
         {shifted:?} vs {out:?}"
    );
}

#[test]
fn test_log_softmax_mixed_nan_gives_finite_logprobs() {
    let out = log_softmax(&[f32::NAN, 1.0, 0.0]);
    assert_eq!(out.len(), 3);
    // The NaN position must be -inf; the finite entries must be finite and
    // ordered (index 1 strictly higher than index 2).
    assert!(out[0] == f32::NEG_INFINITY);
    assert!(out[1].is_finite() && out[2].is_finite());
    assert!(
        out[1] > out[2],
        "argmax logit must have the highest logprob; got {out:?}"
    );
}

#[test]
fn test_renormalized_top_p_logprobs_mixed_nan_gives_finite_logprobs() {
    let out = renormalized_top_p_logprobs(&[f32::NAN, 1.0, 0.0], 0.9);
    assert_eq!(out.len(), 3);
    // The NaN position is excluded from the nucleus -> -inf; the finite
    // entries inside the cutoff have finite, positive-mass logprobs.
    assert!(out[0] == f32::NEG_INFINITY);
    assert!(out[1].is_finite() && out[2].is_finite());
    assert!(
        out[1] > out[2],
        "argmax logit must have the highest logprob; got {out:?}"
    );
}

#[test]
fn test_sample_batch_with_params_all_neg_inf_degenerates_safely() {
    // Public-surface regression: all-`-inf` logits must yield the
    // deterministic first token with `-inf` logprob (not an arbitrary
    // tail index from NaN math), for both nucleus and temperature
    // sampling paths.
    let logits = vec![f32::NEG_INFINITY; 32];
    for params in [
        SamplingParams::builder().with_top_p(0.9).build(),
        SamplingParams::builder().with_temperature(0.8).build(),
        SamplingParams::builder()
            .with_top_k(5)
            .with_temperature(0.8)
            .build(),
    ] {
        let out = sample_batch_with_params(&[logits.clone()], &[params], &[vec![]]);
        assert_eq!(
            out[0].token, 0,
            "all-(-inf) logits must pick the first token"
        );
        assert!(
            !out[0].logprob.is_finite(),
            "degenerate distribution has no finite logprob"
        );
    }
}

#[test]
fn test_temperature_zero_reverts_to_greedy() {
    let logits = &[0.1, 0.9, 0.3];
    let result = temperature_sample(logits, 0.0, 0.5);
    assert_eq!(result, 1);
}

#[test]
fn test_temperature_very_small() {
    let logits = &[0.1, 0.9, 0.3];
    let result = temperature_sample(logits, 0.01, 0.5);
    assert!(result < 3);
}

#[test]
fn test_temperature_very_large() {
    let logits = &[0.1, 0.9, 0.3];
    let _result = temperature_sample(logits, 10.0, 0.5);
}

#[test]
fn test_temperature_empty() {
    let result = temperature_sample(&[], 1.0, 0.5);
    assert_eq!(result, 0);
}

#[test]
fn test_top_p_one_equals_greedy() {
    let logits = &[0.1, 0.9, 0.3];
    let result = top_p_sample(logits, 1.0, 0.5);
    assert_eq!(result, 1);
}

#[test]
fn test_top_p_small() {
    let logits = &[0.9, 0.05, 0.05];
    let result = top_p_sample(logits, 0.5, 0.5);
    assert_eq!(result, 0);
}

#[test]
fn test_top_p_empty() {
    let result = top_p_sample(&[], 0.9, 0.5);
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
    assert!((logits[0] - 0.5).abs() < 1e-6);
    assert!((logits[2] - 0.5).abs() < 1e-6);
}

/// The repeat penalty must reach the LAST index and ignore out-of-range ids.
///
/// The loop guards `idx < logits.len()`. `cargo mutants` relaxed that to `<=`
/// (which indexes out of bounds) and flipped it to `>`, and both survived: no
/// test used a `seen` id at the final position, and none supplied an id past
/// the end. The boundary is load-bearing — a token id equal to the vocabulary
/// size arrives from real tokenizers at the edge of the range.
/// The nucleus cutoff must use a STRICT comparison.
///
/// `cumsum > top_p` stops the nucleus once the mass EXCEEDS `top_p`. Relaxing it
/// to `>=` also stops when the mass equals `top_p` exactly, dropping the token
/// that completes the nucleus — the sampler then never produces it no matter
/// how the threshold falls. `cargo mutants` found the `>=` variant survives:
/// no existing fixture lands exactly on the boundary.
#[test]
fn top_p_cutoff_is_strict_at_the_boundary() {
    // Two equal logits -> probabilities 0.5 / 0.5. With top_p = 0.5 the
    // running mass hits exactly 0.5 after the first token, which is NOT
    // greater than top_p, so both tokens stay in the nucleus.
    let logits = vec![0.0_f32, 0.0];
    // A threshold past 0.5 can only be satisfied by the second token, so it is
    // reachable only when the nucleus kept both.
    let sampled = top_p_sample(&logits, 0.5, 0.75);
    assert_eq!(
        sampled, 1,
        "with mass exactly == top_p the nucleus must keep both tokens; the \
         strict `>` reaches the second one, while `>=` truncates to the first"
    );
}

#[test]
fn test_repeat_penalty_handles_the_last_index_and_ignores_out_of_range() {
    // The last index must be penalised, not skipped.
    let mut logits = vec![0.5, 0.5, 0.5];
    apply_repeat_penalty(&mut logits, &[2], 2.0);
    assert!(
        logits[2] < 0.5,
        "the final index must be penalised; got {logits:?}"
    );
    assert!((logits[0] - 0.5).abs() < 1e-6, "other positions untouched");
    assert!((logits[1] - 0.5).abs() < 1e-6, "other positions untouched");

    // An id past the end is ignored, not applied to a wrong slot.
    let mut logits = vec![0.5, 0.5, 0.5];
    apply_repeat_penalty(&mut logits, &[3, 99], 2.0);
    assert_eq!(
        logits,
        vec![0.5, 0.5, 0.5],
        "out-of-range ids must leave every logit untouched"
    );

    // A negative logit at the last index must move further from zero.
    let mut logits = vec![0.0, -5.0];
    apply_repeat_penalty(&mut logits, &[1], 2.0);
    assert!(
        logits[1] < -5.0,
        "a penalised negative logit must grow more negative; got {logits:?}"
    );
}

#[test]
fn test_repeat_penalty_no_effect_at_one() {
    let mut logits = vec![0.5, 0.5];
    let seen = vec![0];
    apply_repeat_penalty(&mut logits, &seen, 1.0);
    assert!((logits[0] - 0.5).abs() < 1e-6);
    assert!((logits[1] - 0.5).abs() < 1e-6);
}

// P29 v0.3 wire-type follow-up: sign-aware `apply_repeat_penalty`
// implementation, enabling OpenAI `frequency_penalty` boost
// semantics (negative values). The pre-P29 implementation used
// simple division for all logits, which had a sign-flip bug for
// negative logits with `penalty < 1.0` (dividing a negative logit
// by a value < 1.0 makes it MORE negative — opposite of the boost
// direction). These tests pin the corrected sign-aware behavior.

#[test]
fn test_repeat_penalty_penalizes_negative_logits() {
    // Penalty > 1 (penalize): negative logits must become MORE
    // negative (further from zero). Pre-P29 this would be wrong —
    // dividing -5.0 by 2.0 gives -2.5 (boost, opposite of intended).
    // Post-P29 multiplying -5.0 by 2.0 gives -10.0 (correct
    // penalize direction).
    let mut logits = vec![-5.0, -5.0, -5.0];
    let seen = vec![1];
    apply_repeat_penalty(&mut logits, &seen, 2.0);
    assert!((logits[0] - -5.0).abs() < 1e-6, "logit at 0 untouched");
    assert!(
        (logits[1] - -10.0).abs() < 1e-6,
        "logit at 1 must be -10.0 (penalized); got {}",
        logits[1]
    );
    assert!((logits[2] - -5.0).abs() < 1e-6, "logit at 2 untouched");
}

#[test]
fn test_repeat_penalty_boosts_negative_logits() {
    // Penalty < 1 (boost): negative logits must become LESS
    // negative (closer to zero). Pre-P29 this would be wrong —
    // dividing -5.0 by 0.5 gives -10.0 (penalize, opposite of
    // intended). Post-P29 multiplying -5.0 by 0.5 gives -2.5
    // (correct boost direction). This is the v0.3 boost-semantic
    // carve-out closed by P29.
    let mut logits = vec![-5.0, -5.0, -5.0];
    let seen = vec![1];
    apply_repeat_penalty(&mut logits, &seen, 0.5);
    assert!((logits[0] - -5.0).abs() < 1e-6, "logit at 0 untouched");
    assert!(
        (logits[1] - -2.5).abs() < 1e-6,
        "logit at 1 must be -2.5 (boosted); got {}",
        logits[1]
    );
    assert!((logits[2] - -5.0).abs() < 1e-6, "logit at 2 untouched");
}

#[test]
fn test_repeat_penalty_penalizes_positive_logits_unchanged() {
    // Sanity: penalty > 1 on positive logits must still DIVIDE
    // (not multiply) — pins the postive-logit branch of the
    // sign-aware implementation.
    let mut logits = vec![4.0, 4.0, 4.0];
    let seen = vec![1];
    apply_repeat_penalty(&mut logits, &seen, 2.0);
    assert!((logits[0] - 4.0).abs() < 1e-6);
    assert!(
        (logits[1] - 2.0).abs() < 1e-6,
        "logit at 1 must be 2.0 (4.0 / 2.0); got {}",
        logits[1]
    );
    assert!((logits[2] - 4.0).abs() < 1e-6);
}

#[test]
fn test_repeat_penalty_boosts_positive_logits_unchanged() {
    // Sanity: penalty < 1 on positive logits must still DIVIDE
    // (not multiply) — pins the positive-logit branch.
    let mut logits = vec![4.0, 4.0, 4.0];
    let seen = vec![1];
    apply_repeat_penalty(&mut logits, &seen, 0.5);
    assert!((logits[0] - 4.0).abs() < 1e-6);
    assert!(
        (logits[1] - 8.0).abs() < 1e-6,
        "logit at 1 must be 8.0 (4.0 / 0.5); got {}",
        logits[1]
    );
    assert!((logits[2] - 4.0).abs() < 1e-6);
}

#[test]
fn test_repeat_penalty_zero_logit_uses_positive_branch() {
    // A logit of exactly 0.0 is treated as the "positive" branch
    // (the implementation uses `logit >= 0.0`). Penalty > 1: 0 / 2 = 0
    // (no change). Penalty < 1: 0 / 0.5 = 0 (no change). Either way
    // zero logits stay zero — pins the boundary behavior.
    let mut logits = vec![0.0, 0.0];
    let seen = vec![0];
    apply_repeat_penalty(&mut logits, &seen, 2.0);
    assert!(
        logits[0].abs() < 1e-6,
        "zero logit stays zero under penalty > 1"
    );
    apply_repeat_penalty(&mut logits, &seen, 0.5);
    assert!(
        logits[0].abs() < 1e-6,
        "zero logit stays zero under penalty < 1"
    );
}

#[test]
fn test_repeat_penalty_mixed_signs_handled_independently() {
    // The sign-aware implementation must handle each logit
    // independently based on its sign. With seen = [0, 1, 2] and
    // penalty = 0.5 (boost):
    //   - logit[0] = 4.0 (positive): 4.0 / 0.5 = 8.0 (boosted)
    //   - logit[1] = -4.0 (negative): -4.0 * 0.5 = -2.0 (boosted, less negative)
    //   - logit[2] = 0.0 (neutral): unchanged
    let mut logits = vec![4.0, -4.0, 0.0];
    let seen = vec![0, 1, 2];
    apply_repeat_penalty(&mut logits, &seen, 0.5);
    assert!(
        (logits[0] - 8.0).abs() < 1e-6,
        "positive logit at 0 must be boosted to 8.0; got {}",
        logits[0]
    );
    assert!(
        (logits[1] - -2.0).abs() < 1e-6,
        "negative logit at 1 must be boosted to -2.0; got {}",
        logits[1]
    );
    assert!(logits[2].abs() < 1e-6, "zero logit at 2 stays zero");
}

// `apply_presence_penalty` tests (P28 v0.3 wire-type follow-up —
// presence_penalty engine wire-through).
//
// Presence penalty is the OpenAI `presence_penalty` semantic: an
// additive bias subtracted from the logit of every *distinct* token
// already seen in the sequence, regardless of how many times it
// appeared. Positive values discourage repetition (encourage new
// topics); negative values encourage repetition. See the doc-comment
// on `apply_presence_penalty` for the difference from
// `apply_repeat_penalty`.

#[test]
fn test_presence_penalty_basic() {
    // With presence_penalty=0.5 on seen=[1], the logit at 1 is
    // reduced by 0.5 (from 0.5 to 0.0). Logits at positions 0 and 2
    // are untouched.
    let mut logits = vec![0.5, 0.5, 0.5];
    let seen = vec![1];
    apply_presence_penalty(&mut logits, &seen, 0.5);
    assert!((logits[0] - 0.5).abs() < 1e-6, "logit at 0 untouched");
    assert!((logits[1] - 0.0).abs() < 1e-6, "logit at 1 reduced by 0.5");
    assert!((logits[2] - 0.5).abs() < 1e-6, "logit at 2 untouched");
}

#[test]
fn test_presence_penalty_dedupes_seen_tokens() {
    // Presence penalty is presence-style, not frequency-style: the
    // penalty is subtracted ONCE per *distinct* id even when the id
    // appears multiple times in `seen`. Here token 1 appears 3 times
    // in `seen`, but the logit at 1 should only be reduced by 0.5
    // (not by 1.5). This pins the presence/frequency semantic
    // difference from `apply_repeat_penalty`.
    let mut logits = vec![0.5, 0.5, 0.5];
    let seen = vec![1, 1, 1];
    apply_presence_penalty(&mut logits, &seen, 0.5);
    assert!(
        (logits[1] - 0.0).abs() < 1e-6,
        "presence_penalty must subtract 0.5 once per distinct id, got {}",
        logits[1]
    );
}

#[test]
fn test_presence_penalty_no_effect_at_zero() {
    // 0.0 is the no-op default; logits must be untouched.
    let mut logits = vec![0.5, 0.5];
    let seen = vec![0];
    apply_presence_penalty(&mut logits, &seen, 0.0);
    assert!((logits[0] - 0.5).abs() < 1e-6);
    assert!((logits[1] - 0.5).abs() < 1e-6);
}

#[test]
fn test_presence_penalty_empty_seen_is_noop() {
    // An empty `seen` slice means no tokens to penalize; logits must
    // be untouched even when penalty != 0.0.
    let mut logits = vec![0.5, 0.5, 0.5];
    let seen: Vec<TokenId> = vec![];
    apply_presence_penalty(&mut logits, &seen, 1.0);
    assert!((logits[0] - 0.5).abs() < 1e-6);
    assert!((logits[1] - 0.5).abs() < 1e-6);
    assert!((logits[2] - 0.5).abs() < 1e-6);
}

#[test]
fn test_presence_penalty_negative_encourages_repetition() {
    // Negative presence_penalty *encourages* repetition by RAISING
    // the logits of seen tokens (subtracting a negative is the same
    // as adding). With penalty=-0.5 on seen=[1], the logit at 1 is
    // increased from 0.5 to 1.0.
    let mut logits = vec![0.5, 0.5, 0.5];
    let seen = vec![1];
    apply_presence_penalty(&mut logits, &seen, -0.5);
    assert!((logits[0] - 0.5).abs() < 1e-6, "logit at 0 untouched");
    assert!((logits[1] - 1.0).abs() < 1e-6, "logit at 1 raised by 0.5");
    assert!((logits[2] - 0.5).abs() < 1e-6, "logit at 2 untouched");
}

#[test]
fn test_presence_penalty_out_of_range_token_is_ignored() {
    // A seen token id that exceeds the vocab size must be silently
    // ignored (no panic, no out-of-bounds write). Mirrors the
    // `apply_repeat_penalty` contract.
    let mut logits = vec![0.5, 0.5];
    let seen = vec![10u32]; // out of range for vocab_size=2 (TokenId is u32)
    apply_presence_penalty(&mut logits, &seen, 0.5);
    assert!((logits[0] - 0.5).abs() < 1e-6);
    assert!((logits[1] - 0.5).abs() < 1e-6);
}

#[test]
fn test_presence_penalty_combined_with_repeat_penalty() {
    // When both penalties are active on the same seen token, the
    // combined effect should be: logit /= repeat_penalty
    // (frequency-style) then logit -= presence_penalty (presence-
    // style). With logits[1] = 0.5, repeat_penalty=2.0,
    // presence_penalty=0.1, the expected final logit at 1 is
    // 0.5/2.0 - 0.1 = 0.15.
    //
    // This pins the ordering in `sample_one_with_params`:
    // repeat_penalty first, then presence_penalty.
    let mut logits = vec![0.5, 0.5, 0.5];
    let seen = vec![1];
    apply_repeat_penalty(&mut logits, &seen, 2.0);
    apply_presence_penalty(&mut logits, &seen, 0.1);
    assert!((logits[0] - 0.5).abs() < 1e-6);
    assert!(
        (logits[1] - 0.15).abs() < 1e-6,
        "expected logit[1] = 0.5/2.0 - 0.1 = 0.15, got {}",
        logits[1]
    );
    assert!((logits[2] - 0.5).abs() < 1e-6);
}

#[test]
fn test_top_k_only_top_k_selected() {
    let logits = vec![0.1, 0.9, 0.3, 0.05, 0.05];
    let result = top_k_sample(&logits, 2, 0.5);
    assert!(result == 1 || result == 2);
}

#[test]
fn test_top_k_zero_no_effect() {
    let logits = vec![0.1, 0.9, 0.3];
    let result = top_k_sample(&logits, 0, 0.5);
    assert_eq!(result, 1);
}

#[test]
fn test_sample_batch_with_params_matches_scalar_api() {
    // ARCH-02 regression: the per-sequence params API must produce the
    // same answers as the legacy scalar API for the equivalent input.
    let logits = vec![vec![0.1, 0.9, 0.3], vec![0.8, 0.1, 0.1]];
    let seen = vec![vec![], vec![]];

    let params = vec![
        SamplingParams::builder()
            .with_temperature(0.0)
            .with_top_p(1.0)
            .with_top_k(0)
            .with_repeat_penalty(1.0)
            .build(),
        SamplingParams::builder()
            .with_temperature(0.0)
            .with_top_p(1.0)
            .with_top_k(0)
            .with_repeat_penalty(1.0)
            .build(),
    ];

    let with_params = sample_batch_with_params(&logits, &params, &seen);
    let scalar = sample_batch(&logits, 0.0, 1.0, 0, 1.0, &seen);
    // `sample_batch` still returns `Vec<TokenId>` (legacy API); map
    // both sides through `.token` so we can compare on equal footing.
    let with_params_tokens: Vec<TokenId> = with_params.iter().map(|s| s.token).collect();
    assert_eq!(with_params_tokens, scalar);
}

#[test]
fn test_sample_batch_with_params_greedy_default() {
    // Default `SamplingParams::default()` is `temperature = 0` →
    // greedy argmax. Mirrors the legacy `sample_batch` baseline.
    let logits = vec![vec![0.1, 0.9, 0.3], vec![0.8, 0.1, 0.1]];
    let seen = vec![vec![], vec![]];
    let params = vec![SamplingParams::default(); 2];
    let tokens = sample_batch_with_params(&logits, &params, &seen);
    let token_ids: Vec<TokenId> = tokens.iter().map(|s| s.token).collect();
    assert_eq!(token_ids, vec![1, 0]);
}

#[test]
fn test_sample_batch_with_params_empty_params_falls_back_to_greedy() {
    // The `Batch.sampling_params` doc contract (vllm_traits::types):
    // "an empty sampling_params (e.g. on a synthetic test batch) is
    // equivalent to greedy decoding." Pre-fix `sample_batch_with_params`
    // iterated `logits_list.zip(params_list).zip(seen_tokens)`, so an
    // empty params slice truncated the result to ZERO sampled tokens
    // even though real per-sequence logits were present — the engine
    // then emitted nothing for the batch and every sequence stalled
    // (no advance, KV blocks pinned) instead of decoding greedily.
    let logits = vec![vec![0.1, 0.9, 0.3], vec![0.8, 0.1, 0.1]];
    let seen = vec![vec![], vec![]];
    let empty_params: Vec<SamplingParams> = vec![];
    let tokens = sample_batch_with_params(&logits, &empty_params, &seen);
    let token_ids: Vec<TokenId> = tokens.iter().map(|s| s.token).collect();
    // Greedy per sequence, mirroring `test_sample_batch_with_params_greedy_default`.
    assert_eq!(token_ids, vec![1, 0]);
}

#[test]
fn test_sample_batch_with_params_per_sequence_divergence() {
    // Two sequences with different `top_k` must restrict independently.
    let logits = vec![vec![0.1, 0.9, 0.3, 0.05, 0.05]; 2];
    let seen = vec![vec![]; 2];
    let params = vec![
        SamplingParams::builder()
            .with_temperature(0.0)
            .with_top_k(1)
            .build(),
        SamplingParams::builder()
            .with_temperature(0.0)
            .with_top_k(5)
            .build(),
    ];
    let tokens = sample_batch_with_params(&logits, &params, &seen);
    // top_k=1 forces argmax (1); top_k=5 also reaches argmax here, but
    // the contract under test is that both code paths produce a valid
    // token in [0, vocab). The deterministic case still equals argmax.
    assert_eq!(tokens.len(), 2);
    assert_eq!(tokens[0].token, 1);
    assert_eq!(tokens[1].token, 1);
}

#[test]
fn test_sample_batch_with_params_repeat_penalty_per_sequence() {
    // Per-sequence repeat_penalty: seq A penalises token 1, seq B does
    // not. With flat logits and T=0, seq A's argmax should not be 1.
    let logits = vec![vec![0.5, 0.5, 0.5]; 2];
    let seen = vec![vec![1], vec![1]];
    let params = vec![
        SamplingParams::builder()
            .with_temperature(0.0)
            .with_repeat_penalty(2.0)
            .build(),
        SamplingParams::builder()
            .with_temperature(0.0)
            .with_repeat_penalty(1.0)
            .build(),
    ];
    let tokens = sample_batch_with_params(&logits, &params, &seen);
    assert_ne!(
        tokens[0].token, 1,
        "penalty should suppress token 1 for seq 0"
    );
    assert_eq!(tokens[1].token, 0, "no penalty → first argmax wins");
}

#[test]
fn test_sample_one_with_params_empty_seen_is_no_op() {
    // Empty `seen` → repeat_penalty branch skipped, no panic.
    let logits = vec![0.0, 5.0, 0.0];
    let params = SamplingParams::builder()
        .with_temperature(0.0)
        .with_repeat_penalty(2.0)
        .build();
    let token = sample_one_with_params(&logits, &params, &[]);
    assert_eq!(token.token, 1);
}

#[test]
fn test_sample_one_with_params_empty_logits_with_top_k_no_panic() {
    // Empty logits + top_k > 0 must not panic: the top-k mask block
    // is skipped and the pipeline degrades to greedy (token 0).
    let params = SamplingParams::builder().with_top_k(5).build();
    let sampled = sample_one_with_params(&[], &params, &[]);
    assert_eq!(sampled.token, 0);
}

#[test]
fn test_sample_one_with_params_empty_logits_with_top_logprobs_no_panic() {
    // Empty logits + top_logprobs > 0 must not panic: with no
    // distribution to rank, `top_logprobs` is empty, `logprob` is
    // `-inf`, and the token degrades to greedy (token 0).
    let params = SamplingParams::builder().with_top_logprobs(3).build();
    let sampled = sample_one_with_params(&[], &params, &[]);
    assert_eq!(sampled.token, 0);
    assert!(sampled.top_logprobs.is_empty());
    assert!(
        sampled.logprob.is_infinite() && sampled.logprob.is_sign_negative(),
        "empty logits must yield logprob = -inf"
    );
}

#[test]
fn test_sample_batch_empty_logits_row_with_top_k_no_panic() {
    // A short/empty logits row (e.g. malformed model output) plus
    // top_k > 0 must not panic; the row degrades to greedy (token 0).
    let logits = vec![vec![]];
    let seen = vec![vec![]];
    let result = sample_batch(&logits, 0.8, 1.0, 5, 1.0, &seen);
    assert_eq!(result, vec![0]);
}

// `apply_logit_bias` tests (P30 v0.3 wire-type follow-up —
// `logit_bias` engine wire-through).
//
// Logit bias is the OpenAI `logit_bias` semantic: an additive bias
// added to the logit of specific token IDs (keyed by token ID). Per
// OpenAI spec the value range is [-100, 100]; out-of-vocab token IDs
// are silently ignored (the engine doesn't have a vocabulary
// bound here, so any token ID >= logits.len() is ignored). The bias
// is additive, so positive values *increase* the probability of the
// biased tokens and negative values *decrease* it — opposite of the
// "presence_penalty" semantic (which subtracts from seen-token
// logits). See the doc-comment on `apply_logit_bias` for the
// implementation.

#[test]
fn test_logit_bias_basic() {
    // With bias = {1: 1.0}, the logit at 1 is increased by 1.0
    // (from 0.5 to 1.5). Logits at positions 0 and 2 are untouched.
    let mut logits = vec![0.5, 0.5, 0.5];
    let bias: std::collections::HashMap<TokenId, f32> = std::collections::HashMap::from([(1, 1.0)]);
    apply_logit_bias(&mut logits, &bias);
    assert!((logits[0] - 0.5).abs() < 1e-6, "logit at 0 untouched");
    assert!(
        (logits[1] - 1.5).abs() < 1e-6,
        "logit at 1 increased by 1.0 (0.5 → 1.5); got {}",
        logits[1]
    );
    assert!((logits[2] - 0.5).abs() < 1e-6, "logit at 2 untouched");
}

#[test]
fn test_logit_bias_negative_decreases_logit() {
    // Negative bias = suppress the token (matches OpenAI spec:
    // -100 suppresses a token completely, +100 guarantees sampling).
    let mut logits = vec![0.5, 0.5, 0.5];
    let bias: std::collections::HashMap<TokenId, f32> =
        std::collections::HashMap::from([(1, -0.5)]);
    apply_logit_bias(&mut logits, &bias);
    assert!(
        (logits[1] - 0.0).abs() < 1e-6,
        "logit at 1 reduced by 0.5 (0.5 → 0.0); got {}",
        logits[1]
    );
}

#[test]
fn test_logit_bias_empty_map_is_noop() {
    // Empty map → no change. Pins the early-return branch.
    let mut logits = vec![0.5, 0.5, 0.5];
    let bias: std::collections::HashMap<TokenId, f32> = std::collections::HashMap::new();
    apply_logit_bias(&mut logits, &bias);
    assert!((logits[0] - 0.5).abs() < 1e-6);
    assert!((logits[1] - 0.5).abs() < 1e-6);
    assert!((logits[2] - 0.5).abs() < 1e-6);
}

#[test]
fn test_logit_bias_empty_logits_is_noop() {
    // Empty logits + non-empty bias → no panic, no effect (no token
    // IDs to bias).
    let mut logits: Vec<f32> = vec![];
    let bias: std::collections::HashMap<TokenId, f32> =
        std::collections::HashMap::from([(0, 1.0), (1, -1.0)]);
    apply_logit_bias(&mut logits, &bias);
    assert!(logits.is_empty());
}

#[test]
fn test_logit_bias_out_of_range_keys_silently_ignored() {
    // OpenAI spec: out-of-vocab token IDs are silently ignored.
    // Token ID 99 is far beyond the logits length (3); it should
    // have no effect. This matches the engine's actual runtime
    // behavior (token IDs beyond vocab_size are also silently
    // ignored during sampling).
    let mut logits = vec![0.5, 0.5, 0.5];
    let bias: std::collections::HashMap<TokenId, f32> =
        std::collections::HashMap::from([(99, 100.0)]);
    apply_logit_bias(&mut logits, &bias);
    assert!((logits[0] - 0.5).abs() < 1e-6);
    assert!((logits[1] - 0.5).abs() < 1e-6);
    assert!((logits[2] - 0.5).abs() < 1e-6);
}

#[test]
fn test_logit_bias_multiple_tokens_biased_independently() {
    // Multiple biased tokens each get their own delta. The map
    // iteration order is non-deterministic (HashMap) but the
    // *result* is deterministic because each bias is additive and
    // independent.
    let mut logits = vec![0.5, 0.5, 0.5, 0.5];
    let bias: std::collections::HashMap<TokenId, f32> =
        std::collections::HashMap::from([(0, 1.0), (2, -0.5), (3, 2.0)]);
    apply_logit_bias(&mut logits, &bias);
    assert!(
        (logits[0] - 1.5).abs() < 1e-6,
        "logit at 0 increased by 1.0; got {}",
        logits[0]
    );
    assert!(
        (logits[1] - 0.5).abs() < 1e-6,
        "logit at 1 untouched; got {}",
        logits[1]
    );
    assert!(
        (logits[2] - 0.0).abs() < 1e-6,
        "logit at 2 reduced by 0.5; got {}",
        logits[2]
    );
    assert!(
        (logits[3] - 2.5).abs() < 1e-6,
        "logit at 3 increased by 2.0; got {}",
        logits[3]
    );
}

#[test]
fn test_logit_bias_does_not_affect_greedy_when_zero() {
    // Zero bias should be a no-op even though the map is non-empty
    // (covers the "bias map present but empty effect" case where a
    // caller serialised an empty entry).
    let mut logits = vec![0.5, 0.5, 0.5];
    let bias: std::collections::HashMap<TokenId, f32> = std::collections::HashMap::from([(1, 0.0)]);
    apply_logit_bias(&mut logits, &bias);
    assert!((logits[0] - 0.5).abs() < 1e-6);
    assert!((logits[1] - 0.5).abs() < 1e-6);
    assert!((logits[2] - 0.5).abs() < 1e-6);
}

#[test]
fn test_sample_one_with_params_logit_bias_changes_argmax() {
    // Integration test: logit_bias threaded through
    // `sample_one_with_params` flips the argmax. With greedy
    // sampling, `logits = [0.5, 0.5, 0.5]` is a tie and the fold
    // initialises at index 0 (first wins). Applying bias = {2: 1.0}
    // makes the logit at 2 = 1.5 — the new argmax is 2.
    let logits = vec![0.5, 0.5, 0.5];
    let bias: std::collections::HashMap<TokenId, f32> = std::collections::HashMap::from([(2, 1.0)]);
    let params = SamplingParams::builder()
        .with_temperature(0.0)
        .with_logit_bias(Some(bias))
        .build();
    let token = sample_one_with_params(&logits, &params, &[]);
    assert_eq!(
        token.token, 2,
        "logit_bias on token 2 must flip greedy argmax from 0 → 2"
    );
}

#[test]
fn test_sample_one_with_params_logit_bias_none_is_noop() {
    // No logit_bias set → no bias applied. Pins the `None` branch
    // of the dispatch (matches `presence_penalty.abs() > f32::EPSILON`
    // pattern but for the Option<HashMap> variant).
    let logits = vec![0.5, 0.5, 0.5];
    let params = SamplingParams::builder().with_temperature(0.0).build();
    let token = sample_one_with_params(&logits, &params, &[]);
    assert_eq!(token.token, 0, "no logit_bias → first argmax (0) wins");
}

#[test]
fn test_sample_one_with_params_logit_bias_empty_map_is_noop() {
    // Empty map (Some but no entries) → no bias applied. Pins the
    // empty-map early-return branch inside `apply_logit_bias`.
    let logits = vec![0.5, 0.5, 0.5];
    let bias: std::collections::HashMap<TokenId, f32> = std::collections::HashMap::new();
    let params = SamplingParams::builder()
        .with_temperature(0.0)
        .with_logit_bias(Some(bias))
        .build();
    let token = sample_one_with_params(&logits, &params, &[]);
    assert_eq!(token.token, 0, "empty map → first argmax (0) wins");
}

#[test]
fn test_logit_bias_combined_with_repeat_and_presence_penalties() {
    // Verify the full pipeline order: repeat_penalty → presence_penalty
    // → logit_bias → temperature → top-k → top-p → greedy. With
    // greedy sampling and logit_bias flipping the argmax after the
    // penalties, the final token should reflect the bias.
    //
    // Sequence: logits = [1.0, 0.5, 0.5], seen = [0].
    //   - repeat_penalty=2.0: logit at 0 = 1.0/2 = 0.5 → tie at 0/1/2.
    //   - presence_penalty=0.0: no-op.
    //   - logit_bias={2: 1.0}: logit at 2 = 0.5 + 1.0 = 1.5 → argmax = 2.
    let logits = vec![1.0, 0.5, 0.5];
    let bias: std::collections::HashMap<TokenId, f32> = std::collections::HashMap::from([(2, 1.0)]);
    let params = SamplingParams::builder()
        .with_temperature(0.0)
        .with_repeat_penalty(2.0)
        .with_presence_penalty(0.0)
        .with_logit_bias(Some(bias))
        .build();
    let token = sample_one_with_params(&logits, &params, &[0]);
    assert_eq!(
        token.token, 2,
        "combined penalties + logit_bias must yield argmax = 2"
    );
}

// ============================================================================
// P34: seed RNG seeding tests
// ============================================================================
//
// The `seed` field on `SamplingParams` (P34 v0.2 wire-type follow-up engine
// wire-through) gives the OpenAI-spec guarantee that the same seed + same
// model + same prompt produces the same output. These tests pin that
// contract at the sampler layer:
//
// 1. Determinism: same seed + same logits/params/seen → same token.
// 2. Independence: different seeds (with otherwise identical inputs) →
//    usually different tokens (sampled, but the sampler is per-call so
//    the seeded RNG draws different random thresholds).
// 3. `seed = None` falls back to thread RNG (no behaviour change vs.
//    the pre-P34 default).
// 4. `seed = Some(0)` is a valid seed (NOT treated as "unset"; matches
//    OpenAI's "any integer" contract).
// 5. Per-sequence independence in `sample_batch_with_params` (each
//    sequence uses its own RNG seeded from its own `params.seed`).
// 6. Greedy / `temperature = 0` paths bypass the RNG entirely (the
//    seed has no observable effect — same argmax regardless of seed).

/// Multi-logit distribution where greedy argmax is non-trivial and
/// temperature sampling actually exercises the RNG (otherwise the test
/// is degenerate — `temperature = 0` skips the RNG and any seed
/// looks identical).
const SEED_TEST_LOGITS: &[f32] = &[1.0, 2.0, 3.0, 4.0, 5.0];

#[test]
fn test_seed_determinism_same_seed_same_result() {
    // Two SamplingParams with identical seed must produce identical
    // tokens across two independent sample_one_with_params calls.
    let logits = SEED_TEST_LOGITS;
    let seen: &[TokenId] = &[];
    let params_a = SamplingParams::builder()
        .with_temperature(1.0)
        .with_seed(42)
        .build();
    let params_b = SamplingParams::builder()
        .with_temperature(1.0)
        .with_seed(42)
        .build();
    let token_a = sample_one_with_params(logits, &params_a, seen);
    let token_b = sample_one_with_params(logits, &params_b, seen);
    assert_eq!(
        token_a, token_b,
        "same seed must produce the same sampled token (got {token_a:?} vs {token_b:?})"
    );
}

#[test]
fn test_seed_zero_is_valid_seed() {
    // OpenAI spec accepts any i64; the i64 → u64 cast wraps negatives
    // but never produces 0 for non-zero inputs. seed = 0 is a valid
    // seed and must NOT be conflated with seed = None.
    let logits = SEED_TEST_LOGITS;
    let seen: &[TokenId] = &[];
    let params_zero = SamplingParams::builder()
        .with_temperature(1.0)
        .with_seed(0)
        .build();
    let params_nonzero = SamplingParams::builder()
        .with_temperature(1.0)
        .with_seed(1)
        .build();
    let token_zero = sample_one_with_params(logits, &params_zero, seen);
    let token_nonzero = sample_one_with_params(logits, &params_nonzero, seen);
    // Both calls are deterministic; the two values are usually
    // different (because the seeded RNG draws different thresholds
    // from seed 0 vs seed 1). The contract is: seed=0 is honoured.
    // We don't assert inequality (sampling could in principle pick
    // the same token by chance) but we do assert both are valid
    // vocab indices.
    assert!(token_zero.token < logits.len() as TokenId);
    assert!(token_nonzero.token < logits.len() as TokenId);
}

/// `random_f32` must produce values in `[0, 1)`.
///
/// Every temperature/top-p draw multiplies or compares against this value, so a
/// constant outside the unit interval silently breaks unseeded sampling for
/// every request. `cargo mutants` replaced the whole body with `-1.0` and the
/// suite stayed green: the existing seed test only checks that a `seed = None`
/// request still produces a token, which a constant also does.
#[test]
fn random_f32_stays_in_the_unit_interval() {
    let mut seen_distinct = std::collections::HashSet::new();
    for _ in 0..512 {
        let value = random_f32();
        assert!(
            (0.0..1.0).contains(&value),
            "random_f32 must stay in [0, 1); got {value}"
        );
        // Two values in a 512-draw sample colliding would indicate a constant.
        seen_distinct.insert(value.to_bits());
    }
    assert!(
        seen_distinct.len() > 1,
        "random_f32 must vary between draws, not return a constant"
    );
}

/// `splitmix64` must be the documented bijection, not a re-rolled hash.
///
/// The seeded stream is defined by these exact constants and shifts; changing
/// any of them changes every seeded output. `cargo mutants` flipped `>>` to
/// `<<` and `^` to `&` and nothing failed, so the reproducibility contract had
/// no regression test. The values below are computed from the published
/// SplitMix64 finalizer.
#[test]
fn splitmix64_matches_the_reference_finalizer() {
    // Reference values computed with the constants in the implementation.
    assert_eq!(splitmix64(0), 0xE220_A839_7B1D_CDAF);
    assert_eq!(splitmix64(1), 0x910A_2DEC_8902_5CC1);
    assert_eq!(splitmix64(u64::MAX), 0xE4D9_7177_1B65_2C20);
}

#[test]
fn test_seed_none_falls_back_to_thread_rng() {
    // seed = None must NOT crash and must NOT silently use a stale
    // seeded RNG. Two calls with seed = None may produce different
    // tokens (because the thread RNG advances), so we only assert
    // that both calls return valid vocab indices — the test would
    // fail loudly if the None branch were broken (e.g. panic or
    // off-by-one).
    let logits = SEED_TEST_LOGITS;
    let seen: &[TokenId] = &[];
    let params = SamplingParams::builder()
        .with_temperature(1.0)
        .with_seed_none()
        .build();
    let token = sample_one_with_params(logits, &params, seen);
    assert!(
        token.token < logits.len() as TokenId,
        "seed = None must produce a valid vocab index (got {})",
        token.token
    );
}

#[test]
fn test_seed_different_seeds_diverge_in_distribution() {
    // With the same logits + temperature + seen but different seeds,
    // the sampler must draw INDEPENDENT random thresholds (each seed
    // derives its own stream). The load-bearing guarantee is at the
    // threshold level: distinct seeds ⇒ distinct draws. Asserting
    // distinct *tokens* instead is fragile — SEED_TEST_LOGITS at T=1.0
    // is peaked toward token 4, so two independent draws can both land
    // on it (this test previously asserted token inequality
    // and broke when the per-step stream shifted the deterministic
    // draws). If the seeded RNG were broken (shared across seeds), the
    // thresholds below would coincide and this would catch it.
    let threshold_a = sample_random_threshold(Some(42), 0);
    let threshold_b = sample_random_threshold(Some(99), 0);
    assert_ne!(
        threshold_a.to_bits(),
        threshold_b.to_bits(),
        "different seeds must draw independent thresholds (got {threshold_a} \
         for both — RNG is not being seeded correctly)"
    );

    // The sampled tokens are still valid vocab indices and each seed is
    // reproducible call-to-call.
    let logits = SEED_TEST_LOGITS;
    let seen: &[TokenId] = &[];
    let params_a = SamplingParams::builder()
        .with_temperature(1.0)
        .with_seed(42)
        .build();
    let token_a = sample_one_with_params(logits, &params_a, seen);
    assert!(token_a.token < logits.len() as TokenId);
    assert_eq!(
        token_a,
        sample_one_with_params(logits, &params_a, seen),
        "seed 42 must be reproducible"
    );
}

#[test]
fn test_seed_greedy_path_bypasses_rng() {
    // temperature = 0 forces the greedy path which doesn't read
    // from any RNG. The seed field is therefore irrelevant and
    // the result is the deterministic argmax regardless of seed.
    let logits = SEED_TEST_LOGITS;
    let seen: &[TokenId] = &[];
    let params_a = SamplingParams::builder()
        .with_temperature(0.0)
        .with_seed(42)
        .build();
    let params_b = SamplingParams::builder()
        .with_temperature(0.0)
        .with_seed(99)
        .build();
    let token_a = sample_one_with_params(logits, &params_a, seen);
    let token_b = sample_one_with_params(logits, &params_b, seen);
    assert_eq!(
        token_a, token_b,
        "greedy path (T=0) must yield argmax regardless of seed"
    );
    assert_eq!(
        token_a.token, 4,
        "greedy path must yield argmax = 4 for SEED_TEST_LOGITS"
    );
}

#[test]
fn test_temperature_zero_forces_greedy_across_top_p() {
    // `SamplingParams::temperature` doc: "0.0 selects greedy argmax"
    // (unconditional), and the seed doc promises that `temperature = 0`
    // paths "bypass the RNG entirely (deterministic argmax)". Pre-fix the
    // `top_p < 1.0` branch was checked BEFORE temperature, so a request
    // with `temperature = 0` + `top_p < 1` (a common client combination)
    // ran `top_p_sample` on the raw logits with a random seeded threshold:
    // non-greedy AND non-reproducible across seeds, contradicting the
    // documented contract.
    //
    // `[10.0, 10.0, ...]` (a two-way tie for argmax): greedy argmax =
    // token 0 (first-wins). The `top_p = 0.5` nucleus on the raw softmax
    // is `{0, 1}` — P(token₀) ≈ 0.499 ≤ 0.5, so the next token joins the
    // cutoff — and a top-p draw with a seeded threshold ~50% of the time
    // returns token 1. Pre-fix, half the seeds below violate the contract;
    // post-fix every seed yields the argmax.
    let logits = vec![10.0f32, 10.0, 0.0, 0.0, 0.0];
    for seed in (0..256u64).map(|i| i * 37 + 12345) {
        let params = SamplingParams::builder()
            .with_temperature(0.0)
            .with_top_p(0.5)
            .with_seed(seed)
            .build();
        let out = sample_one_with_params(&logits, &params, &[]);
        assert_eq!(
            out.token, 0,
            "temperature=0 must force greedy argmax regardless of top_p (seed {seed})"
        );
    }
}

#[test]
fn test_seed_per_sequence_independence_in_batch() {
    // `sample_batch_with_params` carries per-sequence SamplingParams.
    // Each sequence must use its own RNG seeded from its own
    // params.seed — NOT a shared RNG across sequences.
    let logits_list = vec![SEED_TEST_LOGITS.to_vec(); 3];
    let seen_list = vec![vec![]; 3];
    let params_list = vec![
        SamplingParams::builder()
            .with_temperature(1.0)
            .with_seed(42)
            .build(),
        SamplingParams::builder()
            .with_temperature(1.0)
            .with_seed(42)
            .build(),
        SamplingParams::builder()
            .with_temperature(1.0)
            .with_seed(99)
            .build(),
    ];
    let tokens = sample_batch_with_params(&logits_list, &params_list, &seen_list);
    assert_eq!(tokens.len(), 3);
    // Sequences 0 and 1 share the same seed → must produce the same
    // token (reproducibility; deterministic).
    assert_eq!(
        tokens[0], tokens[1],
        "sequences with the same seed must produce the same token \
         (got {0:?} vs {1:?})",
        tokens[0], tokens[1]
    );
    // Independence is pinned at the threshold level: seed 42 and seed
    // 99 draw different thresholds, so their streams are independent.
    // Asserting different *tokens* would be fragile on the peaked
    // SEED_TEST_LOGITS distribution.
    assert_ne!(
        sample_random_threshold(Some(42), 0).to_bits(),
        sample_random_threshold(Some(99), 0).to_bits(),
        "sequences with different seeds must draw independent thresholds \
         (RNG state is shared across sequences)"
    );
}

#[test]
fn test_seed_top_p_path_uses_seeded_rng() {
    // The seeded RNG must drive the top_p sampler too (not just
    // temperature_sample). With T=1.0 and top_p < 1.0, the same
    // seed must produce the same truncated-nucleus sample.
    let logits = SEED_TEST_LOGITS;
    let seen: &[TokenId] = &[];
    let params_a = SamplingParams::builder()
        .with_temperature(1.0)
        .with_top_p(0.5)
        .with_seed(42)
        .build();
    let params_b = SamplingParams::builder()
        .with_temperature(1.0)
        .with_top_p(0.5)
        .with_seed(42)
        .build();
    let token_a = sample_one_with_params(logits, &params_a, seen);
    let token_b = sample_one_with_params(logits, &params_b, seen);
    assert_eq!(
        token_a, token_b,
        "top_p path must honour seed deterministically \
         (got {token_a:?} vs {token_b:?})"
    );
}

// =============================================================================
// — per-step seeded stream regression tests.
//
// Pre-fix, `random_f32_seeded` re-seeded a fresh `StdRng` with the bare
// request seed on EVERY decode step and read its first value, so a
// seeded generation drew the SAME threshold at every step (per-step
// quantile lock): the nucleus / temperature cutoff froze and cross-step
// independence collapsed. The fix derives a per-step seed from
// `(seed, step)`, so the stream advances each step while staying
// reproducible. These tests pin both halves of that contract directly
// against the (private) threshold helpers, which the child-module
// `use super::*` brings into scope.
// =============================================================================

#[test]
fn test_seeded_stream_advances_across_steps() {
    // THE regression: distinct decode steps must draw distinct
    // thresholds for the same seed. Pre-fix every step returned the
    // identical value (the first draw of `seed_from_u64(seed)`).
    let thresholds: Vec<f32> = (0..8).map(|step| random_f32_seeded(42, step)).collect();
    let distinct: std::collections::HashSet<u32> = thresholds.iter().map(|t| t.to_bits()).collect();
    assert!(
        distinct.len() > 1,
        "a seeded stream must draw different thresholds per step; got a frozen value {thresholds:?} (per-step quantile lock)"
    );
    // The values are deterministic constants (fixed seed + fixed mixer),
    // so this stronger assertion is stable: a good mixer yields 8
    // distinct draws for 8 consecutive steps.
    assert_eq!(
        distinct.len(),
        8,
        "8 consecutive seeded steps should yield 8 distinct thresholds; got {thresholds:?}"
    );
}

#[test]
fn test_seeded_stream_is_reproducible_per_step() {
    // Reproducibility half of the contract: the same (seed, step)
    // always draws the same threshold, across independent calls.
    for step in 0..4 {
        assert_eq!(
            random_f32_seeded(42, step).to_bits(),
            random_f32_seeded(42, step).to_bits(),
            "step {step} must be deterministic for a fixed seed"
        );
    }
}

#[test]
fn test_seeded_stream_is_sensitive_to_seed_at_fixed_step() {
    // Different seeds must derive different streams at the same step
    // (per-sequence independence is preserved by the per-step mix).
    assert_ne!(
        random_f32_seeded(42, 3).to_bits(),
        random_f32_seeded(99, 3).to_bits(),
        "different seeds must diverge at the same step"
    );
}

#[test]
fn test_seeded_threshold_in_unit_interval_across_steps() {
    // Every per-step draw must be a valid `[0, 1)` threshold (the
    // top-p / temperature samplers assume this).
    for step in 0..16 {
        let t = random_f32_seeded(7, step);
        assert!(
            (0.0..1.0).contains(&t),
            "step {step} threshold {t} out of [0,1)"
        );
    }
}

#[test]
fn test_sample_random_threshold_step_drives_seeded_draw() {
    // The dispatch helper must thread `step` into the seeded path.
    assert_ne!(
        sample_random_threshold(Some(42), 0).to_bits(),
        sample_random_threshold(Some(42), 5).to_bits(),
        "seeded threshold must vary with the step index"
    );
    assert_eq!(
        sample_random_threshold(Some(42), 5).to_bits(),
        sample_random_threshold(Some(42), 5).to_bits(),
        "seeded threshold must be reproducible at a fixed step"
    );
}

#[test]
fn test_seeded_multistep_reproducible_via_public_api() {
    // End-to-end at the public sampler: a seeded draw at a NON-zero
    // step (non-empty seen-set) is reproducible call-to-call — the
    // step > 0 analogue of `test_seed_determinism_same_seed_same_result`,
    // proving the per-step derivation did not break determinism.
    let logits = SEED_TEST_LOGITS;
    let seen: &[TokenId] = &[10, 20, 30]; // step = 3
    let params = SamplingParams::builder()
        .with_temperature(1.0)
        .with_seed(42)
        .build();
    let a = sample_one_with_params(logits, &params, seen);
    let b = sample_one_with_params(logits, &params, seen);
    assert_eq!(
        a, b,
        "seeded sampling at step > 0 must be reproducible (got {a:?} vs {b:?})"
    );
}

// =============================================================================
// P38 v0.3 wire-type engine wire-through — engine wire-through:
// `matches_stop_sequences` helper tests. Pure-function unit tests; the
// helper is consumed by the engine's `step_regular` loop after every
// sampled token. Pin the contract end-to-end so future refactors
// (e.g. Aho-Corasick) trip the suite.
// =============================================================================

#[test]
fn test_matches_stop_sequences_no_match_returns_false() {
    // Stop = "xyz" tokens [99], generated = [1, 2, 3] — no suffix match.
    let stops: Vec<Vec<u32>> = vec![vec![99]];
    let generated: Vec<u32> = vec![1, 2, 3];
    assert!(!matches_stop_sequences(&generated, &stops));
}

#[test]
fn test_matches_stop_sequences_single_token_match_returns_true() {
    // Stop = [99] (1 token), generated = [1, 2, 99] — suffix matches.
    let stops: Vec<Vec<u32>> = vec![vec![99]];
    let generated: Vec<u32> = vec![1, 2, 99];
    assert!(matches_stop_sequences(&generated, &stops));
}

#[test]
fn test_matches_stop_sequences_multi_token_match_returns_true() {
    // Stop = [10, 20, 30] (3 tokens), generated ends with [10, 20, 30].
    let stops: Vec<Vec<u32>> = vec![vec![10, 20, 30]];
    let generated: Vec<u32> = vec![1, 2, 3, 10, 20, 30];
    assert!(matches_stop_sequences(&generated, &stops));
}

#[test]
fn test_matches_stop_sequences_partial_match_returns_false() {
    // Stop = [10, 20, 30], generated ends with [10, 20] — only 2 of 3
    // stop tokens match. Must NOT trigger (we require the FULL suffix).
    let stops: Vec<Vec<u32>> = vec![vec![10, 20, 30]];
    let generated: Vec<u32> = vec![1, 2, 10, 20];
    assert!(!matches_stop_sequences(&generated, &stops));
}

#[test]
fn test_matches_stop_sequences_first_match_wins() {
    // Multiple stops; the FIRST one whose suffix matches wins (helper
    // returns true on first hit, doesn't care which one).
    let stops: Vec<Vec<u32>> = vec![vec![99], vec![88]];
    let generated_a: Vec<u32> = vec![1, 2, 99];
    let generated_b: Vec<u32> = vec![1, 2, 88];
    assert!(matches_stop_sequences(&generated_a, &stops));
    assert!(matches_stop_sequences(&generated_b, &stops));
    // Neither matches: stop sequences don't appear as suffixes.
    let generated_c: Vec<u32> = vec![1, 2, 3];
    assert!(!matches_stop_sequences(&generated_c, &stops));
}

#[test]
fn test_matches_stop_sequences_empty_stops_returns_false() {
    // `stops = vec![]` → never matches (helper loops over zero elements).
    let stops: Vec<Vec<u32>> = vec![];
    let generated: Vec<u32> = vec![1, 2, 3];
    assert!(!matches_stop_sequences(&generated, &stops));
}

#[test]
fn test_matches_stop_sequences_stop_longer_than_generated_returns_false() {
    // Stop = [1, 2, 3, 4, 5], generated has only 3 tokens — cannot match.
    let stops: Vec<Vec<u32>> = vec![vec![1, 2, 3, 4, 5]];
    let generated: Vec<u32> = vec![1, 2, 3];
    assert!(!matches_stop_sequences(&generated, &stops));
}

#[test]
fn test_matches_stop_sequences_empty_generated_returns_false() {
    // generated_tokens = &[] — every non-empty stop is "longer than
    // generated", so the helper must return false for any non-empty
    // stops and trivially false for empty stops. Defensive: pins the
    // degenerate case where the engine might call with empty
    // generated (e.g. before any token is sampled in a fresh seq).
    let stops_nonempty: Vec<Vec<u32>> = vec![vec![1, 2, 3]];
    let stops_empty: Vec<Vec<u32>> = vec![];
    assert!(!matches_stop_sequences(&[], &stops_nonempty));
    assert!(!matches_stop_sequences(&[], &stops_empty));
}

/// Regression: temperature must be applied exactly once in the
/// `sample_one_with_params` pipeline. Pre-fix, the logits were divided
/// by temperature in the pre-scaling step AND again inside
/// `temperature_sample`, producing an effective temperature of T².
///
/// Strategy: with a fixed seed the random threshold is deterministic.
/// We pick logits and a temperature where single-scaling and
/// double-scaling disagree on the sampled token, then assert the
/// correct (single-scaled) outcome.
#[test]
fn test_temperature_applied_once_not_twice() {
    // logits = [0.0, 1.0], temperature = 0.5
    // Single-scale: [0.0, 2.0] → softmax ≈ [0.119, 0.881]
    // Double-scale: [0.0, 4.0] → softmax ≈ [0.018, 0.982]
    //
    // We need a seed whose random threshold falls in (0.018, 0.119)
    // so single-scaling picks token 0 but double-scaling picks token 1.
    // Brute-force a few seeds to find one that works.
    let logits = vec![0.0_f32, 1.0];

    // Compute the single-scaled softmax CDF at index 0:
    // exp(0)/(exp(0)+exp(2)) = 1/(1+e²) ≈ 0.1192
    let single_scale_p0 = 1.0 / (1.0 + 2.0_f32.exp());
    // Double-scaled CDF at index 0:
    // exp(0)/(exp(0)+exp(4)) = 1/(1+e⁴) ≈ 0.0180
    let double_scale_p0 = 1.0 / (1.0 + 4.0_f32.exp());

    // Alternative approach: verify statistically over many seeds.
    // With correct single-scaling, P(token=0) ≈ 0.119.
    // With double-scaling, P(token=0) ≈ 0.018.
    // Over 500 seeds, the count of token=0 should be closer to 0.119*500≈60
    // than to 0.018*500≈9.
    let n_trials = 500;
    let mut count_token_0 = 0;
    for seed in 0..n_trials {
        let params = SamplingParams::builder()
            .with_temperature(0.5)
            .with_top_p(1.0)
            .with_seed(seed)
            .build();
        let result = sample_one_with_params(&logits, &params, &[]);
        if result.token == 0 {
            count_token_0 += 1;
        }
    }
    let empirical_p0 = f64::from(count_token_0) / n_trials as f64;
    let expected_p0 = f64::from(single_scale_p0);
    let buggy_p0 = f64::from(double_scale_p0);

    // The empirical rate should be within 0.08 of the correct value
    // (generous margin for 500 trials) and far from the buggy value.
    assert!(
        (empirical_p0 - expected_p0).abs() < 0.08,
        "empirical P(token=0) = {empirical_p0:.4}, expected ≈ {expected_p0:.4} (single-scale), \
         buggy would be ≈ {buggy_p0:.4}; got {count_token_0} hits of token 0 in {n_trials} trials",
    );
    // Sanity: the gap between correct and buggy is large enough to detect.
    assert!(
        (expected_p0 - buggy_p0).abs() > 0.05,
        "test design: gap between correct and buggy must be detectable"
    );
}

/// `top_p_sample` must sample in proportion to the nucleus-renormalized
/// probabilities.
///
/// Sweeping `random_threshold` over `(0, 1)` makes the sampler's decisions
/// deterministic, so the empirical token frequency IS the distribution. Any
/// arithmetic slip — the `x - max_val` stabiliser, the `x / sum` normalisation,
/// the `cumsum += p` accumulation, or the final `p /= total` renormalisation —
/// moves those frequencies. `cargo mutants` found nine variants of exactly
/// those lines survive the suite, because every existing test only checks that
/// the sampled token is *among* the allowed set.
#[test]
fn top_p_sample_follows_the_renormalized_distribution() {
    // logits [1, 0, -1] with top_p = 0.7 keep the two largest tokens; their
    // renormalized probabilities are e/(e+1) and 1/(e+1).
    let logits = vec![1.0_f32, 0.0, -1.0];
    let denom = 1.0_f32.exp() + 1.0;
    let expected = [1.0_f32.exp() / denom, 1.0 / denom];

    const STEPS: usize = 1000;
    let mut counts = [0usize; 3];
    for step in 0..STEPS {
        #[allow(clippy::cast_precision_loss)]
        let threshold = (step as f32 + 0.5) / STEPS as f32;
        let sampled = top_p_sample(&logits, 0.7, threshold);
        counts[usize::try_from(sampled).expect("small token id")] += 1;
    }

    assert_eq!(counts[2], 0, "token 2 (logit -1.0) is outside the nucleus");
    #[allow(clippy::cast_precision_loss)]
    let total = STEPS as f32;
    for (token, want) in expected.iter().enumerate() {
        let got = counts[token] as f32 / total;
        assert!(
            (got - want).abs() < 0.03,
            "token {token} must take {want:.3} of the draws; got {got:.3} \
             (counts {counts:?})"
        );
    }
}

/// `top_p >= 1.0` alone must route to greedy.
///
/// The guard reads `top_p >= 1.0 || logits.is_empty()`. `cargo mutants`
/// replaced the `||` with `&&`, which sends a `top_p = 1.0` request down the
/// nucleus path instead of returning the argmax — so a request that asked for
/// "no nucleus truncation" stops being greedy. The test asserts the argmax on
/// every threshold, which the p-path would violate.
#[test]
fn top_p_at_or_above_one_always_returns_the_argmax() {
    // logit 1 is the argmax; the other two are far enough below that the
    // nucleus path can land on them for some thresholds.
    let logits = vec![0.0_f32, 5.0, 0.0];
    for step in 0..=100 {
        #[allow(clippy::cast_precision_loss)]
        let threshold = step as f32 / 100.0;
        let sampled = top_p_sample(&logits, 1.0, threshold);
        assert_eq!(
            sampled, 1,
            "top_p = 1.0 must be greedy for every threshold; got {sampled} at \
             threshold {threshold}"
        );
    }
    // The `||` also covers an empty vocabulary, which must not panic.
    assert_eq!(top_p_sample(&[], 0.5, 0.5), 0);
}

/// `top_logprobs = Some(n)` must return the n most likely tokens, ranked.
///
/// The guard is `Some(n) if n > 0`. `cargo mutants` replaced it with `true` and
/// `false`, flipped its `>`, and every variant survived: the only existing
/// `top_logprobs` test uses EMPTY logits, where the block produces `Vec::new()`
/// either way. This pins the populated case — the ranking order, the length,
/// and agreement with the sampled token's own `logprob`.
#[test]
fn top_logprobs_returns_the_ranked_most_likely_tokens() {
    // Distinct logits so the ranking is unambiguous: token 2 is most likely.
    let logits = vec![0.0_f32, 2.0, 4.0, 1.0];

    let params = SamplingParams::builder()
        .with_top_logprobs(2)
        .with_seed(7)
        .build();
    let sampled = sample_one_with_params(&logits, &params, &[]);

    assert_eq!(
        sampled.top_logprobs.len(),
        2,
        "asking for 2 must return exactly 2 entries; got {:?}",
        sampled.top_logprobs
    );
    let ranked: Vec<u32> = sampled.top_logprobs.iter().map(|(t, _)| *t).collect();
    assert_eq!(
        ranked,
        vec![2, 1],
        "the two highest logits are tokens 2 then 1; got {ranked:?}"
    );
    assert!(
        sampled.top_logprobs[0].1 > sampled.top_logprobs[1].1,
        "entries must be sorted by logprob descending; got {:?}",
        sampled.top_logprobs
    );

    // The sampled token's own logprob must match its rank entry when it is in
    // the top-N, which is the consistency the block exists to guarantee.
    if let Some((_, ranked_lp)) = sampled
        .top_logprobs
        .iter()
        .find(|(t, _)| *t == sampled.token)
    {
        assert!(
            (sampled.logprob - ranked_lp).abs() < 1e-6,
            "the sampled token's logprob must equal its top_logprobs entry; \
             got {} vs {ranked_lp}",
            sampled.logprob
        );
    }

    // `Some(0)` and `None` must both leave the vector empty.
    let none = SamplingParams::builder().with_seed(7).build();
    assert!(
        sample_one_with_params(&logits, &none, &[])
            .top_logprobs
            .is_empty(),
        "top_logprobs = None must produce an empty vector"
    );
}

/// A penalty with no history must not touch the logits.
///
/// Both guards read `<penalty>… > EPSILON && !seen.is_empty()`. Turning either
/// `&&` into `||` lets an EMPTY history reach `apply_repeat_penalty` /
/// `apply_presence_penalty`, which then iterate nothing — harmless in
/// isolation, but the same mutants also flip the epsilon `>` to `> =`, which
/// treats a penalty of exactly 1.0 as active. Together they are only
/// observable through the sampled distribution, so this test pins the
/// distribution itself for the no-op cases.
#[test]
fn penalties_are_no_ops_without_history() {
    let logits = vec![0.0_f32, 3.0, 1.0];

    // Baseline: no penalties, no history.
    let plain = SamplingParams::builder().with_seed(11).build();
    let baseline = sample_one_with_params(&logits, &plain, &[]);

    // Penalty set, but nothing in `seen`: must behave exactly like baseline.
    let with_penalty = SamplingParams::builder()
        .with_seed(11)
        .with_repeat_penalty(2.0)
        .with_presence_penalty(1.5)
        .build();
    let penalised = sample_one_with_params(&logits, &with_penalty, &[]);

    assert_eq!(
        penalised.token, baseline.token,
        "a penalty with an empty history must not change the draw"
    );
    assert!(
        (penalised.logprob - baseline.logprob).abs() < 1e-6,
        "a penalty with an empty history must not change the distribution; \
         got {} vs {}",
        penalised.logprob,
        baseline.logprob
    );

    // A penalty of exactly 1.0 (and presence 0.0) is a no-op even WITH
    // history: the epsilon comparisons are strict. The baseline here also has
    // history so the comparison isolates the penalty, not the history.
    let history = [0u32, 1];
    let baseline_with_history = sample_one_with_params(&logits, &plain, &history);
    let neutral = SamplingParams::builder()
        .with_seed(11)
        .with_repeat_penalty(1.0)
        .with_presence_penalty(0.0)
        .build();
    let neutral_draw = sample_one_with_params(&logits, &neutral, &history);
    assert!(
        (neutral_draw.logprob - baseline_with_history.logprob).abs() < 1e-6,
        "repeat_penalty = 1.0 and presence_penalty = 0.0 must be exact no-ops \
         even with history; got {} vs {}",
        neutral_draw.logprob,
        baseline_with_history.logprob
    );
}

/// A penalty WITH history must actually be applied.
///
/// The guard is `<penalty>… > EPSILON && !seen.is_empty()`. Deleting the `!`
/// inverts the history test, so a penalty is applied only when there is NO
/// history — the exact opposite of the contract. `cargo mutants` found the
/// variant survives because every existing penalty test either uses an empty
/// history or checks only that a token was produced.
#[test]
fn penalties_apply_when_there_is_history() {
    let logits = vec![0.0_f32, 3.0, 1.0];
    let history = [1u32]; // token 1 is the argmax

    let plain = SamplingParams::builder().with_seed(11).build();
    let baseline = sample_one_with_params(&logits, &plain, &history);

    // A strong repeat penalty on the argmax must push the distribution away
    // from it: the argmax's logit (3.0) becomes 1.5, below token 2's 1.0...
    // actually 1.5 > 1.0, so use a larger penalty to cross over.
    let penalised = SamplingParams::builder()
        .with_seed(11)
        .with_repeat_penalty(8.0)
        .build();
    let after = sample_one_with_params(&logits, &penalised, &history);

    assert!(
        after.logprob < baseline.logprob + 1e-6,
        "penalising the current argmax must not raise its own logprob; \
         got {} vs {}",
        after.logprob,
        baseline.logprob
    );
    // 3.0 / 8.0 = 0.375, now BELOW token 2 (1.0), so the argmax moves to
    // token 2.
    assert_eq!(
        after.token, 2,
        "a repeat penalty of 8.0 must demote the repeated argmax below token \
         2; got token {}",
        after.token
    );
}

/// `renormalized_top_p_logprobs` must emit the renormalized nucleus log-probs.
///
/// The only pre-existing test asserts `out.len()`, that one entry is `-inf`, and
/// that the survivors are ordered — all satisfied by many wrong formulas. This
/// pins the values: with logits `[1, 0, -1]` and `top_p = 0.9` the nucleus is
/// the first two tokens, renormalized to `e/(e+1)` and `1/(e+1)`.
#[test]
fn renormalized_top_p_logprobs_matches_the_reference_values() {
    // `max_val = 2.0` matters: with `max_val = 1.0` a `v / max_val` slip is
    // invisible (dividing by 1 changes nothing), which is exactly how that
    // mutant survived the earlier fixture.
    let out = renormalized_top_p_logprobs(&[2.0_f32, 1.0, 0.0], 0.9);

    let e = 1.0_f32.exp();
    let share0 = e / (e + 1.0);
    let share1 = 1.0 / (e + 1.0);

    assert_eq!(out.len(), 3);
    assert!(
        (out[0] - share0.ln()).abs() < 1e-5,
        "token 0 must be ln(e/(e+1)); got {} want {}",
        out[0],
        share0.ln()
    );
    assert!(
        (out[1] - share1.ln()).abs() < 1e-5,
        "token 1 must be ln(1/(e+1)); got {} want {}",
        out[1],
        share1.ln()
    );
    assert_eq!(
        out[2],
        f32::NEG_INFINITY,
        "token 2 falls outside the nucleus and must be -inf"
    );

    // The nucleus must sum to 1 in probability space.
    let mass: f32 = out.iter().filter(|v| v.is_finite()).map(|v| v.exp()).sum();
    assert!(
        (mass - 1.0).abs() < 1e-5,
        "the nucleus must be renormalized to total probability 1; got {mass}"
    );

    // `top_p = 1.0` keeps everything, so the result equals plain log-softmax.
    let full = renormalized_top_p_logprobs(&[2.0_f32, 1.0, 0.0], 1.0);
    let log_soft = log_softmax(&[2.0_f32, 1.0, 0.0]);
    for (token, (a, b)) in full.iter().zip(log_soft.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-5,
            "top_p = 1.0 must not change the distribution at token {token}; got {a} vs {b}"
        );
    }
}

/// `renormalized_top_p_logprobs` must stay finite for large logits.
///
/// The stabiliser subtracts `max_val` before `exp`. Changing that `-` to `+`
/// is arithmetically a constant shift — softmax cancels it — so small logits
/// look identical. It is NOT equivalent: with `f32`, `exp(v + max_val)`
/// overflows to `inf` once `max_val` passes roughly 88, the sum becomes `inf`,
/// and every probability collapses to `NaN`, which then poisons the logprobs
/// and the `SampleToken` JSON. This is the whole reason the stabiliser exists.
#[test]
fn renormalized_top_p_logprobs_stays_finite_for_large_logits() {
    // 50 is far below the f32 exp limit for `v - max` but far above it for
    // `v + max` (exp(99) overflows).
    let out = renormalized_top_p_logprobs(&[50.0_f32, 49.0], 1.0);

    let e = 1.0_f32.exp();
    let expected0 = (1.0 / (1.0 + 1.0 / e)).ln();
    let expected1 = ((1.0 / e) / (1.0 + 1.0 / e)).ln();
    assert!(
        out[0].is_finite() && (out[0] - expected0).abs() < 1e-5,
        "large logits must stay finite; got {} want {expected0}",
        out[0]
    );
    assert!(
        out[1].is_finite() && (out[1] - expected1).abs() < 1e-5,
        "large logits must stay finite; got {} want {expected1}",
        out[1]
    );
    assert!(
        out.iter().all(|v| !v.is_nan()),
        "no entry may become NaN; got {out:?}"
    );
}

/// The nucleus cutoff must use a STRICT comparison.
///
/// `cumsum > top_p` stops once the mass EXCEEDS `top_p`. Relaxing it to `>=`
/// also stops when the mass equals `top_p` exactly, dropping the token that
/// completes the nucleus. With two equal logits the running mass is exactly
/// 0.5, so `top_p = 0.5` separates the two: the strict form keeps both tokens,
/// the relaxed form keeps only the first.
#[test]
fn renormalized_top_p_cutoff_is_strict_at_the_boundary() {
    let out = renormalized_top_p_logprobs(&[0.0_f32, 0.0], 0.5);

    assert_eq!(out.len(), 2);
    assert!(
        out[0].is_finite() && out[1].is_finite(),
        "mass exactly == top_p must keep BOTH tokens; a strict `>` does, while \
         `>=` truncates to the first and leaves the second at -inf (got {out:?})"
    );
    let ln2 = std::f32::consts::LN_2;
    assert!(
        (out[0] + ln2).abs() < 1e-5 && (out[1] + ln2).abs() < 1e-5,
        "both tokens carry half the mass, so both are -ln 2; got {out:?}"
    );
}

/// A `top_p < 1.0` request must draw from the TRUNCATED nucleus, not the full
/// distribution.
///
/// The dispatch is `temperature <= 0.0` -> greedy, else `top_p < 1.0` ->
/// `top_p_sample`, else `temperature_sample`. `cargo mutants` flipped that
/// `top_p < 1.0` to `==`, `>` and `<=`, and every variant survived: the
/// existing test only checks that SOME token comes back. The consequence is
/// real — a nucleus that excludes a token must never emit it, so sampling
/// repeatedly makes the leak visible as an out-of-nucleus token.
#[test]
fn sample_batch_respects_the_nucleus() {
    // logits [1, 0, -1]: token 0 holds ~0.665 of the mass, so `top_p = 0.5`
    // cuts the nucleus to token 0 ALONE.
    let logits = vec![vec![1.0_f32, 0.0, -1.0]];
    let seen: Vec<Vec<TokenId>> = vec![vec![]];

    for _ in 0..200 {
        let out = sample_batch(&logits, 1.0, 0.5, 0, 1.0, &seen);
        assert_eq!(
            out[0], 0,
            "with top_p = 0.5 only token 0 is inside the nucleus; anything else \
             means the sampler used the untruncated distribution"
        );
    }

    // Sanity: with `top_p = 1.0` the other tokens ARE reachable, so the check
    // above is not vacuously true.
    let mut saw_other = false;
    for _ in 0..400 {
        if sample_batch(&logits, 1.0, 1.0, 0, 1.0, &seen)[0] != 0 {
            saw_other = true;
            break;
        }
    }
    assert!(
        saw_other,
        "with top_p = 1.0 the tail tokens must be reachable, otherwise the \
         nucleus assertion proves nothing"
    );
}

/// `top_k` must mask everything below the k-th largest logit.
///
/// The mask is `if *l < threshold { *l = -inf }`, where `threshold` is the k-th
/// largest. `cargo mutants` flipped that `<` to `==`, `>` and `<=` and every
/// variant survived: existing top-k tests only assert that the result is *in*
/// range. With `top_k = 1` and `top_p = 1.0` the argmax must be the only
/// reachable token, which makes any mis-mask visible as a different token.
#[test]
fn sample_batch_top_k_masks_below_the_threshold() {
    // Distinct logits: token 2 is the strict argmax.
    let logits = vec![vec![0.0_f32, 1.0, 3.0, 2.0]];
    let seen: Vec<Vec<TokenId>> = vec![vec![]];

    for _ in 0..200 {
        let out = sample_batch(&logits, 1.0, 1.0, 1, 1.0, &seen);
        assert_eq!(
            out[0], 2,
            "top_k = 1 must leave only the argmax reachable; any other token \
             means the mask kept the wrong side of the threshold"
        );
    }

    // `top_k = 2` admits the second-largest logit (token 3, logit 2.0) too.
    let mut reached_second = false;
    for _ in 0..400 {
        if sample_batch(&logits, 1.0, 1.0, 2, 1.0, &seen)[0] == 3 {
            reached_second = true;
            break;
        }
    }
    assert!(
        reached_second,
        "top_k = 2 must admit the second-largest logit (token 3), otherwise the \
         k = 1 assertion above proves nothing"
    );
}

/// `sample_batch` must apply temperature scaling, and skip it at `T = 1.0`.
///
/// The guard is `temperature > 0.0 && (temperature - 1.0).abs() > EPSILON`,
/// followed by `*l /= temperature`. `cargo mutants` found six variants of those
/// three operators survive, because existing tests only check that a token is
/// produced. Temperature is what turns a flat distribution into a sharp one, so
/// its effect is measurable: at `T = 0.5` the argmax dominates and the tail
/// becomes unreachable, while at `T = 1.0` the tail stays reachable.
#[test]
fn sample_batch_applies_temperature_scaling() {
    // logits [2, 1, 0]: at T = 0.5 they become [4, 2, 0], so token 0 takes
    // ~0.867 of the mass and the tail is effectively unreachable.
    let logits = vec![vec![2.0_f32, 1.0, 0.0]];
    let seen: Vec<Vec<TokenId>> = vec![vec![]];

    // Cold: the argmax should dominate the sample.
    let mut cold_argmax = 0usize;
    for _ in 0..200 {
        if sample_batch(&logits, 0.5, 1.0, 0, 1.0, &seen)[0] == 0 {
            cold_argmax += 1;
        }
    }
    assert!(
        cold_argmax >= 150,
        "at T = 0.5 the argmax must take most draws (~0.867 expected); got \
         {cold_argmax}/200, which means the logits were NOT divided by the \
         temperature"
    );

    // Hot: the same logits with T = 1.0 must reproduce the unscaled mix, so
    // the tail token is reachable. This is the control that stops the cold
    // assertion above from passing vacuously.
    let mut hot_tail = 0usize;
    for _ in 0..400 {
        if sample_batch(&logits, 1.0, 1.0, 0, 1.0, &seen)[0] == 2 {
            hot_tail += 1;
        }
    }
    assert!(
        hot_tail > 0,
        "at T = 1.0 the tail token (logit 0.0, ~0.09 of the mass) must be \
         reachable, otherwise the cold assertion proves nothing"
    );

    // T = 2.0 flattens: the argmax share must DROP below its T = 1.0 share.
    let mut warm_argmax = 0usize;
    for _ in 0..400 {
        if sample_batch(&logits, 2.0, 1.0, 0, 1.0, &seen)[0] == 0 {
            warm_argmax += 1;
        }
    }
    let cold_share = cold_argmax as f32 / 200.0;
    let warm_share = warm_argmax as f32 / 400.0;
    assert!(
        warm_share < cold_share - 0.1,
        "T = 2.0 must flatten the distribution relative to T = 0.5; got \
         {warm_share:.3} vs {cold_share:.3}"
    );
}

/// `temperature = 0` must select the argmax WITHOUT scaling the logits.
///
/// The guard opens with `temperature > 0.0`. Relaxing it to `>=` lets
/// `T = 0` through into `*l /= temperature`, and a logit of exactly 0.0
/// divides to `0.0 / 0.0 = NaN` rather than `inf`. The dispatch below still
/// routes `T = 0` to `greedy_sample`, but it now runs on a NaN-poisoned row,
/// where the argmax is whatever the comparison happens to favour. The
/// unscaled row has an unambiguous answer.
#[test]
fn sample_batch_greedy_at_zero_temperature_without_scaling() {
    // A logit of exactly 0.0 is what turns division by zero into NaN.
    let logits = vec![vec![1.0_f32, 0.0, -1.0]];
    let seen: Vec<Vec<TokenId>> = vec![vec![]];

    for _ in 0..50 {
        let out = sample_batch(&logits, 0.0, 1.0, 0, 1.0, &seen);
        assert_eq!(
            out[0], 0,
            "T = 0 must pick the argmax (token 0) from the UNSCALED row; a \
             NaN-poisoned row can pick anything"
        );
    }

    // Same row with the argmax NOT first, so a positional accident cannot
    // make the assertion above pass.
    let shifted = vec![vec![-1.0_f32, 0.0, 3.0]];
    for _ in 0..50 {
        let out = sample_batch(&shifted, 0.0, 1.0, 0, 1.0, &seen);
        assert_eq!(
            out[0], 2,
            "T = 0 must pick the argmax (token 2) from the unscaled row"
        );
    }
}

/// `sample_batch` must apply the repeat penalty only when it is BOTH enabled
/// and backed by history.
///
/// The guard is `(repeat_penalty - 1.0).abs() > EPSILON && !seen.is_empty()`.
/// Seven operator mutants of that one line survived, because no test drove
/// `sample_batch` through the penalty at all. `T = 0` makes each case a
/// deterministic greedy pick, so the three arms are cleanly separable:
/// a strong penalty demotes the repeated argmax, a neutral penalty changes
/// nothing, and an empty history disables the penalty regardless of its value.
#[test]
fn sample_batch_repeat_penalty_requires_history_and_a_non_neutral_value() {
    // logits [0, 3, 1]; token 1 is the argmax.
    let logits = vec![vec![0.0_f32, 3.0, 1.0]];

    // Penalising the argmax hard enough (3.0 / 8.0 = 0.375) drops it below
    // token 2 (1.0), so the greedy pick moves.
    let with_history: Vec<Vec<TokenId>> = vec![vec![1]];
    assert_eq!(
        sample_batch(&logits, 0.0, 1.0, 0, 8.0, &with_history)[0],
        2,
        "a repeat penalty of 8.0 on the argmax must demote it below token 2"
    );

    // A neutral penalty is a no-op even with history.
    assert_eq!(
        sample_batch(&logits, 0.0, 1.0, 0, 1.0, &with_history)[0],
        1,
        "repeat_penalty = 1.0 must leave the argmax untouched"
    );

    // Without history the penalty cannot apply, however large it is.
    let no_history: Vec<Vec<TokenId>> = vec![vec![]];
    assert_eq!(
        sample_batch(&logits, 0.0, 1.0, 0, 8.0, &no_history)[0],
        1,
        "an empty history must disable the penalty regardless of its value"
    );
}
