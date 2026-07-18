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
    assert!((logits[0] - 0.5).abs() < 1e-6);
    assert!((logits[2] - 0.5).abs() < 1e-6);
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
    assert!(logits[0].abs() < 1e-6, "zero logit stays zero under penalty > 1");
    apply_repeat_penalty(&mut logits, &seen, 0.5);
    assert!(logits[0].abs() < 1e-6, "zero logit stays zero under penalty < 1");
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
    let result = top_k_sample(&logits, 2);
    assert!(result == 1 || result == 2);
}

#[test]
fn test_top_k_zero_no_effect() {
    let logits = vec![0.1, 0.9, 0.3];
    let result = top_k_sample(&logits, 0);
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
    assert_eq!(with_params, scalar);
}

#[test]
fn test_sample_batch_with_params_greedy_default() {
    // Default `SamplingParams::default()` is `temperature = 0` →
    // greedy argmax. Mirrors the legacy `sample_batch` baseline.
    let logits = vec![vec![0.1, 0.9, 0.3], vec![0.8, 0.1, 0.1]];
    let seen = vec![vec![], vec![]];
    let params = vec![SamplingParams::default(); 2];
    let tokens = sample_batch_with_params(&logits, &params, &seen);
    assert_eq!(tokens, vec![1, 0]);
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
    assert_eq!(tokens[0], 1);
    assert_eq!(tokens[1], 1);
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
    assert_ne!(tokens[0], 1, "penalty should suppress token 1 for seq 0");
    assert_eq!(tokens[1], 0, "no penalty → first argmax wins");
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
    assert_eq!(token, 1);
}
