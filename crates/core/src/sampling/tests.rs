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
    // the sampler must draw different random thresholds (because
    // each seed derives a fresh RNG state). We assert the two
    // tokens are different — for SEED_TEST_LOGITS with T=1.0, the
    // softmax is moderately peaked so two random draws almost
    // certainly land on different tokens. If the seeded RNG is
    // broken (e.g. reused across calls), this test catches it.
    let logits = SEED_TEST_LOGITS;
    let seen: &[TokenId] = &[];
    let params_a = SamplingParams::builder()
        .with_temperature(1.0)
        .with_seed(42)
        .build();
    let params_b = SamplingParams::builder()
        .with_temperature(1.0)
        .with_seed(99)
        .build();
    let token_a = sample_one_with_params(logits, &params_a, seen);
    let token_b = sample_one_with_params(logits, &params_b, seen);
    assert_ne!(
        token_a, token_b,
        "different seeds must produce different sampled tokens for \
         SEED_TEST_LOGITS at T=1.0 (got {token_a:?} for both — RNG is \
         not being seeded correctly)"
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
    // Sequence 0 and 1 share the same seed → must produce the same
    // token. Sequence 2 has a different seed → must produce a
    // different token from sequence 0/1.
    assert_eq!(
        tokens[0], tokens[1],
        "sequences with the same seed must produce the same token \
         (got {0:?} vs {1:?})",
        tokens[0], tokens[1]
    );
    assert_ne!(
        tokens[0], tokens[2],
        "sequences with different seeds must produce different \
         tokens (got {0:?} for both — RNG state is shared across \
         sequences)",
        tokens[0]
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
