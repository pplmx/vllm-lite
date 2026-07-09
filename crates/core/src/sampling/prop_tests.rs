//! Property-based tests (proptest) for the sampling primitives.
//! Companion to `tests.rs`; both extracted from `sampling.rs` to keep
//! the implementation file under the project's 800-line soft cap.
//!
//! Invariants under test:
//! - `sample_batch` output length equals input list length
//! - Every `greedy_sample` result is in `[0, logits.len())`
//! - Greedy `sample_batch` (T=0) returns the same per-row result as
//!   calling `greedy_sample` on each row independently
//! - `apply_repeat_penalty` with penalty=1.0 is a no-op

use super::*;
use proptest::prelude::*;

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
            usize::try_from(result).is_ok_and(|r| r < logits.len()),
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
