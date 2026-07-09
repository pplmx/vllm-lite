//! Unit tests for the sampling primitives (`greedy_sample`,
//! `temperature_sample`, `top_k_sample`, `top_p_sample`,
//! `sample_batch`, `apply_repeat_penalty`).
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
//! - `sample_batch` (basic, T+top_p, top_k, repeat_penalty)
//! - `apply_repeat_penalty` (basic penalty, no-op at 1.0)
//! - Property-based tests (proptest) in the sibling `prop_tests`
//!   module: batch length preservation, greedy index bounds,
//!   batched greedy matches per-row greedy, repeat-penalty no-op.

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
