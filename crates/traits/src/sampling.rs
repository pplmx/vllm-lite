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

use crate::types::TokenId;

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
