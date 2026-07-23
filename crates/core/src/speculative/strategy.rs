#![allow(clippy::module_name_repetitions)]
//! Rejection strategies for speculative decoding
//!
//! Different strategies for determining which draft tokens to accept
//! based on comparing draft and target model probabilities.

/// Strategy pattern implementation for Rejection. Encapsulates one of N interchangeable algorithms.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub enum RejectionStrategy {
    #[default]
    TokenLevel,
    BlockLevel {
        block_size: usize,
    },
}

impl RejectionStrategy {
    /// Create a token-level rejection strategy (the default — each token
    /// is accepted independently based on its probability ratio).
    #[must_use]
    pub const fn new_token_level() -> Self {
        Self::TokenLevel
    }

    /// Create a block-level rejection strategy. Draft tokens are accepted
    /// in blocks of `block_size`; a block is rejected if *any* token in it
    /// fails the acceptance criterion.
    #[must_use]
    pub fn new_block_level(block_size: usize) -> Self {
        Self::BlockLevel {
            block_size: block_size.max(1),
        }
    }

    /// Decide whether to accept a draft token given the draft and target
    /// model probabilities.
    ///
    /// - `TokenLevel` accepts when `target_prob >= draft_prob`.
    /// - `BlockLevel` uses a strict `>` comparison (block rejection
    ///   semantics: one fail rejects the whole block).
    #[must_use]
    pub fn should_accept(&self, draft_prob: f32, target_prob: f32) -> bool {
        match self {
            Self::TokenLevel => target_prob >= draft_prob,
            Self::BlockLevel { .. } => target_prob > draft_prob,
        }
    }

    /// Minimum probability ratio for a token to be accepted without
    /// re-sampling. Used as a numerical-stability floor in the
    /// acceptance sampler.
    #[must_use]
    pub const fn acceptance_threshold(&self) -> f32 {
        match self {
            Self::TokenLevel => 0.0,
            Self::BlockLevel { .. } => 1e-6,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_level_accept_higher() {
        let strategy = RejectionStrategy::TokenLevel;
        assert!(strategy.should_accept(0.1, 0.2));
        assert!(strategy.should_accept(0.5, 0.5));
    }

    #[test]
    fn test_token_level_reject_lower() {
        let strategy = RejectionStrategy::TokenLevel;
        assert!(!strategy.should_accept(0.8, 0.2));
    }

    #[test]
    fn test_block_level_strict() {
        let strategy = RejectionStrategy::BlockLevel { block_size: 4 };
        assert!(!strategy.should_accept(0.5, 0.5));
        assert!(strategy.should_accept(0.5, 0.51));
    }

    #[test]
    fn test_default_is_token_level() {
        let strategy = RejectionStrategy::default();
        assert_eq!(strategy, RejectionStrategy::TokenLevel);
    }

    #[test]
    fn test_block_size_minimum() {
        let strategy = RejectionStrategy::new_block_level(0);
        match strategy {
            RejectionStrategy::BlockLevel { block_size } => {
                assert_eq!(block_size, 1);
            }
            RejectionStrategy::TokenLevel => panic!("Expected BlockLevel variant"),
        }
    }
}
