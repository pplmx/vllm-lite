//! Rejection strategies for speculative decoding
//!
//! Different strategies for determining which draft tokens to accept
//! based on comparing draft and target model probabilities.

/// `RejectionStrategy`: rejection strategy enumeration.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub enum RejectionStrategy {
    #[default]
    TokenLevel,
    BlockLevel {
        block_size: usize,
    },
}

impl RejectionStrategy {
    #[must_use]
    pub const fn new_token_level() -> Self {
        Self::TokenLevel
    }

    #[must_use]
    pub fn new_block_level(block_size: usize) -> Self {
        Self::BlockLevel {
            block_size: block_size.max(1),
        }
    }

    #[must_use]
    pub fn should_accept(&self, draft_prob: f32, target_prob: f32) -> bool {
        match self {
            Self::TokenLevel => target_prob >= draft_prob,
            Self::BlockLevel { .. } => target_prob > draft_prob,
        }
    }

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
            _ => panic!("Expected BlockLevel variant"),
        }
    }
}
