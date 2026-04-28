//! Rejection strategies for speculative decoding
//!
//! Different strategies for determining which draft tokens to accept
//! based on comparing draft and target model probabilities.

#[derive(Clone, Debug, PartialEq, Default)]
pub enum RejectionStrategy {
    #[default]
    TokenLevel,
    BlockLevel { block_size: usize },
}

impl RejectionStrategy {
    pub fn new_token_level() -> Self {
        RejectionStrategy::TokenLevel
    }

    pub fn new_block_level(block_size: usize) -> Self {
        RejectionStrategy::BlockLevel {
            block_size: block_size.max(1),
        }
    }

    pub fn should_accept(&self, draft_prob: f32, target_prob: f32) -> bool {
        match self {
            RejectionStrategy::TokenLevel => target_prob >= draft_prob,
            RejectionStrategy::BlockLevel { .. } => target_prob > draft_prob,
        }
    }

    pub fn acceptance_threshold(&self) -> f32 {
        match self {
            RejectionStrategy::TokenLevel => 0.0,
            RejectionStrategy::BlockLevel { .. } => 1e-6,
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
