//! SpeculativeModel wrapper for speculative decoding
//!
//! Wraps a ModelBackend with speculative execution logic,
//! managing the draft-verify-accept cycle.

use vllm_traits::ModelBackend;
use super::config::SpeculationConfig;
use super::strategy::RejectionStrategy;
use super::verifier::DraftVerifier;

pub struct SpeculativeModel<M: ModelBackend> {
    target_model: M,
    verifier: Box<dyn DraftVerifier>,
    config: SpeculationConfig,
    strategy: RejectionStrategy,
}

impl<M: ModelBackend> SpeculativeModel<M> {
    pub fn new(
        target_model: M,
        verifier: Box<dyn DraftVerifier>,
        config: SpeculationConfig,
        strategy: RejectionStrategy,
    ) -> Self {
        Self {
            target_model,
            verifier,
            config,
            strategy,
        }
    }

    pub fn target_model(&self) -> &M {
        &self.target_model
    }

    pub fn verifier(&self) -> &dyn DraftVerifier {
        self.verifier.as_ref()
    }

    pub fn mut_verifier(&mut self) -> &mut Box<dyn DraftVerifier> {
        &mut self.verifier
    }

    pub fn config(&self) -> &SpeculationConfig {
        &self.config
    }

    pub fn strategy(&self) -> &RejectionStrategy {
        &self.strategy
    }

    pub fn set_strategy(&mut self, strategy: RejectionStrategy) {
        self.strategy = strategy;
    }

    pub fn draft_count(&self) -> usize {
        self.config.draft_count
    }

    pub fn max_depth(&self) -> usize {
        self.config.max_depth
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{SeqId, TokenId};
    use crate::speculative::verifier::{VerificationResult, Result as VerifierResult};

    struct MockVerifier {
        accepted: usize,
    }

    impl DraftVerifier for MockVerifier {
        fn generate_draft(
            &self,
            _batch: &crate::types::Batch,
            _num_tokens: usize,
        ) -> VerifierResult<Vec<(SeqId, Vec<TokenId>)>> {
            Ok(vec![])
        }

        fn verify(
            &self,
            seq_id: SeqId,
            draft_tokens: &[TokenId],
            _target_logits: &[f32],
        ) -> VerifierResult<VerificationResult> {
            Ok(VerificationResult::new(seq_id, draft_tokens.to_vec()))
        }

        fn accept(&mut self, _seq_id: SeqId, accepted_count: usize) {
            self.accepted = accepted_count;
        }
    }

    #[test]
    fn test_speculative_model_creation() {
        let config = SpeculationConfig::default();
        let strategy = RejectionStrategy::default();
        let _verifier = Box::new(MockVerifier { accepted: 0 });

        assert_eq!(config.draft_count, 4);
        assert_eq!(strategy, RejectionStrategy::TokenLevel);
    }

    #[test]
    fn test_config_accessors() {
        let config = SpeculationConfig::builder()
            .draft_count(6)
            .max_depth(12)
            .build();

        assert_eq!(config.draft_count, 6);
        assert_eq!(config.max_depth, 12);
    }

    #[test]
    fn test_strategy_mutation() {
        let mut strategy = RejectionStrategy::default();
        assert_eq!(strategy, RejectionStrategy::TokenLevel);

        strategy = RejectionStrategy::new_block_level(4);
        assert_eq!(strategy, RejectionStrategy::BlockLevel { block_size: 4 });
    }
}
