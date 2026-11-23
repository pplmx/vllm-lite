#![allow(clippy::module_name_repetitions)]
//! `SpeculativeModel` wrapper for speculative decoding
//!
//! Wraps a `ModelBackend` with speculative execution logic,
//! managing the draft-verify-accept cycle.

use super::config::SpeculationConfig;
use super::strategy::RejectionStrategy;
use super::verifier::DraftVerifier;
use vllm_traits::ModelBackend;

/// Wrapper around a draft + target model pair for speculative decoding. Owns the [`DraftVerifier`] used to accept or reject draft tokens.
pub struct SpeculativeModel<M: ModelBackend> {
    target_model: M,
    verifier: Box<dyn DraftVerifier>,
    config: SpeculationConfig,
    strategy: RejectionStrategy,
}

impl<M: ModelBackend + std::fmt::Debug> std::fmt::Debug for SpeculativeModel<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpeculativeModel")
            .field("target_model", &self.target_model)
            .field("verifier", &"<dyn DraftVerifier>")
            .field("config", &self.config)
            .field("strategy", &self.strategy)
            .finish()
    }
}

impl<M: ModelBackend> SpeculativeModel<M> {
    /// Wrap `target_model` with the given verifier, config, and rejection strategy.
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

    /// Borrow the underlying target [`ModelBackend`].
    pub const fn target_model(&self) -> &M {
        &self.target_model
    }

    /// Borrow the draft verifier used for generate/accept cycles.
    pub fn verifier(&self) -> &dyn DraftVerifier {
        self.verifier.as_ref()
    }

    /// Borrow the active [`SpeculationConfig`].
    pub const fn config(&self) -> &SpeculationConfig {
        &self.config
    }

    /// Borrow the active [`RejectionStrategy`].
    pub const fn strategy(&self) -> &RejectionStrategy {
        &self.strategy
    }

    /// Replace the rejection strategy (e.g. switch token-level to block-level).
    pub const fn set_strategy(&mut self, strategy: RejectionStrategy) {
        self.strategy = strategy;
    }

    /// Draft tokens attempted per speculative step from [`SpeculationConfig`].
    pub const fn draft_count(&self) -> usize {
        self.config.draft_count
    }

    /// Maximum speculative tree depth from [`SpeculationConfig`].
    pub const fn max_depth(&self) -> usize {
        self.config.max_depth
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct StubBackend;

    impl ModelBackend for StubBackend {
        fn forward(
            &mut self,
            _seq_ids: &[vllm_traits::SeqId],
            _input_tokens: &[Vec<vllm_traits::TokenId>],
            _positions: &[Vec<usize>],
            _kv_block_ids: &[Vec<usize>],
            _num_computed_tokens: &[usize],
            _is_prefill: &[bool],
        ) -> vllm_traits::Result<vllm_traits::BatchOutput> {
            panic!("StubBackend::forward not used in model tests")
        }
        fn forward_logits(
            &mut self,
            _seq_ids: &[vllm_traits::SeqId],
            _input_tokens: &[Vec<vllm_traits::TokenId>],
            _positions: &[Vec<usize>],
            _kv_block_ids: &[Vec<usize>],
            _num_computed_tokens: &[usize],
            _is_prefill: &[bool],
        ) -> vllm_traits::Result<Vec<Vec<f32>>> {
            panic!("StubBackend::forward_logits not used in model tests")
        }
        fn embed(
            &mut self,
            _input_tokens: &[Vec<vllm_traits::TokenId>],
            _positions: &[Vec<usize>],
        ) -> vllm_traits::Result<Vec<Vec<f32>>> {
            panic!("StubBackend::embed not used in model tests")
        }
        fn vocab_size(&self) -> usize {
            32000
        }
        fn num_layers(&self) -> usize {
            1
        }
        fn num_heads(&self) -> usize {
            1
        }
    }
    use crate::speculative::verifier::{Result as VerifierResult, VerificationResult};
    use crate::types::{SeqId, TokenId};

    struct MockVerifier {
        accepted: usize,
    }

    impl DraftVerifier for MockVerifier {
        fn generate_draft(
            &mut self,
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

    /// The wrapper's accessors must report the values it was CONSTRUCTED with.
    ///
    /// `cargo mutants` replaced `draft_count()` / `max_depth()` with the
    /// constants 0 and 1, `set_strategy` with `()`, and the `Debug` body with
    /// `Ok(())` — five survivors. The existing tests build a bare
    /// `SpeculationConfig`, never a `SpeculativeModel`, so nothing read the
    /// wrapper at all. A `set_strategy` no-op means a runtime switch from
    /// token-level to block-level acceptance silently keeps the old rule.
    #[test]
    fn model_accessors_and_strategy_swap() {
        let config = SpeculationConfig::builder()
            .draft_count(6)
            .max_depth(12)
            .build();
        let mut model = SpeculativeModel::new(
            StubBackend,
            Box::new(MockVerifier { accepted: 0 }),
            config,
            RejectionStrategy::TokenLevel,
        );

        assert_eq!(model.draft_count(), 6, "draft_count must come from config");
        assert_eq!(model.max_depth(), 12, "max_depth must come from config");
        assert_eq!(*model.strategy(), RejectionStrategy::TokenLevel);

        model.set_strategy(RejectionStrategy::new_block_level(4));
        assert_eq!(
            *model.strategy(),
            RejectionStrategy::BlockLevel { block_size: 4 },
            "set_strategy must replace the active strategy"
        );

        let rendered = format!("{model:?}");
        assert!(
            rendered.contains("SpeculativeModel") && rendered.contains("strategy"),
            "Debug must name the struct and its fields; got {rendered}"
        );
        assert!(
            rendered.contains("<dyn DraftVerifier>"),
            "Debug must not try to print the boxed verifier"
        );
    }

    #[test]
    fn test_strategy_mutation() {
        let mut strategy = RejectionStrategy::default();
        assert_eq!(strategy, RejectionStrategy::TokenLevel);

        strategy = RejectionStrategy::new_block_level(4);
        assert_eq!(strategy, RejectionStrategy::BlockLevel { block_size: 4 });
    }
}
