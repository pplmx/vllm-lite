#![allow(clippy::module_name_repetitions)]
//! Draft verification trait for speculative decoding
//!
//! The `DraftVerifier` trait abstracts the draft generation and verification
//! logic, allowing different implementations (self-speculation, small draft model, etc.)

use std::sync::Arc;

use crate::types::{Batch, SeqId, TokenId};
use thiserror::Error;

/// `VerifierError`: verifier error.
#[derive(Debug, Error)]
pub enum VerifierError {
    #[error("draft generation failed: {0}")]
    DraftGeneration(String),
    #[error("verification failed: {0}")]
    Verification(String),
}

/// Result: result.
pub type Result<T> = std::result::Result<T, VerifierError>;

/// `VerificationResult`: verification result alias.
#[derive(Debug, Clone)]
pub struct VerificationResult {
    pub seq_id: SeqId,
    pub draft_tokens: Vec<TokenId>,
    pub accepted_count: usize,
    pub rejected_at: Option<usize>,
}

impl VerificationResult {
    #[must_use]
    pub fn new(seq_id: SeqId, draft_tokens: Vec<TokenId>) -> Self {
        let accepted_count = draft_tokens.len();
        Self {
            seq_id,
            draft_tokens,
            accepted_count,
            rejected_at: None,
        }
    }

    #[must_use]
    pub const fn with_rejection(mut self, rejected_at: usize) -> Self {
        self.rejected_at = Some(rejected_at);
        self.accepted_count = rejected_at;
        self
    }

    #[must_use]
    // invariant: accepted_count and draft_tokens.len() are bounded; f32 precision
    // loss is acceptable for the acceptance-rate metric.
    #[allow(clippy::cast_precision_loss)]
    pub fn acceptance_rate(&self) -> f32 {
        if self.draft_tokens.is_empty() {
            return 1.0;
        }
        self.accepted_count as f32 / self.draft_tokens.len() as f32
    }
}

/// `DraftVerifier`: draft verifier trait.
pub trait DraftVerifier: Send + Sync {
    /// Runs the operation.
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    fn generate_draft(
        &mut self,
        batch: &Batch,
        num_tokens: usize,
    ) -> Result<Vec<(SeqId, Vec<TokenId>)>>;

    /// Runs the operation.
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    fn verify(
        &self,
        seq_id: SeqId,
        draft_tokens: &[TokenId],
        target_logits: &[f32],
    ) -> Result<VerificationResult>;

    fn accept(&mut self, seq_id: SeqId, accepted_count: usize);
}

/// Default stub `DraftVerifier`: accepts all draft tokens, generates no drafts.
///
/// Used by [`dyn DraftVerifier::default_arc`] to construct an `Arc<Self>`
/// default instance.
#[derive(Debug, Default, Clone, Copy)]
pub struct StubDraftVerifier;

impl DraftVerifier for StubDraftVerifier {
    fn generate_draft(
        &mut self,
        batch: &Batch,
        _num_tokens: usize,
    ) -> Result<Vec<(SeqId, Vec<TokenId>)>> {
        Ok(batch.seq_ids.iter().map(|&id| (id, Vec::new())).collect())
    }

    fn verify(
        &self,
        seq_id: SeqId,
        draft_tokens: &[TokenId],
        _target_logits: &[f32],
    ) -> Result<VerificationResult> {
        Ok(VerificationResult::new(seq_id, draft_tokens.to_vec()))
    }

    fn accept(&mut self, _seq_id: SeqId, _accepted_count: usize) {}
}

impl dyn DraftVerifier {
    /// Returns an `Arc<Self>` containing the no-op [`StubDraftVerifier`] stub.
    ///
    /// This is the closest equivalent to `Arc::<dyn DraftVerifier>::default()`;
    /// Rust's orphan rule prevents a direct `impl Default for Arc<dyn ...>`
    /// because `Arc` is foreign and there is no local type appearing before
    /// the uncovered trait-object parameter.
    #[must_use]
    pub fn default_arc() -> Arc<Self> {
        Arc::new(StubDraftVerifier)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verification_result_all_accepted() {
        let tokens = vec![1, 2, 3, 4, 5];
        let result = VerificationResult::new(0, tokens);
        assert_eq!(result.accepted_count, 5);
        assert!((result.acceptance_rate() - 1.0).abs() < 1e-6);
        assert!(result.rejected_at.is_none());
    }

    #[test]
    fn test_verification_result_partial_acceptance() {
        let tokens = vec![1, 2, 3, 4, 5];
        let result = VerificationResult::new(0, tokens).with_rejection(3);
        assert_eq!(result.accepted_count, 3);
        assert!((result.acceptance_rate() - 0.6).abs() < 1e-6);
        assert_eq!(result.rejected_at, Some(3));
    }

    #[test]
    fn test_verification_result_empty() {
        let result = VerificationResult::new(0, vec![]);
        assert_eq!(result.accepted_count, 0);
        assert!((result.acceptance_rate() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn draft_verifier_default_arc_accepts_all() {
        let verifier: Arc<dyn DraftVerifier> = <dyn DraftVerifier>::default_arc();
        let result = verifier
            .verify(42, &[10, 20, 30], &[0.0; 100])
            .expect("stub verify should succeed");
        assert_eq!(result.accepted_count, 3);
        assert!((result.acceptance_rate() - 1.0).abs() < 1e-6);
        assert!(result.rejected_at.is_none());
    }
}
