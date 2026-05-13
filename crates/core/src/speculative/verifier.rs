//! Draft verification trait for speculative decoding
//!
//! The DraftVerifier trait abstracts the draft generation and verification
//! logic, allowing different implementations (self-speculation, small draft model, etc.)

use crate::types::{Batch, SeqId, TokenId};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum VerifierError {
    #[error("draft generation failed: {0}")]
    DraftGeneration(String),
    #[error("verification failed: {0}")]
    Verification(String),
}

pub type Result<T> = std::result::Result<T, VerifierError>;

#[derive(Debug, Clone)]
pub struct VerificationResult {
    pub seq_id: SeqId,
    pub draft_tokens: Vec<TokenId>,
    pub accepted_count: usize,
    pub rejected_at: Option<usize>,
}

impl VerificationResult {
    pub fn new(seq_id: SeqId, draft_tokens: Vec<TokenId>) -> Self {
        let accepted_count = draft_tokens.len();
        Self {
            seq_id,
            draft_tokens,
            accepted_count,
            rejected_at: None,
        }
    }

    pub fn with_rejection(mut self, rejected_at: usize) -> Self {
        self.rejected_at = Some(rejected_at);
        self.accepted_count = rejected_at;
        self
    }

    pub fn acceptance_rate(&self) -> f32 {
        if self.draft_tokens.is_empty() {
            return 1.0;
        }
        self.accepted_count as f32 / self.draft_tokens.len() as f32
    }
}

pub trait DraftVerifier: Send + Sync {
    fn generate_draft(
        &mut self,
        batch: &Batch,
        num_tokens: usize,
    ) -> Result<Vec<(SeqId, Vec<TokenId>)>>;

    fn verify(
        &self,
        seq_id: SeqId,
        draft_tokens: &[TokenId],
        target_logits: &[f32],
    ) -> Result<VerificationResult>;

    fn accept(&mut self, seq_id: SeqId, accepted_count: usize);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verification_result_all_accepted() {
        let tokens = vec![1, 2, 3, 4, 5];
        let result = VerificationResult::new(0, tokens);
        assert_eq!(result.accepted_count, 5);
        assert_eq!(result.acceptance_rate(), 1.0);
        assert!(result.rejected_at.is_none());
    }

    #[test]
    fn test_verification_result_partial_acceptance() {
        let tokens = vec![1, 2, 3, 4, 5];
        let result = VerificationResult::new(0, tokens).with_rejection(3);
        assert_eq!(result.accepted_count, 3);
        assert_eq!(result.acceptance_rate(), 0.6);
        assert_eq!(result.rejected_at, Some(3));
    }

    #[test]
    fn test_verification_result_empty() {
        let result = VerificationResult::new(0, vec![]);
        assert_eq!(result.accepted_count, 0);
        assert_eq!(result.acceptance_rate(), 1.0);
    }
}
