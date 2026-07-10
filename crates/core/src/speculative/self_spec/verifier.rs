// crates/core/src/speculative/self_spec/verifier.rs
//
// `DraftVerifier` trait impl for `SelfSpeculativeModel`. Generation runs
// the draft model `num_tokens` steps per sequence, advancing the position
// each step. Verification is a stub — the engine implements its own
// logit-based verification via `forward_logits()` and argmax.

use super::SelfSpeculativeModel;
use crate::speculative::verifier::{
    DraftVerifier, Result as VerifierResult, VerificationResult, VerifierError,
};
use crate::types::{SeqId, TokenId};
use vllm_traits::ModelBackend;

impl<M: ModelBackend> DraftVerifier for SelfSpeculativeModel<M> {
    fn generate_draft(
        &mut self,
        batch: &vllm_traits::types::Batch,
        num_tokens: usize,
    ) -> VerifierResult<Vec<(SeqId, Vec<TokenId>)>> {
        if num_tokens == 0 || batch.is_empty() {
            return Ok(vec![]);
        }

        let mut drafts: Vec<(SeqId, Vec<TokenId>)> = Vec::new();

        for batch_idx in 0..batch.seq_ids.len() {
            let seq_id = batch.seq_ids[batch_idx];
            let input_tokens = &batch.input_tokens[batch_idx];
            let num_computed = batch.num_computed_tokens[batch_idx];

            let draft_block_ids = self
                .draft_kv_block_ids
                .entry(seq_id)
                .or_insert_with(|| batch.kv_block_ids[batch_idx].clone());

            let mut current_tokens: Vec<TokenId> = input_tokens.clone();
            let mut draft_tokens: Vec<TokenId> = Vec::with_capacity(num_tokens);

            // Use position tracking to compute positions for each draft step
            for current_num_computed in (num_computed..).take(num_tokens) {
                let last_token = vec![*current_tokens.last().unwrap_or(&0)];
                let step_position = vec![current_num_computed];

                let output = self
                    .model
                    .forward_to_layer(
                        &[seq_id],
                        &[last_token],
                        &[step_position],
                        std::slice::from_ref(draft_block_ids),
                        &[current_num_computed],
                        &[false],
                        self.draft_layer_count,
                    )
                    .map_err(|e| VerifierError::DraftGeneration(e.to_string()))?;

                let next_token = output.next_tokens.first().copied().unwrap_or(0);
                draft_tokens.push(next_token);
                current_tokens.push(next_token);
            }

            drafts.push((seq_id, draft_tokens));
        }

        Ok(drafts)
    }

    /// Verification is performed by `Engine::verify_draft_tokens_logits()`
    /// using `forward_logits()` with argmax comparison. This trait method is
    /// a stub — the engine bypasses the `DraftVerifier` trait for verification
    /// and implements its own logit-based path.
    fn verify(
        &self,
        seq_id: SeqId,
        draft_tokens: &[TokenId],
        _target_logits: &[f32],
    ) -> VerifierResult<VerificationResult> {
        Ok(VerificationResult::new(seq_id, draft_tokens.to_vec()))
    }

    fn accept(&mut self, _seq_id: SeqId, _accepted_count: usize) {}
}
