//! Logit-based verification with rejection sampling (Plan 17.1-C).
//!
//! Takes the per-sequence draft tokens and the target model's logits, then
//! greedily accepts drafts whose top-1 matches the target's argmax.
//! On mismatch, the target's argmax is emitted and the remaining drafts are
//! rejected. A bonus token is emitted if all drafts were accepted.

use super::drafts::argmax;
use crate::error::Result;
use crate::sync::lock_mutex;
use vllm_traits::{Batch, SeqId, TokenId};

impl crate::engine::Engine {
    /// Returns `(accepted_tokens, accepted_counts_per_sequence)`.
    pub(crate) fn verify_draft_tokens_logits(
        &mut self,
        batch: &Batch,
        draft_outputs: &[Vec<TokenId>],
    ) -> Result<(Vec<(SeqId, TokenId)>, Vec<usize>)> {
        let mut results = Vec::new();
        let mut accepted_counts = Vec::with_capacity(batch.seq_ids.len());

        for (i, seq_id) in batch.seq_ids.iter().enumerate() {
            let drafts = &draft_outputs[i];

            if drafts.is_empty() {
                let logits = lock_mutex(&self.target_model)?.forward_logits(
                    &[*seq_id],
                    std::slice::from_ref(&batch.input_tokens[i]),
                    std::slice::from_ref(&batch.positions[i]),
                    std::slice::from_ref(&batch.kv_block_ids[i]),
                    std::slice::from_ref(&batch.num_computed_tokens[i]),
                    std::slice::from_ref(&batch.is_prefill[i]),
                )?;
                let token = if let Some(pos_logits) = logits.first() {
                    argmax(pos_logits)
                } else {
                    0
                };
                results.push((*seq_id, token));
                accepted_counts.push(0);
                continue;
            }

            // Concatenate input tokens + draft tokens for verification
            let verify_tokens: Vec<TokenId> = batch.input_tokens[i]
                .iter()
                .chain(drafts.iter())
                .copied()
                .collect();
            let verify_positions: Vec<usize> = (0..verify_tokens.len()).collect();

            // Get logits from target model for all positions
            let logits = lock_mutex(&self.target_model)?.forward_logits(
                &[*seq_id],
                std::slice::from_ref(&verify_tokens),
                std::slice::from_ref(&verify_positions),
                std::slice::from_ref(&batch.kv_block_ids[i]),
                std::slice::from_ref(&batch.num_computed_tokens[i]),
                std::slice::from_ref(&batch.is_prefill[i]),
            )?;

            let logits: &[f32] = logits.first().map(|v| v.as_slice()).unwrap_or(&[]);
            let vocab_size = lock_mutex(&self.target_model)?.vocab_size();

            let mut accepted = 0usize;

            for (j, &draft_token) in drafts.iter().enumerate() {
                let offset = j * vocab_size;
                if offset + vocab_size > logits.len() {
                    break;
                }
                let pos_logits = &logits[offset..offset + vocab_size];
                let target_token = argmax(pos_logits);

                if target_token == draft_token {
                    results.push((*seq_id, draft_token));
                    accepted += 1;
                } else {
                    results.push((*seq_id, target_token));
                    break;
                }
            }

            // Add a bonus token if all drafts were accepted
            if accepted == drafts.len() {
                let bonus_offset = accepted * vocab_size;
                if bonus_offset + vocab_size <= logits.len() {
                    let bonus_logits = &logits[bonus_offset..bonus_offset + vocab_size];
                    let bonus_token = argmax(bonus_logits);
                    results.push((*seq_id, bonus_token));
                }
            }

            accepted_counts.push(accepted);
        }

        Ok((results, accepted_counts))
    }
}
