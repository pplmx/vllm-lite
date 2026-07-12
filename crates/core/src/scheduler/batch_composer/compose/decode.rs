// crates/core/src/scheduler/batch_composer/compose/decode.rs
//
// Decode-phase composition. Produces a `Batch` with `BatchPhase::Decode`:
// one token per sequence (the most recent), with positions pointing at
// the index of the new token (tokens_len - 1).

use super::BatchComposer;
use crate::types::Sequence;
use vllm_traits::{Batch, BatchPhase};

impl BatchComposer {
    /// Compose a decode batch
    pub(super) fn compose_decode_batch(&self, sequences: Vec<Sequence>) -> Batch {
        let batch_size = sequences.len().min(self.config.max_batch_size);

        let mut seq_ids = Vec::with_capacity(batch_size);
        let mut input_tokens = Vec::with_capacity(batch_size);
        let mut positions = Vec::with_capacity(batch_size);
        let mut kv_block_ids = Vec::with_capacity(batch_size);
        let mut num_computed_tokens = Vec::with_capacity(batch_size);
        let mut is_prefill = Vec::with_capacity(batch_size);
        let mut sampling_params = Vec::with_capacity(batch_size);
        let mut total_tokens = 0;
        let mut max_seq_len = 0;

        for seq in sequences.into_iter().take(batch_size) {
            seq_ids.push(seq.id);

            // Decode: only last token
            let last_token = seq.tokens.last().copied().unwrap_or(0);
            // position is the index of the current token (0-indexed)
            // After prefill, seq.tokens has prompt_len tokens, so position = prompt_len (the new token's index)
            // After N decode steps, seq.tokens has prompt_len + N tokens, so position = prompt_len + N - 1
            let tokens_len = seq.tokens.len();
            // tokens_len is the count, position should be the index (0-indexed)
            // If tokens_len=1, position=0 (first token at index 0)
            // If tokens_len=10, position=9 (10th token at index 9)
            // H-14 (CORRECTNESS-FIX): use `saturating_sub` so a sequence
            // with no tokens yields `position = 0` instead of underflowing.
            // Discovered by proptest property suite (v28.0 I-5).
            let position = tokens_len.saturating_sub(1);
            tracing::debug!(
                seq_id = seq.id,
                tokens_len = tokens_len,
                position = position,
                "compose_decode: processing sequence"
            );

            input_tokens.push(vec![last_token]);
            positions.push(vec![position]);
            total_tokens += 1;
            max_seq_len = max_seq_len.max(1);

            kv_block_ids.push(seq.kv_blocks.as_ref().clone());
            // num_computed_tokens is used to determine what to read from KV cache
            // For decode, this should be the number of tokens already in KV cache
            // which is seq.tokens.len() - 1 (before processing this decode step).
            // H-14: saturate to avoid underflow on empty-token sequences.
            num_computed_tokens.push(seq.tokens.len().saturating_sub(1));
            is_prefill.push(false);
            // ARCH-02: carry the per-sequence sampling params so the
            // engine applies them after `forward_logits`. The HTTP
            // layer already received these — the seam that used to drop
            // them closed in `step_regular` /
            // `engine::graph_step::execute_regular`.
            sampling_params.push(seq.sampling_params.clone());
        }

        let total = total_tokens;
        let max_len = max_seq_len;

        tracing::debug!(
            batch_seq_count = seq_ids.len(),
            total_tokens = total,
            "compose_decode: batch built"
        );

        Batch {
            seq_ids,
            input_tokens,
            positions,
            kv_block_ids,
            num_computed_tokens,
            is_prefill,
            sampling_params,
            phase: BatchPhase::Decode,
            total_tokens: total,
            max_seq_len: max_len,
        }
    }
}
