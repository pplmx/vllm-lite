// crates/core/src/scheduler/batch_composer/compose/chunked.rs
//
// Memory-bounded chunked prefill: splits long sequences into chunks that
// fit within `available_memory`. The first chunk uses `is_prefill = true`
// (since `start == 0`); subsequent chunks of the same sequence are flagged
// `is_prefill = false` to continue prefill incrementally.

use super::BatchComposer;
use crate::types::Sequence;
use vllm_traits::{Batch, BatchPhase, TokenId};

impl BatchComposer {
    /// Compose a chunked-prefill batch respecting memory constraints.
    #[allow(dead_code)] // reachable only via test-only `compose_with_chunking`
    pub(super) fn compose_chunked_prefill(
        &self,
        sequences: Vec<Sequence>,
        available_memory: usize,
    ) -> Batch {
        // H-13 (PERF-03 + CORRECTNESS-FIX): pre-size output vecs to
        // `max_batch_size` (matches the decode/prefill patterns), and
        // declare `num_computed_tokens` as `mut` so we can push the
        // start offset per sequence (matches `compose_prefill_batch`).
        // Previously `num_computed_tokens` was declared non-`mut` and
        // left empty — any downstream consumer indexing
        // `batch.num_computed_tokens[i]` for a chunked-prefill batch
        // would have panicked.
        let capacity = self.config.max_batch_size;
        let mut seq_ids = Vec::with_capacity(capacity);
        let mut input_tokens = Vec::with_capacity(capacity);
        let mut positions = Vec::with_capacity(capacity);
        let mut kv_block_ids = Vec::with_capacity(capacity);
        let mut num_computed_tokens = Vec::with_capacity(capacity);
        let mut is_prefill = Vec::with_capacity(capacity);
        let mut total_tokens = 0usize;
        let mut max_seq_len = 0usize;

        for seq in sequences.into_iter().take(self.config.max_batch_size) {
            let start = seq.num_computed_tokens;
            let seq_len = seq.tokens.len();
            let remaining_tokens = seq_len.saturating_sub(start);

            if remaining_tokens == 0 {
                continue;
            }

            // Calculate chunk size for this sequence
            let chunk_size = self
                .chunked_prefill
                .calculate_chunk_size(remaining_tokens, available_memory);

            // Determine how many tokens to process in this chunk
            let tokens_to_process = remaining_tokens.min(chunk_size);

            // Check token budget
            if total_tokens + tokens_to_process > self.config.max_token_budget {
                break;
            }

            // Process this chunk
            seq_ids.push(seq.id);
            let tokens: Vec<TokenId> = seq.tokens[start..start + tokens_to_process].to_vec();
            positions.push((start..start + tokens_to_process).collect());
            total_tokens += tokens_to_process;
            max_seq_len = max_seq_len.max(tokens_to_process);
            input_tokens.push(tokens);

            kv_block_ids.push(seq.kv_blocks.as_ref().clone());
            num_computed_tokens.push(start);
            // Only treat as prefill if this is the first chunk
            // Subsequent chunks use is_prefill=false to continue prefill
            is_prefill.push(start == 0);

            tracing::debug!(
                seq_id = seq.id,
                start = start,
                chunk_size = tokens_to_process,
                remaining = remaining_tokens - tokens_to_process,
                "chunked_prefill: chunked sequence"
            );
        }

        Batch {
            seq_ids,
            input_tokens,
            positions,
            kv_block_ids,
            num_computed_tokens,
            is_prefill,
            phase: BatchPhase::Prefill,
            total_tokens,
            max_seq_len,
        }
    }
}
