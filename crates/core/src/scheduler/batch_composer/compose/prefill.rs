// crates/core/src/scheduler/batch_composer/compose/prefill.rs
//
// Prefill-phase composition. Two entry points:
// - `compose_prefill_with_packing`: packing-aware dispatcher (currently
//   delegates to standard prefill — see comment in body)
// - `compose_prefill_batch`: standard prefill batch builder
//
// Both produce a `Batch` with `BatchPhase::Prefill`. Sequences are sorted
// by remaining-token count (shorter first) so shorter sequences get
// included first when the token budget is tight (improves packing).

use super::BatchComposer;
use crate::types::Sequence;
use vllm_traits::{Batch, BatchPhase, TokenId};

impl BatchComposer {
    /// Compose a prefill batch with sequence packing optimization.
    ///
    /// For now, this delegates to the standard prefill composer — the
    /// packing optimization that would reorder sequences to minimize
    /// padding is deferred. Behaviour is unchanged from
    /// `compose_prefill_batch`; the entry point exists so callers can
    /// opt into packing-aware composition without changing the public
    /// dispatcher.
    pub(super) fn compose_prefill_with_packing(&self, sequences: Vec<Sequence>) -> Batch {
        self.compose_prefill_batch(sequences)
    }

    /// Compose a prefill batch
    pub(super) fn compose_prefill_batch(&self, mut sequences: Vec<Sequence>) -> Batch {
        // Sort by remaining token count (shorter first for better packing)
        // H-13 (PERF-04): `sort_unstable_by_key` drops the stability
        // guarantee that `sort_by_key` provides. Stable ordering is not
        // relied on by downstream consumers of the prefill batch.
        sequences.sort_unstable_by_key(|s| s.tokens.len().saturating_sub(s.num_computed_tokens));

        // H-13 (PERF-03): pre-size the output vecs to `max_batch_size`
        // so the first `max_batch_size` pushes do not trigger
        // reallocation. Matches the decode-path pattern.
        let capacity = self.config.max_batch_size;
        let mut seq_ids = Vec::with_capacity(capacity);
        let mut input_tokens = Vec::with_capacity(capacity);
        let mut positions = Vec::with_capacity(capacity);
        let mut kv_block_ids = Vec::with_capacity(capacity);
        let mut num_computed_tokens = Vec::with_capacity(capacity);
        let mut is_prefill = Vec::with_capacity(capacity);
        let mut sampling_params = Vec::with_capacity(capacity);
        let mut total_tokens = 0usize;
        let mut max_seq_len = 0usize;

        tracing::debug!(
            sequences_count = sequences.len(),
            max_batch_size = self.config.max_batch_size,
            max_token_budget = self.config.max_token_budget,
            "compose_prefill: starting"
        );

        for seq in sequences.into_iter().take(self.config.max_batch_size) {
            let start = seq.num_computed_tokens;
            let seq_len = seq.tokens.len();
            let tokens_to_process = seq_len.saturating_sub(start);

            tracing::debug!(
                seq_id = seq.id,
                start = start,
                seq_len = seq_len,
                tokens_to_process = tokens_to_process,
                total_tokens = total_tokens,
                "compose_prefill: processing sequence"
            );

            if tokens_to_process == 0 {
                tracing::debug!("Skipping: tokens_to_process == 0");
                continue;
            }

            if total_tokens + tokens_to_process > self.config.max_token_budget {
                tracing::debug!(
                    "Breaking: total_tokens {} + tokens_to_process {} > max_token_budget {}",
                    total_tokens,
                    tokens_to_process,
                    self.config.max_token_budget
                );
                break;
            }

            seq_ids.push(seq.id);

            // Prefill: return all remaining tokens
            let tokens: Vec<TokenId> = seq.tokens[start..].to_vec();
            positions.push((start..seq_len).collect());
            total_tokens += tokens.len();
            max_seq_len = max_seq_len.max(tokens.len());
            input_tokens.push(tokens);

            kv_block_ids.push(seq.kv_blocks.as_ref().clone());
            num_computed_tokens.push(start);
            // Only treat as prefill if this is the first chunk of the sequence
            // If start > 0, this is a resume from partial prefill, use decode mode
            is_prefill.push(start == 0);
            // ARCH-02: thread per-sequence sampling params into the
            // batch so the engine applies them after `forward_logits`.
            sampling_params.push(seq.sampling_params.clone());
        }

        let total = total_tokens;
        let max_len = max_seq_len;

        tracing::debug!(
            batch_seq_count = seq_ids.len(),
            total_tokens = total,
            "compose_prefill: batch built"
        );

        Batch {
            seq_ids,
            input_tokens,
            positions,
            kv_block_ids,
            num_computed_tokens,
            is_prefill,
            sampling_params,
            phase: BatchPhase::Prefill,
            total_tokens: total,
            max_seq_len: max_len,
        }
    }
}
