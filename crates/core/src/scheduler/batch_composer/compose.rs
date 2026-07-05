//! `BatchComposer` implementation: builds phase-specific batches (prefill, decode, mixed) from the running + waiting sequence lists.
//!
//! Runs every scheduler tick; the `compose` function is the entry
//! point and is split into prefill-composition + decode-composition
//! sub-passes with a configurable size budget per pass.

// crates/core/src/scheduler/batch_composer/compose.rs
//
// `BatchComposer` implementation: builds phase-specific batches (prefill,
// decode, chunked prefill, packed prefill).

use super::validate::{BatchCompositionConfig, ChunkedPrefillConfig};
use crate::types::{Phase, Sequence, SequencePackingConfig};
use vllm_traits::{Batch, BatchPhase, TokenId};

#[derive(Debug)]
/// Batch composer for building phase-specific batches
pub struct BatchComposer {
    config: BatchCompositionConfig,
    packing_config: SequencePackingConfig,
    chunked_prefill: ChunkedPrefillConfig,
}

impl BatchComposer {
    /// Create a new batch composer with the given configuration
    #[must_use]
    pub fn new(config: BatchCompositionConfig) -> Self {
        Self {
            config,
            packing_config: SequencePackingConfig::default(),
            chunked_prefill: ChunkedPrefillConfig::default(),
        }
    }

    /// Create a new batch composer with custom packing configuration
    #[must_use]
    pub fn with_packing(
        config: BatchCompositionConfig,
        packing_config: SequencePackingConfig,
    ) -> Self {
        Self {
            config,
            packing_config,
            chunked_prefill: ChunkedPrefillConfig::default(),
        }
    }

    /// Create a new batch composer with chunked prefill configuration
    #[must_use]
    pub fn with_chunked_prefill(
        config: BatchCompositionConfig,
        chunked_prefill: ChunkedPrefillConfig,
    ) -> Self {
        Self {
            config,
            packing_config: SequencePackingConfig::default(),
            chunked_prefill,
        }
    }

    /// Compose batch with optional sequence packing for prefill
    #[must_use]
    pub fn compose(&self, sequences: Vec<Sequence>, phase: Phase) -> Batch {
        match phase {
            Phase::Prefill if self.packing_config.enabled && sequences.len() > 1 => {
                self.compose_prefill_with_packing(sequences)
            }
            _ => self.compose_standard(sequences, phase),
        }
    }

    /// Compose batch with chunked prefill support
    /// Returns a batch that respects memory constraints by splitting long sequences
    #[must_use]
    pub fn compose_with_chunking(
        &self,
        sequences: Vec<Sequence>,
        phase: Phase,
        available_memory: usize,
    ) -> Batch {
        match phase {
            Phase::Prefill => {
                if self.chunked_prefill.enabled {
                    self.compose_chunked_prefill(sequences, available_memory)
                } else {
                    self.compose_standard(sequences, phase)
                }
            }
            Phase::Decode => self.compose_standard(sequences, phase),
        }
    }

    fn compose_chunked_prefill(&self, sequences: Vec<Sequence>, available_memory: usize) -> Batch {
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

    fn compose_prefill_with_packing(&self, sequences: Vec<Sequence>) -> Batch {
        // For now, just use standard composition
        // Full packing optimization would reorder sequences to minimize padding
        // but still include all sequences in the batch
        self.build_batch_from_sequences(sequences, Phase::Prefill)
    }

    fn compose_standard(&self, sequences: Vec<Sequence>, phase: Phase) -> Batch {
        self.build_batch_from_sequences(sequences, phase)
    }

    fn build_batch_from_sequences(&self, sequences: Vec<Sequence>, phase: Phase) -> Batch {
        // Call the existing compose_prefill_batch or compose_decode_batch
        match phase {
            Phase::Prefill => self.compose_prefill_batch(sequences),
            Phase::Decode => self.compose_decode_batch(sequences),
        }
    }

    /// Compose a prefill batch
    fn compose_prefill_batch(&self, mut sequences: Vec<Sequence>) -> Batch {
        // Sort by remaining token count (shorter first for better packing)
        // H-13 (PERF-04): `sort_unstable_by_key` drops the stability
        // guarantee that `sort_by_key` provides. Stable ordering is not
        // relied on by downstream consumers of the prefill batch.
        sequences.sort_unstable_by_key(|s| s.tokens.len().saturating_sub(s.num_computed_tokens));

        // H-13 (PERF-03): pre-size the output vecs to `max_batch_size`
        // so the first `max_batch_size` pushes do not trigger
        // reallocation. Matches the decode-path pattern at lines
        // 265-270.
        let capacity = self.config.max_batch_size;
        let mut seq_ids = Vec::with_capacity(capacity);
        let mut input_tokens = Vec::with_capacity(capacity);
        let mut positions = Vec::with_capacity(capacity);
        let mut kv_block_ids = Vec::with_capacity(capacity);
        let mut num_computed_tokens = Vec::with_capacity(capacity);
        let mut is_prefill = Vec::with_capacity(capacity);
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
            phase: BatchPhase::Prefill,
            total_tokens: total,
            max_seq_len: max_len,
        }
    }

    /// Compose a decode batch
    fn compose_decode_batch(&self, sequences: Vec<Sequence>) -> Batch {
        let batch_size = sequences.len().min(self.config.max_batch_size);

        let mut seq_ids = Vec::with_capacity(batch_size);
        let mut input_tokens = Vec::with_capacity(batch_size);
        let mut positions = Vec::with_capacity(batch_size);
        let mut kv_block_ids = Vec::with_capacity(batch_size);
        let mut num_computed_tokens = Vec::with_capacity(batch_size);
        let mut is_prefill = Vec::with_capacity(batch_size);
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
            phase: BatchPhase::Decode,
            total_tokens: total,
            max_seq_len: max_len,
        }
    }
}

impl Default for BatchComposer {
    fn default() -> Self {
        Self::new(BatchCompositionConfig::default())
    }
}

// Tests live in sibling files so this implementation file stays under
// the 800-line soft cap. Property tests are split into a separate file
// so the `proptest` dependency stays scoped to the proptest build.
#[cfg(test)]
#[path = "compose/tests.rs"]
mod tests;

#[cfg(test)]
#[path = "compose/prop_tests.rs"]
mod prop_tests;
