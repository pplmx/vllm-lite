use crate::types::{Phase, Sequence, SequencePackingConfig};
use vllm_traits::{Batch, BatchPhase, TokenId};

/// Batch composition configuration
#[derive(Clone, Debug)]
pub struct BatchCompositionConfig {
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Maximum token budget
    pub max_token_budget: usize,
    /// Enable similarity grouping
    pub enable_similarity_grouping: bool,
}

impl Default for BatchCompositionConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 256,
            max_token_budget: 4096,
            enable_similarity_grouping: false,
        }
    }
}

/// Batch composer for building phase-specific batches
pub struct BatchComposer {
    config: BatchCompositionConfig,
    packing_config: SequencePackingConfig,
}

impl BatchComposer {
    /// Create a new batch composer with the given configuration
    pub fn new(config: BatchCompositionConfig) -> Self {
        Self {
            config,
            packing_config: SequencePackingConfig::default(),
        }
    }

    /// Create a new batch composer with custom packing configuration
    pub fn with_packing(
        config: BatchCompositionConfig,
        packing_config: SequencePackingConfig,
    ) -> Self {
        Self {
            config,
            packing_config,
        }
    }

    /// Compose batch with optional sequence packing for prefill
    pub fn compose(&self, sequences: Vec<Sequence>, phase: Phase) -> Batch {
        match phase {
            Phase::Prefill if self.packing_config.enabled && sequences.len() > 1 => {
                self.compose_prefill_with_packing(sequences)
            }
            _ => self.compose_standard(sequences, phase),
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
        sequences.sort_by_key(|s| s.tokens.len().saturating_sub(s.num_computed_tokens));

        let mut seq_ids = Vec::new();
        let mut input_tokens = Vec::new();
        let mut positions = Vec::new();
        let mut kv_block_ids = Vec::new();
        let mut num_computed_tokens = Vec::new();
        let mut is_prefill = Vec::new();
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

            eprintln!(
                "compose_prefill: seq_id={}, start={}, seq_len={}, tokens_to_process={}, total_tokens={}, max_token_budget={}",
                seq.id,
                start,
                seq_len,
                tokens_to_process,
                total_tokens,
                self.config.max_token_budget
            );

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
                eprintln!(
                    "compose_prefill: BREAKING due to token budget. total={} + to_process={} > budget={}",
                    total_tokens, tokens_to_process, self.config.max_token_budget
                );
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

        eprintln!(
            "compose_prefill: built batch with {} seqs, total_tokens={}, input_tokens[0].len()={}, positions[0].len()={}",
            seq_ids.len(),
            total,
            input_tokens
                .first()
                .map(|t: &Vec<TokenId>| t.len())
                .unwrap_or(0),
            positions.first().map(|p: &Vec<usize>| p.len()).unwrap_or(0)
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
            let position = tokens_len - 1;
            eprintln!(
                "DEBUG compose_decode: seq_id={}, tokens_len={}, position={}",
                seq.id, tokens_len, position
            );

            input_tokens.push(vec![last_token]);
            positions.push(vec![position]);
            total_tokens += 1;
            max_seq_len = max_seq_len.max(1);

            kv_block_ids.push(seq.kv_blocks.as_ref().clone());
            // num_computed_tokens is used to determine what to read from KV cache
            // For decode, this should be the number of tokens already in KV cache
            // which is seq.tokens.len() - 1 (before processing this decode step)
            num_computed_tokens.push(seq.tokens.len() - 1);
            is_prefill.push(false);
        }

        let total = total_tokens;
        let max_len = max_seq_len;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Priority, SamplingParams, Status};
    use std::sync::Arc;

    fn make_sequence(id: u64, tokens: Vec<u32>, status: Status) -> Sequence {
        Sequence {
            id,
            tokens,
            kv_blocks: Arc::new(vec![id as usize]),
            num_computed_tokens: 0,
            prompt_len: 3,
            status,
            max_tokens: 10,
            sampling_params: SamplingParams::default(),
            consecutive_decode_rounds: 0,
            priority: Priority::default(),
        }
    }

    #[test]
    fn test_prefill_batch_includes_all_prompt_tokens() {
        let composer = BatchComposer::default();
        let seq = make_sequence(1, vec![1, 2, 3, 4, 5], Status::Waiting);

        let batch = composer.compose(vec![seq], Phase::Prefill);

        assert_eq!(batch.seq_ids.len(), 1);
        assert_eq!(batch.input_tokens[0], vec![1, 2, 3, 4, 5]);
        assert!(batch.is_prefill[0]);
    }

    #[test]
    fn test_decode_batch_includes_only_last_token() {
        let composer = BatchComposer::default();
        let seq = make_sequence(1, vec![1, 2, 3, 4, 5], Status::Decoding);

        let batch = composer.compose(vec![seq], Phase::Decode);

        assert_eq!(batch.seq_ids.len(), 1);
        assert_eq!(batch.input_tokens[0], vec![5]);
        assert!(!batch.is_prefill[0]);
    }

    #[test]
    fn test_decode_batch_position_is_zero_indexed() {
        let composer = BatchComposer::default();

        // Test case 1: Single token (after prefill)
        let seq1 = make_sequence(1, vec![42], Status::Decoding);
        let batch1 = composer.compose(vec![seq1], Phase::Decode);
        assert_eq!(
            batch1.positions[0],
            vec![0],
            "Position for 1 token should be 0"
        );

        // Test case 2: 5 tokens (3 prompt + 2 generated)
        let seq2 = make_sequence(2, vec![1, 2, 3, 4, 5], Status::Decoding);
        let batch2 = composer.compose(vec![seq2], Phase::Decode);
        assert_eq!(
            batch2.positions[0],
            vec![4],
            "Position for 5 tokens should be 4 (0-indexed)"
        );

        // Test case 3: 10 tokens
        let seq3 = make_sequence(3, vec![0u32; 10], Status::Decoding);
        let batch3 = composer.compose(vec![seq3], Phase::Decode);
        assert_eq!(
            batch3.positions[0],
            vec![9],
            "Position for 10 tokens should be 9 (0-indexed)"
        );
    }

    #[test]
    fn test_batch_respects_max_size() {
        let config = BatchCompositionConfig {
            max_batch_size: 2,
            max_token_budget: 1000,
            enable_similarity_grouping: false,
        };
        let composer = BatchComposer::new(config);

        let seqs: Vec<_> = (1..=5)
            .map(|i| make_sequence(i, vec![i as u32], Status::Decoding))
            .collect();

        let batch = composer.compose(seqs, Phase::Decode);

        assert_eq!(batch.seq_ids.len(), 2);
    }

    #[test]
    fn test_prefill_respects_token_budget() {
        let config = BatchCompositionConfig {
            max_batch_size: 100,
            max_token_budget: 5,
            enable_similarity_grouping: false,
        };
        let composer = BatchComposer::new(config);

        let seqs: Vec<_> = (1..=10)
            .map(|i| make_sequence(i, vec![i as u32; 10], Status::Waiting))
            .collect();

        let batch = composer.compose(seqs, Phase::Prefill);

        let total_tokens: usize = batch.input_tokens.iter().map(|t| t.len()).sum();
        assert!(total_tokens <= 5, "Should respect token budget");
    }
}
