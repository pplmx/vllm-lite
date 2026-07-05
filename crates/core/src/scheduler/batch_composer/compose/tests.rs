//! Unit tests for [`super::BatchComposer`].
//!
//! Lives in a sibling file so the implementation file stays under the
//! 800-line soft cap. Property tests are in a separate file
//! (`prop_tests.rs`) to keep their dependencies (proptest prelude) from
//! leaking into the unit-test build.

use super::*;
use crate::types::{Priority, SamplingParams, Status};
use std::sync::Arc;

fn make_sequence(id: u64, tokens: Vec<u32>, status: Status) -> Sequence {
    Sequence {
        id,
        tokens,
        kv_blocks: Arc::new(vec![usize::try_from(id).expect("bounded test seq id")]),
        num_computed_tokens: 0,
        prompt_len: 3,
        status,
        max_tokens: 10,
        sampling_params: SamplingParams::default(),
        consecutive_decode_rounds: 0,
        priority: Priority::default(),
        degraded_draft: false,
        draft_model_id: None,
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
        .map(|i| {
            // invariant: bounded by configured limit, cannot overflow at runtime.
            let token = u32::try_from(i).expect("bounded test id");
            make_sequence(i, vec![token], Status::Decoding)
        })
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
        .map(|i| {
            // invariant: bounded by configured limit, cannot overflow at runtime.
            let token = u32::try_from(i).expect("bounded test id");
            make_sequence(i, vec![token; 10], Status::Waiting)
        })
        .collect();

    let batch = composer.compose(seqs, Phase::Prefill);

    let total_tokens: usize = batch.input_tokens.iter().map(std::vec::Vec::len).sum();
    assert!(total_tokens <= 5, "Should respect token budget");
}

#[test]
fn test_chunked_prefill_splits_long_sequence() {
    let chunk_config = ChunkedPrefillConfig {
        enabled: true,
        target_chunk_size: 10,
        max_chunk_size: 20,
        min_chunk_size: 5,
    };
    let composer =
        BatchComposer::with_chunked_prefill(BatchCompositionConfig::default(), chunk_config);

    // Long sequence with 50 tokens
    let seq = make_sequence(1, (0..50u32).collect(), Status::Waiting);

    let batch = composer.compose_with_chunking(vec![seq], Phase::Prefill, 10000);

    // Should be chunked into 10-token chunks (target_chunk_size)
    assert_eq!(batch.seq_ids.len(), 1);
    assert!(
        batch.input_tokens[0].len() <= 10,
        "Chunk should respect target size"
    );
    assert!(batch.is_prefill[0], "First chunk should be prefill");
}

#[test]
fn test_chunked_prefill_disabled_uses_full_prefill() {
    let chunk_config = ChunkedPrefillConfig {
        enabled: false,
        target_chunk_size: 10,
        max_chunk_size: 20,
        min_chunk_size: 5,
    };
    let composer =
        BatchComposer::with_chunked_prefill(BatchCompositionConfig::default(), chunk_config);

    let seq = make_sequence(1, (0..50u32).collect(), Status::Waiting);

    let batch = composer.compose_with_chunking(vec![seq], Phase::Prefill, 10000);

    // Should process all tokens when disabled
    assert_eq!(batch.input_tokens[0].len(), 50);
}

#[test]
fn test_chunked_prefill_uses_smaller_chunks_for_very_long_sequences() {
    let chunk_config = ChunkedPrefillConfig {
        enabled: true,
        target_chunk_size: 1024,
        max_chunk_size: 2048,
        min_chunk_size: 64,
    };
    let composer =
        BatchComposer::with_chunked_prefill(BatchCompositionConfig::default(), chunk_config);

    // Very long sequence (16k tokens)
    let seq = make_sequence(1, vec![1u32; 16384], Status::Waiting);

    let batch = composer.compose_with_chunking(vec![seq], Phase::Prefill, 10000);

    // Should use smaller chunks for very long sequences
    assert!(
        batch.input_tokens[0].len() <= 512,
        "Very long sequences should use smaller chunks"
    );
}

#[test]
fn test_chunked_prefill_config_calculation() {
    let config = ChunkedPrefillConfig {
        enabled: true,
        target_chunk_size: 512,
        max_chunk_size: 1024,
        min_chunk_size: 64,
    };

    // Short sequence should not be chunked
    assert_eq!(config.calculate_chunk_size(32, 10000), 32);

    // Medium sequence should use target
    assert_eq!(config.calculate_chunk_size(1000, 10000), 512);

    // Very long sequence should be limited
    let chunk = config.calculate_chunk_size(10000, 10000);
    assert!(chunk <= 1024, "Should respect max chunk size");
}

#[test]
fn test_chunked_prefill_populates_num_computed_tokens() {
    // H-13 (CORRECTNESS-FIX): previously `compose_chunked_prefill`
    // declared `num_computed_tokens` as non-`mut` and left the Vec
    // empty. Any downstream consumer indexing
    // `batch.num_computed_tokens[i]` for a chunked-prefill batch
    // would have panicked. After the fix, the field is populated
    // with the sequence's `num_computed_tokens` (the chunk start).
    let chunk_config = ChunkedPrefillConfig {
        enabled: true,
        target_chunk_size: 10,
        max_chunk_size: 20,
        min_chunk_size: 5,
    };
    let composer =
        BatchComposer::with_chunked_prefill(BatchCompositionConfig::default(), chunk_config);

    let mut seq = make_sequence(1, (0..50u32).collect(), Status::Waiting);
    // Simulate a previously partially-prefilled sequence.
    seq.num_computed_tokens = 30;
    let batch = composer.compose_with_chunking(vec![seq], Phase::Prefill, 10000);

    assert_eq!(batch.num_computed_tokens.len(), batch.seq_ids.len());
    assert_eq!(batch.num_computed_tokens[0], 30);
}

#[test]
fn test_decode_handles_empty_tokens_without_panic() {
    // Regression test: `compose_decode_batch` previously computed
    // `position = tokens_len - 1` which underflowed when the
    // sequence had no tokens. The fix uses `saturating_sub(1)`
    // so an empty sequence produces `position = 0` instead of
    // panicking. Discovered by the proptest property suite.
    let composer = BatchComposer::default();
    let seq = make_sequence(1, vec![], Status::Decoding);

    let batch = composer.compose(vec![seq], Phase::Decode);

    assert_eq!(batch.seq_ids.len(), 1);
    assert_eq!(batch.input_tokens[0], vec![0]);
    assert_eq!(batch.positions[0], vec![0]);
}
