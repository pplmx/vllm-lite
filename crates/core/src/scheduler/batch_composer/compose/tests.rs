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

/// The long-sequence chunk tiers must be exercised at their boundaries.
///
/// `calculate_chunk_size` caps the chunk at 512 above 8192 tokens and at 1024
/// above 4096. The pre-existing test only used `seq_len` 32 and 1000, so BOTH
/// tier guards were unvisited: `cargo mutants` flipped `>` to `>=`, `==` and
/// `<` on those lines and every variant survived, meaning the caps that keep a
/// single prefill chunk from OOMing a long prompt were never verified.
/// Auto mode (`target_chunk_size == 0`) must size the chunk from the memory
/// budget, not from the target.
///
/// Every existing test set a non-zero `target_chunk_size`, so the auto branch
/// was never executed: `cargo mutants` replaced its `/` with `*` and nothing
/// failed, even though the two differ by three orders of magnitude (a 100 KiB
/// budget at 128 bytes/token is a 800-token chunk; multiplying gives 13
/// million, which the clamp then pins to `max_chunk_size` for every input).
#[test]
fn test_auto_chunk_size_derives_from_the_memory_budget() {
    let config = ChunkedPrefillConfig {
        enabled: true,
        target_chunk_size: 0, // auto
        max_chunk_size: 4096,
        min_chunk_size: 64,
    };

    // 103_424 bytes available - 1024 headroom = 102_400; / 128 = 800.
    // The sequence is long enough that neither tier cap applies.
    assert_eq!(
        config.calculate_chunk_size(2000, 103_424),
        800,
        "auto mode must divide the memory budget by the per-token cost"
    );

    // A budget below `min_chunk_size` worth of tokens floors at the minimum.
    assert_eq!(
        config.calculate_chunk_size(2000, 1024),
        64,
        "auto mode must not go below min_chunk_size"
    );

    // A budget above the maximum clamps down to max_chunk_size.
    assert_eq!(
        config.calculate_chunk_size(2000, 10 * 1024 * 1024),
        4096,
        "auto mode must not exceed max_chunk_size"
    );
}

#[test]
fn test_chunk_size_tiers_respect_their_boundaries() {
    let config = ChunkedPrefillConfig {
        enabled: true,
        // Deliberately above both caps so the tier, not the target, decides.
        target_chunk_size: 4096,
        max_chunk_size: 8192,
        min_chunk_size: 64,
    };

    // At or below 4096 the tier caps do not apply: the target survives.
    assert_eq!(
        config.calculate_chunk_size(4096, 1_000_000),
        4096,
        "4096 is NOT above the first tier (guard is `> 4096`)"
    );

    // Just above 4096 -> capped at 1024.
    assert_eq!(
        config.calculate_chunk_size(4097, 1_000_000),
        1024,
        "4097 must take the 1024 cap"
    );

    // At 8192 the second tier has not applied yet (guard is `> 8192`), so the
    // first tier's cap still governs.
    assert_eq!(
        config.calculate_chunk_size(8192, 1_000_000),
        1024,
        "8192 must still use the 1024 cap, not 512"
    );

    // Just above 8192 -> capped at 512.
    assert_eq!(
        config.calculate_chunk_size(8193, 1_000_000),
        512,
        "8193 must take the 512 cap"
    );
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
fn test_partial_prefill_resume_keeps_is_prefill_true() {
    // Regression: a sequence resuming prefill after a partial
    // prefix-cache hit (num_computed_tokens > 0 with a multi-token suffix)
    // must keep is_prefill = true so the model dispatches to forward_prefill
    // -> forward_prefill_continue (reads cached KV, processes the chunk).
    // Pre-fix classify_prefill_seq set is_prefill = (start == 0), routing the
    // multi-token suffix to the single-token forward_decode path.
    let composer = BatchComposer::default();
    let mut seq = make_sequence(1, (0..10u32).collect(), Status::Prefilling);
    seq.prompt_len = 10;
    seq.num_computed_tokens = 4; // partial hit: 4 cached, 6-token suffix
    let batch = composer.compose(vec![seq], Phase::Prefill);
    assert_eq!(batch.seq_ids.len(), 1);
    assert!(
        batch.is_prefill[0],
        "a resumed prefill (num_computed > 0) must keep is_prefill=true so it routes to forward_prefill_continue, not single-token forward_decode"
    );
    // The full 6-token suffix is processed in this prefill batch.
    assert_eq!(batch.input_tokens[0].len(), 6);
    assert_eq!(batch.num_computed_tokens[0], 4);
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
