use vllm_traits::{
    BLOCK_SIZE, Batch, BatchOutput, BatchPhase, ModelError, SampledToken, SamplingParams, SeqId,
    TokenId,
};

#[test]
fn test_batch_output_creation() {
    let output = BatchOutput {
        seq_ids: vec![1, 2, 3],
        // P36: BatchOutput::next_tokens is `Vec<SampledToken>`.
        next_tokens: vec![
            SampledToken {
                token: 10,
                logprob: 0.0,
                top_logprobs: Vec::new(),
            },
            SampledToken {
                token: 20,
                logprob: 0.0,
                top_logprobs: Vec::new(),
            },
            SampledToken {
                token: 30,
                logprob: 0.0,
                top_logprobs: Vec::new(),
            },
        ],
    };
    assert_eq!(output.seq_ids.len(), 3);
    assert_eq!(output.next_tokens.len(), 3);
    assert_eq!(output.seq_ids[0], 1);
    assert_eq!(output.next_tokens[0].token, 10);
}

#[test]
fn test_batch_output_empty() {
    let output = BatchOutput {
        seq_ids: vec![],
        next_tokens: vec![],
    };
    assert!(output.seq_ids.is_empty());
    assert!(output.next_tokens.is_empty());
}

#[test]
fn test_seq_id_type() {
    let seq_id: SeqId = 42;
    assert_eq!(seq_id, 42);
}

#[test]
fn test_token_id_type() {
    let token_id: TokenId = 100;
    assert_eq!(token_id, 100);
}

#[test]
fn test_block_size_constant() {
    assert_eq!(BLOCK_SIZE, 16);
}

#[test]
fn test_batch_creation() {
    let batch = Batch {
        seq_ids: vec![1, 2],
        input_tokens: vec![vec![10, 20], vec![30, 40]],
        positions: vec![vec![0, 1], vec![0, 1]],
        kv_block_ids: vec![vec![0], vec![1]],
        num_computed_tokens: vec![0, 0],
        is_prefill: vec![true, false],
        sampling_params: vec![SamplingParams::default(); 2],
        phase: BatchPhase::Mixed,
        total_tokens: 4,
        max_seq_len: 2,
    };
    assert_eq!(batch.seq_ids.len(), 2);
    assert_eq!(batch.input_tokens.len(), 2);
}

#[test]
fn test_batch_is_empty() {
    let empty_batch = Batch {
        seq_ids: vec![],
        input_tokens: vec![],
        positions: vec![],
        kv_block_ids: vec![],
        num_computed_tokens: vec![],
        is_prefill: vec![],
        sampling_params: vec![],
        phase: BatchPhase::Mixed,
        total_tokens: 0,
        max_seq_len: 0,
    };
    assert!(empty_batch.is_empty());

    let non_empty_batch = Batch {
        seq_ids: vec![1],
        input_tokens: vec![vec![10]],
        positions: vec![vec![0]],
        kv_block_ids: vec![vec![0]],
        num_computed_tokens: vec![0],
        is_prefill: vec![true],
        sampling_params: vec![SamplingParams::default()],
        phase: BatchPhase::Prefill,
        total_tokens: 1,
        max_seq_len: 1,
    };
    assert!(!non_empty_batch.is_empty());
}

#[test]
fn test_batch_has_prefill() {
    let prefill_batch = Batch {
        seq_ids: vec![1, 2],
        input_tokens: vec![vec![10], vec![20]],
        positions: vec![vec![0], vec![0]],
        kv_block_ids: vec![vec![0], vec![0]],
        num_computed_tokens: vec![0, 0],
        is_prefill: vec![true, false],
        sampling_params: vec![SamplingParams::default(); 2],
        phase: BatchPhase::Mixed,
        total_tokens: 2,
        max_seq_len: 1,
    };
    assert!(prefill_batch.has_prefill());

    let decode_only_batch = Batch {
        seq_ids: vec![1],
        input_tokens: vec![vec![10]],
        positions: vec![vec![5]],
        kv_block_ids: vec![vec![0]],
        num_computed_tokens: vec![5],
        is_prefill: vec![false],
        sampling_params: vec![SamplingParams::default()],
        phase: BatchPhase::Decode,
        total_tokens: 1,
        max_seq_len: 1,
    };
    assert!(!decode_only_batch.has_prefill());
}

#[test]
fn test_batch_has_decode() {
    let decode_batch = Batch {
        seq_ids: vec![1, 2],
        input_tokens: vec![vec![10], vec![20]],
        positions: vec![vec![0], vec![5]],
        kv_block_ids: vec![vec![0], vec![0]],
        num_computed_tokens: vec![0, 5],
        is_prefill: vec![true, false],
        sampling_params: vec![SamplingParams::default(); 2],
        phase: BatchPhase::Mixed,
        total_tokens: 2,
        max_seq_len: 1,
    };
    assert!(decode_batch.has_decode());

    let prefill_only_batch = Batch {
        seq_ids: vec![1],
        input_tokens: vec![vec![10, 20, 30]],
        positions: vec![vec![0, 1, 2]],
        kv_block_ids: vec![vec![0]],
        num_computed_tokens: vec![0],
        is_prefill: vec![true],
        sampling_params: vec![SamplingParams::default()],
        phase: BatchPhase::Prefill,
        total_tokens: 3,
        max_seq_len: 3,
    };
    assert!(!prefill_only_batch.has_decode());
}

#[test]
fn test_model_error_creation() {
    let err = ModelError::new("test error");
    assert_eq!(err.to_string(), "test error");
}

#[test]
fn test_model_error_display() {
    let err = ModelError::new("connection failed");
    let formatted = format!("{err}");
    assert!(formatted.contains("connection failed"));
}

/// `Batch::len` must report the number of sequences.
///
/// `cargo mutants` replaced it with the constants 0 and 1 and both survived:
/// the crate's own tests only ever call `is_empty`, and every `len()` call site
/// in the workspace is in a test that is free to assert whatever the
/// implementation returns. `len` is what the server uses to size per-request
/// accounting and what the batching code uses to decide whether a batch is
/// worth a forward pass.
#[test]
fn test_batch_len_counts_sequences() {
    let mut batch = Batch::empty();
    assert_eq!(batch.len(), 0);

    batch.seq_ids = vec![7];
    batch.input_tokens = vec![vec![10]];
    batch.is_prefill = vec![true];
    assert_eq!(batch.len(), 1);

    batch.seq_ids = vec![7, 9, 11];
    batch.input_tokens = vec![vec![10], vec![20], vec![30]];
    batch.is_prefill = vec![true, false, false];
    assert_eq!(batch.len(), 3, "len must track seq_ids, not a constant");

    // `is_empty` and `len` must agree — they read the same field.
    assert!(!batch.is_empty());
    batch.seq_ids.clear();
    assert!(batch.is_empty());
    assert_eq!(batch.len(), 0);
}

/// The phase helpers must agree with the `is_prefill` flags they read.
///
/// `has_prefill` / `has_decode` were each replaced by the constants `true` and
/// `false`, and `has_decode`'s `!p` had its `!` deleted — five survivors. On a
/// mixed batch all of them look right, so the fixture needs the three SHAPES:
/// all-prefill, all-decode, and mixed.
#[test]
fn test_batch_phase_helpers_cover_every_shape() {
    let make = |is_prefill: Vec<bool>| Batch {
        seq_ids: (0..is_prefill.len() as u64).collect(),
        input_tokens: is_prefill.iter().map(|_| vec![1]).collect(),
        positions: is_prefill.iter().map(|_| vec![0]).collect(),
        kv_block_ids: is_prefill.iter().map(|_| vec![0]).collect(),
        num_computed_tokens: is_prefill.iter().map(|_| 0).collect(),
        is_prefill: is_prefill.clone(),
        sampling_params: is_prefill
            .iter()
            .map(|_| SamplingParams::default())
            .collect(),
        phase: BatchPhase::Mixed,
        total_tokens: is_prefill.len(),
        max_seq_len: 1,
    };

    let all_prefill = make(vec![true, true]);
    assert!(all_prefill.has_prefill());
    assert!(
        !all_prefill.has_decode(),
        "an all-prefill batch has no decode sequence"
    );

    let all_decode = make(vec![false, false]);
    assert!(!all_decode.has_prefill());
    assert!(all_decode.has_decode());

    let mixed = make(vec![true, false]);
    assert!(mixed.has_prefill(), "the first sequence prefills");
    assert!(mixed.has_decode(), "the second sequence decodes");

    // Order must not matter: an all-decode batch preceded by one prefill flag
    // still has both phases.
    let mixed_reversed = make(vec![false, true]);
    assert!(mixed_reversed.has_prefill());
    assert!(mixed_reversed.has_decode());

    // An empty batch has neither phase.
    let empty = make(vec![]);
    assert!(!empty.has_prefill());
    assert!(!empty.has_decode());
}
