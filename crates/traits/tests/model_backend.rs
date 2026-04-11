use vllm_traits::{BLOCK_SIZE, Batch, BatchOutput, BatchPhase, ModelError, SeqId, TokenId};

#[test]
fn test_batch_output_creation() {
    let output = BatchOutput {
        seq_ids: vec![1, 2, 3],
        next_tokens: vec![10, 20, 30],
    };
    assert_eq!(output.seq_ids.len(), 3);
    assert_eq!(output.next_tokens.len(), 3);
    assert_eq!(output.seq_ids[0], 1);
    assert_eq!(output.next_tokens[0], 10);
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
        phase: BatchPhase::Prefill,
        total_tokens: 3,
        max_seq_len: 3,
    };
    assert!(!prefill_only_batch.has_decode());
}

#[test]
fn test_model_error_creation() {
    let err = ModelError::new("test error");
    assert_eq!(err.to_string(), "Model error: test error");
}

#[test]
fn test_model_error_display() {
    let err = ModelError::new("connection failed");
    let formatted = format!("{}", err);
    assert!(formatted.contains("connection failed"));
}
