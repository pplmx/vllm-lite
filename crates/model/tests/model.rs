use vllm_model::fake::FakeModel;
use vllm_traits::ModelBackend;

#[test]
fn test_fake_model_output_count() {
    let mut model = FakeModel::new(1000);
    let seq_ids = vec![1u64, 2, 3];
    let input_tokens = vec![vec![1u32, 2], vec![3, 4], vec![5, 6]];
    let positions = vec![vec![0usize, 1], vec![0, 1], vec![0, 1]];
    let kv_block_ids = vec![vec![0usize], vec![0], vec![0]];
    let num_computed_tokens = vec![0usize, 0, 0];
    let is_prefill = vec![true, true, true];

    let output = model
        .forward(
            &seq_ids,
            &input_tokens,
            &positions,
            &kv_block_ids,
            &num_computed_tokens,
            &is_prefill,
        )
        .unwrap();
    assert_eq!(output.seq_ids.len(), 3);
    assert_eq!(output.next_tokens.len(), 3);
}

#[test]
fn test_fake_model_deterministic() {
    let mut model = FakeModel::new(42);
    let seq_ids = vec![1u64];
    let input_tokens = vec![vec![1u32, 2, 3]];
    let positions = vec![vec![0usize, 1, 2]];
    let kv_block_ids = vec![vec![0usize]];
    let num_computed_tokens = vec![0usize];
    let is_prefill = vec![true];

    let output1 = model
        .forward(
            &seq_ids,
            &input_tokens,
            &positions,
            &kv_block_ids,
            &num_computed_tokens,
            &is_prefill,
        )
        .unwrap();
    let output2 = model
        .forward(
            &seq_ids,
            &input_tokens,
            &positions,
            &kv_block_ids,
            &num_computed_tokens,
            &is_prefill,
        )
        .unwrap();

    // Same input should produce same output
    assert_eq!(output1.next_tokens, output2.next_tokens);
}

#[test]
fn test_fake_model_different_seqs_different_output() {
    let mut model = FakeModel::new(100);
    let seq_ids = vec![1u64, 2u64];
    let input_tokens = vec![vec![1u32], vec![1u32]];
    let positions = vec![vec![0usize], vec![0usize]];
    let kv_block_ids = vec![vec![0usize], vec![0]];
    let num_computed_tokens = vec![0usize, 0];
    let is_prefill = vec![true, true];

    let output = model
        .forward(
            &seq_ids,
            &input_tokens,
            &positions,
            &kv_block_ids,
            &num_computed_tokens,
            &is_prefill,
        )
        .unwrap();

    // Different sequence IDs should produce different tokens
    assert_ne!(
        output.next_tokens[0], output.next_tokens[1],
        "different seq_ids should produce different tokens"
    );
}

#[test]
fn test_fake_model_batch_size_respected() {
    let mut model = FakeModel::new(1000);

    // Test various batch sizes
    for batch_size in [1, 2, 5, 10] {
        let seq_ids: Vec<u64> = (0..batch_size).map(|i| i as u64).collect();
        let input_tokens: Vec<Vec<u32>> = (0..batch_size).map(|_| vec![1]).collect();
        let positions: Vec<Vec<usize>> = (0..batch_size).map(|_| vec![0]).collect();
        let kv_block_ids: Vec<Vec<usize>> = (0..batch_size).map(|_| vec![0]).collect();
        let num_computed_tokens: Vec<usize> = (0..batch_size).map(|_| 0).collect();
        let is_prefill: Vec<bool> = (0..batch_size).map(|_| true).collect();

        let output = model
            .forward(
                &seq_ids,
                &input_tokens,
                &positions,
                &kv_block_ids,
                &num_computed_tokens,
                &is_prefill,
            )
            .unwrap();
        assert_eq!(
            output.seq_ids.len(),
            batch_size,
            "batch size {} not respected",
            batch_size
        );
        assert_eq!(
            output.next_tokens.len(),
            batch_size,
            "batch size {} not respected",
            batch_size
        );
    }
}

#[test]
fn test_model_empty_batch() {
    let mut model = FakeModel::new(1000);
    let seq_ids: Vec<u64> = vec![];
    let input_tokens: Vec<Vec<u32>> = vec![];
    let positions: Vec<Vec<usize>> = vec![];
    let kv_block_ids: Vec<Vec<usize>> = vec![];
    let num_computed_tokens: Vec<usize> = vec![];
    let is_prefill: Vec<bool> = vec![];

    let output = model
        .forward(
            &seq_ids,
            &input_tokens,
            &positions,
            &kv_block_ids,
            &num_computed_tokens,
            &is_prefill,
        )
        .unwrap();
    assert_eq!(output.seq_ids.len(), 0);
    assert_eq!(output.next_tokens.len(), 0);
}

#[test]
fn test_model_single_token_batch() {
    let mut model = FakeModel::new(1000);
    let seq_ids = vec![1u64];
    let input_tokens = vec![vec![42u32]];
    let positions = vec![vec![0usize]];
    let kv_block_ids = vec![vec![0usize]];
    let num_computed_tokens = vec![0usize];
    let is_prefill = vec![true];

    let output = model
        .forward(
            &seq_ids,
            &input_tokens,
            &positions,
            &kv_block_ids,
            &num_computed_tokens,
            &is_prefill,
        )
        .unwrap();
    assert_eq!(output.seq_ids.len(), 1);
    assert_eq!(output.next_tokens.len(), 1);
    assert_eq!(output.next_tokens[0], 1);
}

#[test]
fn test_model_large_batch() {
    let mut model = FakeModel::new(1000);
    let batch_size = 32;
    let seq_ids: Vec<u64> = (0..batch_size).map(|i| i as u64).collect();
    let input_tokens: Vec<Vec<u32>> = (0..batch_size).map(|i| vec![i as u32]).collect();
    let positions: Vec<Vec<usize>> = (0..batch_size).map(|_| vec![0]).collect();
    let kv_block_ids: Vec<Vec<usize>> = (0..batch_size).map(|_| vec![0]).collect();
    let num_computed_tokens: Vec<usize> = (0..batch_size).map(|_| 0).collect();
    let is_prefill: Vec<bool> = (0..batch_size).map(|_| true).collect();

    let output = model
        .forward(
            &seq_ids,
            &input_tokens,
            &positions,
            &kv_block_ids,
            &num_computed_tokens,
            &is_prefill,
        )
        .unwrap();
    assert_eq!(output.seq_ids.len(), batch_size);
    assert_eq!(output.next_tokens.len(), batch_size);
}
