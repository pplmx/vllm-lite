use vllm_core::engine::ModelBackend;
use vllm_model::fake::FakeModel;

#[test]
fn test_fake_model_output_count() {
    let model = FakeModel::new(1000);
    let seq_ids = vec![1u64, 2, 3];
    let input_tokens = vec![vec![1u32, 2], vec![3, 4], vec![5, 6]];
    let positions = vec![vec![0usize, 1], vec![0, 1], vec![0, 1]];

    let output = model.forward(&seq_ids, &input_tokens, &positions).unwrap();
    assert_eq!(output.seq_ids.len(), 3);
    assert_eq!(output.next_tokens.len(), 3);
}
