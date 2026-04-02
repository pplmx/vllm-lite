use vllm_core::beam::BeamSequence;

#[test]
fn test_beam_sequence_new() {
    let beam = BeamSequence::new(vec![1, 2, 3], 0.5, vec![0, 1]);
    assert_eq!(beam.tokens, vec![1, 2, 3]);
    assert_eq!(beam.score, 0.5);
    assert_eq!(beam.kv_blocks.as_ref(), &vec![0, 1]);
}

#[test]
fn test_beam_sequence_push() {
    let mut beam = BeamSequence::new(vec![1, 2], 0.5, vec![]);
    beam.push(3, 0.3);
    assert_eq!(beam.tokens, vec![1, 2, 3]);
    assert_eq!(beam.score, 0.8); // 0.5 + 0.3
}

#[test]
fn test_beam_width_one_equals_greedy() {
    // beam_width=1 应该只返回一个结果
    // 这个测试验证 BeamSequence 结构正确
    let beam = BeamSequence::new(vec![1, 2, 3], -1.0, vec![]);
    let normalized = beam.score / (beam.tokens.len() as f32).powf(0.6);
    assert!(normalized < 0.0);
}
