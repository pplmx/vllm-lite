use candle_core::{DType, Device, Tensor};
use vllm_core::types::{Batch, Request};

#[test]
fn test_logits_extraction_1d() {
    let device = Device::Cpu;

    // Simulate logits: [batch=1, seq=1, vocab=5]
    // Values: [[0.1, 0.5, 0.3, 0.8, 0.2]]
    // Max should be at index 3 (value 0.8)
    let logits_data = vec![0.1f32, 0.5, 0.3, 0.8, 0.2];
    let logits = Tensor::from_slice(&logits_data, (1, 1, 5), &device).unwrap();

    // Extract last token: get(batch=0) -> [1,5], get(seq=0) -> [5]
    let batch_size = logits.dims()[0];
    let seq_len = logits.dims()[1];
    let last_logits = logits
        .get(batch_size - 1)
        .unwrap()
        .get(seq_len - 1)
        .unwrap();

    let max_idx = last_logits.argmax(0).unwrap().to_scalar::<u32>().unwrap();

    assert_eq!(max_idx, 3);
}

#[test]
fn test_logits_extraction_multiple_tokens() {
    let device = Device::Cpu;

    // Simulate logits: [batch=1, seq=3, vocab=4]
    // Last token (index 2): [0.1, 0.9, 0.2, 0.3] -> max at index 1
    let logits_data = vec![
        // seq 0
        0.1f32, 0.2, 0.3, 0.4, // seq 1
        0.5, 0.6, 0.7, 0.8, // seq 2 (last)
        0.1, 0.9, 0.2, 0.3,
    ];
    let logits = Tensor::from_slice(&logits_data, (1, 3, 4), &device).unwrap();

    let batch_size = logits.dims()[0];
    let seq_len = logits.dims()[1];
    let last_logits = logits
        .get(batch_size - 1)
        .unwrap()
        .get(seq_len - 1)
        .unwrap();

    let max_idx = last_logits.argmax(0).unwrap().to_scalar::<u32>().unwrap();

    assert_eq!(max_idx, 1);
}

#[test]
fn test_logits_extraction_batch() {
    let device = Device::Cpu;

    // Simulate logits: [batch=2, seq=1, vocab=3]
    // Batch 0: [0.5, 0.1, 0.2] -> max at 0
    // Batch 1: [0.1, 0.8, 0.3] -> max at 1
    let logits_data = vec![
        // batch 0
        0.5f32, 0.1, 0.2, // batch 1
        0.1, 0.8, 0.3,
    ];
    let logits = Tensor::from_slice(&logits_data, (2, 1, 3), &device).unwrap();

    // Get first batch's last token
    let last_logits = logits.get(0).unwrap().get(0).unwrap();
    let max_idx = last_logits.argmax(0).unwrap().to_scalar::<u32>().unwrap();

    assert_eq!(max_idx, 0);
}

#[test]
fn test_wrong_dimension_argmax() {
    // This test demonstrates the bug we fixed
    let device = Device::Cpu;

    // logits: [batch=1, seq=3, vocab=4]
    let logits_data = vec![
        0.1f32, 0.2, 0.3, 0.4, // seq 0
        0.5, 0.6, 0.7, 0.8, // seq 1 - max=0.8 at idx 3
        0.1, 0.9, 0.2, 0.3, // seq 2 - max=0.9 at idx 1
    ];
    let logits = Tensor::from_slice(&logits_data, (1, 3, 4), &device).unwrap();

    // WRONG: Getting first element then argmax over dim 0
    // logits.get(0) -> [3, 4] (all seqs for first batch)
    // argmax(0) -> reduces seq dim -> [4] (one max per vocab)
    let wrong_way = logits.get(0).unwrap();
    let wrong_max = wrong_way.argmax(0).unwrap();
    // wrong_max is [4], each value is max from each vocab position across seqs

    // RIGHT: Get last token then argmax over vocab
    let batch_size = logits.dims()[0];
    let seq_len = logits.dims()[1];
    let right_way = logits
        .get(batch_size - 1)
        .unwrap()
        .get(seq_len - 1)
        .unwrap();
    let right_max = right_way.argmax(0).unwrap().to_scalar::<u32>().unwrap();

    // Right way: correctly finds max at vocab index 1 (value 0.9)
    assert_eq!(right_max, 1);
}
