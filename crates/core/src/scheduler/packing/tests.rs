use super::*;
use crate::types::{Priority, SamplingParams, Status};
use std::sync::Arc;

fn create_sequence(id: u64, len: usize) -> Sequence {
    Sequence {
        id,
        tokens: vec![1u32; len],
        kv_blocks: Arc::new(vec![]),
        num_computed_tokens: 0,
        prompt_len: len,
        status: Status::Waiting,
        max_tokens: 100,
        sampling_params: SamplingParams::default(),
        consecutive_decode_rounds: 0,
        priority: Priority::default(),
    }
}

#[test]
fn test_packer_reduces_padding_waste() {
    let sequences = vec![
        create_sequence(1, 1000),
        create_sequence(2, 100),
        create_sequence(3, 95),
        create_sequence(4, 10),
        create_sequence(5, 200),
    ];

    let config = SequencePackingConfig::default();
    let packer = SequencePacker::new(config);
    let batches = packer.pack_sequences(sequences);

    // Calculate total waste
    let total_waste: usize = batches.iter().map(|b| b.padding_waste).sum();

    // FIFO would have waste = (1000-10)+(1000-100)+(1000-95)+(1000-200) = 3495
    // BFD should have significantly less waste
    assert!(
        total_waste < 500,
        "Expected waste < 500, got {}",
        total_waste
    );
}

#[test]
fn test_packer_respects_max_batch_size() {
    let sequences: Vec<_> = (0..50)
        .map(|i| create_sequence(i as u64, 100 + i))
        .collect();

    let config = SequencePackingConfig {
        max_batch_size: 10,
        ..Default::default()
    };
    let packer = SequencePacker::new(config);
    let batches = packer.pack_sequences(sequences);

    for batch in &batches {
        assert!(
            batch.batch_size <= 10,
            "Batch size {} exceeds max {}",
            batch.batch_size,
            10
        );
    }
}

#[test]
fn test_packer_disabled_returns_single_batch() {
    let sequences = vec![create_sequence(1, 100), create_sequence(2, 200)];

    let config = SequencePackingConfig {
        enabled: false,
        ..Default::default()
    };
    let packer = SequencePacker::new(config);
    let batches = packer.pack_sequences(sequences);

    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].batch_size, 2);
}

#[test]
fn test_packer_empty_sequences() {
    let config = SequencePackingConfig::default();
    let packer = SequencePacker::new(config);
    let batches = packer.pack_sequences(vec![]);

    assert!(batches.is_empty());
}

#[test]
fn test_packer_single_sequence() {
    let sequences = vec![create_sequence(1, 100)];

    let config = SequencePackingConfig::default();
    let packer = SequencePacker::new(config);
    let batches = packer.pack_sequences(sequences);

    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].batch_size, 1);
    assert_eq!(batches[0].padding_waste, 0);
}

#[test]
fn test_similar_sequences_packed_together() {
    // Sequences with similar lengths should be packed together
    let sequences = vec![
        create_sequence(1, 100),
        create_sequence(2, 105),
        create_sequence(3, 110),
        create_sequence(4, 1000),
    ];

    let config = SequencePackingConfig {
        similarity_threshold: 0.2,
        ..Default::default()
    };
    let packer = SequencePacker::new(config);
    let batches = packer.pack_sequences(sequences);

    // Should have 2 batches: [1000] and [110, 105, 100]
    assert_eq!(batches.len(), 2);

    // Find batch with 3 sequences
    let large_batch = batches
        .iter()
        .find(|b| b.batch_size == 3)
        .expect("Should have batch with 3 sequences");

    assert!(large_batch.max_seq_len <= 110);
}
