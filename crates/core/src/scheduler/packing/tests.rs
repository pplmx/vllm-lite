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
        degraded_draft: false,
        draft_model_id: None,
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
    assert!(total_waste < 500, "Expected waste < 500, got {total_waste}");
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

/// Best-fit must pick the batch that needs the LEAST extra padding.
///
/// `find_best_fit` scores each candidate batch with
/// `a.max_seq_len.max(seq_len) - a.max_seq_len` — the padding the new sequence
/// would add. `cargo mutants` relaxed both subtractions to `+` and `/`, and all
/// four variants survived because no test drove `find_best_fit` with two
/// candidate batches at all. With `+` the score becomes a SUM, which ranking-
/// inverses the choice: the batch that needs more padding wins; with `/` the
/// score is a ratio, which also inverts.
///
/// Fixture: two compatible batches (max lengths 10 and 12) and a 100-token
/// sequence LONGER than both, so each candidate's cost is a real difference
/// (90 vs 88) rather than a tie.
#[test]
fn find_best_fit_picks_the_batch_with_least_added_padding() {
    let config = SequencePackingConfig {
        // Both candidates must pass the similarity filter for the padding
        // ranking to be the only thing that decides.
        similarity_threshold: 1.0,
        ..Default::default()
    };
    let packer = SequencePacker::new(config);

    let mut narrow = PackedBatch::new();
    narrow.add_sequence(create_sequence(1, 10));
    let mut wide = PackedBatch::new();
    wide.add_sequence(create_sequence(2, 12));
    assert_eq!(
        narrow.max_seq_len, 10,
        "fixture: first candidate is shorter"
    );
    assert_eq!(wide.max_seq_len, 12, "fixture: second candidate is longer");
    let batches = vec![narrow, wide];

    // A sequence longer than every candidate: adding it to the 12-wide batch
    // costs 88 padding tokens, to the 10-wide batch 90.
    let incoming = create_sequence(9, 100);
    assert_eq!(
        packer.find_best_fit(&batches, &incoming),
        Some(1),
        "the batch with the larger max_seq_len needs less extra padding"
    );

    // Control: a sequence that fits inside both costs nothing either way, so
    // the first candidate wins on the tie (and this cannot pass by accident if
    // the filter rejects everything).
    let short = create_sequence(10, 5);
    assert_eq!(
        packer.find_best_fit(&batches, &short),
        Some(0),
        "a shorter sequence ties at zero padding, so the first batch wins"
    );

    // Second shape, because the first cannot separate every mutant: here the
    // FIRST candidate has the larger max length (21 vs 20), so it wins with 79
    // tokens of padding against 80. The `-` -> `/` variant scores the longer
    // candidate as `100 / 20 = 5` and picks it instead — a division is not a
    // padding difference, and it inverts the ranking.
    let mut first = PackedBatch::new();
    first.add_sequence(create_sequence(3, 21));
    let mut second = PackedBatch::new();
    second.add_sequence(create_sequence(4, 20));
    let batches = vec![first, second];
    assert_eq!(
        packer.find_best_fit(&batches, &incoming),
        Some(0),
        "the 21-wide batch needs 79 tokens of padding against the 20-wide \
         batch's 80, so it must win"
    );
}
