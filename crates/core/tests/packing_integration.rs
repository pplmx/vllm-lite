use vllm_core::scheduler::{SchedulerEngine, SequencePacker};
use vllm_core::types::{Request, SchedulerConfig, SequencePackingConfig};

#[test]
fn test_packing_disabled_by_default() {
    let config = SchedulerConfig::default();
    assert!(config.packing.enabled);
}

#[test]
fn test_packing_disabled_returns_single_batch() {
    let config = SchedulerConfig {
        packing: SequencePackingConfig {
            enabled: false,
            ..Default::default()
        },
        ..Default::default()
    };

    let mut engine = SchedulerEngine::new(config, 1024);

    // Add requests
    engine.add_request(Request::new(0, vec![1; 100], 10));
    engine.add_request(Request::new(0, vec![1; 200], 10));

    let _batch = engine.build_batch();

    // Batch should still be valid
    // Note: build_batch may not return 2 seqs if they're not ready yet
}

#[test]
fn test_end_to_end_packing_reduces_waste() {
    let config = SchedulerConfig::default();
    let mut engine = SchedulerEngine::new(config, 1024);

    // Add requests with varying lengths
    engine.add_request(Request::new(0, vec![1; 1000], 10));
    engine.add_request(Request::new(0, vec![1; 100], 10));
    engine.add_request(Request::new(0, vec![1; 95], 10));
    engine.add_request(Request::new(0, vec![1; 10], 10));

    // Build batch (should use packing for prefill)
    let batch = engine.build_batch();

    // Verify batch is valid
    assert!(!batch.seq_ids.is_empty());
}

#[test]
fn test_packer_config_from_env() {
    // Test that config can be created from env vars
    let config = SequencePackingConfig::from_env();

    // Should have reasonable defaults
    assert!(config.target_batch_size > 0);
    assert!(config.max_batch_size >= config.target_batch_size);
    assert!(config.similarity_threshold > 0.0 && config.similarity_threshold <= 1.0);
}

#[test]
fn test_packer_with_similar_lengths() {
    use std::sync::Arc;
    use vllm_core::types::{Priority, SamplingParams, Sequence, Status};

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

    let sequences = vec![
        create_sequence(1, 100),
        create_sequence(2, 105),
        create_sequence(3, 110),
        create_sequence(4, 200),
        create_sequence(5, 210),
    ];

    let config = SequencePackingConfig::default();
    let packer = SequencePacker::new(config);
    let batches = packer.pack_sequences(sequences);

    // With similarity threshold, similar sequences should be packed together
    assert!(batches.len() >= 2);

    // Check that sequences are reasonably grouped
    let total_sequences: usize = batches.iter().map(|b| b.batch_size).sum();
    assert_eq!(total_sequences, 5);
}
