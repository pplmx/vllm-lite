use vllm_core::engine::Engine;
use vllm_core::types::{AdaptiveDraftConfig, Request, SchedulerConfig};
use vllm_testing::IncrementModel;

#[test]
fn test_adaptive_speculative_disabled_by_default() {
    let config = SchedulerConfig::default();
    let engine = Engine::with_config(IncrementModel, None, config, 4, 1024);

    assert!(!engine.is_adaptive_speculative_enabled());
}

#[test]
fn test_enable_adaptive_speculative() {
    let config = SchedulerConfig::default();
    let mut engine = Engine::with_config(IncrementModel, None, config, 4, 1024);

    engine.enable_adaptive_speculative(AdaptiveDraftConfig::default());
    assert!(engine.is_adaptive_speculative_enabled());
    assert!(engine.speculative_mode);
}

#[test]
fn test_disable_adaptive_speculative() {
    let config = SchedulerConfig::default();
    let mut engine = Engine::with_config(IncrementModel, None, config, 4, 1024);

    engine.enable_adaptive_speculative(AdaptiveDraftConfig::default());
    assert!(engine.is_adaptive_speculative_enabled());

    engine.disable_adaptive_speculative();
    assert!(!engine.is_adaptive_speculative_enabled());
    assert!(!engine.speculative_mode);
}

#[test]
fn test_adaptive_speculative_basic() {
    let config = SchedulerConfig::default();
    let mut engine = Engine::with_config(IncrementModel, None, config, 4, 1024);

    engine.enable_adaptive_speculative(AdaptiveDraftConfig::default());

    assert!(engine.adaptive_decoder.is_some());
    assert!(engine.is_adaptive_speculative_enabled());
}

#[test]
fn test_adaptive_speculative_adjusts_draft_count() {
    let config = SchedulerConfig::default();
    let mut engine = Engine::with_config(IncrementModel, None, config, 4, 1024);

    engine.enable_adaptive_speculative(AdaptiveDraftConfig {
        min_draft_tokens: 2,
        max_draft_tokens: 6,
        target_acceptance_rate: 0.5,
        accuracy_window_size: 5,
        adjustment_step: 1,
        cooldown_steps: 2,
        ewma_alpha: 0.1,
        deadband_threshold: 0.05,
    });

    assert!(engine.adaptive_decoder.is_some());
}

#[test]
fn test_adaptive_speculative_with_same_model_for_draft() {
    use tokio::sync::mpsc;

    let config = SchedulerConfig::default();
    let target_model = IncrementModel;
    let draft_model = IncrementModel;

    let mut engine = Engine::with_config(target_model, Some(draft_model), config, 4, 1024);

    engine.enable_adaptive_speculative(AdaptiveDraftConfig::default());

    assert!(engine.is_adaptive_speculative_enabled());
    assert!(engine.adaptive_decoder.is_some());
    assert!(engine.draft_model.is_some());

    let (tx, _rx) = mpsc::channel(64);
    engine.add_request(Request::new(0, vec![10, 20], 5), tx);

    // Run prefill first with regular step
    let prefill_result = engine.step();
    assert!(prefill_result.is_ok());
    let prefill_outputs = prefill_result.unwrap();
    assert!(
        !prefill_outputs.is_empty(),
        "Prefill should produce at least one output"
    );

    // Then use adaptive speculative for decode steps
    let mut decode_iterations = 0;
    while engine.has_pending() && decode_iterations < 50 {
        let result = engine.step_adaptive_speculative();
        assert!(
            result.is_ok(),
            "step_adaptive_speculative failed: {:?}",
            result
        );
        decode_iterations += 1;
    }

    assert!(
        !engine.has_pending(),
        "Expected completion after {} decode iterations, but still pending. running={}",
        decode_iterations,
        engine.scheduler.running_count()
    );
}

#[test]
fn test_adaptive_speculative_run_loop_uses_adaptive() {
    use tokio::sync::mpsc;

    let config = SchedulerConfig::default();
    let target_model = IncrementModel;
    let draft_model = IncrementModel;

    let mut engine = Engine::with_config(target_model, Some(draft_model), config, 4, 1024);

    engine.enable_adaptive_speculative(AdaptiveDraftConfig {
        min_draft_tokens: 1,
        max_draft_tokens: 3,
        ..Default::default()
    });

    let (tx, _rx) = mpsc::channel(64);
    engine.add_request(Request::new(0, vec![10], 3), tx);

    for _ in 0..10 {
        if !engine.has_pending() {
            break;
        }
        let _ = engine.step_adaptive_speculative();
    }

    assert!(!engine.has_pending());
}

#[test]
fn test_adaptive_speculative_max_draft_tokens() {
    let config = SchedulerConfig::default();
    let mut engine = Engine::with_config(IncrementModel, None, config, 4, 1024);

    engine.enable_adaptive_speculative(AdaptiveDraftConfig {
        min_draft_tokens: 1,
        max_draft_tokens: 5,
        ..Default::default()
    });

    let max = engine
        .adaptive_decoder
        .as_ref()
        .map(|d| d.current_max_draft_tokens());
    assert_eq!(max, Some(5));
}

#[test]
fn test_speculative_verification_with_multiple_drafts() {
    use tokio::sync::mpsc;

    let config = SchedulerConfig::default();
    let target_model = IncrementModel;
    let draft_model = IncrementModel;

    let mut engine = Engine::with_config(target_model, Some(draft_model), config, 4, 1024);

    engine.enable_adaptive_speculative(AdaptiveDraftConfig {
        min_draft_tokens: 2,
        max_draft_tokens: 4,
        ..Default::default()
    });

    let (tx, _rx) = mpsc::channel(64);
    engine.add_request(Request::new(0, vec![10, 20, 30], 10), tx);

    let prefill_result = engine.step();
    assert!(prefill_result.is_ok(), "Prefill should succeed");

    for _ in 0..20 {
        if !engine.has_pending() {
            break;
        }
        let result = engine.step_adaptive_speculative();
        assert!(
            result.is_ok(),
            "Speculative step failed: {:?}",
            result.err()
        );
    }

    assert!(!engine.has_pending(), "Should complete after max_tokens");
}

#[test]
fn test_speculative_verify_batch_dimensions_consistency() {
    use std::sync::Arc;
    use vllm_core::scheduler::BatchComposer;
    use vllm_core::types::{Priority, SamplingParams, Sequence, Status};

    fn make_seq(id: u64, tokens: Vec<u32>, status: Status) -> Sequence {
        Sequence {
            id,
            tokens,
            kv_blocks: Arc::new(vec![id as usize]),
            num_computed_tokens: 0,
            prompt_len: 3,
            status,
            max_tokens: 10,
            sampling_params: SamplingParams::default(),
            consecutive_decode_rounds: 0,
            priority: Priority::default(),
        }
    }

    let composer = BatchComposer::default();

    let seq1 = make_seq(1, vec![1, 2, 3], Status::Decoding);
    let batch = composer.compose(vec![seq1], vllm_core::types::Phase::Decode);

    assert_eq!(batch.seq_ids.len(), 1, "Batch should have 1 sequence");
    assert_eq!(
        batch.input_tokens.len(),
        1,
        "input_tokens should have 1 entry"
    );
    assert_eq!(batch.positions.len(), 1, "positions should have 1 entry");
    assert_eq!(
        batch.kv_block_ids.len(),
        1,
        "kv_block_ids should have 1 entry"
    );
    assert_eq!(
        batch.num_computed_tokens.len(),
        1,
        "num_computed_tokens should have 1 entry"
    );
    assert_eq!(
        batch.is_prefill.len(),
        1,
        "is_prefill should have 1 entry for single sequence batch"
    );
}

#[test]
fn test_prefill_batch_single_sequence() {
    use std::sync::Arc;
    use vllm_core::scheduler::BatchComposer;
    use vllm_core::types::{Priority, SamplingParams, Sequence, Status};

    fn make_seq(id: u64, tokens: Vec<u32>, status: Status) -> Sequence {
        Sequence {
            id,
            tokens,
            kv_blocks: Arc::new(vec![id as usize]),
            num_computed_tokens: 0,
            prompt_len: 5,
            status,
            max_tokens: 10,
            sampling_params: SamplingParams::default(),
            consecutive_decode_rounds: 0,
            priority: Priority::default(),
        }
    }

    let composer = BatchComposer::default();

    // Test prefill with single sequence
    let seq1 = make_seq(1, vec![1, 2, 3, 4, 5], Status::Waiting);
    let batch = composer.compose(vec![seq1], vllm_core::types::Phase::Prefill);

    assert_eq!(batch.seq_ids.len(), 1, "Batch should have 1 sequence");
    assert_eq!(
        batch.input_tokens[0],
        vec![1, 2, 3, 4, 5],
        "All tokens should be in prefill batch"
    );
    assert!(
        batch.is_prefill[0],
        "is_prefill should be true for first prefill"
    );
}

#[test]
fn test_prefill_batch_with_partial_computed() {
    use std::sync::Arc;
    use vllm_core::scheduler::BatchComposer;
    use vllm_core::types::{Priority, SamplingParams, Sequence, Status};

    fn make_seq(id: u64, tokens: Vec<u32>, status: Status, num_computed: usize) -> Sequence {
        Sequence {
            id,
            tokens,
            kv_blocks: Arc::new(vec![id as usize]),
            num_computed_tokens: num_computed,
            prompt_len: 10,
            status,
            max_tokens: 20,
            sampling_params: SamplingParams::default(),
            consecutive_decode_rounds: 0,
            priority: Priority::default(),
        }
    }

    let composer = BatchComposer::default();

    // Test prefill with partial computed tokens (simulating chunked prefill)
    let seq1 = make_seq(
        1,
        vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        Status::Prefilling,
        5,
    );
    let batch = composer.compose(vec![seq1], vllm_core::types::Phase::Prefill);

    assert_eq!(batch.seq_ids.len(), 1, "Batch should have 1 sequence");
    assert_eq!(
        batch.input_tokens[0],
        vec![6, 7, 8, 9, 10],
        "Only remaining tokens should be in batch"
    );
    assert_eq!(
        batch.positions[0],
        vec![5, 6, 7, 8, 9],
        "Positions should start from num_computed"
    );
    assert!(
        !batch.is_prefill[0],
        "is_prefill should be false for partial prefill"
    );
}

#[test]
fn test_decode_batch_contains_only_last_token() {
    use std::sync::Arc;
    use vllm_core::scheduler::BatchComposer;
    use vllm_core::types::{Priority, SamplingParams, Sequence, Status};

    fn make_seq(id: u64, tokens: Vec<u32>, status: Status, prompt_len: usize) -> Sequence {
        Sequence {
            id,
            tokens,
            kv_blocks: Arc::new(vec![id as usize]),
            num_computed_tokens: 0,
            prompt_len,
            status,
            max_tokens: 15,
            sampling_params: SamplingParams::default(),
            consecutive_decode_rounds: 0,
            priority: Priority::default(),
        }
    }

    let composer = BatchComposer::default();

    // After prefill (prompt_len=5) and 2 decode steps, tokens=[1,2,3,4,5,6,7]
    let seq1 = make_seq(1, vec![1, 2, 3, 4, 5, 6, 7], Status::Decoding, 5);
    let batch = composer.compose(vec![seq1], vllm_core::types::Phase::Decode);

    assert_eq!(batch.seq_ids.len(), 1);
    // Should only contain the last token
    assert_eq!(
        batch.input_tokens[0],
        vec![7],
        "Decode should only have last token"
    );
    // Position should be tokens_len - 1 = 6 (0-indexed)
    assert_eq!(
        batch.positions[0],
        vec![6],
        "Position should be last token index"
    );
    // num_computed should be tokens_len - 1 = 6 (tokens already in cache)
    assert_eq!(
        batch.num_computed_tokens[0], 6,
        "num_computed should be all but last token"
    );
    assert!(
        !batch.is_prefill[0],
        "is_prefill should be false for decode"
    );
}

#[test]
fn test_decode_batch_position_is_tokens_len_minus_one() {
    use std::sync::Arc;
    use vllm_core::scheduler::BatchComposer;
    use vllm_core::types::{Priority, SamplingParams, Sequence, Status};

    fn make_seq(id: u64, tokens: Vec<u32>) -> Sequence {
        let prompt_len = tokens.len();
        Sequence {
            id,
            tokens,
            kv_blocks: Arc::new(vec![id as usize]),
            num_computed_tokens: 0,
            prompt_len,
            status: Status::Decoding,
            max_tokens: 20,
            sampling_params: SamplingParams::default(),
            consecutive_decode_rounds: 0,
            priority: Priority::default(),
        }
    }

    let composer = BatchComposer::default();

    // Test various token lengths
    let test_cases = vec![
        (1, vec![1], 0),         // 1 token -> position 0
        (2, vec![1, 2], 1),      // 2 tokens -> position 1
        (3, vec![1, 2, 3], 2),   // 3 tokens -> position 2
        (10, vec![0u32; 10], 9), // 10 tokens -> position 9
    ];

    for (id, tokens, expected_pos) in test_cases {
        let seq = make_seq(id, tokens.clone());
        let batch = composer.compose(vec![seq], vllm_core::types::Phase::Decode);
        assert_eq!(
            batch.positions[0][0],
            expected_pos,
            "For {} tokens, position should be {}, got {}",
            tokens.len(),
            expected_pos,
            batch.positions[0][0]
        );
    }
}
