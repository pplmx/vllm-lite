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
