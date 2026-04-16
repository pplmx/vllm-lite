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
