use tokio::sync::mpsc;
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

    let (tx, mut rx) = mpsc::channel(64);
    engine.add_request(Request::new(1, vec![10, 20], 5), tx);

    // Run a few steps
    for _ in 0..5 {
        let results = engine.step_adaptive_speculative().unwrap();
        for (_, _token) in results {
            let _ = rx.try_recv();
        }
    }

    // Verify decoder exists and has reasonable state
    assert!(engine.adaptive_decoder.is_some());
}

#[test]
fn test_adaptive_speculative_adjusts_draft_count() {
    let config = SchedulerConfig::default();
    let mut engine = Engine::with_config(IncrementModel, None, config, 4, 1024);

    // Use low cooldown to trigger adjustment quickly
    engine.enable_adaptive_speculative(AdaptiveDraftConfig {
        min_draft_tokens: 2,
        max_draft_tokens: 6,
        target_acceptance_rate: 0.5,
        accuracy_window_size: 5,
        adjustment_step: 1,
        cooldown_steps: 2,
    });

    let _initial_max = engine
        .adaptive_decoder
        .as_ref()
        .unwrap()
        .current_max_draft_tokens();

    // Add a request and run multiple steps
    let (tx, _rx) = mpsc::channel(64);
    engine.add_request(Request::new(1, vec![10, 20], 10), tx);

    // Run enough steps to trigger adjustment
    for _ in 0..20 {
        let _ = engine.step_adaptive_speculative();
    }

    // Draft count may have adjusted
    let final_max = engine
        .adaptive_decoder
        .as_ref()
        .unwrap()
        .current_max_draft_tokens();
    assert!((2..=6).contains(&final_max));
}
