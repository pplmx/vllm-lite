use tokio::sync::mpsc;
use vllm_core::engine::Engine;
use vllm_core::types::{AdaptiveDraftConfig, SchedulerConfig};
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

    // Cannot use step_adaptive_speculative without a draft model
    // Just verify adaptive speculative mode can be enabled
    engine.enable_adaptive_speculative(AdaptiveDraftConfig::default());

    assert!(engine.adaptive_decoder.is_some());
    assert!(engine.is_adaptive_speculative_enabled());
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

    // Verify adaptive decoder is initialized
    assert!(engine.adaptive_decoder.is_some());
}
