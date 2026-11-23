//! Unit tests for the adaptive speculative decoding primitives.
//!
//! Extracted from `adaptive.rs` to keep the implementation file under the
//! project's 800-line soft cap. The tests cover the sliding-window
//! accuracy tracker, the EWMA smoothing path, the deadband hysteresis
//! adjustment logic, and the `record_verification` bool-return contract.

use super::*;

#[test]
fn test_accuracy_tracker_empty() {
    let tracker = DraftAccuracyTracker::new(5);
    assert!(tracker.acceptance_rate().abs() < 1e-6);
}

#[test]
fn test_accuracy_tracker_calculation() {
    let mut tracker = DraftAccuracyTracker::new(5);
    tracker.record(true);
    tracker.record(true);
    tracker.record(false);
    tracker.record(true);
    tracker.record(false);
    assert!((tracker.acceptance_rate() - 0.6).abs() < 1e-6);
}

#[test]
fn test_accuracy_tracker_window() {
    let mut tracker = DraftAccuracyTracker::new(3);
    tracker.record(true);
    tracker.record(true);
    tracker.record(true);
    tracker.record(false);
    let rate = tracker.acceptance_rate();
    assert!((rate - 0.67).abs() < 0.01);
}

// ---- EWMA Tests (Plan 17.3-A / 17.3-G) ----

#[test]
fn test_ewma_initialization() {
    let mut tracker = DraftAccuracyTracker::with_alpha(10, 0.1);
    // Before any records, EWMA should return 0.0
    assert!(tracker.acceptance_rate_ewma().abs() < 1e-6);

    // First record initializes smoothed_rate to the observed sliding window rate
    tracker.record(true);
    let ewma_after_first = tracker.acceptance_rate_ewma();
    assert!(
        (ewma_after_first - 1.0).abs() < 0.001,
        "First record should set EWMA to {:.3}, got {:.3}",
        1.0,
        ewma_after_first
    );

    // Second record updates EWMA but sliding window is still 1.0
    tracker.record(true);
    let sliding = tracker.acceptance_rate();
    let ewma = tracker.acceptance_rate_ewma();
    // EWMA = 0.1*1.0 + 0.9*1.0 = 1.0, sliding = 2/2 = 1.0
    assert!((ewma - sliding).abs() < 0.001);
}

#[test]
fn test_ewma_convergence() {
    let mut tracker = DraftAccuracyTracker::with_alpha(20, 0.3);
    // Record 10 times with 80% acceptance
    for _ in 0..10 {
        tracker.record(true);
        tracker.record(true);
        tracker.record(true);
        tracker.record(true);
        tracker.record(false); // 4/5 = 0.8
    }
    let ewma = tracker.acceptance_rate_ewma();
    // Should have converged close to 0.8
    assert!((ewma - 0.8).abs() < 0.05, "EWMA {ewma} should ≈ 0.8");
}

#[test]
fn test_ewma_smoothing() {
    let mut tracker = DraftAccuracyTracker::with_alpha(10, 0.1);
    // Oscillating input: alternate between 100% and 0%
    for _ in 0..5 {
        // 100% acceptance: 5 accepted
        for _ in 0..5 {
            tracker.record(true);
        }
        // 0% acceptance: 5 rejected
        for _ in 0..5 {
            tracker.record(false);
        }
    }
    let ewma = tracker.acceptance_rate_ewma();
    let sliding = tracker.acceptance_rate();
    // Sliding window on 10 entries with 50% rate
    assert!((sliding - 0.5).abs() < 0.01);
    // EWMA should also be around 0.5 (smoothed)
    assert!((ewma - 0.5).abs() < 0.1, "EWMA {ewma} should ≈ 0.5");
}

#[test]
fn test_ewma_configurable_alpha() {
    // alpha=1.0: instant, no smoothing
    let mut instant = DraftAccuracyTracker::with_alpha(10, 1.0);
    instant.record(true);
    instant.record(false);
    let sliding = instant.acceptance_rate();
    let ewma = instant.acceptance_rate_ewma();
    // With alpha=1.0, ewma always equals the latest sliding window rate
    assert!((ewma - sliding).abs() < 0.001);

    // alpha=0.01: heavy smoothing, changes slowly
    let mut heavy = DraftAccuracyTracker::with_alpha(10, 0.01);
    for _ in 0..5 {
        heavy.record(true);
    }
    let first_ewma = heavy.acceptance_rate_ewma();
    // Then a run of false
    for _ in 0..5 {
        heavy.record(false);
    }
    let second_ewma = heavy.acceptance_rate_ewma();
    // With heavy smoothing, the change should be small
    let change = (first_ewma - second_ewma).abs();
    assert!(
        change < 0.1,
        "Heavy smoothing should dampen change, got {change}"
    );
}

// ---- Deadband Hysteresis Tests (Plan 17.3-B / 17.3-G) ----

#[test]
fn test_deadband_no_adjustment() {
    let config = AdaptiveDraftConfig {
        min_draft_tokens: 2,
        max_draft_tokens: 8,
        target_acceptance_rate: 0.7,
        accuracy_window_size: 5,
        adjustment_step: 1,
        cooldown_steps: 1,
        ewma_alpha: 0.5,
        deadband_threshold: 0.1,
    };
    let mut decoder = AdaptiveSpeculativeDecoder::new(config);
    // Start below max so we have room
    let initial = decoder.current_max_draft_tokens();
    // Record rate = 0.68, target = 0.7, threshold = 0.1
    // |0.68 - 0.7| = 0.02 <= 0.1 → within deadband, no adjustment
    // 4/5 ≈ 0.8 rate initially from sliding window, but with alpha=0.5 EWMA converges fast
    // Use exactly: 3/5 accepted → 0.6 rate, deviation = -0.1 → not above threshold
    decoder.record_verification(5, 3); // 3/5 = 0.6

    // Should still be at initial value since within deadband
    assert_eq!(decoder.current_max_draft_tokens(), initial);
}

#[test]
fn test_deadband_resets_cooldown() {
    let config = AdaptiveDraftConfig {
        min_draft_tokens: 2,
        max_draft_tokens: 8,
        target_acceptance_rate: 0.7,
        accuracy_window_size: 5,
        adjustment_step: 1,
        cooldown_steps: 1,
        ewma_alpha: 0.9,
        deadband_threshold: 0.2,
    };
    let mut decoder = AdaptiveSpeculativeDecoder::new(config);

    // Record a rate that is within deadband (rate ≈ 0.7, target ±0.2)
    // 3/5 = 0.6 acceptance → deviation = |0.6-0.7| = 0.1 ≤ 0.2 deadband
    decoder.record_verification(5, 3);
    // maybe_adjust() runs (cooldown=1), finds rate within deadband,
    // and resets steps_since_adjustment to 0
    assert_eq!(
        decoder.steps_since_adjustment, 0,
        "Within deadband should reset cooldown to 0"
    );
}

#[test]
fn test_deadband_above_triggers_increase() {
    let config = AdaptiveDraftConfig {
        min_draft_tokens: 2,
        max_draft_tokens: 8,
        target_acceptance_rate: 0.5,
        accuracy_window_size: 5,
        adjustment_step: 1,
        cooldown_steps: 1,
        ewma_alpha: 0.5,
        deadband_threshold: 0.1,
    };
    let mut decoder = AdaptiveSpeculativeDecoder::new(config);
    // Start at max=8, first decrease to give room to increase
    decoder.record_verification(5, 0); // 0% → deviation = -0.5 < -0.1 → decrease
    let decreased = decoder.current_max_draft_tokens();
    assert!(decreased < 8);

    // Now 100% acceptance → rate=1.0, deviation=0.5 > 0.1 → increase
    decoder.record_verification(5, 5);
    assert!(decoder.current_max_draft_tokens() > decreased);
}

#[test]
fn test_deadband_below_decreases() {
    let config = AdaptiveDraftConfig {
        min_draft_tokens: 2,
        max_draft_tokens: 8,
        target_acceptance_rate: 0.7,
        accuracy_window_size: 5,
        adjustment_step: 1,
        cooldown_steps: 1,
        ewma_alpha: 0.5,
        deadband_threshold: 0.1,
    };
    let mut decoder = AdaptiveSpeculativeDecoder::new(config);
    let initial = decoder.current_max_draft_tokens();

    // 0% acceptance → deviation = -0.7 < -0.1 → decrease
    decoder.record_verification(5, 0);
    assert!(decoder.current_max_draft_tokens() < initial);
}

// ---- Adaptive decoder existing tests ----

#[test]
fn test_adaptive_decoder_initial_state() {
    let config = AdaptiveDraftConfig::default();
    let decoder = AdaptiveSpeculativeDecoder::new(config.clone());
    assert_eq!(decoder.current_max_draft_tokens(), config.max_draft_tokens);
}

#[test]
fn test_adaptive_decoder_increases_on_high_accuracy() {
    let config = AdaptiveDraftConfig {
        min_draft_tokens: 2,
        max_draft_tokens: 10,
        target_acceptance_rate: 0.5,
        accuracy_window_size: 5,
        adjustment_step: 1,
        cooldown_steps: 1,
        ewma_alpha: 0.5,
        deadband_threshold: 0.05,
    };
    let mut decoder = AdaptiveSpeculativeDecoder::new(config);
    assert_eq!(decoder.current_max_draft_tokens(), 10);

    for _ in 0..5 {
        decoder.record_verification(5, 1); // 20% acceptance
    }
    let decreased_value = decoder.current_max_draft_tokens();
    assert!(decreased_value < 10);

    for _ in 0..5 {
        decoder.record_verification(5, 5); // 100% acceptance
    }
    assert!(decoder.current_max_draft_tokens() > decreased_value);
}

#[test]
fn test_adaptive_decoder_decreases_on_low_accuracy() {
    let config = AdaptiveDraftConfig {
        min_draft_tokens: 2,
        max_draft_tokens: 8,
        target_acceptance_rate: 0.7,
        accuracy_window_size: 5,
        adjustment_step: 1,
        cooldown_steps: 1,
        ewma_alpha: 0.5,
        deadband_threshold: 0.05,
    };
    let mut decoder = AdaptiveSpeculativeDecoder::new(config);

    for _ in 0..5 {
        decoder.record_verification(5, 1); // 20% acceptance
    }

    assert!(decoder.current_max_draft_tokens() < 8);
}

#[test]
fn test_adaptive_decoder_respects_min_bound() {
    let config = AdaptiveDraftConfig {
        min_draft_tokens: 2,
        max_draft_tokens: 4,
        target_acceptance_rate: 0.7,
        accuracy_window_size: 5,
        adjustment_step: 1,
        cooldown_steps: 1,
        ewma_alpha: 0.5,
        deadband_threshold: 0.05,
    };
    let mut decoder = AdaptiveSpeculativeDecoder::new(config);

    for _ in 0..10 {
        decoder.record_verification(5, 1);
    }

    assert_eq!(decoder.current_max_draft_tokens(), 2);
}

#[test]
fn test_adaptive_decoder_respects_max_bound() {
    let config = AdaptiveDraftConfig {
        min_draft_tokens: 2,
        max_draft_tokens: 4,
        target_acceptance_rate: 0.5,
        accuracy_window_size: 5,
        adjustment_step: 1,
        cooldown_steps: 1,
        ewma_alpha: 0.5,
        deadband_threshold: 0.05,
    };
    let mut decoder = AdaptiveSpeculativeDecoder::new(config);

    for _ in 0..10 {
        decoder.record_verification(5, 5);
    }

    assert_eq!(decoder.current_max_draft_tokens(), 4);
}

#[test]
fn test_adaptive_decoder_cooldown() {
    let config = AdaptiveDraftConfig {
        min_draft_tokens: 2,
        max_draft_tokens: 10,
        target_acceptance_rate: 0.5,
        accuracy_window_size: 5,
        adjustment_step: 1,
        cooldown_steps: 3,
        ewma_alpha: 0.5,
        deadband_threshold: 0.1,
    };
    let mut decoder = AdaptiveSpeculativeDecoder::new(config);

    for _ in 0..5 {
        decoder.record_verification(5, 1); // Low accuracy
    }
    let decreased_value = decoder.current_max_draft_tokens();
    assert!(decreased_value < 10);

    decoder.steps_since_adjustment = 0;

    decoder.record_verification(5, 5);
    assert_eq!(decoder.steps_since_adjustment, 1);

    decoder.record_verification(5, 5);
    assert_eq!(decoder.steps_since_adjustment, 2);

    decoder.record_verification(5, 5);
    assert!(decoder.current_max_draft_tokens() > decreased_value);
}

#[test]
fn test_adaptive_config_defaults() {
    let config = AdaptiveDraftConfig::default();
    assert!((config.ewma_alpha - 0.1).abs() < 1e-6);
    assert!((config.deadband_threshold - 0.05).abs() < 1e-6);
}

#[test]
fn test_adaptive_full_cycle() {
    let config = AdaptiveDraftConfig {
        min_draft_tokens: 1,
        max_draft_tokens: 5,
        target_acceptance_rate: 0.6,
        accuracy_window_size: 10,
        adjustment_step: 1,
        cooldown_steps: 2,
        ewma_alpha: 0.3,
        deadband_threshold: 0.1,
    };
    let mut decoder = AdaptiveSpeculativeDecoder::new(config);

    // Phase 1: Low acceptance → decrease
    for _ in 0..5 {
        decoder.record_verification(5, 1); // 20%
    }
    let after_low = decoder.current_max_draft_tokens();
    assert!(
        after_low < 5,
        "Should decrease after low acceptance, got {after_low}"
    );

    // Phase 2: High acceptance → increase
    for _ in 0..5 {
        decoder.record_verification(5, 5); // 100%
    }
    let after_high = decoder.current_max_draft_tokens();
    assert!(
        after_high > after_low,
        "Should increase after high acceptance, got {after_high} > {after_low}"
    );
}

// ---- Bool return contract tests (Wave 2 D2-3) ----

#[test]
fn test_record_verification_returns_true_on_increase() {
    let config = AdaptiveDraftConfig {
        min_draft_tokens: 2,
        max_draft_tokens: 8,
        target_acceptance_rate: 0.5,
        accuracy_window_size: 5,
        adjustment_step: 1,
        cooldown_steps: 1,
        ewma_alpha: 0.5,
        deadband_threshold: 0.1,
    };
    let mut decoder = AdaptiveSpeculativeDecoder::new(config);
    // Pre-decrease to create room (test starts at config.max_draft_tokens=8).
    // 0% acceptance drops to 7; otherwise high-acceptance step would be
    // clamped to the same value and the test would not exercise a true
    // adjustment.
    decoder.record_verification(5, 0);
    let before = decoder.current_max_draft_tokens();
    assert!(before < 8, "pre-condition: must be below max");

    // 100% acceptance → deviation > threshold → should adjust up to 8
    let adjusted = decoder.record_verification(5, 5);
    assert!(
        adjusted,
        "100% acceptance with room to grow should return true"
    );
    assert!(
        decoder.current_max_draft_tokens() > before,
        "max_draft_tokens should increase on high acceptance when below max"
    );
}

#[test]
fn test_record_verification_returns_true_on_decrease() {
    let config = AdaptiveDraftConfig {
        min_draft_tokens: 2,
        max_draft_tokens: 8,
        target_acceptance_rate: 0.7,
        accuracy_window_size: 5,
        adjustment_step: 1,
        cooldown_steps: 1,
        ewma_alpha: 0.5,
        deadband_threshold: 0.1,
    };
    let mut decoder = AdaptiveSpeculativeDecoder::new(config);

    let adjusted = decoder.record_verification(5, 0);
    assert!(
        adjusted,
        "0% acceptance should trigger adjustment (return true)"
    );
    assert!(
        decoder.current_max_draft_tokens() < 8,
        "max_draft_tokens should decrease below initial max on 0% acceptance"
    );
}

#[test]
fn test_record_verification_returns_false_within_deadband() {
    let config = AdaptiveDraftConfig {
        min_draft_tokens: 2,
        max_draft_tokens: 8,
        target_acceptance_rate: 0.7,
        accuracy_window_size: 5,
        adjustment_step: 1,
        cooldown_steps: 1,
        ewma_alpha: 0.5,
        deadband_threshold: 0.5, // very wide deadband
    };
    let mut decoder = AdaptiveSpeculativeDecoder::new(config);
    let initial = decoder.current_max_draft_tokens();

    let adjusted = decoder.record_verification(5, 3);
    assert!(
        !adjusted,
        "Within deadband should NOT trigger adjustment (return false)"
    );
    assert_eq!(
        decoder.current_max_draft_tokens(),
        initial,
        "max_draft_tokens should be unchanged within deadband"
    );
}

/// The tracker's accessors and `reset` must reflect the recorded history.
///
/// `cargo mutants` replaced `len()` with the constants 0 / 1, `is_empty()` with
/// `true` / `false`, and `reset` with `()` — five survivors, because the suite
/// only ever read `acceptance_rate()` and `acceptance_rate_ewma()`. `len()`
/// is the sliding window's fill level and `reset` is what the engine calls
/// between requests: a no-op reset keeps a previous request's accuracy (and its
/// EWMA) and silently biases the next request's draft-token adjustment.
#[test]
fn tracker_accessors_track_history_and_reset_clears_it() {
    let mut tracker = DraftAccuracyTracker::with_alpha(4, 0.5);
    assert_eq!(tracker.len(), 0, "a fresh tracker holds no results");
    assert!(tracker.is_empty());

    tracker.record(true);
    tracker.record(false);
    assert_eq!(tracker.len(), 2, "len must count recorded results");
    assert!(!tracker.is_empty(), "two results means not empty");

    // The window is bounded: overflow keeps `len` at the window size.
    tracker.record(true);
    tracker.record(true);
    tracker.record(true);
    assert_eq!(tracker.len(), 4, "len must saturate at the window size");

    let rate_before = tracker.acceptance_rate_ewma();
    assert!(rate_before > 0.0, "fixture: a rate was recorded");

    tracker.reset();
    assert_eq!(tracker.len(), 0, "reset must clear the history");
    assert!(tracker.is_empty(), "reset must leave the tracker empty");
    assert!(
        tracker.acceptance_rate_ewma().abs() < f32::EPSILON,
        "reset must clear the EWMA too, not just the window"
    );
    assert!(
        tracker.acceptance_rate().abs() < f32::EPSILON,
        "a reset tracker reports a zero sliding-window rate"
    );
}

/// The decomposition's `reset` must restore the CONFIGURED max, not drift.
///
/// `AdaptiveSpeculativeDecoder::reset` replaced by `()` survived: the decoder's
/// `current_max_draft_tokens` mutates with accuracy, and a reset that leaves it
/// where the previous request put it makes the next request start with a draft
/// budget the operator never chose. It also has to clear the tracker and the
/// cooldown, so the next `record_verification` counts from zero.
#[test]
fn decoder_reset_restores_configured_max_and_clears_tracker() {
    let config = AdaptiveDraftConfig {
        min_draft_tokens: 2,
        max_draft_tokens: 8,
        target_acceptance_rate: 0.5,
        accuracy_window_size: 4,
        adjustment_step: 1,
        cooldown_steps: 1,
        ewma_alpha: 0.5,
        deadband_threshold: 0.1,
    };
    let mut decoder = AdaptiveSpeculativeDecoder::new(config);
    assert_eq!(decoder.current_max_draft_tokens(), 8);

    // All drafts rejected: the deviation is -0.5, below -0.1, so it decreases.
    decoder.reset();
    decoder.record_verification(4, 0);
    let decreased = decoder.current_max_draft_tokens();
    assert!(
        decreased < 8,
        "setup: accuracy drove the max down, got {decreased}"
    );
    assert!(
        decoder.accuracy_tracker().len() > 0,
        "setup: results recorded"
    );

    decoder.reset();

    assert_eq!(
        decoder.current_max_draft_tokens(),
        8,
        "reset must restore the configured max_draft_tokens"
    );
    assert_eq!(
        decoder.accuracy_tracker().len(),
        0,
        "reset must clear the accuracy tracker"
    );
    assert_eq!(
        decoder.steps_since_adjustment, 0,
        "reset must clear the cooldown counter"
    );
}

/// The deadband must be STRICT: a deviation exactly at the threshold is inside.
///
/// `maybe_adjust` uses `deviation > threshold` to grow and
/// `deviation < -threshold` to shrink, so equality is the deadband. `cargo
/// mutants` relaxed the lower test to `<=` and the upper to `>=`; both survived
/// because every existing fixture picks rates that land clearly inside or
/// clearly outside, never exactly on the boundary. The boundary is reachable in
/// `f32`: with `ewma_alpha = 1.0` the smoothed rate equals the window rate, and
/// `1/4 - 0.5 == -0.25` / `3/4 - 0.25 == 0.5 - 0.25` are exact (measured).
///
/// Relaxed, the decoder changes its draft budget on a deviation the configured
/// deadband says to ignore — the hysteresis the whole struct exists for.
#[test]
fn deadband_boundary_is_strict() {
    let config = AdaptiveDraftConfig {
        min_draft_tokens: 2,
        max_draft_tokens: 8,
        target_acceptance_rate: 0.5,
        accuracy_window_size: 4,
        adjustment_step: 1,
        cooldown_steps: 1,
        // Exact EWMA: the smoothed rate IS the window rate, no interpolation.
        ewma_alpha: 1.0,
        deadband_threshold: 0.25,
    };

    // Rate 0.25, target 0.5 -> deviation exactly -threshold -> deadband.
    let mut lower = AdaptiveSpeculativeDecoder::new(config.clone());
    let before = lower.current_max_draft_tokens();
    lower.record_verification(4, 1);
    assert_eq!(
        lower.current_max_draft_tokens(),
        before,
        "a deviation EQUAL to the threshold must not shrink the draft budget"
    );

    // Rate 0.75, target 0.5 -> +threshold. Start below the max so a spurious
    // increase would be visible: reject everything once to shrink first.
    let mut upper = AdaptiveSpeculativeDecoder::new(config);
    upper.record_verification(4, 0);
    let shrunken = upper.current_max_draft_tokens();
    assert!(shrunken < 8, "setup: a clear low rate shrinks the budget");
    upper.record_verification(4, 3);
    assert_eq!(
        upper.current_max_draft_tokens(),
        shrunken,
        "a deviation EQUAL to the threshold must not grow the draft budget"
    );
}
