//! Adaptive Speculative Decoding
//!
//! Implements dynamic draft token count adjustment based on acceptance rate tracking.

pub use crate::types::AdaptiveDraftConfig;
use std::collections::VecDeque;

/// Tracks draft token acceptance accuracy using a sliding window
#[derive(Clone, Debug)]
pub struct DraftAccuracyTracker {
    /// Recent acceptance results (true = accepted, false = rejected)
    history: VecDeque<bool>,
    /// Window size
    window_size: usize,
}

impl DraftAccuracyTracker {
    /// Create a new accuracy tracker with the given window size
    pub fn new(window_size: usize) -> Self {
        Self {
            history: VecDeque::with_capacity(window_size),
            window_size,
        }
    }

    /// Record a verification result
    pub fn record(&mut self, accepted: bool) {
        if self.history.len() >= self.window_size {
            self.history.pop_front();
        }
        self.history.push_back(accepted);
    }

    /// Calculate current acceptance rate
    pub fn acceptance_rate(&self) -> f32 {
        if self.history.is_empty() {
            return 0.0;
        }
        let accepted: usize = self.history.iter().filter(|&&b| b).count();
        accepted as f32 / self.history.len() as f32
    }

    /// Reset tracking
    pub fn reset(&mut self) {
        self.history.clear();
    }

    /// Get number of tracked results
    pub fn len(&self) -> usize {
        self.history.len()
    }

    /// Check if tracker is empty
    pub fn is_empty(&self) -> bool {
        self.history.is_empty()
    }
}

/// Adaptive speculative decoder with dynamic draft token adjustment
#[derive(Clone, Debug)]
pub struct AdaptiveSpeculativeDecoder {
    config: AdaptiveDraftConfig,
    /// Current max draft tokens
    current_max_draft_tokens: usize,
    /// Accuracy tracker
    accuracy_tracker: DraftAccuracyTracker,
    /// Steps since last adjustment
    steps_since_adjustment: usize,
}

impl AdaptiveSpeculativeDecoder {
    /// Create a new adaptive speculative decoder
    pub fn new(config: AdaptiveDraftConfig) -> Self {
        let window_size = config.accuracy_window_size;
        let initial_max = config.max_draft_tokens;
        Self {
            config: config.clone(),
            current_max_draft_tokens: initial_max,
            accuracy_tracker: DraftAccuracyTracker::new(window_size),
            steps_since_adjustment: 0,
        }
    }

    /// Get current max draft tokens
    pub fn current_max_draft_tokens(&self) -> usize {
        self.current_max_draft_tokens
    }

    /// Get accuracy tracker (for testing)
    pub fn accuracy_tracker(&self) -> &DraftAccuracyTracker {
        &self.accuracy_tracker
    }

    /// Record verification results and potentially adjust
    pub fn record_verification(&mut self, num_draft: usize, num_accepted: usize) {
        // Record each draft token result
        for i in 0..num_draft {
            let accepted = i < num_accepted;
            self.accuracy_tracker.record(accepted);
        }

        // Check if we should adjust
        self.steps_since_adjustment += 1;
        if self.steps_since_adjustment >= self.config.cooldown_steps {
            self.maybe_adjust();
        }
    }

    /// Potentially adjust draft token count based on accuracy
    fn maybe_adjust(&mut self) {
        let rate = self.accuracy_tracker.acceptance_rate();
        let target = self.config.target_acceptance_rate;

        // Calculate adjustment
        let adjustment: i32 = if rate > target + 0.1 {
            // High accuracy: increase draft tokens
            self.config.adjustment_step as i32
        } else if rate < target - 0.1 {
            // Low accuracy: decrease draft tokens
            -(self.config.adjustment_step as i32)
        } else {
            // Within acceptable range: no change
            0
        };

        if adjustment != 0 {
            let new_max = (self.current_max_draft_tokens as i32 + adjustment).clamp(
                self.config.min_draft_tokens as i32,
                self.config.max_draft_tokens as i32,
            ) as usize;

            if new_max != self.current_max_draft_tokens {
                tracing::info!(
                    "Adjusted max_draft_tokens: {} -> {} (acceptance_rate: {:.2})",
                    self.current_max_draft_tokens,
                    new_max,
                    rate
                );
                self.current_max_draft_tokens = new_max;
                self.steps_since_adjustment = 0;
            }
        }
    }

    /// Reset the decoder state
    pub fn reset(&mut self) {
        self.current_max_draft_tokens = self.config.max_draft_tokens;
        self.accuracy_tracker.reset();
        self.steps_since_adjustment = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accuracy_tracker_empty() {
        let tracker = DraftAccuracyTracker::new(5);
        assert_eq!(tracker.acceptance_rate(), 0.0);
    }

    #[test]
    fn test_accuracy_tracker_calculation() {
        let mut tracker = DraftAccuracyTracker::new(5);
        tracker.record(true);
        tracker.record(true);
        tracker.record(false);
        tracker.record(true);
        tracker.record(false);
        assert_eq!(tracker.acceptance_rate(), 0.6); // 3/5
    }

    #[test]
    fn test_accuracy_tracker_window() {
        let mut tracker = DraftAccuracyTracker::new(3);
        tracker.record(true);
        tracker.record(true);
        tracker.record(true);
        tracker.record(false); // Pushes out first true
        let rate = tracker.acceptance_rate();
        assert!((rate - 0.67).abs() < 0.01); // 2/3
    }

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
        };
        let mut decoder = AdaptiveSpeculativeDecoder::new(config);
        // Starts at max
        assert_eq!(decoder.current_max_draft_tokens(), 10);

        // First simulate low accuracy to decrease below max
        // With cooldown=1, each verification triggers adjustment
        // 5 verifications with 20% rate (1/5) -> rate = 0.2 < 0.4 target
        // Each triggers decrease by 1: 10 -> 9 -> 8 -> 7 -> 6 -> 5
        for _ in 0..5 {
            decoder.record_verification(5, 1); // 20% acceptance
        }
        let decreased_value = decoder.current_max_draft_tokens();
        assert!(decreased_value < 10); // Should have decreased

        // Now simulate high accuracy to increase
        for _ in 0..5 {
            decoder.record_verification(5, 5); // 100% acceptance
        }
        // Should have increased from the decreased value
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
        };
        let mut decoder = AdaptiveSpeculativeDecoder::new(config);

        // Simulate low accuracy
        for _ in 0..5 {
            decoder.record_verification(5, 1); // 20% acceptance
        }

        assert!(decoder.current_max_draft_tokens() < 8); // Should decrease
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
        };
        let mut decoder = AdaptiveSpeculativeDecoder::new(config);

        // Try to decrease below min
        for _ in 0..10 {
            decoder.record_verification(5, 1);
        }

        assert_eq!(decoder.current_max_draft_tokens(), 2); // Should not go below min
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
        };
        let mut decoder = AdaptiveSpeculativeDecoder::new(config);

        // Try to increase above max
        for _ in 0..10 {
            decoder.record_verification(5, 5);
        }

        assert_eq!(decoder.current_max_draft_tokens(), 4); // Should not go above max
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
        };
        let mut decoder = AdaptiveSpeculativeDecoder::new(config);

        // First decrease below max so we can test increase
        for _ in 0..5 {
            decoder.record_verification(5, 1); // Low accuracy
        }
        let decreased_value = decoder.current_max_draft_tokens();
        assert!(decreased_value < 10); // Should have decreased

        // Reset cooldown
        decoder.steps_since_adjustment = 0;

        // Record high accuracy - should count cooldown
        decoder.record_verification(5, 5);
        assert_eq!(decoder.steps_since_adjustment, 1);

        decoder.record_verification(5, 5);
        assert_eq!(decoder.steps_since_adjustment, 2);

        decoder.record_verification(5, 5);
        // After cooldown, should have increased from the decreased value
        assert!(decoder.current_max_draft_tokens() > decreased_value);
    }
}
