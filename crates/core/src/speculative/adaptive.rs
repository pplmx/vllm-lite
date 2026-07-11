#![allow(clippy::module_name_repetitions)]
//! Adaptive Speculative Decoding
//!
//! Implements dynamic draft token count adjustment based on acceptance rate tracking.
//!
//! Tests live in `tests.rs` (sibling file) to keep this module under the
//! 800-line soft cap.

pub use crate::types::AdaptiveDraftConfig;
use std::collections::VecDeque;

#[cfg(test)]
mod tests;

/// Tracks draft token acceptance accuracy using a sliding window and EWMA smoothing
#[derive(Clone, Debug)]
pub struct DraftAccuracyTracker {
    /// Recent acceptance results (true = accepted, false = rejected)
    history: VecDeque<bool>,
    /// Window size
    window_size: usize,
    /// EWMA smoothing factor (0.0-1.0)
    ewma_alpha: f32,
    /// Smoothed rate from EWMA
    smoothed_rate: Option<f32>,
}

impl DraftAccuracyTracker {
    /// Create a new accuracy tracker with the given window size
    #[must_use]
    pub fn new(window_size: usize) -> Self {
        Self {
            history: VecDeque::with_capacity(window_size),
            window_size,
            ewma_alpha: 0.1,
            smoothed_rate: None,
        }
    }

    /// Create a new accuracy tracker with configurable EWMA alpha
    #[must_use]
    #[allow(dead_code)] // test-only helper; reachable under cfg(test) only
    pub(crate) fn with_alpha(window_size: usize, ewma_alpha: f32) -> Self {
        Self {
            history: VecDeque::with_capacity(window_size),
            window_size,
            ewma_alpha,
            smoothed_rate: None,
        }
    }

    /// Record a verification result
    pub fn record(&mut self, accepted: bool) {
        if self.history.len() >= self.window_size {
            self.history.pop_front();
        }
        self.history.push_back(accepted);

        // Update EWMA smoothed rate
        let current_rate = self.acceptance_rate();
        self.smoothed_rate = Some(match self.smoothed_rate {
            None => current_rate,
            Some(prev) => self
                .ewma_alpha
                .mul_add(current_rate, (1.0 - self.ewma_alpha) * prev),
        });
    }

    /// Calculate current acceptance rate (sliding window, for debugging)
    #[must_use]
    // invariant: history length is bounded by window_size; f32 precision loss
    // is acceptable for the sliding-window rate metric.
    #[allow(clippy::cast_precision_loss)]
    pub fn acceptance_rate(&self) -> f32 {
        if self.history.is_empty() {
            return 0.0;
        }
        let accepted: usize = self.history.iter().filter(|&&b| b).count();
        accepted as f32 / self.history.len() as f32
    }

    /// Get the EWMA-smoothed acceptance rate
    #[must_use]
    #[allow(dead_code)]
    pub(crate) fn acceptance_rate_ewma(&self) -> f32 {
        self.smoothed_rate.unwrap_or(0.0)
    }

    /// Reset tracking
    pub fn reset(&mut self) {
        self.history.clear();
        self.smoothed_rate = None;
    }

    /// Get number of tracked results
    #[must_use]
    pub fn len(&self) -> usize {
        self.history.len()
    }

    /// Check if tracker is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.history.is_empty()
    }
}

/// Adaptive speculative decoder with dynamic draft token adjustment
#[derive(Clone, Debug)]
pub struct AdaptiveSpeculativeDecoder {
    /// Configuration for adjustment thresholds and bounds.
    config: AdaptiveDraftConfig,
    /// Current max draft tokens
    current_max_draft_tokens: usize,
    /// Accuracy tracker
    accuracy_tracker: DraftAccuracyTracker,
    /// Steps since last adjustment
    steps_since_adjustment: usize,
}

impl AdaptiveSpeculativeDecoder {
    /// Create a new adaptive speculative decoder. Starts with
    /// `current_max_draft_tokens = config.max_draft_tokens` and seeds the
    /// accuracy tracker with the configured window and EWMA alpha.
    #[must_use]
    pub fn new(config: AdaptiveDraftConfig) -> Self {
        let window_size = config.accuracy_window_size;
        let initial_max = config.max_draft_tokens;
        let tracker = DraftAccuracyTracker::with_alpha(window_size, config.ewma_alpha);
        Self {
            config,
            current_max_draft_tokens: initial_max,
            accuracy_tracker: tracker,
            steps_since_adjustment: 0,
        }
    }

    /// Get current max draft tokens
    #[must_use]
    pub const fn current_max_draft_tokens(&self) -> usize {
        self.current_max_draft_tokens
    }

    /// Get accuracy tracker (for testing)
    #[must_use]
    pub const fn accuracy_tracker(&self) -> &DraftAccuracyTracker {
        &self.accuracy_tracker
    }

    /// Record verification results and potentially adjust.
    /// Returns `true` if `current_max_draft_tokens` was actually changed,
    /// `false` if within deadband, clamped to bound, or cooldown not elapsed.
    pub fn record_verification(&mut self, num_draft: usize, num_accepted: usize) -> bool {
        // Record each draft token result
        for i in 0..num_draft {
            let accepted = i < num_accepted;
            self.accuracy_tracker.record(accepted);
        }

        // Check if we should adjust
        self.steps_since_adjustment += 1;
        if self.steps_since_adjustment >= self.config.cooldown_steps {
            self.maybe_adjust()
        } else {
            false
        }
    }

    /// Potentially adjust draft token count based on EWMA accuracy and deadband hysteresis.
    /// Returns `true` if `current_max_draft_tokens` was actually changed.
    fn maybe_adjust(&mut self) -> bool {
        let rate = self.accuracy_tracker.acceptance_rate_ewma();
        let target = self.config.target_acceptance_rate;
        let threshold = self.config.deadband_threshold;
        let deviation = rate - target;

        // Deadband hysteresis: only adjust if deviation exceeds threshold.
        // invariant: adjustment_step / current_max_draft_tokens / min/max_draft_tokens
        // are bounded config values; i32/usize truncation is not reachable.
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let adjustment: i32 = if deviation > threshold {
            // Rate too high (above target + threshold): increase draft tokens
            self.config.adjustment_step as i32
        } else if deviation < -threshold {
            // Rate too low (below target - threshold): decrease draft tokens
            -(self.config.adjustment_step as i32)
        } else {
            // Within deadband: no change
            0
        };

        if adjustment != 0 {
            // invariant: clamp(..., min_draft_tokens, max_draft_tokens) yields a
            // non-negative i32 in a small config-defined range, so the i32 ->
            // usize cast is sign-safe and bounded.
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_possible_wrap,
                clippy::cast_sign_loss
            )]
            let new_max = (self.current_max_draft_tokens as i32 + adjustment).clamp(
                self.config.min_draft_tokens as i32,
                self.config.max_draft_tokens as i32,
            ) as usize;

            if new_max == self.current_max_draft_tokens {
                // Clamped to bound: adjustment would not change value
                false
            } else {
                tracing::info!(
                    "Adjusted max_draft_tokens: {} -> {} (rate: {:.3}, target: {:.2}, threshold: {:.2})",
                    self.current_max_draft_tokens,
                    new_max,
                    rate,
                    target,
                    threshold,
                );
                self.current_max_draft_tokens = new_max;
                self.steps_since_adjustment = 0;
                true
            }
        } else {
            // Within deadband: reset cooldown to prevent stale accumulation
            self.steps_since_adjustment = 0;
            false
        }
    }

    /// Reset the decoder state
    pub fn reset(&mut self) {
        self.current_max_draft_tokens = self.config.max_draft_tokens;
        self.accuracy_tracker.reset();
        self.steps_since_adjustment = 0;
    }
}
