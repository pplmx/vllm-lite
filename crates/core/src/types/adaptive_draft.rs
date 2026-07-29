//! Adaptive draft speculative-decoding configuration.

/// Configuration for adaptive speculative decoding
#[derive(Clone, Debug)]
pub struct AdaptiveDraftConfig {
    /// Minimum number of draft tokens
    pub min_draft_tokens: usize,
    /// Maximum number of draft tokens
    pub max_draft_tokens: usize,
    /// Target acceptance rate (0.0-1.0)
    pub target_acceptance_rate: f32,
    /// Window size for accuracy tracking
    pub accuracy_window_size: usize,
    /// Adjustment step size
    pub adjustment_step: usize,
    /// Cooldown steps between adjustments
    pub cooldown_steps: usize,
    /// EWMA smoothing factor (0.0-1.0). Higher = more responsive to recent changes.
    pub ewma_alpha: f32,
    /// Deadband threshold for hysteresis. Only adjusts when |rate - target| > threshold.
    pub deadband_threshold: f32,
}

impl Default for AdaptiveDraftConfig {
    fn default() -> Self {
        Self {
            min_draft_tokens: 2,
            max_draft_tokens: 8,
            target_acceptance_rate: 0.7,
            accuracy_window_size: 20,
            adjustment_step: 1,
            cooldown_steps: 5,
            ewma_alpha: 0.1,
            deadband_threshold: 0.05,
        }
    }
}

impl AdaptiveDraftConfig {
    /// Returns a builder for configuring this type with the documented field defaults.
    /// Use `with_*(...)` to override individual fields, then `build()` to produce the type.
    #[must_use]
    pub fn builder() -> AdaptiveDraftConfigBuilder {
        AdaptiveDraftConfigBuilder::default()
    }
}

/// Builder for [`AdaptiveDraftConfig`].
#[derive(Debug, Clone, Default)]
pub struct AdaptiveDraftConfigBuilder {
    inner: AdaptiveDraftConfig,
}

impl AdaptiveDraftConfigBuilder {
    /// Set the minimum number of draft tokens per step.
    #[must_use]
    pub const fn with_min_draft_tokens(mut self, v: usize) -> Self {
        self.inner.min_draft_tokens = v;
        self
    }
    /// Set the maximum number of draft tokens per step.
    #[must_use]
    pub const fn with_max_draft_tokens(mut self, v: usize) -> Self {
        self.inner.max_draft_tokens = v;
        self
    }
    /// Set the target acceptance rate for draft tokens (0.0–1.0).
    #[must_use]
    pub const fn with_target_acceptance_rate(mut self, v: f32) -> Self {
        self.inner.target_acceptance_rate = v;
        self
    }
    /// Set the moving-average window size for accuracy tracking.
    #[must_use]
    pub const fn with_accuracy_window_size(mut self, v: usize) -> Self {
        self.inner.accuracy_window_size = v;
        self
    }
    /// Set the step size for adjusting draft count based on acceptance rate.
    #[must_use]
    pub const fn with_adjustment_step(mut self, v: usize) -> Self {
        self.inner.adjustment_step = v;
        self
    }
    /// Set the number of cooldown steps after an adjustment before re-evaluating.
    #[must_use]
    pub const fn with_cooldown_steps(mut self, v: usize) -> Self {
        self.inner.cooldown_steps = v;
        self
    }
    /// Set the EMA smoothing factor (0.0–1.0) for the acceptance-rate tracker.
    #[must_use]
    pub const fn with_ewma_alpha(mut self, v: f32) -> Self {
        self.inner.ewma_alpha = v;
        self
    }
    /// Builder-style setter for the deadband threshold below which
    /// acceptance-rate changes are ignored (avoids thrashing).
    #[must_use]
    pub const fn with_deadband_threshold(mut self, v: f32) -> Self {
        self.inner.deadband_threshold = v;
        self
    }
    /// build: build the [`AdaptiveDraftConfig`].
    #[must_use]
    pub const fn build(self) -> AdaptiveDraftConfig {
        self.inner
    }
}

impl PartialEq for AdaptiveDraftConfig {
    fn eq(&self, other: &Self) -> bool {
        self.min_draft_tokens == other.min_draft_tokens
            && self.max_draft_tokens == other.max_draft_tokens
            && self.target_acceptance_rate.to_bits() == other.target_acceptance_rate.to_bits()
            && self.accuracy_window_size == other.accuracy_window_size
            && self.adjustment_step == other.adjustment_step
            && self.cooldown_steps == other.cooldown_steps
            && self.ewma_alpha.to_bits() == other.ewma_alpha.to_bits()
            && self.deadband_threshold.to_bits() == other.deadband_threshold.to_bits()
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn default_values() {
        let cfg = AdaptiveDraftConfig::default();
        assert_eq!(cfg.min_draft_tokens, 2);
        assert_eq!(cfg.max_draft_tokens, 8);
        assert!((cfg.target_acceptance_rate - 0.7).abs() < f32::EPSILON);
        assert_eq!(cfg.accuracy_window_size, 20);
        assert_eq!(cfg.adjustment_step, 1);
        assert_eq!(cfg.cooldown_steps, 5);
        assert!((cfg.ewma_alpha - 0.1).abs() < f32::EPSILON);
        assert!((cfg.deadband_threshold - 0.05).abs() < f32::EPSILON);
    }

    #[test]
    fn builder_overrides() {
        let cfg = AdaptiveDraftConfig::builder()
            .with_min_draft_tokens(1)
            .with_max_draft_tokens(4)
            .with_target_acceptance_rate(0.9)
            .with_accuracy_window_size(10)
            .with_adjustment_step(2)
            .with_cooldown_steps(3)
            .with_ewma_alpha(0.3)
            .with_deadband_threshold(0.1)
            .build();
        assert_eq!(cfg.min_draft_tokens, 1);
        assert_eq!(cfg.max_draft_tokens, 4);
        assert!((cfg.target_acceptance_rate - 0.9).abs() < f32::EPSILON);
        assert_eq!(cfg.accuracy_window_size, 10);
        assert_eq!(cfg.adjustment_step, 2);
        assert_eq!(cfg.cooldown_steps, 3);
        assert!((cfg.ewma_alpha - 0.3).abs() < f32::EPSILON);
        assert!((cfg.deadband_threshold - 0.1).abs() < f32::EPSILON);
    }

    #[test]
    fn builder_defaults_match_struct_default() {
        assert_eq!(
            AdaptiveDraftConfig::builder().build(),
            AdaptiveDraftConfig::default()
        );
    }

    #[test]
    fn default_is_consistent() {
        let cfg = AdaptiveDraftConfig::default();
        assert!(cfg.min_draft_tokens <= cfg.max_draft_tokens);
        assert!(
            (0.0..=1.0).contains(&cfg.target_acceptance_rate),
            "target_acceptance_rate must be in [0.0, 1.0]"
        );
        assert!(
            (0.0..=1.0).contains(&cfg.ewma_alpha),
            "ewma_alpha must be in [0.0, 1.0]"
        );
    }
}
