#![allow(dead_code)]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// `BackpressureConfig`: backpressure configuration.
#[derive(Debug, Clone)]
pub(crate) struct BackpressureConfig {
    pub max_buffer_size: usize,
    pub high_water_mark: usize,
    pub resume_threshold: usize,
}

impl Default for BackpressureConfig {
    fn default() -> Self {
        Self {
            max_buffer_size: 64,
            high_water_mark: 48,
            resume_threshold: 16,
        }
    }
}

impl BackpressureConfig {
    pub const fn new(max_buffer_size: usize) -> Self {
        // invariant: integer arithmetic preserves precision for the 75% / 25%
        // water marks; no float conversion needed.
        let high_water_mark = max_buffer_size * 3 / 4;
        let resume_threshold = max_buffer_size / 4;
        Self {
            max_buffer_size,
            high_water_mark,
            resume_threshold,
        }
    }
}

/// `FlowControlState`: flow control state enumeration.
#[derive(Debug, Clone)]
pub(crate) enum FlowControlState {
    Normal,
    Throttled,
    Resumed,
}

/// `BackpressureState`: backpressure state.
#[derive(Debug)]
pub(crate) struct BackpressureState {
    pending_tokens: Arc<AtomicUsize>,
    config: BackpressureConfig,
    last_state: std::sync::Mutex<FlowControlState>,
}

impl BackpressureState {
    pub fn new(config: BackpressureConfig) -> Self {
        Self {
            pending_tokens: Arc::new(AtomicUsize::new(0)),
            config,
            last_state: std::sync::Mutex::new(FlowControlState::Normal),
        }
    }

    pub fn increment(&self) -> FlowControlState {
        let pending = self.pending_tokens.fetch_add(1, Ordering::SeqCst);
        self.evaluate_state(pending + 1)
    }

    pub fn decrement(&self) -> FlowControlState {
        let pending = self.pending_tokens.fetch_sub(1, Ordering::SeqCst);
        self.evaluate_state(pending.saturating_sub(1))
    }

    pub fn batch_increment(&self, count: usize) -> FlowControlState {
        let pending = self.pending_tokens.fetch_add(count, Ordering::SeqCst);
        self.evaluate_state(pending + count)
    }

    pub fn batch_decrement(&self, count: usize) -> FlowControlState {
        let pending = self.pending_tokens.fetch_sub(count, Ordering::SeqCst);
        self.evaluate_state(pending.saturating_sub(count))
    }

    fn evaluate_state(&self, pending: usize) -> FlowControlState {
        let mut last = self
            .last_state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let new_state = if pending >= self.config.max_buffer_size
            || (pending >= self.config.high_water_mark && matches!(*last, FlowControlState::Normal))
        {
            FlowControlState::Throttled
        } else if pending <= self.config.resume_threshold
            && matches!(*last, FlowControlState::Throttled)
        {
            FlowControlState::Resumed
        } else {
            FlowControlState::Normal
        };
        *last = new_state.clone();
        new_state
    }

    pub fn should_throttle(&self) -> bool {
        let pending = self.pending_tokens.load(Ordering::SeqCst);
        pending >= self.config.high_water_mark
    }

    pub fn pending_count(&self) -> usize {
        self.pending_tokens.load(Ordering::SeqCst)
    }

    pub fn reset(&self) {
        self.pending_tokens.store(0, Ordering::SeqCst);
        *self
            .last_state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = FlowControlState::Normal;
    }
}

/// `StreamingBackpressure`: streaming backpressure.
pub(crate) struct StreamingBackpressure {
    state: Arc<BackpressureState>,
}

impl StreamingBackpressure {
    pub fn new(config: BackpressureConfig) -> Self {
        Self {
            state: Arc::new(BackpressureState::new(config)),
        }
    }

    pub const fn state(&self) -> &Arc<BackpressureState> {
        &self.state
    }

    pub fn before_send(&self) -> FlowControlState {
        self.state.increment()
    }

    pub fn after_send(&self) -> FlowControlState {
        self.state.decrement()
    }

    pub fn can_send(&self) -> bool {
        !self.state.should_throttle()
    }
}

impl Default for StreamingBackpressure {
    fn default() -> Self {
        Self::new(BackpressureConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backpressure_config_defaults() {
        let config = BackpressureConfig::default();
        assert_eq!(config.max_buffer_size, 64);
        assert_eq!(config.high_water_mark, 48);
        assert_eq!(config.resume_threshold, 16);
    }

    #[test]
    fn test_backpressure_config_custom() {
        let config = BackpressureConfig::new(100);
        assert_eq!(config.max_buffer_size, 100);
        assert_eq!(config.high_water_mark, 75);
        assert_eq!(config.resume_threshold, 25);
    }

    #[test]
    fn test_backpressure_state_increment() {
        let config = BackpressureConfig::default();
        let state = BackpressureState::new(config);

        let result = state.increment();
        assert!(matches!(result, FlowControlState::Normal));
        assert_eq!(state.pending_count(), 1);
    }

    #[test]
    fn test_backpressure_state_decrement() {
        let config = BackpressureConfig::default();
        let state = BackpressureState::new(config);

        state.increment();
        state.increment();
        state.decrement();
        assert_eq!(state.pending_count(), 1);
    }

    #[test]
    fn test_backpressure_state_throttle() {
        let config = BackpressureConfig::new(10);
        let state = BackpressureState::new(config);

        for _ in 0..8 {
            state.increment();
        }
        assert!(state.should_throttle());
    }

    #[test]
    fn test_backpressure_state_resume() {
        let config = BackpressureConfig::new(10);
        let state = BackpressureState::new(config);

        for _ in 0..8 {
            state.increment();
        }

        for _ in 0..6 {
            state.decrement();
        }

        assert!(!state.should_throttle());
    }

    #[test]
    fn test_streaming_backpressure_flow() {
        let bp = StreamingBackpressure::default();

        assert!(bp.can_send());
        let state = bp.before_send();
        assert!(matches!(state, FlowControlState::Normal));
        bp.after_send();
        assert!(bp.can_send());
    }

    #[test]
    fn test_backpressure_reset() {
        let config = BackpressureConfig::default();
        let state = BackpressureState::new(config);

        state.increment();
        state.increment();
        state.increment();

        assert_eq!(state.pending_count(), 3);

        state.reset();
        assert_eq!(state.pending_count(), 0);
    }

    #[test]
    fn test_batch_operations() {
        let config = BackpressureConfig::default();
        let state = BackpressureState::new(config);

        state.batch_increment(10);
        assert_eq!(state.pending_count(), 10);

        state.batch_decrement(5);
        assert_eq!(state.pending_count(), 5);
    }
}
