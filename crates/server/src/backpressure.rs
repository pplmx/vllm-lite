//! Request backpressure: token-bucket + concurrency-cap admission control that protects the engine from overload.
//!
//! The handlers wrap each request through the backpressure layer before
//! forwarding to the engine; rejected requests get a 429 with a
//! `Retry-After` header. The implementation lives in `pub(crate)`
//! helpers and is not part of the public API.
#![allow(dead_code)]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Configuration for Backpressure. Constructed via the `builder()` associated function or by deserializing from JSON / TOML. Pass-by-value to construction APIs.
#[derive(Debug, Clone)]
pub(crate) struct BackpressureConfig {
    /// Maximum in-flight requests before the backpressure gate trips.
    pub max_buffer_size: usize,
    /// Buffer-fill level at which new requests start being throttled.
    pub high_water_mark: usize,
    /// Buffer-fill level at which throttling is lifted.
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
    /// Build config with 75% high-water and 25% resume thresholds.
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

/// Internal state of `FlowControl`. Mutated under a lock; read via accessor methods on the parent type.
#[derive(Debug, Clone)]
pub(crate) enum FlowControlState {
    Normal,
    Throttled,
    Resumed,
}

/// Internal state of Backpressure. Mutated under a lock; read via accessor methods on the parent type.
#[derive(Debug)]
pub(crate) struct BackpressureState {
    pending_tokens: Arc<AtomicUsize>,
    config: BackpressureConfig,
    last_state: std::sync::Mutex<FlowControlState>,
}

impl BackpressureState {
    /// Create state tracking in-flight requests against `config` limits.
    pub fn new(config: BackpressureConfig) -> Self {
        Self {
            pending_tokens: Arc::new(AtomicUsize::new(0)),
            config,
            last_state: std::sync::Mutex::new(FlowControlState::Normal),
        }
    }

    /// Register one in-flight request and return the updated flow-control state.
    pub fn increment(&self) -> FlowControlState {
        let pending = self.pending_tokens.fetch_add(1, Ordering::SeqCst);
        self.evaluate_state(pending + 1)
    }

    /// Release one in-flight request and return the updated flow-control state.
    pub fn decrement(&self) -> FlowControlState {
        let pending = self.pending_tokens.fetch_sub(1, Ordering::SeqCst);
        self.evaluate_state(pending.saturating_sub(1))
    }

    /// Register `count` in-flight requests in a single atomic step.
    pub fn batch_increment(&self, count: usize) -> FlowControlState {
        let pending = self.pending_tokens.fetch_add(count, Ordering::SeqCst);
        self.evaluate_state(pending + count)
    }

    /// Release `count` in-flight requests in a single atomic step.
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

    /// Whether pending count has reached the high-water throttle threshold.
    pub fn should_throttle(&self) -> bool {
        let pending = self.pending_tokens.load(Ordering::SeqCst);
        pending >= self.config.high_water_mark
    }

    /// Current number of in-flight requests tracked by this gate.
    pub fn pending_count(&self) -> usize {
        self.pending_tokens.load(Ordering::SeqCst)
    }

    /// Clear pending count and return flow control to [`FlowControlState::Normal`].
    pub fn reset(&self) {
        self.pending_tokens.store(0, Ordering::SeqCst);
        *self
            .last_state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = FlowControlState::Normal;
    }
}

/// `StreamingBackpressure`. See the type definition for fields and behavior.
pub(crate) struct StreamingBackpressure {
    state: Arc<BackpressureState>,
}

impl StreamingBackpressure {
    /// Wrap a new [`BackpressureState`] for streaming response admission control.
    pub fn new(config: BackpressureConfig) -> Self {
        Self {
            state: Arc::new(BackpressureState::new(config)),
        }
    }

    /// Shared backpressure state for inspection in tests and middleware.
    pub const fn state(&self) -> &Arc<BackpressureState> {
        &self.state
    }

    /// Acquire a send slot before emitting the next streaming chunk.
    pub fn before_send(&self) -> FlowControlState {
        self.state.increment()
    }

    /// Release a send slot after a streaming chunk has been flushed.
    pub fn after_send(&self) -> FlowControlState {
        self.state.decrement()
    }

    /// Whether a new streaming chunk may be sent without throttling.
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
