// crates/core/src/circuit_breaker/strategy.rs
use std::fmt;
use std::time::Duration;

/// Trait for fallback strategies when circuit breaker is open
pub trait FallbackStrategy: Send + Sync {
    type Output;
    fn fallback(&self) -> Self::Output;
}

/// Retry strategy configuration
#[derive(Debug, Clone)]
pub struct RetryStrategy {
    pub max_retries: usize,
    pub initial_delay: Duration,
    pub max_delay: Duration,
    pub backoff_multiplier: f64,
}

impl Default for RetryStrategy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
            backoff_multiplier: 2.0,
        }
    }
}

impl RetryStrategy {
    /// Calculate delay for a specific retry attempt
    pub fn delay_for(&self, attempt: usize) -> Duration {
        if attempt == 0 {
            Duration::ZERO
        } else {
            let delay = self.initial_delay.as_millis() as f64
                * self.backoff_multiplier.powi(attempt as i32 - 1);
            let delay = delay.min(self.max_delay.as_millis() as f64) as u64;
            Duration::from_millis(delay)
        }
    }

    /// Check if we should retry
    pub fn should_retry(&self, attempt: usize, _error: &dyn fmt::Display) -> bool {
        attempt < self.max_retries
    }
}

/// Degrade strategy for graceful degradation
#[derive(Debug, Clone)]
pub struct DegradeStrategy {
    pub enable_degraded_mode: bool,
    pub degraded_capacity: usize, // percentage 0-100
    pub response_timeout: Duration,
}

impl Default for DegradeStrategy {
    fn default() -> Self {
        Self {
            enable_degraded_mode: true,
            degraded_capacity: 50,
            response_timeout: Duration::from_secs(5),
        }
    }
}

impl DegradeStrategy {
    /// Check if degraded mode should be used
    pub fn should_degrade(&self, error_rate: f64) -> bool {
        self.enable_degraded_mode && error_rate > 0.3
    }

    /// Get the degraded capacity as a fraction
    pub fn capacity_fraction(&self) -> f64 {
        (self.degraded_capacity as f64) / 100.0
    }
}

/// Fail-fast strategy for immediate rejection
#[derive(Debug, Clone)]
pub struct FailFastStrategy {
    pub enable_fail_fast: bool,
    pub max_concurrent: usize,
    pub queue_size: usize,
}

impl Default for FailFastStrategy {
    fn default() -> Self {
        Self {
            enable_fail_fast: true,
            max_concurrent: 100,
            queue_size: 1000,
        }
    }
}

impl FailFastStrategy {
    /// Check if request should be rejected immediately
    pub fn should_fail_fast(&self, current_load: usize) -> bool {
        self.enable_fail_fast && current_load >= self.max_concurrent + self.queue_size
    }
}
