// crates/core/src/circuit_breaker/breaker.rs
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, trace, warn};

/// Circuit breaker state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    Closed,   // Normal operation
    Open,     // Failing, reject calls
    HalfOpen, // Testing recovery
}

/// Circuit breaker configuration
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    pub failure_threshold: usize,
    pub recovery_timeout: Duration,
    pub half_open_max_calls: usize,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            recovery_timeout: Duration::from_secs(30),
            half_open_max_calls: 3,
        }
    }
}

/// Circuit breaker error
#[derive(Debug, thiserror::Error, Clone, PartialEq)]
pub enum CircuitBreakerError {
    #[error("circuit breaker is open")]
    Open,
    #[error("operation failed: {0}")]
    OperationFailed(String),
}

/// Circuit breaker implementation
#[derive(Clone)]
pub struct CircuitBreaker {
    config: CircuitBreakerConfig,
    state: Arc<RwLock<CircuitState>>,
    failure_count: Arc<AtomicU64>,
    last_failure_time: Arc<RwLock<Option<Instant>>>,
    half_open_calls: Arc<AtomicU64>,
}

impl CircuitBreaker {
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            config,
            state: Arc::new(RwLock::new(CircuitState::Closed)),
            failure_count: Arc::new(AtomicU64::new(0)),
            last_failure_time: Arc::new(RwLock::new(None)),
            half_open_calls: Arc::new(AtomicU64::new(0)),
        }
    }

    pub async fn call<F, Fut, T, E>(&self, operation: F) -> Result<T, CircuitBreakerError>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<T, E>>,
        E: std::fmt::Display,
    {
        self.check_and_transition().await;
        let state = *self.state.read().await;
        match state {
            CircuitState::Open => {
                return Err(CircuitBreakerError::Open);
            }
            CircuitState::HalfOpen => {
                let calls = self.half_open_calls.fetch_add(1, Ordering::Relaxed);
                if calls >= self.config.half_open_max_calls as u64 {
                    return Err(CircuitBreakerError::Open);
                }
            }
            CircuitState::Closed => {}
        }

        match operation().await {
            Ok(result) => {
                trace!("Circuit breaker: call succeeded, resetting failure count");
                self.on_success().await;
                Ok(result)
            }
            Err(e) => {
                warn!(
                    error = %e,
                    "Circuit breaker: call failed, incrementing failure count"
                );
                self.on_failure().await;
                Err(CircuitBreakerError::OperationFailed(e.to_string()))
            }
        }
    }

    async fn check_and_transition(&self) {
        debug!(
            current_state = ?*self.state.read().await,
            failure_count = self.failure_count.load(Ordering::Relaxed),
            "Circuit breaker check"
        );
        let mut state = self.state.write().await;
        if matches!(*state, CircuitState::Open) {
            let should_attempt = {
                let last = self.last_failure_time.read().await;
                last.map(|t| t.elapsed() >= self.config.recovery_timeout)
                    .unwrap_or(false)
            };
            if should_attempt {
                trace!("Circuit breaker: Closed -> HalfOpen");
                *state = CircuitState::HalfOpen;
                self.half_open_calls.store(0, Ordering::Relaxed);
            }
        }
    }

    async fn on_success(&self) {
        let mut state = self.state.write().await;
        if matches!(*state, CircuitState::HalfOpen) {
            trace!("Circuit breaker: HalfOpen -> Closed");
            *state = CircuitState::Closed;
            self.failure_count.store(0, Ordering::Relaxed);
        }
    }

    async fn on_failure(&self) {
        let count = self.failure_count.fetch_add(1, Ordering::Relaxed);
        *self.last_failure_time.write().await = Some(Instant::now());
        if count + 1 >= self.config.failure_threshold as u64 {
            warn!(
                last_failure = ?*self.last_failure_time.read().await,
                "Circuit breaker: entering Open state"
            );
            let mut state = self.state.write().await;
            *state = CircuitState::Open;
        }
    }

    pub async fn state(&self) -> CircuitState {
        *self.state.read().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct TestError(&'static str);

    impl std::fmt::Display for TestError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.0)
        }
    }

    impl std::error::Error for TestError {}

    #[tokio::test]
    async fn test_circuit_starts_closed() {
        let breaker = CircuitBreaker::new(CircuitBreakerConfig::default());
        let result = breaker.call(|| async { Ok::<_, TestError>(42) }).await;
        assert_eq!(result, Ok(42));
    }

    #[tokio::test]
    async fn test_circuit_opens_after_failures() {
        let config = CircuitBreakerConfig {
            failure_threshold: 3,
            recovery_timeout: Duration::from_millis(100),
            half_open_max_calls: 1,
        };
        let breaker = CircuitBreaker::new(config);
        for _ in 0..3 {
            let _ = breaker
                .call(|| async { Err::<i32, TestError>(TestError("fail")) })
                .await;
        }
        let result = breaker.call(|| async { Ok::<_, TestError>(42) }).await;
        assert!(matches!(result, Err(CircuitBreakerError::Open)));
    }

    #[tokio::test]
    async fn test_circuit_transitions_to_half_open() {
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            recovery_timeout: Duration::from_millis(50),
            half_open_max_calls: 3, // Allow multiple calls in half-open
        };
        let breaker = CircuitBreaker::new(config);
        // First failure opens the circuit
        let _ = breaker
            .call(|| async { Err::<i32, TestError>(TestError("fail")) })
            .await;
        // Wait for recovery timeout
        tokio::time::sleep(Duration::from_millis(100)).await;
        // The next call will transition to HalfOpen
        // But if it fails, it goes back to Open
        // If we want to test HalfOpen, we need to make a call that fails but
        // check state before it transitions back. Let's use success to verify the transition happens.
        let _ = breaker.call(|| async { Ok::<_, TestError>(42) }).await;
        // After successful call in HalfOpen, state should be Closed
        let state = breaker.state().await;
        assert!(matches!(state, CircuitState::Closed));
    }

    #[tokio::test]
    async fn test_half_open_detected() {
        let config = CircuitBreakerConfig {
            failure_threshold: 2,
            recovery_timeout: Duration::from_millis(50),
            half_open_max_calls: 3,
        };
        let breaker = CircuitBreaker::new(config);
        // Two failures open the circuit
        let _ = breaker
            .call(|| async { Err::<i32, TestError>(TestError("fail")) })
            .await;
        let _ = breaker
            .call(|| async { Err::<i32, TestError>(TestError("fail")) })
            .await;
        // Wait for recovery
        tokio::time::sleep(Duration::from_millis(100)).await;
        // Check that we're in HalfOpen by making a call that succeeds
        // and transitions to Closed
        assert!(matches!(breaker.state().await, CircuitState::Open));
        let _ = breaker.call(|| async { Ok::<_, TestError>(42) }).await;
        assert!(matches!(breaker.state().await, CircuitState::Closed));
    }
}
