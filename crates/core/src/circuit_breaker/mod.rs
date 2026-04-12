// crates/core/src/circuit_breaker/mod.rs
//! Circuit breaker pattern for fault tolerance
pub mod breaker;
pub mod strategy;
pub use breaker::{CircuitBreaker, CircuitBreakerConfig, CircuitBreakerError, CircuitState};
pub use strategy::{DegradeStrategy, FailFastStrategy, FallbackStrategy, RetryStrategy};
