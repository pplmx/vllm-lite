// crates/core/src/circuit_breaker/mod.rs
//! Circuit breaker pattern for fault tolerance
pub mod breaker;
pub mod strategy;
pub use breaker::{CircuitBreaker, CircuitBreakerConfig, CircuitState, CircuitBreakerError};
pub use strategy::{FallbackStrategy, RetryStrategy, DegradeStrategy, FailFastStrategy};
