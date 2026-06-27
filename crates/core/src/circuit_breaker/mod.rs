//! mod: module.

// crates/core/src/circuit_breaker/mod.rs
//! Circuit breaker pattern for fault tolerance
/// breaker: breaker module.
pub mod breaker;
/// strategy: strategy module.
pub mod strategy;
pub use breaker::{CircuitBreaker, CircuitBreakerConfig, CircuitBreakerError, CircuitState};
pub use strategy::{DegradeStrategy, FailFastStrategy, FallbackStrategy, RetryStrategy};
