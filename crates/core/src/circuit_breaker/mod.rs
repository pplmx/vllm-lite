#![allow(unused_imports)]
#![allow(dead_code)]
// crates/core/src/circuit_breaker/mod.rs
//! Circuit breaker pattern for fault tolerance
pub mod breaker;
pub mod strategy;
pub(crate) use breaker::{CircuitBreaker, CircuitBreakerConfig, CircuitBreakerError, CircuitState};
pub(crate) use strategy::{
    AsyncFallbackStrategy, DegradeStrategy, FailFastStrategy, FallbackStrategy, RetryStrategy,
    RetryStrategyBuilder,
};
