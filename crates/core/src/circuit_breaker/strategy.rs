#![allow(dead_code,clippy::module_name_repetitions)]

//! Fallback strategies for the circuit breaker.
//!
//! # Sync vs Async Trait Split (Phase 32 / API-08)
//!
//! Two distinct traits exist for callers to pick based on runtime requirements:
//!
//! - [`FallbackStrategy`] — purely-computational fallbacks. Sync. The
//!   operation is a plain `fn() -> Result<T, E>`. Useful for in-process
//!   retry/fail-fast on synchronous computations.
//! - [`AsyncFallbackStrategy`] — I/O-bound fallbacks. Async. The operation
//!   is `Fn() -> Future<Result<T, E>>`. Useful for retries with `tokio::sleep`
//!   or external resource access.
//!
//! # Object Safety
//!
//! These traits are intentionally **not object-safe** — they have generic
//! `execute` methods for caller ergonomics. Callers who need runtime dispatch
//! should box the concrete strategy type (`Box<RetryStrategy>`, `Box<FailFastStrategy>`),
//! not the trait. Both concrete types provide `Default` (Phase 32 / API-06).

use std::time::Duration;

/// Sync fallback strategy for purely-computational operations.
///
/// `op` is a plain function pointer — no closures, no futures. Use this
/// when the fallback wraps a synchronous computation (e.g., retrying a
/// pure calculation, fail-fast on a sync validation).
pub trait FallbackStrategy {
    /// Execute the operation, applying the fallback policy on failure.
    /// On success, returns the operation's result. On failure, the policy
    /// decides whether to retry, fail-fast, or degrade.
    fn execute<T, E>(&self, op: fn() -> Result<T, E>) -> Result<T, E>;
}

/// Async fallback strategy for I/O-bound operations.
#[async_trait::async_trait]
pub trait AsyncFallbackStrategy {
    /// Execute the operation, applying the fallback policy on failure.
    async fn execute<F, Fut, T, E>(&self, operation: F) -> Result<T, E>
    where
        F: Fn() -> Fut + Send,
        Fut: std::future::Future<Output = Result<T, E>> + Send,
        T: Send,
        E: Send;
}

// ─────────────────────── RetryStrategy (async, with sleep) ──────────────────

/// Retry strategy with exponential backoff. Async (uses `tokio::time::sleep`).
pub struct RetryStrategy {
    pub(crate) max_attempts: usize,
    pub(crate) base_delay: Duration,
}

impl RetryStrategy {
    /// new: construct with explicit max attempts and base delay.
    pub const fn new(max_attempts: usize, base_delay: Duration) -> Self {
        Self {
            max_attempts,
            base_delay,
        }
    }

    /// Returns a builder for configuring this type with the documented field defaults.
    /// Use `with_*(...)` to override individual fields, then `build()` to produce the type.
    pub fn builder() -> RetryStrategyBuilder {
        RetryStrategyBuilder::default()
    }

    fn calculate_delay(&self, attempt: usize) -> Duration {
        let multiplier = 2_u32.pow(attempt as u32);
        self.base_delay * multiplier
    }
}

impl Default for RetryStrategy {
    /// Default: 3 attempts, 100ms base delay.
    fn default() -> Self {
        Self::new(3, Duration::from_millis(100))
    }
}

/// Builder for [`RetryStrategy`].
#[derive(Debug, Clone)]
pub struct RetryStrategyBuilder {
    max_attempts: usize,
    base_delay: Duration,
}

impl Default for RetryStrategyBuilder {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            base_delay: Duration::from_millis(100),
        }
    }
}

impl RetryStrategyBuilder {
    pub const fn with_max_attempts(mut self, n: usize) -> Self {
        self.max_attempts = n;
        self
    }

    pub const fn with_base_delay(mut self, d: Duration) -> Self {
        self.base_delay = d;
        self
    }

    pub const fn build(self) -> RetryStrategy {
        RetryStrategy::new(self.max_attempts, self.base_delay)
    }
}

#[async_trait::async_trait]
impl AsyncFallbackStrategy for RetryStrategy {
    async fn execute<F, Fut, T, E>(&self, operation: F) -> Result<T, E>
    where
        F: Fn() -> Fut + Send,
        Fut: std::future::Future<Output = Result<T, E>> + Send,
        T: Send,
        E: Send,
    {
        let mut last_error = None;
        for attempt in 0..self.max_attempts {
            match operation().await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    last_error = Some(e);
                    if attempt < self.max_attempts - 1 {
                        tokio::time::sleep(self.calculate_delay(attempt)).await;
                    }
                }
            }
        }
        // invariant: loop populates `last_error` on every iteration that doesn't return Ok;
        // if the loop exits without setting it, the success path has already returned.
        Err(last_error.unwrap())
    }
}

// ─────────────────────── FailFastStrategy (sync passthrough) ──────────────────

/// Fail-fast strategy — no fallback, propagate immediately.
///
/// Implements the sync trait because there's no I/O involved.
#[derive(Debug, Clone, Copy, Default)]
pub struct FailFastStrategy;

impl FallbackStrategy for FailFastStrategy {
    fn execute<T, E>(&self, op: fn() -> Result<T, E>) -> Result<T, E> {
        op()
    }
}

// ─────────────────────── DegradeStrategy (sync) ──────────────────────────────

/// Degrade strategy — fallback to a simpler implementation on error.
///
/// Not a `FallbackStrategy` impl in the original trait because it changes
/// the output type. Re-introduced here as a sync helper that takes both
/// the primary op (sync) and a fallback closure (sync, infallible).
pub struct DegradeStrategy<F> {
    fallback: F,
}

impl<F> DegradeStrategy<F> {
    pub const fn new<T>(fallback: F) -> Self
    where
        F: Fn() -> T,
    {
        Self { fallback }
    }

    /// execute: execute the primary operation; on error, run the fallback.
    pub fn execute<T, E>(&self, op: fn() -> Result<T, E>) -> Result<T, T>
    where
        F: Fn() -> T,
    {
        match op() {
            Ok(result) => Ok(result),
            Err(_) => Ok((self.fallback)()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ────── Sync FallbackStrategy ──────

    #[test]
    fn test_fail_fast_sync_success() {
        let strategy = FailFastStrategy;
        let result: Result<i32, ()> = strategy.execute(|| Ok(42));
        assert_eq!(result, Ok(42));
    }

    #[test]
    fn test_fail_fast_sync_propagates_error() {
        let strategy = FailFastStrategy;
        let result: Result<i32, ()> = strategy.execute(|| Err(()));
        assert_eq!(result, Err(()));
    }

    #[test]
    fn test_degrade_strategy_sync() {
        let strategy = DegradeStrategy::new(|| 42);
        let result: Result<i32, i32> = strategy.execute(|| Err(99));
        assert_eq!(result, Ok(42));
    }

    // ────── Async AsyncFallbackStrategy ──────

    #[tokio::test]
    async fn test_retry_strategy_success() {
        let strategy = RetryStrategy::new(3, Duration::from_millis(10));
        let result: Result<i32, ()> = strategy.execute(|| async { Ok(42) }).await;
        assert_eq!(result, Ok(42));
    }

    #[tokio::test]
    async fn test_retry_strategy_eventually_succeeds() {
        let strategy = RetryStrategy::new(3, Duration::from_millis(1));
        let attempts = std::sync::atomic::AtomicUsize::new(0);
        let result: Result<i32, ()> = strategy
            .execute(|| async {
                let count = attempts.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if count < 2 { Err(()) } else { Ok(42) }
            })
            .await;
        assert_eq!(result, Ok(42));
    }

    #[tokio::test]
    async fn test_retry_strategy_default_impl() {
        let strategy = RetryStrategy::default();
        let result: Result<i32, ()> = strategy.execute(|| async { Ok(1) }).await;
        assert_eq!(result, Ok(1));
    }

    // ────── Builder ──────

    #[test]
    fn test_retry_strategy_builder() {
        let strategy = RetryStrategy::builder()
            .with_max_attempts(5)
            .with_base_delay(Duration::from_millis(50))
            .build();
        assert_eq!(strategy.max_attempts, 5);
        assert_eq!(strategy.base_delay, Duration::from_millis(50));
    }

    // ────── Object safety ──────

    // ────── Object safety (compile-only) ──────
    //
    // Note: The generic `execute<T, E>` / `execute<F, Fut, T, E>` methods
    // mean these traits are NOT object-safe by design — they trade dyn
    // compatibility for caller ergonomics. Callers who need `dyn` dispatch
    // should box the concrete strategy type, not the trait.

    #[test]
    fn test_sync_fallback_erased_box() {
        let _boxed: Box<FailFastStrategy> = Box::default();
    }

    #[tokio::test]
    async fn test_async_fallback_erased_box() {
        let _boxed: Box<RetryStrategy> = Box::default();
    }
}
