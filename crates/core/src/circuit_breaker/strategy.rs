#![allow(dead_code, clippy::module_name_repetitions)]

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
        // invariant: attempt is bounded by max_attempts (small integer), so the
        // cast to u32 is safe.
        #[allow(clippy::cast_possible_truncation)]
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
        let attempts = self.max_attempts.max(1);
        for attempt in 0..attempts {
            match operation().await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    last_error = Some(e);
                    if attempt < attempts - 1 {
                        tokio::time::sleep(self.calculate_delay(attempt)).await;
                    }
                }
            }
        }
        // invariant: loop populates `last_error` on every iteration that doesn't return Ok;
        // `attempts` is at least 1, so the loop always runs at least once.
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
        op().map_or_else(|_| Ok((self.fallback)()), |result| Ok(result))
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

    #[tokio::test]
    async fn test_retry_strategy_zero_attempts_runs_once() {
        let strategy = RetryStrategy::new(0, Duration::from_millis(1));
        let attempts = std::sync::atomic::AtomicUsize::new(0);
        let result: Result<i32, ()> = strategy
            .execute(|| async {
                attempts.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Err(())
            })
            .await;
        assert_eq!(result, Err(()));
        assert_eq!(attempts.load(std::sync::atomic::Ordering::Relaxed), 1);
    }

    /// The retry delay is exponential backoff, and the LAST attempt must not
    /// sleep.
    ///
    /// `calculate_delay` was only reachable through a successful retry, and the
    /// suite used 1 ms delays, so both `base_delay * 2^attempt` operators and
    /// the whole function replaced by `Duration::default()` survived. The guard
    /// `attempt < attempts - 1` decides whether the final failed attempt sleeps:
    /// sleeping there delays the error by a whole backoff step for nothing.
    ///
    /// The clock is paused, so `Instant::now()` advances only when every task is
    /// idle — measured elapsed time is then the sum of the requested sleeps, not
    /// a wall-clock guess.
    #[tokio::test(start_paused = true)]
    async fn retry_strategy_backs_off_between_attempts_but_not_after_the_last() {
        let strategy = RetryStrategy::new(3, Duration::from_millis(100));
        let attempts = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let seen = std::sync::Arc::clone(&attempts);
        let started = tokio::time::Instant::now();
        let result: Result<i32, ()> = strategy
            .execute(|| {
                let seen = std::sync::Arc::clone(&seen);
                async move {
                    seen.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    Err(())
                }
            })
            .await;
        let elapsed = started.elapsed();

        assert_eq!(result, Err(()));
        assert_eq!(
            attempts.load(std::sync::atomic::Ordering::Relaxed),
            3,
            "3 attempts are made"
        );
        // Two sleeps: 100ms * 2^0 and 100ms * 2^1. The final attempt is the
        // error path, so it must NOT add a third (400ms) sleep.
        assert_eq!(
            elapsed,
            Duration::from_millis(300),
            "backoff is 100ms then 200ms; the last attempt must not sleep"
        );
    }

    /// A single attempt never sleeps, whatever the base delay is.
    #[tokio::test(start_paused = true)]
    async fn retry_strategy_single_attempt_does_not_sleep() {
        let strategy = RetryStrategy::new(1, Duration::from_secs(60));
        let started = tokio::time::Instant::now();
        let result: Result<i32, ()> = strategy.execute(|| async { Err(()) }).await;
        assert_eq!(result, Err(()));
        assert_eq!(
            started.elapsed(),
            Duration::ZERO,
            "one attempt is the last attempt and must not back off"
        );
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

    /// `RetryStrategy::builder()` starts from `RetryStrategy::default()`'s values.
    #[test]
    fn retry_strategy_builder_starts_from_the_documented_defaults() {
        let built = RetryStrategy::builder().build();
        assert_eq!(built.max_attempts, 3);
        assert_eq!(built.base_delay, Duration::from_millis(100));
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
