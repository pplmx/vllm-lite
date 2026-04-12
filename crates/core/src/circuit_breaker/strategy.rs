// crates/core/src/circuit_breaker/strategy.rs
use std::time::Duration;

/// Trait for fallback strategies
#[async_trait::async_trait]
pub trait FallbackStrategy {
    async fn execute<F, Fut, T, E>(&self, operation: F) -> Result<T, E>
    where
        F: Fn() -> Fut + Send,
        Fut: std::future::Future<Output = Result<T, E>> + Send,
        T: Send,
        E: Send;
}

/// Retry strategy with exponential backoff
pub struct RetryStrategy {
    max_attempts: usize,
    base_delay: Duration,
}

impl RetryStrategy {
    pub fn new(max_attempts: usize, base_delay: Duration) -> Self {
        Self {
            max_attempts,
            base_delay,
        }
    }

    fn calculate_delay(&self, attempt: usize) -> Duration {
        let multiplier = 2_u32.pow(attempt as u32);
        self.base_delay * multiplier
    }
}

#[async_trait::async_trait]
impl FallbackStrategy for RetryStrategy {
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
        Err(last_error.unwrap())
    }
}

/// Fail-fast strategy - no fallback, propagate immediately
pub struct FailFastStrategy;

#[async_trait::async_trait]
impl FallbackStrategy for FailFastStrategy {
    async fn execute<F, Fut, T, E>(&self, operation: F) -> Result<T, E>
    where
        F: Fn() -> Fut + Send,
        Fut: std::future::Future<Output = Result<T, E>> + Send,
        T: Send,
        E: Send,
    {
        operation().await
    }
}

/// Degrade strategy - fallback to simpler implementation
/// This is not a FallbackStrategy impl because it changes the output type
pub struct DegradeStrategy<F> {
    fallback: F,
}

impl<F> DegradeStrategy<F> {
    pub fn new<T>(fallback: F) -> Self
    where
        F: Fn() -> T,
    {
        Self { fallback }
    }

    pub async fn execute<Op, Fut, T, E>(&self, operation: Op) -> Result<T, E>
    where
        Op: Fn() -> Fut + Send,
        Fut: std::future::Future<Output = Result<T, E>> + Send,
        F: Fn() -> T,
        T: Send,
        E: Send,
    {
        match operation().await {
            Ok(result) => Ok(result),
            Err(_) => Ok((self.fallback)()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_retry_strategy_success() {
        let strategy = RetryStrategy::new(3, Duration::from_millis(10));
        let result = strategy.execute(|| async { Ok::<_, ()>(42) }).await;
        assert_eq!(result, Ok(42));
    }

    #[tokio::test]
    async fn test_retry_strategy_eventually_succeeds() {
        let strategy = RetryStrategy::new(3, Duration::from_millis(1));
        let attempts = std::sync::atomic::AtomicUsize::new(0);
        let result = strategy
            .execute(|| async {
                let count =
                    attempts.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if count < 2 {
                    Err::<i32, ()>(())
                } else {
                    Ok(42)
                }
            })
            .await;
        assert_eq!(result, Ok(42));
    }

    #[tokio::test]
    async fn test_degrade_strategy_fallback() {
        let strategy = DegradeStrategy::new(|| 42);
        let result = strategy.execute(|| async { Err::<i32, ()>(()) }).await;
        assert_eq!(result, Ok(42));
    }

    #[tokio::test]
    async fn test_degrade_strategy_uses_original_on_success() {
        let strategy = DegradeStrategy::new(|| 42);
        let result = strategy.execute(|| async { Ok::<_, ()>(100) }).await;
        assert_eq!(result, Ok(100));
    }
}
