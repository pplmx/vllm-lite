// tests/e2e/common/mod.rs
//! Shared E2E test utilities

pub mod mock_model;

use std::time::Duration;

/// Default timeout for E2E operations
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

/// Generate a test request with specified token count
pub fn generate_test_request(token_count: usize) -> Request {
    Request {
        id: generate_seq_id(),
        tokens: vec![1u64; token_count],
        max_tokens: token_count + 50,
    }
}

fn generate_seq_id() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(1);
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// Test request structure
#[derive(Debug, Clone)]
pub struct Request {
    pub id: u64,
    pub tokens: Vec<u64>,
    pub max_tokens: usize,
}

/// Wait for condition with timeout
pub async fn wait_for<F, Fut>(condition: F, max_wait: Duration) -> Result<(), String>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = bool>,
{
    let start = std::time::Instant::now();
    while start.elapsed() < max_wait {
        if condition().await {
            return Ok(());
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    Err("Timeout waiting for condition".to_string())
}
