// tests/e2e_concurrent.rs
//! Concurrent request handling E2E tests

use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{Mutex, mpsc};
use vllm_core::engine::Engine;
use vllm_core::types::{Request, SchedulerConfig};
use vllm_testing::IncrementModel;

/// Thread-safe engine wrapper for concurrent tests
struct ConcurrentEngine {
    inner: Arc<Mutex<Engine<IncrementModel>>>,
}

impl ConcurrentEngine {
    fn new() -> Self {
        let config = SchedulerConfig::default();
        let engine = Engine::with_config(IncrementModel, None, config, 4, 1024);
        Self {
            inner: Arc::new(Mutex::new(engine)),
        }
    }

    async fn add_request(&self, max_tokens: usize) -> Result<u64, String> {
        let prompt: Vec<u32> = (1..=10).collect();
        let (tx, _rx) = mpsc::channel(64);
        let mut engine = self.inner.lock().await;
        let seq_id = engine.add_request(Request::new(1, prompt, max_tokens), tx);
        if seq_id > 0 {
            Ok(seq_id)
        } else {
            Err("Failed to add request".to_string())
        }
    }

    async fn process_single(&self, seq_id: u64, max_tokens: usize) -> Result<(), String> {
        let timeout = Duration::from_secs(30);
        let start = std::time::Instant::now();
        let mut iterations = 0;
        let max_iterations = max_tokens * 10; // Generous limit

        while start.elapsed() < timeout && iterations < max_iterations {
            let mut engine = self.inner.lock().await;

            if let Ok(_results) = engine.step() {
                // Check if our sequence is finished (not in running)
                let still_running = engine.scheduler.running().iter().any(|s| s.id == seq_id);
                if !still_running {
                    return Ok(());
                }
            }

            iterations += 1;
            drop(engine);
            tokio::time::sleep(Duration::from_millis(1)).await;
        }

        Err(format!(
            "Timeout waiting for completion after {} iterations",
            iterations
        ))
    }
}

impl Clone for ConcurrentEngine {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

#[tokio::test]
async fn test_concurrent_requests() {
    let concurrency = 10;
    let engine = ConcurrentEngine::new();

    let handles: Vec<_> = (0..concurrency)
        .map(|_| {
            let eng = engine.clone();
            tokio::spawn(async move {
                let id = eng.add_request(5).await?;
                eng.process_single(id, 5).await
            })
        })
        .collect();

    let mut success_count = 0;
    let mut errors = Vec::new();

    for (i, handle) in handles.into_iter().enumerate() {
        match handle.await {
            Ok(Ok(())) => success_count += 1,
            Ok(Err(e)) => errors.push(format!("Task {} failed: {}", i, e)),
            Err(e) => errors.push(format!("Task {} panicked: {}", i, e)),
        }
    }

    assert_eq!(
        success_count, concurrency,
        "Expected all {} requests to succeed, but {} succeeded. Errors: {:?}",
        concurrency, success_count, errors
    );
}

#[tokio::test]
async fn test_mixed_workload() {
    let engine = ConcurrentEngine::new();
    let count = 20;

    let handles: Vec<_> = (0..count)
        .map(|i| {
            let eng = engine.clone();
            let max_tokens = if i % 2 == 0 { 3 } else { 8 };
            tokio::spawn(async move {
                let id = eng.add_request(max_tokens).await?;
                eng.process_single(id, max_tokens).await
            })
        })
        .collect();

    let mut success_count = 0;
    let mut errors = Vec::new();

    for (i, handle) in handles.into_iter().enumerate() {
        match handle.await {
            Ok(Ok(())) => success_count += 1,
            Ok(Err(e)) => errors.push(format!("Task {}: {}", i, e)),
            Err(e) => errors.push(format!("Task {} panicked: {}", i, e)),
        }
    }

    let success_rate = success_count as f64 / count as f64;
    assert!(
        success_rate >= 0.9,
        "Expected at least 90% success rate, got {:.1}% ({} of {}). Errors: {:?}",
        success_rate * 100.0,
        success_count,
        count,
        errors
    );
}

#[tokio::test]
async fn test_staggered_requests() {
    let engine = ConcurrentEngine::new();
    let mut handles = Vec::new();

    // Add requests staggered over time
    for i in 0..5 {
        let eng = engine.clone();
        let handle = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(i * 10)).await;
            let id = eng.add_request(3).await?;
            eng.process_single(id, 3).await
        });
        handles.push(handle);
    }

    let mut success_count = 0;
    for handle in handles {
        if let Ok(Ok(())) = handle.await {
            success_count += 1;
        }
    }

    assert_eq!(success_count, 5, "All staggered requests should complete");
}

#[test]
fn test_batch_processing() {
    let config = SchedulerConfig::default();
    let mut engine = Engine::with_config(IncrementModel, None, config, 4, 1024);

    // Add multiple requests
    let num_requests = 10;
    let mut receivers = Vec::new();

    for i in 0..num_requests {
        let (tx, rx) = mpsc::channel(64);
        let seq_id = engine.add_request(Request::new(i, vec![10, 20], 5), tx);
        assert!(seq_id > 0);
        receivers.push(rx);
    }

    // Process all in batch
    let mut total_tokens = 0;
    let max_iterations = 50;

    for _ in 0..max_iterations {
        if let Ok(results) = engine.step() {
            total_tokens += results.len();
        }

        // Check if all done
        if engine.scheduler.running().is_empty() && !engine.has_pending() {
            break;
        }
    }

    assert!(
        total_tokens >= num_requests as usize,
        "Should process tokens for all {} requests, got {}",
        num_requests,
        total_tokens
    );
}
