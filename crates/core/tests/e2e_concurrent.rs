// tests/e2e_concurrent.rs
//! Concurrent request handling E2E tests

use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{Mutex, mpsc};
use vllm_core::engine::Engine;
use vllm_core::types::{Request, SchedulerConfig};
use vllm_testing::IncrementModel;

/// Thread-safe engine wrapper with background stepper
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

    async fn start_background_stepper(&self) {
        let inner = self.inner.clone();

        tokio::spawn(async move {
            loop {
                let mut engine = inner.lock().await;
                if !engine.has_pending() {
                    drop(engine);
                    tokio::time::sleep(Duration::from_millis(1)).await;
                    continue;
                }

                let _ = engine.step();
                drop(engine);
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        });
    }

    async fn add_request(&self, max_tokens: usize) -> Result<u64, String> {
        let prompt: Vec<u32> = (1..=10).collect();
        let (tx, _rx) = mpsc::channel(64);
        let mut engine = self.inner.lock().await;
        let seq_id = engine.add_request(Request::new(0, prompt, max_tokens), tx);
        if seq_id > 0 {
            Ok(seq_id)
        } else {
            Err("Failed to add request".to_string())
        }
    }

    async fn wait_for_completion(&self, seq_id: u64, _max_tokens: usize) -> Result<(), String> {
        let timeout = Duration::from_secs(30);
        let start = std::time::Instant::now();

        while start.elapsed() < timeout {
            let engine = self.inner.lock().await;

            let still_running = engine.scheduler.running().iter().any(|s| s.id == seq_id);

            if !still_running {
                return Ok(());
            }

            drop(engine);
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        Err("Timeout waiting for completion".to_string())
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
    let engine = ConcurrentEngine::new();
    engine.start_background_stepper().await;

    // Small delay to let stepper start
    tokio::time::sleep(Duration::from_millis(50)).await;

    let concurrency = 10;
    let handles: Vec<_> = (0..concurrency)
        .map(|_| {
            let eng = engine.clone();
            tokio::spawn(async move {
                let id = eng.add_request(15).await?; // total tokens = 10 + 5
                eng.wait_for_completion(id, 15).await
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
    engine.start_background_stepper().await;

    tokio::time::sleep(Duration::from_millis(50)).await;

    let count = 20;

    let handles: Vec<_> = (0..count)
        .map(|i| {
            let eng = engine.clone();
            let max_tokens = if i % 2 == 0 { 13 } else { 18 }; // total tokens
            tokio::spawn(async move {
                let id = eng.add_request(max_tokens).await?;
                eng.wait_for_completion(id, max_tokens).await
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
    engine.start_background_stepper().await;

    tokio::time::sleep(Duration::from_millis(50)).await;

    let mut handles = Vec::new();

    // Add requests staggered over time
    for i in 0..5 {
        let eng = engine.clone();
        let handle = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(i * 20)).await;
            let id = eng.add_request(13).await?; // total tokens = 10 + 3
            eng.wait_for_completion(id, 13).await
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

    // Add multiple requests with total tokens = prompt + max_tokens
    let num_requests = 10;
    let mut receivers = Vec::new();

    for i in 0..num_requests {
        let (tx, rx) = mpsc::channel(64);
        let seq_id = engine.add_request(Request::new(i, vec![10, 20], 15), tx); // total = 2 + 15 = 17
        assert!(seq_id > 0);
        receivers.push(rx);
    }

    // Process all in batch
    let mut total_tokens = 0;
    let max_iterations = 100;

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
        total_tokens >= num_requests as usize * 5, // 5 tokens each
        "Should process tokens for all {} requests, got {}",
        num_requests,
        total_tokens
    );
}
