// tests/e2e_graceful_shutdown.rs
//! Graceful shutdown E2E tests

use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{Mutex, mpsc};
use vllm_core::engine::Engine;
use vllm_core::types::{EngineMessage, Request, SchedulerConfig};
use vllm_testing::IncrementModel;

/// Engine with shutdown capabilities using actor pattern
struct ShutdownEngine {
    msg_tx: mpsc::UnboundedSender<EngineMessage>,
    shutdown_complete: Arc<Mutex<bool>>,
}

impl ShutdownEngine {
    fn new() -> Self {
        let config = SchedulerConfig::default();
        let mut engine = Engine::with_config(IncrementModel, IncrementModel, config, 4, 1024);

        let (msg_tx, msg_rx) = mpsc::unbounded_channel::<EngineMessage>();
        let shutdown_complete = Arc::new(Mutex::new(false));
        let shutdown_complete_clone = Arc::clone(&shutdown_complete);

        // Spawn the engine loop in a blocking task
        tokio::task::spawn_blocking(move || {
            engine.run(msg_rx);
            // Mark shutdown complete
            if let Ok(mut guard) = shutdown_complete_clone.try_lock() {
                *guard = true;
            }
        });

        Self {
            msg_tx,
            shutdown_complete,
        }
    }

    async fn add_request(&self, max_tokens: usize) -> Result<u64, String> {
        let (tx, _rx) = mpsc::channel(64);
        let request = Request::new(1, vec![10, 20, 30], max_tokens);

        self.msg_tx
            .send(EngineMessage::AddRequest {
                request,
                response_tx: tx,
            })
            .map_err(|_| "Failed to send request")?;

        // Return placeholder - actual seq_id assigned by engine
        Ok(1)
    }

    async fn shutdown(&self) -> Result<(), String> {
        self.msg_tx
            .send(EngineMessage::Shutdown)
            .map_err(|_| "Failed to send shutdown")?;
        Ok(())
    }

    async fn is_shutdown_complete(&self) -> bool {
        *self.shutdown_complete.lock().await
    }
}

impl Clone for ShutdownEngine {
    fn clone(&self) -> Self {
        Self {
            msg_tx: self.msg_tx.clone(),
            shutdown_complete: Arc::clone(&self.shutdown_complete),
        }
    }
}

#[tokio::test]
async fn test_graceful_shutdown_signal() {
    let engine = ShutdownEngine::new();

    // Add some requests
    for _ in 0..5 {
        let _ = engine.add_request(3).await;
    }

    // Send shutdown signal
    let result = engine.shutdown().await;
    assert!(result.is_ok(), "Should successfully send shutdown signal");

    // Wait for shutdown to complete
    let timeout = Duration::from_secs(5);
    let start = std::time::Instant::now();
    while !engine.is_shutdown_complete().await && start.elapsed() < timeout {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    assert!(
        engine.is_shutdown_complete().await,
        "Engine should shut down within timeout"
    );
}

#[tokio::test]
async fn test_shutdown_with_in_flight_requests() {
    let engine = ShutdownEngine::new();

    // Add multiple in-flight requests
    let mut added = 0;
    for _ in 0..5 {
        if engine.add_request(5).await.is_ok() {
            added += 1;
        }
    }

    assert!(added > 0, "Should have added at least one request");

    // Initiate shutdown
    let _ = engine.shutdown().await;

    // Wait for completion with timeout
    let timeout = Duration::from_secs(10);
    let start = std::time::Instant::now();

    while !engine.is_shutdown_complete().await && start.elapsed() < timeout {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    assert!(
        engine.is_shutdown_complete().await,
        "Should complete shutdown within timeout"
    );
}

#[tokio::test]
async fn test_shutdown_completes_within_timeout() {
    let engine = ShutdownEngine::new();

    // Add requests with varying token counts
    let _ = engine.add_request(3).await;
    let _ = engine.add_request(5).await;
    let _ = engine.add_request(2).await;

    let start = std::time::Instant::now();
    let _ = engine.shutdown().await;

    let timeout = Duration::from_secs(5);
    while !engine.is_shutdown_complete().await && start.elapsed() < timeout {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    let elapsed = start.elapsed();
    assert!(
        engine.is_shutdown_complete().await,
        "Shutdown should complete within 5 seconds"
    );
    assert!(
        elapsed < Duration::from_secs(5),
        "Shutdown took {:?}, expected under 5s",
        elapsed
    );
}

#[tokio::test]
async fn test_multiple_shutdown_signals_idempotent() {
    let engine = ShutdownEngine::new();

    // Send multiple shutdown signals
    for _ in 0..3 {
        let _ = engine.shutdown().await;
    }

    // Wait for shutdown
    let timeout = Duration::from_secs(5);
    let start = std::time::Instant::now();
    while !engine.is_shutdown_complete().await && start.elapsed() < timeout {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    assert!(
        engine.is_shutdown_complete().await,
        "Engine should shut down even with multiple signals"
    );
}

#[tokio::test]
async fn test_new_requests_before_shutdown() {
    let engine = ShutdownEngine::new();

    // Verify we can add requests before shutdown
    let mut added = 0;
    for _ in 0..10 {
        if engine.add_request(3).await.is_ok() {
            added += 1;
        }
    }

    assert!(added > 0, "Should be able to add requests before shutdown");

    // Now shutdown
    let _ = engine.shutdown().await;

    let timeout = Duration::from_secs(5);
    let start = std::time::Instant::now();
    while !engine.is_shutdown_complete().await && start.elapsed() < timeout {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    assert!(engine.is_shutdown_complete().await);
}

#[tokio::test]
async fn test_shutdown_no_pending_requests() {
    let engine = ShutdownEngine::new();

    // Shutdown immediately without any requests
    let start = std::time::Instant::now();
    let _ = engine.shutdown().await;

    let timeout = Duration::from_secs(3);
    while !engine.is_shutdown_complete().await && start.elapsed() < timeout {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    let elapsed = start.elapsed();
    assert!(
        engine.is_shutdown_complete().await,
        "Shutdown without requests should complete"
    );
    assert!(
        elapsed < Duration::from_secs(3),
        "Shutdown without requests took {:?}, expected under 3s",
        elapsed
    );
}

#[test]
fn test_engine_shutdown_synchronously() {
    let config = SchedulerConfig::default();
    let mut engine = Engine::with_config(IncrementModel, IncrementModel, config, 4, 1024);

    // Add some requests
    let (tx, _rx) = mpsc::channel(64);
    for i in 0..5 {
        engine.add_request(Request::new(i, vec![10, 20], 3), tx.clone());
    }

    // Process until all done
    let start = std::time::Instant::now();
    let timeout = Duration::from_secs(10);

    while start.elapsed() < timeout {
        if let Ok(results) = engine.step() {
            if results.is_empty() && !engine.has_pending() {
                break;
            }
        }
    }

    // Verify no pending work
    assert!(
        !engine.has_pending(),
        "Engine should have no pending work after processing"
    );
}

#[test]
fn test_drain_in_flight_requests() {
    let config = SchedulerConfig::default();
    let mut engine = Engine::with_config(IncrementModel, IncrementModel, config, 4, 1024);

    // Add requests
    let (tx, _rx) = mpsc::channel(64);
    let mut seq_ids = Vec::new();
    for i in 0..10 {
        let seq_id = engine.add_request(Request::new(i, vec![10, 20, 30], 5), tx.clone());
        seq_ids.push(seq_id);
    }

    // Process until completion (simulating graceful shutdown)
    let start = std::time::Instant::now();
    let timeout = Duration::from_secs(30);
    let mut completed_count = 0;

    while start.elapsed() < timeout && completed_count < seq_ids.len() {
        if let Ok(_results) = engine.step() {
            // Check for completions (not in running)
            completed_count = seq_ids
                .iter()
                .filter(|&&id| !engine.scheduler.running().iter().any(|s| s.id == id))
                .count();
        }

        if !engine.has_pending() {
            break;
        }
    }

    // All sequences should be completed or removed
    assert!(
        !engine.has_pending(),
        "All in-flight requests should be drained"
    );
}
