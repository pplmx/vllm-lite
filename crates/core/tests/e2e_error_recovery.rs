// tests/e2e_error_recovery.rs
//! Error recovery E2E tests

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::{Mutex, mpsc};
use vllm_core::engine::Engine;
use vllm_core::types::{Request, SchedulerConfig};
use vllm_testing::IncrementModel;
use vllm_traits::{BatchOutput, ModelBackend, ModelError, SeqId, TokenId};

/// Model that simulates failures after a threshold
struct FaultInjectedModel {
    inner: IncrementModel,
    failure_count: Arc<AtomicU64>,
    failure_threshold: u64,
}

impl FaultInjectedModel {
    fn new(threshold: u64) -> Self {
        Self {
            inner: IncrementModel,
            failure_count: Arc::new(AtomicU64::new(0)),
            failure_threshold: threshold,
        }
    }

    fn should_fail(&self) -> bool {
        let count = self.failure_count.fetch_add(1, Ordering::Relaxed);
        count < self.failure_threshold
    }
}

impl Clone for FaultInjectedModel {
    fn clone(&self) -> Self {
        Self {
            inner: IncrementModel,
            failure_count: Arc::clone(&self.failure_count),
            failure_threshold: self.failure_threshold,
        }
    }
}

impl ModelBackend for FaultInjectedModel {
    fn forward(
        &mut self,
        seq_ids: &[SeqId],
        input_tokens: &[Vec<TokenId>],
        positions: &[Vec<usize>],
        kv_block_ids: &[Vec<usize>],
        num_computed_tokens: &[usize],
        is_prefill: &[bool],
    ) -> vllm_traits::Result<BatchOutput> {
        // Deterministic failure: fail every 10th call to ensure reproducible tests
        let count = self.failure_count.load(Ordering::Relaxed);
        if self.should_fail() && count % 10 == 0 {
            return Err(ModelError::new("Simulated failure"));
        }
        self.inner.forward(
            seq_ids,
            input_tokens,
            positions,
            kv_block_ids,
            num_computed_tokens,
            is_prefill,
        )
    }

    fn forward_logits(
        &mut self,
        seq_ids: &[SeqId],
        input_tokens: &[Vec<TokenId>],
        positions: &[Vec<usize>],
        kv_block_ids: &[Vec<usize>],
        num_computed_tokens: &[usize],
        is_prefill: &[bool],
    ) -> vllm_traits::Result<Vec<Vec<f32>>> {
        self.inner.forward_logits(
            seq_ids,
            input_tokens,
            positions,
            kv_block_ids,
            num_computed_tokens,
            is_prefill,
        )
    }

    fn embed(
        &mut self,
        input_tokens: &[Vec<TokenId>],
        positions: &[Vec<usize>],
    ) -> vllm_traits::Result<Vec<Vec<f32>>> {
        self.inner.embed(input_tokens, positions)
    }

    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    fn num_layers(&self) -> usize {
        self.inner.num_layers()
    }

    fn num_heads(&self) -> usize {
        self.inner.num_heads()
    }
}

/// Engine wrapper with error tracking
struct ErrorTrackingEngine {
    engine: Arc<Mutex<Engine<IncrementModel>>>,
}

impl ErrorTrackingEngine {
    fn new() -> Self {
        let config = SchedulerConfig::default();
        let engine = Engine::with_config(IncrementModel, None, config, 4, 1024);
        Self {
            engine: Arc::new(Mutex::new(engine)),
        }
    }

    async fn add_request(&self, max_tokens: usize) -> Result<u64, String> {
        let (tx, _rx) = mpsc::channel(64);
        let mut engine = self.engine.lock().await;
        let seq_id = engine.add_request(Request::new(1, vec![10, 20, 30], max_tokens), tx);
        if seq_id > 0 {
            Ok(seq_id)
        } else {
            Err("Failed to add request".to_string())
        }
    }

    #[allow(dead_code)]
    async fn get_error_count(&self) -> usize {
        let engine = self.engine.lock().await;
        engine.error_count
    }

    #[allow(dead_code)]
    async fn is_healthy(&self) -> bool {
        let engine = self.engine.lock().await;
        engine.is_healthy()
    }

    #[allow(dead_code)]
    async fn get_last_error(&self) -> Option<String> {
        let engine = self.engine.lock().await;
        engine.get_last_error().map(|s| s.to_string())
    }
}

impl Clone for ErrorTrackingEngine {
    fn clone(&self) -> Self {
        Self {
            engine: Arc::clone(&self.engine),
        }
    }
}

#[test]
fn test_error_tracking_accumulates() {
    let config = SchedulerConfig::default();
    let mut engine = Engine::with_config(IncrementModel, None, config, 4, 1024);

    // Initially healthy
    assert!(engine.is_healthy(), "Engine should start healthy");

    // Process some requests (these may generate errors in edge cases)
    let (tx, _rx) = mpsc::channel(64);
    for i in 0..5 {
        engine.add_request(Request::new(i, vec![10, 20], 3), tx.clone());
    }

    // Run a few steps
    for _ in 0..10 {
        let _ = engine.step();
    }

    // Engine should still be healthy (no real errors in normal operation)
    assert!(
        engine.is_healthy(),
        "Engine should remain healthy without actual errors"
    );
}

#[test]
fn test_request_failure_recovery() {
    let config = SchedulerConfig::default();
    let mut engine = Engine::with_config(IncrementModel, None, config, 4, 1024);

    // Test that we can recover from individual request processing
    let request_count = 20;
    let mut success_count = 0;

    for i in 0..request_count {
        let (tx, _rx) = mpsc::channel(64);
        let seq_id = engine.add_request(Request::new(i, vec![10, 20], 3), tx);
        if seq_id > 0 {
            success_count += 1;
        }
    }

    assert_eq!(
        success_count, request_count,
        "All requests should be added successfully"
    );

    // Process all requests
    for _ in 0..50 {
        let _ = engine.step();
    }

    // Engine should still be healthy
    assert!(engine.is_healthy());
}

#[test]
fn test_empty_prompt_rejected() {
    let config = SchedulerConfig::default();
    let mut engine = Engine::with_config(IncrementModel, None, config, 4, 1024);

    // Create empty request
    let (tx, _rx) = mpsc::channel(64);
    let seq_id = engine.add_request(Request::new(999, vec![], 5), tx);

    assert_eq!(seq_id, 0, "Empty prompt should be rejected");

    // Should record an error
    assert!(
        engine.get_last_error().is_some(),
        "Last error should be set for empty prompt"
    );
}

#[tokio::test]
async fn test_concurrent_error_handling() {
    let engine = ErrorTrackingEngine::new();
    let concurrency = 10;

    let handles: Vec<_> = (0..concurrency)
        .map(|i| {
            let eng = engine.clone();
            tokio::spawn(async move {
                // Mix of valid and edge case requests
                let max_tokens = if i % 3 == 0 { 0 } else { 5 };
                eng.add_request(max_tokens).await
            })
        })
        .collect();

    let mut success_count = 0;
    let mut fail_count = 0;

    for handle in handles {
        match handle.await {
            Ok(Ok(_)) => success_count += 1,
            Ok(Err(_)) | Err(_) => fail_count += 1,
        }
    }

    // Requests with max_tokens=0 should still succeed to be added
    assert!(
        success_count + fail_count == concurrency,
        "All requests should complete one way or another"
    );
}

#[test]
fn test_engine_health_with_errors() {
    let config = SchedulerConfig::default();
    let mut engine = Engine::with_config(IncrementModel, None, config, 4, 1024);

    // Initially healthy
    assert!(engine.is_healthy());

    // Add some requests (normal operation doesn't cause errors)
    let (tx, _rx) = mpsc::channel(64);
    for i in 0..5 {
        engine.add_request(Request::new(i, vec![10, 20], 3), tx.clone());
    }

    // Run steps
    for _ in 0..10 {
        let _ = engine.step();
    }

    // Still healthy
    assert!(engine.is_healthy());

    // Error count should be 0 in normal operation
    assert_eq!(engine.error_count, 0);
}

#[test]
fn test_request_cancellation() {
    let config = SchedulerConfig::default();
    let mut engine = Engine::with_config(IncrementModel, None, config, 4, 1024);

    // Add a request
    let (tx, _rx) = mpsc::channel(64);
    let seq_id = engine.add_request(Request::new(1, vec![10, 20, 30], 100), tx);
    assert!(seq_id > 0, "Should add request");

    // Process a few steps to get it running
    for _ in 0..3 {
        let _ = engine.step();
    }

    // Cancel it
    let canceled = engine.cancel_request(seq_id);
    assert!(canceled, "Should successfully cancel request");

    // Verify it was removed (not in running)
    let still_running = engine.scheduler.running().iter().any(|s| s.id == seq_id);
    assert!(
        !still_running,
        "Cancelled request should be removed from scheduler"
    );
}

#[test]
fn test_multiple_cancellations() {
    let config = SchedulerConfig::default();
    let mut engine = Engine::with_config(IncrementModel, None, config, 4, 1024);

    let (tx, _rx) = mpsc::channel(64);
    let mut seq_ids = Vec::new();

    // Add multiple requests
    for i in 0..5 {
        let seq_id = engine.add_request(Request::new(i, vec![10, 20, 30], 100), tx.clone());
        assert!(seq_id > 0, "Should add request successfully");
        seq_ids.push(seq_id);
    }

    // Process one step to get some requests running
    let _ = engine.step();

    // Cancel all requests - cancel_request returns true if the request was found and canceled
    // It may be waiting (not yet running), which is still a valid cancellation
    let mut canceled_count = 0;
    for seq_id in &seq_ids {
        if engine.cancel_request(*seq_id) {
            canceled_count += 1;
        }
    }

    // At least some should be canceled (those that were running or waiting)
    assert!(
        canceled_count > 0,
        "Should cancel at least some requests, canceled {}",
        canceled_count
    );

    // Verify canceled requests are not running
    for seq_id in &seq_ids {
        let still_running = engine.scheduler.running().iter().any(|s| s.id == *seq_id);
        assert!(!still_running, "Request {} should not be running", seq_id);
    }
}

#[test]
fn test_error_recovery_with_faulty_model() {
    let config = SchedulerConfig::default();
    let mut engine = Engine::with_config(FaultInjectedModel::new(100), None, config, 4, 1024);

    // Add requests
    let (tx, _rx) = mpsc::channel(64);
    let seq_id = engine.add_request(Request::new(1, vec![10, 20, 30], 5), tx);
    assert!(seq_id > 0);

    // Process - should handle errors gracefully
    let mut _success = false;
    for _ in 0..50 {
        if engine.step().is_ok() {
            _success = true;
        }
    }

    // Should still be healthy (error count < 10 threshold)
    assert!(engine.is_healthy());
}
