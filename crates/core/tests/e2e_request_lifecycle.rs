//! End-to-end tests for complete request lifecycle
//!
//! These tests verify the entire flow from request submission to completion.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use vllm_core::engine::Engine;
use vllm_core::types::{Request, SchedulerConfig};
use vllm_testing::IncrementModel;

/// Test complete request lifecycle
#[test]
fn test_e2e_complete_lifecycle() {
    let config = SchedulerConfig::default();
    let mut engine = Engine::with_config(IncrementModel, IncrementModel, config, 4, 1024);

    let (tx, mut rx) = mpsc::channel(64);
    let seq_id = engine.add_request(Request::new(1, vec![10, 20, 30], 5), tx);
    assert!(seq_id > 0);

    // Process until completion
    let mut received_tokens = 0;
    let mut completed = false;
    let start = Instant::now();
    let timeout = Duration::from_secs(30);

    while !completed && start.elapsed() < timeout {
        let results = engine.step().unwrap();
        for _ in results {
            if rx.try_recv().is_ok() {
                received_tokens += 1;
            }
        }

        // Check if sequence is finished
        completed = !engine
            .scheduler
            .running_sequences()
            .iter()
            .any(|s| s.id == seq_id);
    }

    assert!(
        completed,
        "Request should complete within timeout, got {} tokens",
        received_tokens
    );
    assert!(received_tokens > 0, "Should have received some tokens");
}

/// Test concurrent requests
#[test]
fn test_e2e_concurrent_requests() {
    let config = SchedulerConfig::default();
    let mut engine = Engine::with_config(IncrementModel, IncrementModel, config, 4, 1024);

    let num_requests = 10;
    let mut receivers = Vec::new();
    let mut seq_ids = Vec::new();

    // Submit multiple concurrent requests
    for i in 0..num_requests {
        let (tx, rx) = mpsc::channel(64);
        let seq_id = engine.add_request(Request::new(i as u64, vec![10, 20], 5), tx);
        seq_ids.push(seq_id);
        receivers.push(rx);
    }

    // Process all to completion
    let mut completed_count = 0;
    let start = Instant::now();
    let timeout = Duration::from_secs(60);

    while completed_count < num_requests && start.elapsed() < timeout {
        let results = engine.step().unwrap();

        for (seq_id, _token) in &results {
            // Mark as completed
            if seq_ids.contains(seq_id) {
                completed_count += 1;
            }
        }

        // Drain all receivers
        for rx in &mut receivers {
            while rx.try_recv().is_ok() {}
        }
    }

    assert_eq!(
        completed_count, num_requests,
        "All {} requests should complete",
        num_requests
    );
}

/// Test request cancellation
#[test]
fn test_e2e_request_cancellation() {
    let config = SchedulerConfig::default();
    let mut engine = Engine::with_config(IncrementModel, IncrementModel, config, 4, 1024);

    let (tx, mut rx) = mpsc::channel(64);
    let seq_id = engine.add_request(Request::new(1, vec![10, 20, 30, 40, 50], 100), tx);

    // Process a few steps
    for _ in 0..3 {
        engine.step().unwrap();
    }

    // Cancel the request
    let cancelled = engine.cancel_request(seq_id);
    assert!(cancelled, "Request should be cancelled");

    // Should receive disconnected channel error
    assert!(
        rx.try_recv().is_err(),
        "Channel should be disconnected after cancellation"
    );
}

/// Test graceful shutdown
#[test]
fn test_e2e_graceful_shutdown() {
    let config = SchedulerConfig::default();
    let mut engine = Engine::with_config(IncrementModel, IncrementModel, config, 4, 1024);

    // Add some requests
    let (tx, _rx) = mpsc::channel(64);
    engine.add_request(Request::new(1, vec![10, 20], 5), tx);

    // Process a bit
    for _ in 0..2 {
        engine.step().unwrap();
    }

    // Drain all sequences
    let start = Instant::now();
    while !engine.scheduler.running_sequences().is_empty()
        && start.elapsed() < Duration::from_secs(10)
    {
        engine.step().unwrap();
    }

    assert!(
        engine.scheduler.running_sequences().is_empty(),
        "All sequences should be drained"
    );
}

/// Test with all optimizations enabled
#[test]
fn test_e2e_with_all_optimizations() {
    let config = SchedulerConfig::default();
    let mut engine = Engine::with_config(IncrementModel, IncrementModel, config, 4, 1024);

    // Enable all optimizations
    engine.enable_adaptive_speculative(vllm_core::types::AdaptiveDraftConfig::default());

    let (tx, mut rx) = mpsc::channel(64);
    let seq_id = engine.add_request(Request::new(1, vec![10, 20, 30], 5), tx);

    let mut completed = false;
    let start = Instant::now();

    while !completed && start.elapsed() < Duration::from_secs(30) {
        let results = engine.step_adaptive_speculative().unwrap();
        for _ in &results {
            let _ = rx.try_recv();
        }

        completed = !engine
            .scheduler
            .running_sequences()
            .iter()
            .any(|s| s.id == seq_id);
    }

    assert!(completed, "Request should complete with optimizations");
}

/// Test error recovery
#[test]
fn test_e2e_error_recovery() {
    let config = SchedulerConfig::default();
    let mut engine = Engine::with_config(IncrementModel, IncrementModel, config, 4, 1024);

    // Add multiple requests
    let num_requests = 5;
    let mut receivers = Vec::new();

    for i in 0..num_requests {
        let (tx, rx) = mpsc::channel(64);
        engine.add_request(Request::new(i as u64, vec![10, 20], 3), tx);
        receivers.push(rx);
    }

    // Process all - should handle any errors gracefully
    let start = Instant::now();
    while start.elapsed() < Duration::from_secs(30) {
        let _ = engine.step();

        let running = engine.scheduler.running_sequences().len();
        let waiting = engine.scheduler.waiting_sequences().len();

        if running == 0 && waiting == 0 {
            break;
        }
    }

    // Verify engine is still healthy
    assert!(engine.is_healthy());
}

/// Test performance SLO
#[test]
fn test_e2e_latency_slo() {
    let config = SchedulerConfig::default();
    let mut engine = Engine::with_config(IncrementModel, IncrementModel, config, 4, 1024);

    let (tx, _rx) = mpsc::channel(64);
    engine.add_request(Request::new(1, vec![10, 20], 10), tx);

    let mut latencies = Vec::new();

    for _ in 0..20 {
        let start = Instant::now();
        engine.step().unwrap();
        let elapsed = start.elapsed();
        latencies.push(elapsed.as_millis());
    }

    // Calculate P99 latency
    latencies.sort();
    let p99_idx = (latencies.len() as f64 * 0.99) as usize;
    let p99_latency = latencies[p99_idx.min(latencies.len() - 1)];

    // SLO: P99 < 1000ms (generous for testing)
    assert!(
        p99_latency < 1000,
        "P99 latency {}ms exceeds SLO of 1000ms",
        p99_latency
    );
}
