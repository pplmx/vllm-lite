// tests/e2e_lifecycle.rs
//! Complete request lifecycle E2E tests

use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use vllm_core::engine::Engine;
use vllm_core::types::{Request, SchedulerConfig};
use vllm_testing::IncrementModel;

/// Output from processing a request
#[derive(Debug)]
struct RequestOutput {
    tokens: Vec<u32>,
    finish_reason: Option<String>,
}

/// Engine wrapper for testing
struct TestEngine {
    engine: Engine<IncrementModel>,
}

impl TestEngine {
    fn new() -> Self {
        let config = SchedulerConfig::default();
        let engine = Engine::with_config(IncrementModel, IncrementModel, config, 4, 1024);
        Self { engine }
    }

    fn add_request(&mut self, max_tokens: usize) -> u64 {
        let prompt: Vec<u32> = (1..=10).collect();
        let (tx, _rx) = mpsc::channel(64);
        self.engine
            .add_request(Request::new(1, prompt, max_tokens), tx)
    }

    fn process_until_complete(&mut self, seq_id: u64, max_tokens: usize) -> RequestOutput {
        let mut tokens = Vec::new();
        let mut finished = false;
        let max_iterations = max_tokens * 2; // Safety limit

        for _ in 0..max_iterations {
            if let Ok(results) = self.engine.step() {
                for (result_seq_id, token) in results {
                    if result_seq_id == seq_id {
                        tokens.push(token);
                    }
                }
            }

            // Check if sequence is finished (not in running)
            let still_running = self
                .engine
                .scheduler
                .running()
                .iter()
                .any(|s| s.id == seq_id);
            if !still_running && !self.engine.has_pending() {
                finished = true;
                break;
            }
        }

        RequestOutput {
            tokens,
            finish_reason: if finished {
                Some("stop".to_string())
            } else {
                None
            },
        }
    }
}

#[test]
fn test_complete_request_lifecycle() {
    let mut engine = TestEngine::new();
    let seq_id = engine.add_request(5);

    assert!(seq_id > 0, "Should get valid sequence ID");

    let output = engine.process_until_complete(seq_id, 5);

    assert!(!output.tokens.is_empty(), "Should have output tokens");
    assert_eq!(
        output.finish_reason,
        Some("stop".to_string()),
        "Should have finish reason"
    );
}

#[test]
fn test_request_with_different_token_counts() {
    for token_count in [1, 3, 5, 10] {
        let mut engine = TestEngine::new();
        let seq_id = engine.add_request(token_count);

        let output = engine.process_until_complete(seq_id, token_count + 5);

        assert!(
            !output.tokens.is_empty(),
            "Failed for {} tokens - no output",
            token_count
        );
        assert!(
            output.finish_reason.is_some(),
            "Failed for {} tokens - not finished",
            token_count
        );
    }
}

#[test]
fn test_multiple_requests_lifecycle() {
    let config = SchedulerConfig::default();
    let mut engine = Engine::with_config(IncrementModel, IncrementModel, config, 4, 1024);

    let _seq_ids: Vec<u64> = (0..5)
        .map(|i| {
            let (tx, _rx) = mpsc::channel(64);
            engine.add_request(Request::new(i, vec![10, 20, 30], 3), tx)
        })
        .collect();

    // Process all requests
    let max_iterations = 100;

    for _ in 0..max_iterations {
        let _ = engine.step();

        // Check if all sequences are done (not running and not waiting)
        if engine.scheduler.running().is_empty() && !engine.has_pending() {
            break;
        }
    }

    // Verify all sequences are not running (completed)
    assert!(
        engine.scheduler.running().is_empty(),
        "All sequences should be completed, but {} still running",
        engine.scheduler.running().len()
    );
}

#[test]
fn test_empty_request_rejected() {
    let config = SchedulerConfig::default();
    let mut engine = Engine::with_config(IncrementModel, IncrementModel, config, 4, 1024);
    let (tx, _rx) = mpsc::channel(64);

    // Empty prompt should return 0
    let seq_id = engine.add_request(Request::new(999, vec![], 5), tx);

    assert_eq!(seq_id, 0, "Empty prompt should be rejected with seq_id 0");
}

#[test]
fn test_request_cancellation_lifecycle() {
    let config = SchedulerConfig::default();
    let mut engine = Engine::with_config(IncrementModel, IncrementModel, config, 4, 1024);

    let (tx, _rx) = mpsc::channel(64);
    let seq_id = engine.add_request(Request::new(1, vec![10, 20, 30], 100), tx);

    assert!(seq_id > 0, "Should get valid sequence ID");

    // Process a few steps
    for _ in 0..3 {
        let _ = engine.step();
    }

    // Cancel the request
    let cancelled = engine.cancel_request(seq_id);
    assert!(cancelled, "Request should be cancelled");

    // Verify it's not running
    let still_running = engine.scheduler.running().iter().any(|s| s.id == seq_id);
    assert!(!still_running, "Cancelled sequence should not be running");
}

#[test]
fn test_streaming_tokens() {
    let config = SchedulerConfig::default();
    let mut engine = Engine::with_config(IncrementModel, IncrementModel, config, 4, 1024);

    let (tx, mut rx) = mpsc::channel(64);
    let seq_id = engine.add_request(Request::new(1, vec![10, 20, 30], 5), tx);

    let mut received_tokens = 0;
    let start = Instant::now();
    let timeout = Duration::from_secs(30);

    while start.elapsed() < timeout {
        let results = engine.step().unwrap();
        for _ in results {
            if rx.try_recv().is_ok() {
                received_tokens += 1;
            }
        }

        let completed = !engine.scheduler.running().iter().any(|s| s.id == seq_id);
        if completed {
            break;
        }
    }

    assert!(
        received_tokens > 0,
        "Should receive streaming tokens, got {}",
        received_tokens
    );
}
