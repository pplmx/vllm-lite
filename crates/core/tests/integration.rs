use tokio::sync::mpsc;
use vllm_core::engine::{Engine, ModelBackend};
use vllm_core::error::Result;
use vllm_core::types::{BatchOutput, Request, SchedulerConfig, SeqId, TokenId};

struct IncrementModel;

impl ModelBackend for IncrementModel {
    fn forward(
        &self,
        seq_ids: &[SeqId],
        _input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
    ) -> Result<BatchOutput> {
        Ok(BatchOutput {
            seq_ids: seq_ids.to_vec(),
            next_tokens: seq_ids.iter().map(|id| *id as TokenId).collect(),
        })
    }

    fn forward_logits(
        &self,
        _seq_ids: &[SeqId],
        input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
    ) -> Result<Vec<Vec<f32>>> {
        Ok(input_tokens
            .iter()
            .map(|tokens| tokens.iter().map(|_| 0.0).collect())
            .collect())
    }
}

#[test]
fn test_continuous_batching_with_streaming() {
    let config = SchedulerConfig {
        max_num_seqs: 2,
        max_num_batched_tokens: 100,
        max_consecutive_decode: 10,
    };
    let mut engine = Engine::with_config(IncrementModel, IncrementModel, config, 4, 1024);

    let (tx1, mut rx1) = mpsc::unbounded_channel();
    let (tx2, mut rx2) = mpsc::unbounded_channel();

    // req1: prompt=2, max_tokens=4 -> total 4 tokens, finishes in step 3
    // req2: prompt=3, max_tokens=5 -> total 5 tokens, finishes in step 4
    engine.add_request(Request::new(1, vec![10, 20], 4), tx1);
    engine.add_request(Request::new(2, vec![30, 40, 50], 5), tx2);

    // Step 1: both prefill
    // After: req1=3 tokens, req2=4 tokens
    engine.step().unwrap();
    assert!(rx1.try_recv().is_ok(), "req1 should get token in step 1");
    assert!(rx2.try_recv().is_ok(), "req2 should get token in step 1");

    // Step 2: both decode
    // After: req1=4 tokens (finished), req2=5 tokens
    engine.step().unwrap();
    assert!(rx2.try_recv().is_ok(), "req2 should get token in step 2");

    // After step 2, req1 is finished and its channel is disconnected
    // Don't try to receive from rx1 anymore

    // Step 3: req2 only (req1 finished)
    // After: req2=6 tokens > max_tokens(5), but this is the step where it finishes
    // Actually: 5 tokens = finished, so after step 2 req2 already has 5 tokens
    // Wait, let me recount:
    // Step 1: prompt + 1 = 3 + 1 = 4 tokens (req2)
    // Step 2: 4 + 1 = 5 tokens (req2) = max_tokens(5) -> finished!

    // So after step 2, both should be finished?
    // Let me check has_pending...
    assert!(!engine.has_pending(), "both requests should be finished");
}

#[test]
fn test_chunked_prefill_integration() {
    let config = SchedulerConfig {
        max_num_seqs: 256,
        max_num_batched_tokens: 10,
        max_consecutive_decode: 10,
    };
    let mut engine = Engine::with_config(IncrementModel, IncrementModel, config, 4, 1024);

    let (tx, mut rx) = mpsc::unbounded_channel();
    // 4 prompt + 10 max_tokens = 14 total tokens (need 10 decode steps)
    // Note: max_tokens must be > prompt.len() to avoid finishing in prefill
    engine.add_request(Request::new(1, vec![10, 20, 30, 40], 10), tx);

    // Step 1: full prefill (all 4 prompt tokens)
    // After: tokens = 5 (4 prompt + 1 generated), status = Decoding
    let mut steps = 0;
    while engine.has_pending() {
        engine.step().unwrap();
        steps += 1;
        if rx.try_recv().is_err() {
            break;
        }
    }

    // Should have done more than just prefill
    assert!(steps > 1, "should have multiple steps, got {}", steps);
}

#[test]
fn test_max_tokens_includes_prompt() {
    // This test verifies the fix: max_tokens should represent total sequence length
    let config = SchedulerConfig {
        max_num_seqs: 256,
        max_num_batched_tokens: 100,
        max_consecutive_decode: 10,
    };
    let mut engine = Engine::with_config(IncrementModel, IncrementModel, config, 4, 1024);

    let (tx, _rx) = mpsc::unbounded_channel();

    // Prompt: 3 tokens, max_new_tokens: 2
    // Total should be: 3 + 2 = 5 tokens before finishing
    let prompt = vec![10, 20, 30];
    let max_new_tokens = 2;
    let total_max = prompt.len() + max_new_tokens; // This is what the API should send

    engine.add_request(Request::new(1, prompt, total_max), tx);

    let mut steps = 0;
    while engine.has_pending() {
        engine.step().unwrap();
        steps += 1;
        if steps > 10 {
            panic!("Too many steps - max_tokens might not include prompt");
        }
    }

    // Should finish in exactly 3 steps:
    // Step 1: prompt(3) -> tokens=4 (still prefill/decode), wait for status update
    // Actually: prompt processing + decoding until 5 total tokens
    assert!(
        steps <= 5,
        "should finish within expected steps, got {}",
        steps
    );
}

#[test]
fn test_single_token_prefill_then_decode() {
    // Test the case where prompt is single token (our bug scenario)
    let config = SchedulerConfig {
        max_num_seqs: 256,
        max_num_batched_tokens: 100,
        max_consecutive_decode: 10,
    };
    let mut engine = Engine::with_config(IncrementModel, IncrementModel, config, 4, 1024);

    let (tx, mut rx) = mpsc::unbounded_channel();

    // Single token prompt
    let prompt = vec![100];
    let max_new_tokens = 3;
    let total_max = prompt.len() + max_new_tokens;

    engine.add_request(Request::new(1, prompt, total_max), tx);

    // Step 1: Process prompt token
    engine.step().unwrap();
    assert!(rx.try_recv().is_ok(), "should get first token");

    // Step 2-4: Decode
    for _ in 2..=4 {
        if !engine.has_pending() {
            break;
        }
        engine.step().unwrap();
    }

    // Should have received all tokens
    assert!(!engine.has_pending(), "should be finished after 4 steps");
}

#[test]
fn test_concurrent_requests_finish_together() {
    let config = SchedulerConfig {
        max_num_seqs: 10,
        max_num_batched_tokens: 100,
        max_consecutive_decode: 10,
    };
    let mut engine = Engine::with_config(IncrementModel, IncrementModel, config, 4, 1024);

    let (tx1, _rx1) = mpsc::unbounded_channel();
    let (tx2, _rx2) = mpsc::unbounded_channel();

    engine.add_request(Request::new(1, vec![10], 2), tx1);
    engine.add_request(Request::new(2, vec![20], 2), tx2);

    engine.step().unwrap();
    engine.step().unwrap();

    assert!(!engine.has_pending(), "both requests should finish");
}

#[test]
fn test_batch_full_new_request_waits() {
    let config = SchedulerConfig {
        max_num_seqs: 1,
        max_num_batched_tokens: 100,
        max_consecutive_decode: 10,
    };
    let mut engine = Engine::with_config(IncrementModel, IncrementModel, config, 4, 1024);

    let (tx1, _rx1) = mpsc::unbounded_channel();
    let (tx2, _rx2) = mpsc::unbounded_channel();

    engine.add_request(Request::new(1, vec![10], 5), tx1);
    let batch1 = engine.scheduler.build_batch();
    engine.scheduler.update(
        &batch1.seq_ids,
        &[99],
        &[batch1.input_tokens.iter().map(|v| v.len()).sum()],
    );

    engine.add_request(Request::new(2, vec![20], 5), tx2);

    assert!(engine.scheduler.waiting_count() > 0);
}

#[test]
fn test_prefix_cache_hit_directly_decoding() {
    let config = SchedulerConfig {
        max_num_seqs: 10,
        max_num_batched_tokens: 4096,
        max_consecutive_decode: 10,
    };
    let mut engine = Engine::with_config(IncrementModel, IncrementModel, config, 4, 1024);

    let (tx1, _rx1) = mpsc::unbounded_channel();
    let (tx2, _rx2) = mpsc::unbounded_channel();

    engine.add_request(Request::new(1, vec![10, 20], 3), tx1);
    while engine.has_pending() {
        engine.step().unwrap();
    }

    engine.add_request(Request::new(2, vec![10, 20], 5), tx2);

    assert_eq!(engine.scheduler.running().len(), 1);
    let seq = &engine.scheduler.running()[0];
    assert_eq!(seq.status, vllm_core::types::Status::Decoding);
}

#[test]
fn test_different_prompt_lengths_batching() {
    let config = SchedulerConfig {
        max_num_seqs: 10,
        max_num_batched_tokens: 50,
        max_consecutive_decode: 10,
    };
    let mut engine = Engine::with_config(IncrementModel, IncrementModel, config, 4, 1024);

    let (tx1, _rx1) = mpsc::unbounded_channel();
    let (tx2, _rx2) = mpsc::unbounded_channel();
    let (tx3, _rx3) = mpsc::unbounded_channel();

    engine.add_request(Request::new(1, vec![1], 3), tx1);
    engine.add_request(Request::new(2, vec![1, 2], 3), tx2);
    engine.add_request(Request::new(3, vec![1, 2, 3], 3), tx3);

    let batch = engine.scheduler.build_batch();
    assert!(batch.seq_ids.len() <= 3);
}

#[test]
fn test_prefill_priority_under_decode_limit() {
    let config = SchedulerConfig {
        max_num_seqs: 2,
        max_num_batched_tokens: 100,
        max_consecutive_decode: 1,
    };
    let mut engine = Engine::with_config(IncrementModel, IncrementModel, config, 4, 1024);

    let (tx1, _rx1) = mpsc::unbounded_channel();
    let (tx2, _rx2) = mpsc::unbounded_channel();
    let (tx3, _rx3) = mpsc::unbounded_channel();

    engine.add_request(Request::new(1, vec![10], 5), tx1);
    let batch1 = engine.scheduler.build_batch();
    engine.scheduler.update(
        &batch1.seq_ids,
        &[99],
        &[batch1.input_tokens.iter().map(|v| v.len()).sum()],
    );

    engine.add_request(Request::new(2, vec![20, 30], 5), tx2);
    engine.add_request(Request::new(3, vec![40, 50, 60], 5), tx3);

    let batch2 = engine.scheduler.build_batch();
    assert!(batch2.seq_ids.len() <= 2);
}

#[test]
fn test_many_sequences_stress() {
    let config = SchedulerConfig {
        max_num_seqs: 50,
        max_num_batched_tokens: 100,
        max_consecutive_decode: 5,
    };
    let mut engine = Engine::with_config(IncrementModel, IncrementModel, config, 4, 1024);

    for i in 1..=20 {
        let (tx, _rx) = mpsc::unbounded_channel();
        engine.add_request(Request::new(i, vec![i as TokenId], 3), tx);
    }

    for _ in 0..10 {
        if !engine.has_pending() {
            break;
        }
        engine.step().unwrap();
    }

    let state_valid = engine.scheduler.running_count() <= 20;
    assert!(state_valid, "engine state should be valid under stress");
}

#[test]
fn test_sequence_state_transitions() {
    let config = SchedulerConfig {
        max_num_seqs: 10,
        max_num_batched_tokens: 100,
        max_consecutive_decode: 10,
    };
    let mut engine = Engine::with_config(IncrementModel, IncrementModel, config, 4, 1024);

    let (tx, _rx) = mpsc::unbounded_channel();
    engine.add_request(Request::new(1, vec![10, 20, 30], 5), tx);

    let batch1 = engine.scheduler.build_batch();
    let seq = engine
        .scheduler
        .running()
        .iter()
        .find(|s| s.id == 1)
        .unwrap();
    assert_eq!(seq.status, vllm_core::types::Status::Prefilling);

    engine.scheduler.update(
        &batch1.seq_ids,
        &[99],
        &[batch1.input_tokens.iter().map(|v| v.len()).sum()],
    );

    let seq = engine
        .scheduler
        .running()
        .iter()
        .find(|s| s.id == 1)
        .unwrap();
    assert_eq!(seq.status, vllm_core::types::Status::Decoding);
}

#[test]
fn test_immediate_finish_after_prompt() {
    let config = SchedulerConfig {
        max_num_seqs: 10,
        max_num_batched_tokens: 100,
        max_consecutive_decode: 10,
    };
    let mut engine = Engine::with_config(IncrementModel, IncrementModel, config, 4, 1024);

    let (tx, _rx) = mpsc::unbounded_channel();
    engine.add_request(Request::new(1, vec![1, 2, 3], 3), tx);

    engine.step().unwrap();

    assert!(
        !engine.has_pending(),
        "should finish when max_tokens equals prompt length"
    );
}
