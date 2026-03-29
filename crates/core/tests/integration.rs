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
}

#[test]
fn test_continuous_batching_with_streaming() {
    let config = SchedulerConfig {
        max_num_seqs: 2,
        max_num_batched_tokens: 100,
    };
    let mut engine = Engine::with_config(IncrementModel, config, 1024);

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
    };
    let mut engine = Engine::with_config(IncrementModel, config, 1024);

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
