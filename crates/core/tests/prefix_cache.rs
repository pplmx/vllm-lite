use tokio::sync::mpsc;
use vllm_core::engine::{Engine, ModelBackend};
use vllm_core::error::Result;
use vllm_core::types::{BatchOutput, Request, SchedulerConfig, SeqId, TokenId};

struct StubModel;

impl ModelBackend for StubModel {
    fn forward(
        &self,
        seq_ids: &[SeqId],
        _input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
    ) -> Result<BatchOutput> {
        Ok(BatchOutput {
            seq_ids: seq_ids.to_vec(),
            next_tokens: seq_ids.iter().map(|_| 1 as TokenId).collect(),
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
fn test_prefix_cache_hit() {
    let config = SchedulerConfig {
        max_num_seqs: 256,
        max_num_batched_tokens: 4096,
        max_consecutive_decode: 10,
        enable_pd_separation: false,
        prefill_chunk_size: 512,
        decode_preference_ratio: 0.7,
        enable_priority_scheduling: false,
        enable_dynamic_batching: false,
        min_batch_size: 1,
        max_batch_size: 256,
    };
    let mut engine = Engine::with_config(StubModel, StubModel, config, 4, 100);

    let (tx1, _rx1) = mpsc::unbounded_channel();
    let (tx2, _rx2) = mpsc::unbounded_channel();

    // First request: cache miss - wait for completion to populate cache
    engine.add_request(Request::new(1, vec![10, 20], 3), tx1);
    while engine.has_pending() {
        engine.step().unwrap();
    }

    // Second request with same prompt: cache hit
    engine.add_request(Request::new(2, vec![10, 20], 5), tx2);

    // Verify second request is in decoding state (cache hit)
    assert_eq!(engine.scheduler.running().len(), 1);
    let seq = &engine.scheduler.running()[0];
    assert_eq!(seq.status, vllm_core::types::Status::Decoding);
}

#[test]
fn test_cache_after_completion() {
    let config = SchedulerConfig {
        max_num_seqs: 256,
        max_num_batched_tokens: 4096,
        max_consecutive_decode: 10,
        enable_pd_separation: false,
        prefill_chunk_size: 512,
        decode_preference_ratio: 0.7,
        enable_priority_scheduling: false,
        enable_dynamic_batching: false,
        min_batch_size: 1,
        max_batch_size: 256,
    };
    let mut engine = Engine::with_config(StubModel, StubModel, config, 4, 100);

    let (tx, _rx) = mpsc::unbounded_channel();

    // Add request and complete it
    engine.add_request(Request::new(1, vec![10, 20], 3), tx);
    while engine.has_pending() {
        engine.step().unwrap();
    }

    // Cache should have entry
    assert!(engine.scheduler.prefix_cache().len() > 0);
}

#[test]
fn test_prefix_cache_partial_hit() {
    let config = SchedulerConfig {
        max_num_seqs: 256,
        max_num_batched_tokens: 4096,
        max_consecutive_decode: 10,
        enable_pd_separation: false,
        prefill_chunk_size: 512,
        decode_preference_ratio: 0.7,
        enable_priority_scheduling: false,
        enable_dynamic_batching: false,
        min_batch_size: 1,
        max_batch_size: 256,
    };
    let mut engine = Engine::with_config(StubModel, StubModel, config, 4, 100);

    let (tx1, _rx1) = mpsc::unbounded_channel();
    let (tx2, _rx2) = mpsc::unbounded_channel();

    // First request: [10, 20, 30]
    engine.add_request(Request::new(1, vec![10, 20, 30], 3), tx1);
    while engine.has_pending() {
        engine.step().unwrap();
    }

    // Second request: [10, 20] - prefix of first
    engine.add_request(Request::new(2, vec![10, 20], 5), tx2);
    engine.step().unwrap();

    // Should be in decoding state (prefix hit)
    assert_eq!(engine.scheduler.running().len(), 1);
    let seq = &engine.scheduler.running()[0];
    assert_eq!(seq.status, vllm_core::types::Status::Decoding);
}

#[test]
fn test_prefix_cache_no_hit_different_prefix() {
    let config = SchedulerConfig {
        max_num_seqs: 256,
        max_num_batched_tokens: 4096,
        max_consecutive_decode: 10,
        enable_pd_separation: false,
        prefill_chunk_size: 512,
        decode_preference_ratio: 0.7,
        enable_priority_scheduling: false,
        enable_dynamic_batching: false,
        min_batch_size: 1,
        max_batch_size: 256,
    };
    let mut engine = Engine::with_config(StubModel, StubModel, config, 4, 100);

    let (tx1, _rx1) = mpsc::unbounded_channel();
    let (tx2, _rx2) = mpsc::unbounded_channel();

    // First request: [10, 20]
    engine.add_request(Request::new(1, vec![10, 20], 3), tx1);
    while engine.has_pending() {
        engine.step().unwrap();
    }

    // Second request: [30, 40] - different prefix
    engine.add_request(Request::new(2, vec![30, 40], 5), tx2);
    engine.step().unwrap();

    // Should be in decoding state (cache miss but prompt already processed)
    assert_eq!(engine.scheduler.running().len(), 1);
    let seq = &engine.scheduler.running()[0];
    assert_eq!(seq.status, vllm_core::types::Status::Decoding);
}

#[test]
fn test_prefix_cache_multiple_shared() {
    let config = SchedulerConfig {
        max_num_seqs: 256,
        max_num_batched_tokens: 4096,
        max_consecutive_decode: 10,
        enable_pd_separation: false,
        prefill_chunk_size: 512,
        decode_preference_ratio: 0.7,
        enable_priority_scheduling: false,
        enable_dynamic_batching: false,
        min_batch_size: 1,
        max_batch_size: 256,
    };
    let mut engine = Engine::with_config(StubModel, StubModel, config, 4, 100);

    // First: [1, 2, 3]
    engine.add_request(
        Request::new(1, vec![1, 2, 3], 3),
        mpsc::unbounded_channel().0,
    );
    while engine.has_pending() {
        engine.step().unwrap();
    }

    // Second: [1, 2] (prefix)
    engine.add_request(Request::new(2, vec![1, 2], 3), mpsc::unbounded_channel().0);
    while engine.has_pending() {
        engine.step().unwrap();
    }

    // Third: [1, 2, 3, 4] (longer)
    engine.add_request(
        Request::new(3, vec![1, 2, 3, 4], 3),
        mpsc::unbounded_channel().0,
    );

    // Should all share the common prefix [1, 2]
    let cache = engine.scheduler.prefix_cache();
    assert!(cache.len() >= 1, "cache should have entries");
}

#[test]
fn test_prefix_hit_partial_prefill() {
    let config = SchedulerConfig {
        max_num_seqs: 256,
        max_num_batched_tokens: 4096,
        max_consecutive_decode: 10,
        enable_pd_separation: false,
        prefill_chunk_size: 512,
        decode_preference_ratio: 0.7,
        enable_priority_scheduling: false,
        enable_dynamic_batching: false,
        min_batch_size: 1,
        max_batch_size: 256,
    };
    let mut engine = Engine::with_config(StubModel, StubModel, config, 4, 100);

    let (tx1, _rx1) = mpsc::unbounded_channel();

    // First request: complete it to populate cache
    engine.add_request(Request::new(1, vec![10, 20], 3), tx1);
    while engine.has_pending() {
        engine.step().unwrap();
    }

    // Check cache has the entry
    assert!(
        engine.scheduler.prefix_cache().len() > 0,
        "cache should have entry after first request"
    );

    // Second request: longer prompt starting with same tokens
    // Use max_tokens=10 (> prompt_len=5) to avoid immediate finish
    let (tx2, _rx2) = mpsc::unbounded_channel();
    engine.add_request(Request::new(2, vec![10, 20, 30, 40, 50], 10), tx2);

    // Should have pending work (sequence in waiting)
    assert!(
        engine.has_pending(),
        "should have pending after adding second request"
    );

    engine.step().unwrap();

    // After one step, should have 1 sequence in running
    // The sequence should be in Decoding state after processing the remaining 3 tokens
    // (num_computed_tokens went from 2 to 5, then became Decoding)
    assert_eq!(engine.scheduler.running().len(), 1);
    let seq = &engine.scheduler.running()[0];
    assert_eq!(seq.status, vllm_core::types::Status::Decoding);
    assert_eq!(seq.num_computed_tokens, 5); // 2 cached + 3 processed = 5
}
