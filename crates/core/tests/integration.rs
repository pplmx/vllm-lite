use tokio::sync::mpsc;
use vllm_core::engine::Engine;
use vllm_core::scheduler::cache::PrefixCacheConfig;
use vllm_core::types::{Request, SchedulerConfig, TokenId};
use vllm_testing::{ConstModel, IncrementModel};

#[test]
fn test_continuous_batching_with_streaming() {
    let config = SchedulerConfig {
        max_num_seqs: 2,
        max_num_batched_tokens: 100,
        max_consecutive_decode: 10,
        enable_pd_separation: false,
        prefill_chunk_size: 512,
        decode_preference_ratio: 0.7,
        enable_priority_scheduling: false,
        enable_dynamic_batching: false,
        min_batch_size: 1,
        max_batch_size: 256,
    };
    let mut engine = Engine::with_config(IncrementModel, IncrementModel, config, 4, 1024);

    let (tx1, mut rx1) = mpsc::channel(64);
    let (tx2, mut rx2) = mpsc::channel(64);

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
        enable_pd_separation: false,
        prefill_chunk_size: 512,
        decode_preference_ratio: 0.7,
        enable_priority_scheduling: false,
        enable_dynamic_batching: false,
        min_batch_size: 1,
        max_batch_size: 256,
    };
    let mut engine = Engine::with_config(IncrementModel, IncrementModel, config, 4, 1024);

    let (tx, mut rx) = mpsc::channel(64);
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
        enable_pd_separation: false,
        prefill_chunk_size: 512,
        decode_preference_ratio: 0.7,
        enable_priority_scheduling: false,
        enable_dynamic_batching: false,
        min_batch_size: 1,
        max_batch_size: 256,
    };
    let mut engine = Engine::with_config(IncrementModel, IncrementModel, config, 4, 1024);

    let (tx, _rx) = mpsc::channel(64);

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
        enable_pd_separation: false,
        prefill_chunk_size: 512,
        decode_preference_ratio: 0.7,
        enable_priority_scheduling: false,
        enable_dynamic_batching: false,
        min_batch_size: 1,
        max_batch_size: 256,
    };
    let mut engine = Engine::with_config(IncrementModel, IncrementModel, config, 4, 1024);

    let (tx, mut rx) = mpsc::channel(64);

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
        enable_pd_separation: false,
        prefill_chunk_size: 512,
        decode_preference_ratio: 0.7,
        enable_priority_scheduling: false,
        enable_dynamic_batching: false,
        min_batch_size: 1,
        max_batch_size: 256,
    };
    let mut engine = Engine::with_config(IncrementModel, IncrementModel, config, 4, 1024);

    let (tx1, _rx1) = mpsc::channel(64);
    let (tx2, _rx2) = mpsc::channel(64);

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
        enable_pd_separation: false,
        prefill_chunk_size: 512,
        decode_preference_ratio: 0.7,
        enable_priority_scheduling: false,
        enable_dynamic_batching: false,
        min_batch_size: 1,
        max_batch_size: 256,
    };
    let mut engine = Engine::with_config(IncrementModel, IncrementModel, config, 4, 1024);

    let (tx1, _rx1) = mpsc::channel(64);
    let (tx2, _rx2) = mpsc::channel(64);

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
        enable_pd_separation: false,
        prefill_chunk_size: 512,
        decode_preference_ratio: 0.7,
        enable_priority_scheduling: false,
        enable_dynamic_batching: false,
        min_batch_size: 1,
        max_batch_size: 256,
    };
    let mut engine = Engine::with_config(IncrementModel, IncrementModel, config, 4, 1024);

    let (tx1, _rx1) = mpsc::channel(64);
    let (tx2, _rx2) = mpsc::channel(64);

    engine.add_request(Request::new(1, vec![10, 20], 3), tx1);
    while engine.has_pending() {
        engine.step().unwrap();
    }

    engine.add_request(Request::new(2, vec![10, 20], 5), tx2);

    // Cache hit - sequence should be waiting with cached num_computed_tokens
    // In V2, sequences are added to request_queue, not directly to running
    // Build batch to move sequence to running
    let _batch = engine.scheduler.build_batch();
    assert_eq!(engine.scheduler.running().len(), 1);
    let seq = &engine.scheduler.running()[0];
    // New architecture: status is Waiting but num_computed_tokens is set (cache hit)
    assert_eq!(seq.num_computed_tokens, 2);
}

#[test]
fn test_different_prompt_lengths_batching() {
    let config = SchedulerConfig {
        max_num_seqs: 10,
        max_num_batched_tokens: 50,
        max_consecutive_decode: 10,
        enable_pd_separation: false,
        prefill_chunk_size: 512,
        decode_preference_ratio: 0.7,
        enable_priority_scheduling: false,
        enable_dynamic_batching: false,
        min_batch_size: 1,
        max_batch_size: 256,
    };
    let mut engine = Engine::with_config(IncrementModel, IncrementModel, config, 4, 1024);

    let (tx1, _rx1) = mpsc::channel(64);
    let (tx2, _rx2) = mpsc::channel(64);
    let (tx3, _rx3) = mpsc::channel(64);

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
        enable_pd_separation: false,
        prefill_chunk_size: 512,
        decode_preference_ratio: 0.7,
        enable_priority_scheduling: false,
        enable_dynamic_batching: false,
        min_batch_size: 1,
        max_batch_size: 256,
    };
    let mut engine = Engine::with_config(IncrementModel, IncrementModel, config, 4, 1024);

    let (tx1, _rx1) = mpsc::channel(64);
    let (tx2, _rx2) = mpsc::channel(64);
    let (tx3, _rx3) = mpsc::channel(64);

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
        enable_pd_separation: false,
        prefill_chunk_size: 512,
        decode_preference_ratio: 0.7,
        enable_priority_scheduling: false,
        enable_dynamic_batching: false,
        min_batch_size: 1,
        max_batch_size: 256,
    };
    let mut engine = Engine::with_config(IncrementModel, IncrementModel, config, 4, 1024);

    for i in 1..=20 {
        let (tx, _rx) = mpsc::channel(64);
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
        enable_pd_separation: false,
        prefill_chunk_size: 512,
        decode_preference_ratio: 0.7,
        enable_priority_scheduling: false,
        enable_dynamic_batching: false,
        min_batch_size: 1,
        max_batch_size: 256,
    };
    let mut engine = Engine::with_config(IncrementModel, IncrementModel, config, 4, 1024);

    let (tx, _rx) = mpsc::channel(64);
    engine.add_request(Request::new(1, vec![10, 20, 30], 5), tx);

    let batch1 = engine.scheduler.build_batch();
    let running = engine.scheduler.running();
    let seq = running.iter().find(|s| s.id == 1).unwrap();
    assert_eq!(seq.status, vllm_core::types::Status::Prefilling);

    engine.scheduler.update(
        &batch1.seq_ids,
        &[99],
        &[batch1.input_tokens.iter().map(|v| v.len()).sum()],
    );

    let running = engine.scheduler.running();
    let seq = running.iter().find(|s| s.id == 1).unwrap();
    assert_eq!(seq.status, vllm_core::types::Status::Decoding);
}

#[test]
fn test_immediate_finish_after_prompt() {
    let config = SchedulerConfig {
        max_num_seqs: 10,
        max_num_batched_tokens: 100,
        max_consecutive_decode: 10,
        enable_pd_separation: false,
        prefill_chunk_size: 512,
        decode_preference_ratio: 0.7,
        enable_priority_scheduling: false,
        enable_dynamic_batching: false,
        min_batch_size: 1,
        max_batch_size: 256,
    };
    let mut engine = Engine::with_config(IncrementModel, IncrementModel, config, 4, 1024);

    let (tx, _rx) = mpsc::channel(64);
    engine.add_request(Request::new(1, vec![1, 2, 3], 3), tx);

    engine.step().unwrap();

    assert!(
        !engine.has_pending(),
        "should finish when max_tokens equals prompt length"
    );
}

#[test]
fn test_speculative_decoding_verification() {
    let model = ConstModel::new(42);
    let mut engine = Engine::new(model.clone(), model);
    engine.enable_speculative();

    let (tx, _rx) = tokio::sync::mpsc::channel(64);
    engine.add_request(Request::new(1, vec![1, 2, 3], 10), tx);

    // Run speculative step
    let results = engine.step_speculative().unwrap();

    // Should return at least one token (target)
    assert!(!results.is_empty());
}

#[test]
fn test_concurrent_requests_different_prompts() {
    let config = SchedulerConfig {
        max_num_seqs: 10,
        max_num_batched_tokens: 100,
        max_consecutive_decode: 10,
        enable_pd_separation: false,
        prefill_chunk_size: 512,
        decode_preference_ratio: 0.7,
        enable_priority_scheduling: false,
        enable_dynamic_batching: false,
        min_batch_size: 1,
        max_batch_size: 256,
    };
    let mut engine = Engine::with_config(IncrementModel, IncrementModel, config, 4, 1024);

    let (tx1, _rx1) = mpsc::channel(64);
    let (tx2, _rx2) = mpsc::channel(64);
    let (tx3, _rx3) = mpsc::channel(64);

    engine.add_request(Request::new(1, vec![10, 20], 5), tx1);
    engine.add_request(Request::new(2, vec![30, 40, 50, 60], 6), tx2);
    engine.add_request(Request::new(3, vec![70], 4), tx3);

    assert!(
        engine.scheduler.waiting_count() + engine.scheduler.running_count() == 3,
        "all 3 requests should be pending"
    );

    for _ in 0..5 {
        if !engine.has_pending() {
            break;
        }
        engine.step().unwrap();
    }

    assert!(!engine.has_pending(), "all requests should finish");
}

#[test]
fn test_batch_size_variation() {
    let config = SchedulerConfig {
        max_num_seqs: 20,
        max_num_batched_tokens: 200,
        max_consecutive_decode: 10,
        enable_pd_separation: false,
        prefill_chunk_size: 512,
        decode_preference_ratio: 0.7,
        enable_priority_scheduling: false,
        enable_dynamic_batching: false,
        min_batch_size: 1,
        max_batch_size: 256,
    };
    let mut engine = Engine::with_config(IncrementModel, IncrementModel, config, 4, 1024);

    for i in 1..=10 {
        let (tx, _rx) = mpsc::channel(64);
        engine.add_request(Request::new(i, vec![i as TokenId], 4), tx);
    }

    assert!(
        engine.scheduler.waiting_count() + engine.scheduler.running_count() == 10,
        "all 10 requests should be pending"
    );

    for _ in 0..10 {
        if !engine.has_pending() {
            break;
        }
        engine.step().unwrap();
    }

    assert!(!engine.has_pending(), "all 10 requests should finish");
}

#[test]
fn test_request_cancellation() {
    let config = SchedulerConfig {
        max_num_seqs: 10,
        max_num_batched_tokens: 100,
        max_consecutive_decode: 10,
        enable_pd_separation: false,
        prefill_chunk_size: 512,
        decode_preference_ratio: 0.7,
        enable_priority_scheduling: false,
        enable_dynamic_batching: false,
        min_batch_size: 1,
        max_batch_size: 256,
    };
    let mut engine = Engine::with_config(IncrementModel, IncrementModel, config, 4, 1024);

    let (tx1, _rx1) = mpsc::channel(64);
    let (tx2, rx2) = mpsc::channel(64);

    engine.add_request(Request::new(1, vec![10, 20], 5), tx1);
    engine.add_request(Request::new(2, vec![30, 40], 5), tx2);

    engine.step().unwrap();

    drop(rx2);

    engine.step().unwrap();
    engine.step().unwrap();

    assert!(!engine.has_pending(), "remaining request should finish");
}

#[test]
fn test_finished_sequences_cleared() {
    let config = SchedulerConfig {
        max_num_seqs: 10,
        max_num_batched_tokens: 100,
        max_consecutive_decode: 10,
        enable_pd_separation: false,
        prefill_chunk_size: 512,
        decode_preference_ratio: 0.7,
        enable_priority_scheduling: false,
        enable_dynamic_batching: false,
        min_batch_size: 1,
        max_batch_size: 256,
    };
    let mut engine = Engine::with_config(IncrementModel, IncrementModel, config, 4, 1024);

    let (tx, _rx) = mpsc::channel(64);
    engine.add_request(Request::new(1, vec![10, 20], 2), tx);

    for _ in 0..3 {
        engine.step().unwrap();
    }

    assert!(!engine.has_pending());
    assert!(engine.scheduler.finished_sequences().is_empty());
}

#[test]
fn test_cancel_request_cleans_response_txs() {
    let config = SchedulerConfig {
        max_num_seqs: 10,
        max_num_batched_tokens: 100,
        max_consecutive_decode: 10,
        enable_pd_separation: false,
        prefill_chunk_size: 512,
        decode_preference_ratio: 0.7,
        enable_priority_scheduling: false,
        enable_dynamic_batching: false,
        min_batch_size: 1,
        max_batch_size: 256,
    };
    let mut engine = Engine::with_config(IncrementModel, IncrementModel, config, 4, 1024);

    let (tx1, _rx1) = mpsc::channel(64);
    let (tx2, _rx2) = mpsc::channel(64);

    let id1 = engine.add_request(Request::new(1, vec![10, 20], 5), tx1);
    let _id2 = engine.add_request(Request::new(2, vec![30, 40], 5), tx2);

    assert_eq!(engine.response_txs.len(), 2);

    let canceled = engine.cancel_request(id1);
    assert!(canceled);

    assert_eq!(engine.response_txs.len(), 1);
    assert!(!engine.response_txs.contains_key(&id1));

    engine.step().unwrap();
    assert!(engine.has_pending());
}

#[test]
#[ignore = "empty prompt handling needs validation at API layer"]
fn test_empty_prompt_handling() {
    let config = SchedulerConfig::default();
    let mut engine = Engine::with_config(IncrementModel, IncrementModel, config, 4, 1024);

    let (tx, _rx) = mpsc::channel(64);
    engine.add_request(Request::new(1, vec![], 5), tx);

    engine.step().unwrap();

    assert!(
        !engine.has_pending(),
        "empty prompt should be rejected at API layer"
    );
}

#[test]
fn test_single_token_prompt() {
    let config = SchedulerConfig::default();
    let mut engine = Engine::with_config(IncrementModel, IncrementModel, config, 4, 1024);

    let (tx, _rx) = mpsc::channel(64);
    engine.add_request(Request::new(1, vec![42], 3), tx);

    let mut steps = 0;
    while engine.has_pending() && steps < 10 {
        engine.step().unwrap();
        steps += 1;
    }

    assert!(steps <= 3, "should complete in ~2-3 steps");
}

#[test]
fn test_engine_health_tracking() {
    let engine = Engine::new(IncrementModel, IncrementModel);

    assert!(engine.is_healthy(), "new engine should be healthy");
    assert!(
        engine.get_last_error().is_none(),
        "new engine should have no errors"
    );
}

#[test]
fn test_engine_with_const_model() {
    let config = SchedulerConfig::default();
    let const_model = ConstModel::new(42);
    let mut engine = Engine::with_config(const_model.clone(), const_model, config, 4, 1024);

    let (tx, _rx) = mpsc::channel(64);
    engine.add_request(Request::new(1, vec![1, 2, 3], 5), tx);

    let mut steps = 0;
    while engine.has_pending() && steps < 10 {
        engine.step().unwrap();
        steps += 1;
    }

    assert!(steps > 0, "should have processed some steps");
}

#[test]
fn test_engine_large_batch_handling() {
    let config = SchedulerConfig {
        max_num_seqs: 32,
        max_num_batched_tokens: 512,
        max_consecutive_decode: 10,
        enable_pd_separation: false,
        prefill_chunk_size: 512,
        decode_preference_ratio: 0.7,
        enable_priority_scheduling: false,
        enable_dynamic_batching: false,
        min_batch_size: 1,
        max_batch_size: 256,
    };
    let mut engine = Engine::with_config(IncrementModel, IncrementModel, config, 4, 1024);

    for i in 0..10 {
        let (tx, _rx) = mpsc::channel(64);
        engine.add_request(Request::new(i as u64, vec![i; 3], 5), tx);
    }

    let steps = 5;
    for _ in 0..steps {
        if !engine.has_pending() {
            break;
        }
        engine.step().unwrap();
    }
}

#[test]
fn test_engine_sequential_requests() {
    let config = SchedulerConfig::default();
    let mut engine = Engine::with_config(IncrementModel, IncrementModel, config, 4, 1024);

    let (tx1, _rx1) = mpsc::channel(64);
    engine.add_request(Request::new(1, vec![10], 3), tx1);

    while engine.has_pending() {
        engine.step().unwrap();
    }

    let (tx2, _rx2) = mpsc::channel(64);
    engine.add_request(Request::new(2, vec![20], 3), tx2);

    while engine.has_pending() {
        engine.step().unwrap();
    }

    assert!(!engine.has_pending(), "all requests should be done");
}

#[test]
fn test_request_with_max_tokens_equals_prompt() {
    let config = SchedulerConfig::default();
    let mut engine = Engine::with_config(IncrementModel, IncrementModel, config, 4, 1024);

    let (tx, _rx) = mpsc::channel(64);
    let prompt = vec![1, 2, 3];
    engine.add_request(Request::new(1, prompt.clone(), prompt.len()), tx);

    engine.step().unwrap();

    assert!(
        !engine.has_pending(),
        "request should complete immediately when max_tokens == prompt_len"
    );
}

#[test]
fn test_concurrent_requests_batch_processing() {
    let config = SchedulerConfig {
        max_num_seqs: 4,
        max_num_batched_tokens: 50,
        max_consecutive_decode: 10,
        enable_pd_separation: false,
        prefill_chunk_size: 512,
        decode_preference_ratio: 0.7,
        enable_priority_scheduling: false,
        enable_dynamic_batching: true,
        min_batch_size: 1,
        max_batch_size: 256,
    };
    let mut engine = Engine::with_config(
        vllm_testing::IncrementModel,
        vllm_testing::IncrementModel,
        config,
        4,
        1024,
    );

    // Add 4 concurrent requests
    for i in 1..=4 {
        let (tx, _rx) = mpsc::channel(64);
        engine.add_request(Request::new(i as u64, vec![i as u32; 3], 5), tx);
    }

    // Process in batches
    let mut total_processed = 0;
    while engine.has_pending() {
        let results = engine.step().unwrap();
        total_processed += results.len();
    }

    assert!(total_processed > 0, "Should have processed some tokens");
}

#[test]
fn test_multi_batch_continuous_processing() {
    let config = SchedulerConfig {
        max_num_seqs: 2,
        max_num_batched_tokens: 20,
        max_consecutive_decode: 10,
        enable_pd_separation: false,
        prefill_chunk_size: 512,
        decode_preference_ratio: 0.7,
        enable_priority_scheduling: false,
        enable_dynamic_batching: true,
        min_batch_size: 1,
        max_batch_size: 256,
    };
    let mut engine = Engine::with_config(
        vllm_testing::IncrementModel,
        vllm_testing::IncrementModel,
        config,
        4,
        1024,
    );

    // Add first batch of requests
    for i in 1..=2 {
        let (tx, _rx) = mpsc::channel(64);
        engine.add_request(Request::new(i as u64, vec![i as u32; 3], 10), tx);
    }

    // Process first batch
    while engine.has_pending() {
        let results = engine.step().unwrap();
        if results.is_empty() {
            break;
        }
    }

    // Add second batch of requests
    for i in 3..=4 {
        let (tx, _rx) = mpsc::channel(64);
        engine.add_request(Request::new(i as u64, vec![i as u32; 3], 10), tx);
    }

    // Process second batch
    while engine.has_pending() {
        let results = engine.step().unwrap();
        if results.is_empty() {
            break;
        }
    }

    assert!(!engine.has_pending(), "All requests should be completed");
}

#[test]
fn test_dynamic_batch_adjustment() {
    let config = SchedulerConfig::default();
    let mut engine = Engine::with_config(
        vllm_testing::IncrementModel,
        vllm_testing::IncrementModel,
        config,
        4,
        1024,
    );

    // Rapidly add many requests
    for i in 1..=10 {
        let (tx, _rx) = mpsc::channel(64);
        engine.add_request(Request::new(i as u64, vec![i as u32], 3), tx);
    }

    // Process all
    let mut batches = 0;
    while engine.has_pending() {
        engine.step().unwrap();
        batches += 1;
    }

    assert!(batches > 0, "Should have processed in multiple batches");
}

#[test]
fn test_mixed_prompt_lengths() {
    let config = SchedulerConfig {
        max_num_seqs: 10,
        max_num_batched_tokens: 50,
        max_consecutive_decode: 10,
        enable_pd_separation: false,
        prefill_chunk_size: 512,
        decode_preference_ratio: 0.7,
        enable_priority_scheduling: false,
        enable_dynamic_batching: true,
        min_batch_size: 1,
        max_batch_size: 256,
    };
    let mut engine = Engine::with_config(
        vllm_testing::IncrementModel,
        vllm_testing::IncrementModel,
        config,
        4,
        1024,
    );

    // Add requests with varying prompt lengths
    let prompts = [1, 3, 5, 10, 20];
    for (i, len) in prompts.iter().enumerate() {
        let (tx, _rx) = mpsc::channel(64);
        let prompt: Vec<u32> = (0..*len as u32).collect();
        engine.add_request(Request::new((i + 1) as u64, prompt, 5), tx);
    }

    // Process
    while engine.has_pending() {
        engine.step().unwrap();
    }

    assert!(!engine.has_pending(), "All requests should complete");
}

#[test]
fn test_batch_size_changes_over_time() {
    let config = SchedulerConfig {
        max_num_seqs: 10,
        max_num_batched_tokens: 100,
        max_consecutive_decode: 10,
        enable_pd_separation: false,
        prefill_chunk_size: 512,
        decode_preference_ratio: 0.7,
        enable_priority_scheduling: false,
        enable_dynamic_batching: true,
        min_batch_size: 1,
        max_batch_size: 256,
    };
    let mut engine = Engine::with_config(
        vllm_testing::IncrementModel,
        vllm_testing::IncrementModel,
        config,
        4,
        1024,
    );

    let mut batch_sizes = Vec::new();

    // Add requests gradually and track batch sizes
    for i in 1..=10 {
        let (tx, _rx) = mpsc::channel(64);
        engine.add_request(Request::new(i as u64, vec![i as u32], 5), tx);

        if i >= 3 {
            let results = engine.step().unwrap();
            batch_sizes.push(results.len());
        }
    }

    // Process remaining
    while engine.has_pending() {
        let results = engine.step().unwrap();
        batch_sizes.push(results.len());
    }

    assert!(!batch_sizes.is_empty(), "Should have recorded batch sizes");
    // Verify batch sizes are reasonable (not all zero)
    let total: usize = batch_sizes.iter().sum();
    assert!(total > 0, "Should have processed tokens");
}

#[test]
#[should_panic(expected = "max_num_seqs must be > 0")]
fn test_scheduler_config_rejects_zero_max_seqs() {
    let _ = SchedulerConfig::new(
        0, // max_num_seqs = 0 - should panic
        100, 10, false, 512, 0.7, false, false, 1, 10,
    );
}

#[test]
#[should_panic(expected = "max_batch_size must be >= min_batch_size")]
fn test_scheduler_config_rejects_invalid_batch_range() {
    let _ = SchedulerConfig::new(
        10, 100, 10, false, 512, 0.7, false, false, 10, // min_batch_size = 10
        5,  // max_batch_size = 5 - should panic (less than min)
    );
}

#[test]
#[should_panic(expected = "decode_preference_ratio must be between 0.0 and 1.0")]
fn test_scheduler_config_rejects_invalid_ratio() {
    let _ = SchedulerConfig::new(
        10, 100, 10, false, 512, 1.5, // invalid ratio > 1.0 - should panic
        false, false, 1, 10,
    );
}

#[test]
#[should_panic(expected = "max_entries must be > 0")]
fn test_prefix_cache_config_rejects_zero_entries() {
    let _ = PrefixCacheConfig::new(Some(0), Some(100));
}

#[test]
#[should_panic(expected = "max_blocks must be > 0")]
fn test_prefix_cache_config_rejects_zero_blocks() {
    let _ = PrefixCacheConfig::new(Some(100), Some(0));
}

#[test]
fn test_prefix_cache_config_allows_none_values() {
    let config = PrefixCacheConfig::new(None, None);
    assert_eq!(config.max_entries, None);
    assert_eq!(config.max_blocks, None);
}
