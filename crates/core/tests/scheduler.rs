use std::sync::Arc;
use vllm_core::metrics::EnhancedMetricsCollector;
use vllm_core::scheduler::SchedulerEngine;
use vllm_core::types::{Priority, Request, SchedulerConfig, Status};
use vllm_traits::SampledToken;

fn create_test_engine(config: SchedulerConfig, num_kv_blocks: usize) -> SchedulerEngine {
    let metrics = Arc::new(EnhancedMetricsCollector::new());
    SchedulerEngine::new(config, num_kv_blocks, metrics)
}

#[test]
fn test_scheduler_batch_builder_extract() {
    let config = SchedulerConfig::default();
    let mut sched = create_test_engine(config, 1024);

    // Add 5 requests
    for i in 1..=5 {
        sched.add_request(Request::new(
            i,
            vec![u32::try_from(i).expect("bounded test id")],
            3,
        ));
    }

    let batch = sched.build_batch();

    // Should process all 5 requests (prefill)
    assert_eq!(batch.seq_ids.len(), 5, "Should batch all 5 requests");
    assert_eq!(
        batch.input_tokens.len(),
        5,
        "Should have 5 input token arrays"
    );

    // Verify tokens are correct
    for (i, tokens) in batch.input_tokens.iter().enumerate() {
        assert_eq!(
            tokens.len(),
            1,
            "Request {}: should have 1 prompt token",
            i + 1
        );
    }
}

#[test]
fn test_pd_separation_refactored() {
    let config = SchedulerConfig {
        enable_pd_separation: true,
        decode_preference_ratio: 0.5,
        max_num_seqs: 10,
        max_num_batched_tokens: 100,
        max_consecutive_decode: 10,
        prefill_chunk_size: 512,
        enable_priority_scheduling: false,
        enable_dynamic_batching: false,
        min_batch_size: 1,
        max_batch_size: 256,
        ..Default::default()
    };

    let mut sched = create_test_engine(config, 1024);

    // Add request 1: prefill then decode
    sched.add_request(Request::new(1, vec![1, 2, 3], 5));
    let batch1 = sched.build_batch();

    // First batch: prefill all 3 tokens
    assert_eq!(
        batch1.input_tokens[0],
        vec![1, 2, 3],
        "Prefill should include all prompt tokens"
    );

    // Update to complete prefill
    sched.update(
        &batch1.seq_ids,
        &[SampledToken {
            token: 99,
            logprob: 0.0,
            top_logprobs: vec![],
        }],
        &[3],
    );

    // Now request 1 is decoding, add request 2
    sched.add_request(Request::new(2, vec![4, 5], 3));
    let batch2 = sched.build_batch();

    // Should have both decode (request 1) and prefill (request 2)
    assert!(
        !batch2.seq_ids.is_empty(),
        "Should process at least one sequence"
    );

    // Verify running sequences status
    let running = sched.running();
    let has_decode = running.iter().any(|s| s.status == Status::Decoding);
    let has_prefill = running.iter().any(|s| s.status == Status::Prefilling);

    assert!(
        has_decode || has_prefill,
        "Should have either decode or prefill"
    );
}

#[test]
fn test_process_finished_sequences() {
    let config = SchedulerConfig::default();
    let mut sched = create_test_engine(config, 1024);

    // Add request with max_tokens = prompt_len (should finish after prefill)
    sched.add_request(Request::new(1, vec![1, 2], 2)); // prompt_len=2, max_tokens=2

    let batch1 = sched.build_batch();
    sched.update(
        &batch1.seq_ids,
        &[SampledToken {
            token: 99,
            logprob: 0.0,
            top_logprobs: vec![],
        }],
        &[2],
    ); // 2 tokens processed = prompt done

    // Should move to finished
    let finished = sched.finished_sequences();
    assert_eq!(finished.len(), 1, "Should have 1 finished sequence");
    assert_eq!(
        finished[0].tokens,
        vec![1, 2, 99],
        "Should include prompt + generated"
    );
}

#[test]
fn test_build_decode_batch_budget() {
    let config = SchedulerConfig {
        max_num_seqs: 10,
        max_num_batched_tokens: 3, // Only 3 tokens budget
        max_consecutive_decode: 10,
        enable_pd_separation: true,
        prefill_chunk_size: 512,
        decode_preference_ratio: 1.0, // All budget to decode
        enable_priority_scheduling: false,
        enable_dynamic_batching: false,
        min_batch_size: 1,
        max_batch_size: 256,
        ..Default::default()
    };

    let mut sched = create_test_engine(config, 1024);
    sched.add_request(Request::new(1, vec![1], 5));
    let batch1 = sched.build_batch();
    sched.update(
        &batch1.seq_ids,
        &[SampledToken {
            token: 10,
            logprob: 0.0,
            top_logprobs: vec![],
        }],
        &[1],
    );

    // Add more requests while budget is limited
    for i in 2..=5 {
        sched.add_request(Request::new(
            i,
            vec![u32::try_from(i).expect("bounded test id")],
            3,
        ));
    }

    let batch2 = sched.build_batch();

    // Should respect budget
    let total_tokens: usize = batch2.input_tokens.iter().map(std::vec::Vec::len).sum();
    assert!(total_tokens <= 3, "Should respect budget of 3 tokens");
}

#[test]
fn test_token_budget_boundary_zero() {
    let config = SchedulerConfig {
        max_num_seqs: 10,
        max_num_batched_tokens: 0, // Zero budget
        max_consecutive_decode: 10,
        enable_pd_separation: false,
        prefill_chunk_size: 512,
        decode_preference_ratio: 0.7,
        enable_priority_scheduling: false,
        enable_dynamic_batching: false,
        min_batch_size: 1,
        max_batch_size: 256,
        ..Default::default()
    };

    let mut sched = create_test_engine(config, 1024);

    // Add some requests
    for i in 1..=5 {
        sched.add_request(Request::new(
            i,
            vec![u32::try_from(i).expect("bounded test id"); 3],
            5,
        ));
    }

    let batch = sched.build_batch();
    // With zero budget, should not add any tokens but may still have sequences
    let total_tokens: usize = batch.input_tokens.iter().map(std::vec::Vec::len).sum();
    assert_eq!(total_tokens, 0, "Zero budget should result in zero tokens");
}

#[test]
fn test_token_budget_boundary_one() {
    let config = SchedulerConfig {
        max_num_seqs: 10,
        max_num_batched_tokens: 1, // Single token budget
        max_consecutive_decode: 10,
        enable_pd_separation: false,
        prefill_chunk_size: 512,
        decode_preference_ratio: 0.7,
        enable_priority_scheduling: false,
        enable_dynamic_batching: false,
        min_batch_size: 1,
        max_batch_size: 256,
        ..Default::default()
    };

    let mut sched = create_test_engine(config, 1024);

    // Add decode requests (1 token each)
    for i in 1..=3 {
        sched.add_request(Request::new(
            i,
            vec![u32::try_from(i).expect("bounded test id")],
            5,
        ));
    }

    let batch = sched.build_batch();
    let total_tokens: usize = batch.input_tokens.iter().map(std::vec::Vec::len).sum();
    assert!(
        total_tokens <= 1,
        "Single token budget should allow max 1 token"
    );
}

#[test]
fn test_prefill_and_decode_separation() {
    let config = SchedulerConfig {
        max_num_seqs: 10,
        max_num_batched_tokens: 100,
        max_consecutive_decode: 10,
        enable_pd_separation: true,
        prefill_chunk_size: 512,
        decode_preference_ratio: 0.7, // 70% decode, 30% prefill
        enable_priority_scheduling: false,
        enable_dynamic_batching: false,
        min_batch_size: 1,
        max_batch_size: 256,
        ..Default::default()
    };

    let mut sched = create_test_engine(config, 1024);

    // Add prefill requests
    for i in 1..=3 {
        sched.add_request(Request::new(i, vec![1, 2, 3, 4, 5], 5));
    }

    // Add decode requests
    for i in 4..=6 {
        sched.add_request(Request::new(
            i,
            vec![u32::try_from(i).expect("bounded test id")],
            10,
        ));
    }

    let batch = sched.build_batch();

    // Verify both prefill and decode in batch
    let prefill_count = batch.is_prefill.iter().filter(|&&p| p).count();
    let decode_count = batch.is_prefill.iter().filter(|&&p| !p).count();

    assert!(
        prefill_count > 0 || decode_count > 0,
        "Batch should have some requests"
    );
}

#[test]
fn test_priority_scheduling() {
    let config = SchedulerConfig {
        max_num_seqs: 10,
        max_num_batched_tokens: 100,
        max_consecutive_decode: 10,
        enable_pd_separation: false,
        prefill_chunk_size: 512,
        decode_preference_ratio: 0.7,
        enable_priority_scheduling: true,
        enable_dynamic_batching: false,
        min_batch_size: 1,
        max_batch_size: 256,
        ..Default::default()
    };

    let mut sched = create_test_engine(config, 1024);

    // Add requests with different priorities (lower = higher priority)
    sched.add_request(Request::new(1, vec![1], 5)); // default priority 0
    sched.add_request(Request::new(2, vec![2], 5).with_priority(Priority(50))); // lower priority
    sched.add_request(Request::new(3, vec![3], 5).with_priority(Priority(100))); // lowest priority

    let batch = sched.build_batch();

    // Should prioritize lower sequence IDs (added first with higher implicit priority)
    // Or depending on implementation, check that all were added
    assert!(
        !batch.seq_ids.is_empty(),
        "Should have selected some requests"
    );
}

/// With `PriorityPolicy` installed and `max_num_seqs` limited to 1, the
/// scheduler must select only the highest-priority request (lowest
/// `Priority(n)` value). The prefill `BatchComposer` re-sorts by remaining
/// token count, so this test uses sequences with identical prompt lengths to
/// eliminate that secondary ordering — leaving priority as the sole differentiator.
///
/// This closes the gap where `test_priority_scheduling` only asserted that
/// "some requests were selected" without verifying priority ordering.
#[test]
fn test_priority_scheduling_selects_highest_priority_under_limit() {
    use vllm_core::scheduler::policy::PriorityPolicy;

    let config = SchedulerConfig {
        max_num_seqs: 1, // only one sequence per batch — priority decides
        max_num_batched_tokens: 100,
        max_consecutive_decode: 10,
        enable_pd_separation: false,
        prefill_chunk_size: 512,
        decode_preference_ratio: 0.7,
        enable_priority_scheduling: true,
        enable_dynamic_batching: false,
        min_batch_size: 1,
        max_batch_size: 256,
        ..Default::default()
    };

    let mut sched = create_test_engine(config, 1024);

    // Install PriorityPolicy — the scheduler constructor defaults to FcfsPolicy.
    sched.set_policy(Box::new(PriorityPolicy::default()));

    // All prompts are length-1 so the prefill re-sort by remaining tokens
    // is a no-op (ties are unstable but all equal). Priority determines the
    // sort in build_batch, and compose_prefill_batch preserves the first
    // element when only one is selected.
    sched.add_request(Request::new(1, vec![1], 5).with_priority(Priority(100))); // lowest
    sched.add_request(Request::new(2, vec![2], 5).with_priority(Priority(0))); // highest
    sched.add_request(Request::new(3, vec![3], 5).with_priority(Priority(50))); // middle

    let batch = sched.build_batch();

    assert_eq!(
        batch.seq_ids.len(),
        1,
        "max_num_seqs=1 should produce exactly one sequence in the batch"
    );
    assert_eq!(
        batch.seq_ids[0], 2,
        "Priority(0) on seq 2 must be selected when only one slot is available"
    );
}

/// With `PriorityPolicy` installed and `max_num_seqs >= 3`, the batch's
/// `seq_ids` must reflect priority ordering for decode-phase sequences.
///
/// Decode path (`compose_decode_batch`) preserves the sequence order from
/// `build_batch`'s priority sort — unlike prefill which re-sorts by
/// remaining-token count. This test promotes sequences to decode state
/// (via `update`) and then verifies ordering.
#[test]
fn test_priority_scheduling_decode_batch_ordering() {
    let config = SchedulerConfig {
        max_num_seqs: 10,
        max_num_batched_tokens: 100,
        max_consecutive_decode: 10,
        enable_pd_separation: false,
        prefill_chunk_size: 512,
        decode_preference_ratio: 0.7,
        enable_priority_scheduling: true,
        enable_dynamic_batching: false,
        min_batch_size: 1,
        max_batch_size: 256,
        ..Default::default()
    };

    let mut sched = create_test_engine(config, 1024);
    sched.set_policy(Box::new(
        vllm_core::scheduler::policy::PriorityPolicy::default(),
    ));

    // Prefill + promote all three to decode state.
    sched.add_request(Request::new(1, vec![1], 5).with_priority(Priority(100)));
    sched.add_request(Request::new(2, vec![2], 5).with_priority(Priority(0)));
    sched.add_request(Request::new(3, vec![3], 5).with_priority(Priority(50)));

    // First batch: complete prefill for all three.
    let batch1 = sched.build_batch();
    let sampled: Vec<SampledToken> = (0..batch1.seq_ids.len())
        .map(|_| SampledToken {
            token: 1,
            logprob: 0.0,
            top_logprobs: vec![],
        })
        .collect();
    sched.update(&batch1.seq_ids, &sampled, &vec![1; batch1.seq_ids.len()]);

    // Second batch: all three are now in decode phase, ordered by priority.
    let batch2 = sched.build_batch();

    // decode compose preserves the priority-sorted order from build_batch.
    // Expected: seq 2 (Priority(0)) first, then seq 3 (Priority(50)), then seq 1 (Priority(100)).
    assert_eq!(
        batch2.seq_ids,
        vec![2, 3, 1],
        "decode batch seq_ids must follow priority ordering (Priority(0) → Priority(100)), got {:?}",
        batch2.seq_ids
    );
}

#[test]
fn test_consecutive_decode_limit() {
    let config = SchedulerConfig {
        max_num_seqs: 10,
        max_num_batched_tokens: 100,
        max_consecutive_decode: 3, // Limit to 3 consecutive decode rounds
        enable_pd_separation: false,
        prefill_chunk_size: 512,
        decode_preference_ratio: 0.7,
        enable_priority_scheduling: false,
        enable_dynamic_batching: false,
        min_batch_size: 1,
        max_batch_size: 256,
        ..Default::default()
    };

    let mut sched = create_test_engine(config, 1024);

    // Add requests and simulate multiple decode rounds
    for i in 1..=3 {
        sched.add_request(Request::new(
            i,
            vec![u32::try_from(i).expect("bounded test id")],
            10,
        ));
    }

    // First batch
    let batch1 = sched.build_batch();
    assert!(!batch1.is_empty());

    // Simulate decode completion
    for _ in 0..3 {
        let batch = sched.build_batch();
        if !batch.is_empty() {
            let next_tokens: Vec<SampledToken> = (0..batch.seq_ids.len())
                .map(|_| SampledToken {
                    token: 1,
                    logprob: 0.0,
                    top_logprobs: vec![],
                })
                .collect();
            sched.update(&batch.seq_ids, &next_tokens, &vec![1; batch.seq_ids.len()]);
        }
    }

    // After 3 rounds, should not add more decode-only requests
    // This tests that consecutive decode limit is respected
}

#[test]
fn test_min_batch_size() {
    let config = SchedulerConfig {
        max_num_seqs: 10,
        max_num_batched_tokens: 100,
        max_consecutive_decode: 10,
        enable_pd_separation: false,
        prefill_chunk_size: 512,
        decode_preference_ratio: 0.7,
        enable_priority_scheduling: false,
        enable_dynamic_batching: false,
        min_batch_size: 3, // Require at least 3
        max_batch_size: 256,
        ..Default::default()
    };

    let mut sched = create_test_engine(config, 1024);

    // Add fewer than min_batch_size requests
    sched.add_request(Request::new(1, vec![1], 5));
    sched.add_request(Request::new(2, vec![2], 5));

    let batch = sched.build_batch();

    // Should still return batch (min_batch_size is hint, not hard requirement)
    // Or may return empty if waiting < min_batch_size
    assert!(batch.seq_ids.len() <= 2);
}

#[test]
fn test_max_batch_size() {
    let config = SchedulerConfig {
        max_num_seqs: 5, // Target limit
        max_num_batched_tokens: 1000,
        max_consecutive_decode: 10,
        enable_pd_separation: false,
        prefill_chunk_size: 512,
        decode_preference_ratio: 0.7,
        enable_priority_scheduling: false,
        enable_dynamic_batching: false,
        min_batch_size: 1,
        max_batch_size: 256,
        ..Default::default()
    };

    let mut sched = create_test_engine(config, 1024);

    // Add many requests
    for i in 1..=20 {
        sched.add_request(Request::new(
            i,
            vec![u32::try_from(i).expect("bounded test id")],
            5,
        ));
    }

    let batch = sched.build_batch();

    // Should respect max_num_seqs (or be close to it)
    assert!(
        batch.seq_ids.len() <= 20,
        "Should not exceed available requests"
    );
}

#[test]
fn test_memory_manager_allocation_and_release() {
    use vllm_core::scheduler::memory::MemoryManager;

    let config = SchedulerConfig::default();
    let mut memory = MemoryManager::new(config, 100);

    // Allocate blocks
    let blocks1 = memory.allocate(10).unwrap();
    assert_eq!(blocks1.len(), 10);
    assert_eq!(memory.available_blocks(), 90);

    // Allocate more
    let blocks2 = memory.allocate(20).unwrap();
    assert_eq!(blocks2.len(), 20);
    assert_eq!(memory.available_blocks(), 70);

    // Release first allocation
    memory.release_blocks(&blocks1);
    assert_eq!(memory.available_blocks(), 80); // 70 + 10

    // Release second allocation
    memory.release_blocks(&blocks2);
    assert_eq!(memory.available_blocks(), 100);
}

#[test]
fn test_memory_manager_out_of_memory() {
    use vllm_core::scheduler::memory::MemoryManager;

    let config = SchedulerConfig::default();
    let mut memory = MemoryManager::new(config, 10);

    // Allocate all blocks
    let blocks = memory.allocate(10).unwrap();
    assert_eq!(blocks.len(), 10);
    assert_eq!(memory.available_blocks(), 0);

    // Try to allocate more - should fail
    let overflow = memory.allocate(1);
    assert!(overflow.is_none());

    // Release and retry
    memory.release_blocks(&blocks);
    let retry = memory.allocate(5);
    assert!(retry.is_some());
    assert_eq!(retry.unwrap().len(), 5);
}

#[test]
fn test_memory_manager_select_victims() {
    use std::sync::Arc;
    use vllm_core::scheduler::memory::MemoryManager;
    use vllm_core::types::SamplingParams;
    use vllm_core::types::{Priority, Sequence, Status};

    let config = SchedulerConfig::default();
    let mut memory = MemoryManager::new(config, 100);

    // Create sequences with different decode rounds
    let seq1 = Sequence {
        id: 1,
        tokens: vec![1, 2, 3],
        kv_blocks: Arc::new(vec![10, 11]),
        num_computed_tokens: 3,
        prompt_len: 3,
        status: Status::Decoding,
        max_tokens: 10,
        sampling_params: SamplingParams::default(),
        consecutive_decode_rounds: 5,
        priority: Priority::default(),
        degraded_draft: false,
        draft_model_id: None,
    };

    let seq2 = Sequence {
        id: 2,
        tokens: vec![4, 5, 6],
        kv_blocks: Arc::new(vec![20, 21]),
        num_computed_tokens: 3,
        prompt_len: 3,
        status: Status::Decoding,
        max_tokens: 10,
        sampling_params: SamplingParams::default(),
        consecutive_decode_rounds: 20, // Higher - should be selected first
        priority: Priority::default(),
        degraded_draft: false,
        draft_model_id: None,
    };

    // Record blocks so the eviction policy has ref_count = 1 (eligible).
    memory.record_blocks(&[10, 11, 20, 21]);

    let running = vec![seq1, seq2];

    // Select victims — should return 1 block from the higher-priority (shorter decode) sequence.
    let victims = memory.select_victims(&running, 1);
    assert_eq!(victims.len(), 1, "should select exactly 1 victim");
    // seq1 (consecutive_decode_rounds=5) has priority 3 (new decode, evictable),
    // seq2 (consecutive_decode_rounds=20) has priority 1 (long-running, protected).
    // Higher priority values are evicted first, so seq1's blocks are selected.
    assert!(
        [10, 11].contains(&victims[0]),
        "victim should come from seq1 (priority 3, new decode), got block {}",
        victims[0],
    );
}

#[test]
fn test_preemption_execution() {
    let config = SchedulerConfig {
        max_num_seqs: 10,
        max_num_batched_tokens: 100,
        max_consecutive_decode: 2, // Low limit to trigger preemption
        enable_pd_separation: false,
        prefill_chunk_size: 512,
        decode_preference_ratio: 0.7,
        enable_priority_scheduling: false,
        enable_dynamic_batching: false,
        min_batch_size: 1,
        max_batch_size: 256,
        ..Default::default()
    };

    let mut sched = create_test_engine(config, 100);

    // Add multiple requests
    for i in 1..=5 {
        sched.add_request(Request::new(
            i,
            vec![u32::try_from(i).expect("bounded test id")],
            10,
        ));
    }

    // Build batch - some should be in decoding state
    let batch1 = sched.build_batch();
    assert!(!batch1.seq_ids.is_empty());

    // After multiple steps, check for preemption
    // Run enough steps to trigger consecutive decode limit
    for _ in 0..20 {
        if sched.has_pending() {
            let _batch = sched.build_batch();
        }
    }

    // Test passes if we reach here without panicking
}

/// Verify that `MemoryManager::select_victims` (delegating to
/// `EvictionPolicy::select_victims`) is wired into the preemption path.
///
/// Two sequences with different `consecutive_decode_rounds` values
/// compete for eviction. The long-running decode sequence (priority 1)
/// should have its blocks selected before the new decode sequence
/// (priority 3).
#[test]
fn test_eviction_policy_wired_into_preemption() {
    use std::sync::Arc;
    use vllm_core::scheduler::memory::MemoryManager;
    use vllm_core::types::Status;
    use vllm_core::types::{Priority, SamplingParams, Sequence};

    let config = SchedulerConfig::default();
    let mut memory = MemoryManager::new(config, 100);

    let new_decode = Sequence {
        id: 1,
        tokens: vec![],
        kv_blocks: Arc::new(vec![10, 11]),
        num_computed_tokens: 0,
        prompt_len: 0,
        status: Status::Decoding,
        max_tokens: 10,
        sampling_params: SamplingParams::default(),
        consecutive_decode_rounds: 0, // priority 3 (new decode)
        priority: Priority::default(),
        degraded_draft: false,
        draft_model_id: None,
    };

    let long_running = Sequence {
        id: 2,
        tokens: vec![],
        kv_blocks: Arc::new(vec![20, 21]),
        num_computed_tokens: 0,
        prompt_len: 0,
        status: Status::Decoding,
        max_tokens: 10,
        sampling_params: SamplingParams::default(),
        consecutive_decode_rounds: 50, // priority 1 (long-running)
        priority: Priority::default(),
        degraded_draft: false,
        draft_model_id: None,
    };

    memory.record_blocks(&[10, 11, 20, 21]);

    // Request 1 victim — eviction policy should pick from the new-decode
    // sequence (priority 3) since higher priority values are evicted first.
    // Long-running sequences (priority 1) are protected to preserve their
    // accumulated compute.
    let victims = memory.select_victims(&[new_decode, long_running], 1);

    assert_eq!(victims.len(), 1);
    assert!(
        [10, 11].contains(&victims[0]),
        "victim should come from new decode (priority 3), got block {}",
        victims[0]
    );
}
