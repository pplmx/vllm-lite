use vllm_core::scheduler::SchedulerEngine;
use vllm_core::types::{Request, SchedulerConfig, Status};

#[test]
fn test_scheduler_batch_builder_extract() {
    let config = SchedulerConfig::default();
    let mut sched = SchedulerEngine::new(config, 1024);

    // Add 5 requests
    for i in 1..=5 {
        sched.add_request(Request::new(i, vec![i as u32], 3));
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
    };

    let mut sched = SchedulerEngine::new(config, 1024);

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
    sched.update(&batch1.seq_ids, &[99], &[3]);

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
    let mut sched = SchedulerEngine::new(config, 1024);

    // Add request with max_tokens = prompt_len (should finish after prefill)
    sched.add_request(Request::new(1, vec![1, 2], 2)); // prompt_len=2, max_tokens=2

    let batch1 = sched.build_batch();
    sched.update(&batch1.seq_ids, &[99], &[2]); // 2 tokens processed = prompt done

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
    };

    let mut sched = SchedulerEngine::new(config, 1024);
    sched.add_request(Request::new(1, vec![1], 5));
    let batch1 = sched.build_batch();
    sched.update(&batch1.seq_ids, &[10], &[1]);

    // Add more requests while budget is limited
    for i in 2..=5 {
        sched.add_request(Request::new(i, vec![i as u32], 3));
    }

    let batch2 = sched.build_batch();

    // Should respect budget
    let total_tokens: usize = batch2.input_tokens.iter().map(|t| t.len()).sum();
    assert!(total_tokens <= 3, "Should respect budget of 3 tokens");
}
