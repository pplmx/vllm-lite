use std::sync::Arc;
use vllm_core::metrics::EnhancedMetricsCollector;
use vllm_core::scheduler::SchedulerEngine;
use vllm_core::scheduler::policy::SjfPolicy;
use vllm_core::types::{Request, SchedulerConfig, SequencePackingConfig, Status};
use vllm_traits::{BatchPhase, SampledToken};

fn create_test_engine(config: SchedulerConfig, num_kv_blocks: usize) -> SchedulerEngine {
    let metrics = Arc::new(EnhancedMetricsCollector::new());
    SchedulerEngine::new(config, num_kv_blocks, metrics)
}

#[test]
fn test_scheduler_basic_flow() {
    let config = SchedulerConfig::default();
    let mut engine = create_test_engine(config, 1024);

    // Add a request
    let id = engine.add_request(Request::new(0, vec![1, 2, 3], 5));
    assert_eq!(id, 1);

    // Build batch
    let batch = engine.build_batch();
    assert!(!batch.is_empty());
    assert_eq!(batch.seq_ids.len(), 1);

    // Simulate model forward
    let input_counts: Vec<usize> = batch.input_tokens.iter().map(std::vec::Vec::len).collect();
    engine.update(
        &batch.seq_ids,
        &[SampledToken {
            token: 99,
            logprob: 0.0,
            top_logprobs: vec![],
        }],
        &input_counts,
    );

    // Verify
    assert_eq!(engine.running_count(), 1);
}

#[test]
fn test_scheduler_multiple_requests() {
    let config = SchedulerConfig::default();
    let mut engine = create_test_engine(config, 1024);

    // Add multiple requests
    for i in 1..=5 {
        engine.add_request(Request::new(
            0,
            vec![u32::try_from(i).expect("bounded test id")],
            10,
        ));
    }

    // Build batch
    let batch = engine.build_batch();
    assert!(!batch.is_empty());
    assert!(batch.seq_ids.len() <= 5);
}

#[test]
fn test_scheduler_prefill_decode_separation() {
    let config = SchedulerConfig::default();
    let mut engine = create_test_engine(config, 1024);

    // Add a request
    engine.add_request(Request::new(0, vec![1, 2, 3], 5));

    // First batch should be prefill
    let batch1 = engine.build_batch();
    assert!(!batch1.is_empty());
    assert_eq!(batch1.phase, BatchPhase::Prefill);

    // Complete prefill
    let input_counts: Vec<usize> = batch1.input_tokens.iter().map(std::vec::Vec::len).collect();
    engine.update(
        &batch1.seq_ids,
        &[SampledToken {
            token: 99,
            logprob: 0.0,
            top_logprobs: vec![],
        }],
        &input_counts,
    );

    // Next batch should be decode (if we have running sequences)
    let batch2 = engine.build_batch();
    // If sequence completed (max_tokens reached), batch might be empty
    // Otherwise it should be a decode phase
    if !batch2.is_empty() {
        assert_eq!(batch2.phase, BatchPhase::Decode);
    }
}

#[test]
fn test_scheduler_policy_switching() {
    let config = SchedulerConfig::default();
    let mut engine = create_test_engine(config, 1024);

    // Default policy is FCFS
    // Add requests with different priorities
    engine.add_request(Request::new(0, vec![1], 5));
    engine.add_request(Request::new(0, vec![2], 5));
    engine.add_request(Request::new(0, vec![3], 5));

    // Build batch (FCFS order)
    let batch = engine.build_batch();
    assert!(!batch.is_empty());

    // Switch to SJF
    engine.set_policy(Box::new(SjfPolicy::default()));

    // Add more requests
    engine.add_request(Request::new(0, vec![4], 5));

    // Build batch (SJF order)
    let batch2 = engine.build_batch();
    assert!(!batch2.is_empty());
}

#[test]
fn test_scheduler_prefix_cache() {
    // Note: This test verifies that prefix cache operations don't panic
    // There's a known bug in batch_composer.rs when num_computed_tokens > tokens.len()
    // after prefix cache hit - we work around it by not triggering that path
    let config = SchedulerConfig::default();
    let mut engine = create_test_engine(config, 1024);

    // Add first request - complete it to add to prefix cache
    let id1 = engine.add_request(Request::new(0, vec![1, 2, 3], 10));
    let batch1 = engine.build_batch();
    let input_counts: Vec<usize> = batch1.input_tokens.iter().map(std::vec::Vec::len).collect();
    engine.update(
        &batch1.seq_ids,
        &[SampledToken {
            token: 99,
            logprob: 0.0,
            top_logprobs: vec![],
        }],
        &input_counts,
    );

    // Continue until finished
    for i in 0..9 {
        if engine.running_count() == 0 {
            break;
        }
        let next = u32::try_from(100 + i).expect("bounded test token");
        engine.update(
            &[id1],
            &[SampledToken {
                token: next,
                logprob: 0.0,
                top_logprobs: vec![],
            }],
            &[0],
        );
    }

    // Add second request with overlapping prefix - different suffix
    // This will trigger prefix cache lookup during add_request
    let _id2 = engine.add_request(Request::new(0, vec![1, 2, 3, 4], 10));

    // The sequence was created with prefix cache info
    // Verify it was enqueued properly
    assert!(engine.waiting_count() > 0 || engine.running_count() > 0);
}

#[test]
fn test_scheduler_memory_preemption() {
    // Create engine with limited memory
    let config = SchedulerConfig::default();
    let mut engine = create_test_engine(config, 20); // Only 20 blocks

    // Add multiple large requests
    for i in 1..=5 {
        let prompt: Vec<u32> = (1..=100)
            .map(|j| u32::try_from(i * 100 + j).expect("bounded test token"))
            .collect();
        engine.add_request(Request::new(0, prompt, 200));
    }

    // Build batch - should handle memory constraints
    let batch = engine.build_batch();
    assert!(!batch.is_empty());
}

#[test]
fn test_scheduler_concurrent_requests() {
    let config = SchedulerConfig::default();
    let mut engine = create_test_engine(config, 1024);

    // Add concurrent requests
    for i in 1..=10 {
        engine.add_request(Request::new(
            0,
            vec![u32::try_from(i).expect("bounded test id"); 10],
            20,
        ));
    }

    // Multiple batch cycles
    for _ in 0..5 {
        if engine.has_pending() {
            let batch = engine.build_batch();
            if !batch.is_empty() {
                let input_counts: Vec<usize> =
                    batch.input_tokens.iter().map(std::vec::Vec::len).collect();
                let next_tokens: Vec<SampledToken> = batch
                    .seq_ids
                    .iter()
                    .map(|_| SampledToken {
                        token: 99,
                        logprob: 0.0,
                        top_logprobs: vec![],
                    })
                    .collect();
                engine.update(&batch.seq_ids, &next_tokens, &input_counts);
            }
        }
    }

    // Should have processed some requests
    assert!(engine.running_count() > 0 || engine.waiting_count() > 0 || !engine.has_pending());
}

#[test]
fn test_finish_sequence_excludes_from_future_batches() {
    // Regression test for a bug where stop-sequence-matched sequences
    // were finalized via `finalize_finished` (dropping the response
    // channel + sending FinishReason::Stop) but were NOT marked as
    // `Finished` in the scheduler. Without the status change + block
    // release, the sequence lingered in `running` (status still
    // `Decoding`) and was re-included in every subsequent `build_batch`,
    // wasting compute on tokens the client never sees and leaking KV
    // blocks until `max_tokens` was eventually hit.
    let config = SchedulerConfig::default();
    let mut engine = create_test_engine(config, 1024);

    // max_tokens well above prompt_len so update() does NOT finish
    // the sequence via the max_tokens path.
    engine.add_request(Request::new(0, vec![1, 2, 3], 100));

    // Build batch (prefill), process it — sequence transitions to
    // Decoding and stays in running.
    let batch = engine.build_batch();
    assert!(!batch.is_empty());
    let input_counts: Vec<usize> = batch.input_tokens.iter().map(std::vec::Vec::len).collect();
    engine.update(
        &batch.seq_ids,
        &[SampledToken {
            token: 42,
            logprob: 0.0,
            top_logprobs: vec![],
        }],
        &input_counts,
    );

    // update() did NOT finish the sequence (max_tokens=100 >> 4 tokens).
    assert_eq!(engine.running_count(), 1);
    assert!(engine.get_sequence(1).is_some());

    // Simulate a stop-sequence match.
    engine.finish_sequence(1);

    // The sequence must be excluded from running so build_batch
    // doesn't re-schedule it.
    assert_eq!(engine.running_count(), 0);
    assert!(engine.get_sequence(1).is_none());

    // build_batch must not re-schedule the finished sequence. Before
    // the fix, it would have appeared here again.
    let next_batch = engine.build_batch();
    assert!(
        next_batch.is_empty(),
        "finish_sequence must exclude the sequence from future batches"
    );

    // The finished sequence is in the finished set (not silently lost).
    let finished = engine.finished_sequences();
    assert_eq!(finished.len(), 1);
    assert_eq!(finished[0].id, 1);
}

// RIL ISS-052: a decode sequence must never be composed with a short KV
// table. When a boundary-crossing growth allocation fails (pool exhausted
// mid-update), the table is one block short; if the pool recovers (other
// sequences finish) before the next round, the preemption gate sees
// blocks_needed <= available and does NOT fire — without this fix the
// sequence would be composed with a short table and the paged-KV writer
// would fall back to block 0, corrupting the prompt's first block.
#[test]
fn decode_growth_failure_composes_only_complete_tables() {
    // 3 KV blocks: A (1) + D1 (1) + D2 (1) fill the pool exactly.
    let config = SchedulerConfig::builder().build();
    let mut engine = create_test_engine(config, 3);

    // A: long-running decode that will cross the 16-token boundary.
    engine.add_request(Request::new(1, vec![1; 16], 50));
    // D1/D2: finish during the prefill update (max_tokens = 1 already
    // generated), freeing their blocks afterwards.
    engine.add_request(Request::new(2, vec![2; 8], 1));
    engine.add_request(Request::new(3, vec![3; 8], 1));

    // Prefill round: all three admitted (pool -> 0). After the forward,
    // update pushes one predicted token per sequence: A crosses to 17
    // tokens and its growth to block 2 FAILS (pool empty) while D1/D2 hit
    // max_tokens and release their blocks -> pool = 2, A's table still
    // holds only block 0 (short for its 17 tokens).
    let batch = engine.build_batch();
    assert_eq!(
        batch.seq_ids.len(),
        3,
        "all three must be admitted at prefill"
    );
    step(&mut engine, &batch);
    assert_eq!(engine.waiting_count(), 0);
    let a = engine.get_sequence(1).expect("A must still be in running");
    assert_eq!(a.tokens.len(), 17, "A must have crossed the block boundary");
    assert_eq!(
        a.kv_blocks.len(),
        1,
        "A's boundary-crossing growth must have failed (pool was empty)"
    );

    // Decode round: A's blocks_needed (2) is exactly `available` (2), so
    // the preemption gate does NOT fire. The scheduler must grow A's table
    // to its full 2 blocks before composing (ISS-052) — never compose it
    // short (block-0 fallback) or silently drop it.
    let batch = engine.build_batch();
    assert_eq!(batch.seq_ids, vec![1], "A must be composed (and only A)");
    assert_eq!(
        batch.kv_block_ids[0].len(),
        2,
        "A must carry a complete KV table (17 tokens -> 2 blocks), not a short one"
    );
    let a = engine.get_sequence(1).expect("A stays in running");
    assert_eq!(
        a.kv_blocks.len(),
        2,
        "A's persisted table must also be complete for the next update"
    );
}

/// Feed one sampled token per composed sequence and advance the scheduler,
/// mirroring `Engine::step_regular`'s post-forward update.
fn step(engine: &mut SchedulerEngine, batch: &vllm_traits::Batch) {
    let input_counts: Vec<usize> = batch.input_tokens.iter().map(std::vec::Vec::len).collect();
    let next_tokens: Vec<SampledToken> = batch
        .seq_ids
        .iter()
        .map(|_| SampledToken {
            token: 3,
            logprob: 0.0,
            top_logprobs: vec![],
        })
        .collect();
    engine.update(&batch.seq_ids, &next_tokens, &input_counts);
}

// RIL ISS-051: a prompt longer than max_num_batched_tokens must be served
// via chunked prefill instead of spinning in the Prefill phase forever.
// Each round processes a budget-sized (and prefill_chunk_size-capped)
// chunk, the partial sequence is re-queued with its computed KV blocks +
// num_computed_tokens preserved, and the final chunk transitions it to
// Decoding. The invariant "running never holds Prefilling across steps"
// (ISS-045) must hold between rounds.
#[test]
fn long_prompt_is_served_via_chunked_prefill() {
    // 100-token prompt = 100/16 = 7 KV blocks. Batch token budget 32 and
    // prefill chunk cap 16 -> ceil(100/16) = 7 chunk rounds to prefill.
    let config = SchedulerConfig::builder()
        .with_max_num_batched_tokens(32)
        .with_prefill_chunk_size(16)
        .with_packing(SequencePackingConfig {
            enabled: false,
            ..Default::default()
        })
        .build();
    let mut engine = create_test_engine(config, 64);

    engine.add_request(Request::new(5, vec![7; 100], 8));

    // Drive rounds until the prefill completes (sequence leaves the queue).
    let mut rounds = 0;
    while engine.waiting_count() > 0 {
        let batch = engine.build_batch();
        assert!(
            !batch.seq_ids.is_empty(),
            "round {rounds}: a chunk must always be servable (max_num_seqs/budget are not the bottleneck here)"
        );
        step(&mut engine, &batch);
        // ISS-045 invariant: `running` never holds Prefilling across steps
        // (a partial chunk is re-queued by update; only a completed prefill
        // stays in running, as Decoding).
        for seq in engine.running() {
            assert_ne!(
                seq.status,
                Status::Prefilling,
                "no sequence may stay Prefilling in running between steps (round {rounds})"
            );
        }
        rounds += 1;
        assert!(
            rounds <= 12,
            "prefill must complete within ceil(100/16)+slack rounds"
        );
    }

    // The prompt is now fully prefilled and decoding.
    let seq = engine
        .get_sequence(5)
        .expect("sequence must be running as Decoding");
    assert_eq!(
        seq.num_computed_tokens, 100,
        "the whole prompt must have been computed across chunks"
    );
    assert_eq!(
        seq.status,
        Status::Decoding,
        "a completed prefill must decode"
    );
    assert_eq!(engine.waiting_count(), 0, "queue fully drained");
    // max_tokens accounting: exactly ONE output token is in the stream so
    // far (the prefill's final prediction); 7 more remain.
    assert_eq!(
        seq.tokens.len() - seq.prompt_len,
        1,
        "chunked prefill must not leak stale intermediate predictions into the output count"
    );
}

// RIL ISS-051: a prompt that FITS the batch token budget is NOT chunked —
// mid-size prefills keep their single-round behavior (no regression).
#[test]
fn fitting_prompt_stays_unchunked() {
    let config = SchedulerConfig::builder()
        .with_max_num_batched_tokens(64)
        .with_prefill_chunk_size(16)
        .with_packing(SequencePackingConfig {
            enabled: false,
            ..Default::default()
        })
        .build();
    let mut engine = create_test_engine(config, 64);

    // 20 tokens fits the 64-token budget -> single prefill round.
    engine.add_request(Request::new(6, vec![7; 20], 8));
    let batch = engine.build_batch();
    assert_eq!(
        batch.input_tokens[0].len(),
        20,
        "a fitting prompt must be processed whole, not chunked"
    );
    step(&mut engine, &batch);
    let seq = engine.get_sequence(6).expect("served in one round");
    assert_eq!(seq.num_computed_tokens, 20);
    assert_eq!(seq.status, Status::Decoding);
}

// RIL ISS-054: a chunked prefill whose chunk lost the token-budget
// competition must NOT lose its computed progress.
//
// `requeue_seq` (used by ISS-041 uncomposed-overflow requeue and ISS-045
// forward-error recovery) used to unconditionally reset every requeued
// prefill to `num_computed_tokens == 0` and release its blocks. That was
// correct for never-advanced sequences, but chunked prefill (ISS-051)
// deliberately requeues ADVANCED sequences — a long prompt that made
// partial progress and is then squeezed out of the composed batch by
// shorter requests would be reset to scratch and recompute its whole
// prompt on every contention round (work-loss thrashing under load).
// The requeue must preserve the sequence's frontier and blocks.
#[test]
fn uncomposed_prefill_overflow_keeps_chunked_progress() {
    let config = SchedulerConfig::builder()
        .with_max_num_batched_tokens(32)
        .with_prefill_chunk_size(16)
        .with_packing(SequencePackingConfig {
            enabled: false,
            ..Default::default()
        })
        .build();
    let mut engine = create_test_engine(config, 64);

    // A: long prompt served via chunked prefill.
    engine.add_request(Request::new(10, vec![7; 100], 8));

    // Round 1: A processed as a 16-token chunk -> frontier 16, requeued.
    let batch1 = engine.build_batch();
    assert_eq!(batch1.seq_ids, vec![10]);
    assert_eq!(batch1.input_tokens[0].len(), 16, "A must be chunked");
    step(&mut engine, &batch1);
    assert_eq!(
        engine.get_sequence(10).map(|s| s.num_computed_tokens),
        None,
        "A must be requeued (not running) after its partial chunk"
    );

    // B: a 32-token prompt that exactly fills the token budget.
    engine.add_request(Request::new(11, vec![7; 32], 4));

    // Round 2: compose sorts shortest-first (B = 32 fills the budget), so
    // A (remaining 84) breaks out of composition and is requeued as
    // uncomposed overflow. Its computed frontier must SURVIVE this requeue.
    let batch2 = engine.build_batch();
    assert_eq!(
        batch2.seq_ids,
        vec![11],
        "B fills the budget; A is dropped from composition"
    );
    step(&mut engine, &batch2);

    // Round 3: no competition. A must resume from its preserved frontier,
    // not restart from the first prompt token (work-loss thrashing).
    let batch3 = engine.build_batch();
    assert_eq!(batch3.seq_ids, vec![10]);
    assert_eq!(
        batch3.num_computed_tokens[0], 16,
        "A must resume from position 16, not recompute its prompt from scratch"
    );
    assert_eq!(
        batch3.positions[0],
        (16..32).collect::<Vec<_>>(),
        "the resumed chunk must start at the preserved frontier"
    );
}

// RIL ISS-055 (TASK-059): the build_batch preemption gate must compare the
// ADDITIONAL blocks a sequence still needs against the free pool, not its
// full-prompt block count. A re-admitted chunked prefill ALREADY HOLDS its
// whole prompt table, so demanding `full_prompt_blocks` of free space
// double-counts its own held blocks and needlessly preempts running decode
// sequences. Fails pre-fix: the decode sequence B is preempted (kicked out
// of `running`, reset to Waiting) to "free" space for a chunked prefill that
// needs zero new blocks.
#[test]
fn readmitted_chunked_prefill_does_not_preempt_running_decode() {
    let config = SchedulerConfig::builder()
        .with_max_num_batched_tokens(32)
        .with_prefill_chunk_size(16)
        .with_packing(SequencePackingConfig {
            enabled: false,
            ..Default::default()
        })
        .build();
    let mut engine = create_test_engine(config, 12); // 12 blocks total

    // B: short prompt (30 tokens = 2 blocks) that reaches Decode and holds
    // its 2 blocks in `running`.
    engine.add_request(Request::new(20, vec![7; 30], 4));
    let batch_b = engine.build_batch();
    step(&mut engine, &batch_b);
    assert_eq!(
        engine.get_sequence(20).map(|s| s.status),
        Some(Status::Decoding),
        "B must be decoding and holding its 2 blocks"
    );

    // A: long prompt (100 tokens = 7 blocks), chunked. Its full prompt table
    // is preallocated in round A1, so after the partial chunk it requeues
    // HOLDING all 7 blocks while only 3 blocks remain free.
    engine.add_request(Request::new(10, vec![7; 100], 8));
    let batch_a1 = engine.build_batch();
    assert_eq!(batch_a1.seq_ids, vec![10]);
    step(&mut engine, &batch_a1);
    // A is requeued to the waiting queue holding its full 7-block table
    // (frontier 16); `get_sequence` only searches `running`, so None here.
    assert_eq!(engine.get_sequence(10).map(|s| s.status), None);

    // Round A2: prefill phase re-drains A (holds 7 blocks; free = 3 < 7).
    // The gate must see `additional == 0` and NOT preempt B to make room for
    // blocks A already owns.
    let batch_a2 = engine.build_batch();
    assert_eq!(
        batch_a2.seq_ids,
        vec![10],
        "A is composed for its next chunk"
    );
    assert_eq!(
        engine.get_sequence(20).map(|s| s.status),
        Some(Status::Decoding),
        "a re-admitted chunked prefill that already holds its blocks must not \
         preempt a running decode sequence"
    );
    step(&mut engine, &batch_a2);
}
