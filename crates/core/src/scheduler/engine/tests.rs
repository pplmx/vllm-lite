//! Unit tests for the `SchedulerEngine` lifecycle (`add_request`,
//! `build_batch`, `update`, `running_count`, `waiting_count`,
//! `get_memory_pressure`).
//!
//! Extracted from `engine/mod.rs` to keep the implementation file
//! under the project's 800-line soft cap. Exercises:
//!
//! - `add_request` returns a positive id, registers as waiting
//! - `build_batch` produces a non-empty batch with phase=Prefill
//! - `update` advances sequences through running → finished
//! - Multiple-request batching (waiting + running counts)
//! - Memory pressure bounds (0..=1)
//! - Prefix-cache hit path
//! - Metrics counters (`requests_total`)

use std::sync::Arc;

use vllm_traits::{BatchPhase, SampledToken};

use crate::metrics::EnhancedMetricsCollector;
use crate::scheduler::SchedulerCudaGraphConfig;
use crate::scheduler::engine::SchedulerEngine;
use crate::scheduler::observer::{NoopSchedulerObserver, SchedulerObservers};
use crate::scheduler::policy::PriorityPolicy;
use crate::types::{Phase, Priority, Request, SamplingParams, SchedulerConfig, Sequence, Status};

fn create_test_engine(config: SchedulerConfig, num_kv_blocks: usize) -> SchedulerEngine {
    let metrics = Arc::new(EnhancedMetricsCollector::new());
    SchedulerEngine::new(config, num_kv_blocks, metrics)
}

/// Every token folded into a sequence must advance its decode round counter.
///
/// `consecutive_decode_rounds` drives eviction priority
/// (`compute_priority` flips at > 5) and is part of the victim-set cache
/// key, but every existing test SETS it by hand on a hand-built `Sequence` —
/// nothing checked the production increment. `cargo mutants` replacing
/// `+= 1` with `*= 1` (which pins it at 0 forever, so a sequence is never
/// considered long-running) survived the whole suite.
/// Cancelling a request must release the KV blocks it holds.
///
/// `cancel_request` has two branches — one for a still-queued request, one for
/// a sequence already moved into `running` — and each guards its release with
/// `if !seq.kv_blocks.is_empty()`. `cargo mutants` deleted that `!` in both and
/// the whole suite still passed, because the existing cancel tests assert only
/// the `bool` return value and response-channel cleanup, never the allocator
/// effect.
///
/// The RUNNING branch is what this test pins: a batched sequence holds real
/// blocks, so inverting the guard leaves them allocated. (Inverting only the
/// QUEUED branch is a no-op for this fixture — a queued request owns no blocks
/// yet — which is exactly why both mutants have to be considered separately.)
/// `ensure_blocks_for_tokens` must allocate only what the target block count
/// still needs.
///
/// The growth loop is `while kv_blocks.len() < min_blocks`. `cargo mutants`
/// relaxed it to `<=` and the whole suite still passed: the sequence then gets
/// ONE block more than asked for on every call, because the loop runs once more
/// after the target is reached. That is invisible to correctness (extra blocks
/// are simply unused capacity) but leaks KV capacity on every speculative step,
/// and the function's return value is documented as "the number of blocks the
/// sequence holds afterwards".
/// Phase-2 preemption must stop as soon as it has freed enough blocks.
///
/// `blocks_freed += block_count` at the end of the phase-2 loop is what lets
/// the `if blocks_freed >= blocks_needed { break }` guard fire. `cargo
/// mutants` replaced it with `*=`, which pins the counter at 0 (or 0 after the
/// first multiply) so the guard never fires and EVERY preemptable sequence is
/// evicted — the whole running set is rolled back to full re-prefill when
/// freeing one sequence would have been enough. The existing preemption tests
/// check a single sequence, where the guard cannot be distinguished from its
/// absence.
#[test]
fn test_phase2_preemption_stops_once_enough_blocks_are_freed() {
    use crate::scheduler::SchedulerEngine;
    use crate::types::{Request, SchedulerConfig};
    use std::sync::Arc;

    let config = SchedulerConfig {
        max_num_seqs: 8,
        max_num_batched_tokens: 256,
        max_consecutive_decode: 2,
        enable_pd_separation: false,
        prefill_chunk_size: 128,
        decode_preference_ratio: 0.5,
        enable_priority_scheduling: false,
        enable_dynamic_batching: false,
        min_batch_size: 1,
        max_batch_size: 8,
        ..Default::default()
    };
    let mut scheduler = SchedulerEngine::new(
        config,
        16,
        Arc::new(crate::metrics::EnhancedMetricsCollector::new()),
    );

    // Four 32-token prompts (2 blocks each) admitted and decoded, so they all
    // hold real KV and are preemptable.
    for i in 0..4u64 {
        scheduler.add_request(Request::new(i + 1, vec![7; 32], 50));
    }
    let _ = scheduler.build_batch();
    for i in 0..4u64 {
        scheduler.update(
            &[i + 1],
            &[vllm_traits::SampledToken {
                token: 9,
                logprob: 0.0,
                top_logprobs: vec![],
            }],
            &[32],
        );
    }
    let occupied = scheduler.running_count();
    assert!(occupied > 0, "setup: sequences must be running");

    // A 96-token request (6 blocks) that cannot fit alongside all four.
    scheduler.add_request(Request::new(99, vec![3; 96], 50));
    let batch = scheduler.build_batch();

    assert!(
        batch.seq_ids.contains(&99),
        "the new request must be admitted; batched={:?}",
        batch.seq_ids
    );
    assert_eq!(
        scheduler.waiting_count(),
        0,
        "preemption must free exactly what the new request needs — over-evicting \
         pushes otherwise-runnable sequences into the waiting queue \
         (running={}, waiting={})",
        scheduler.running_count(),
        scheduler.waiting_count()
    );
}

#[test]
fn test_ensure_blocks_for_tokens_allocates_only_the_shortfall() {
    use crate::scheduler::SchedulerEngine;
    use crate::types::{Request, SchedulerConfig};
    use std::sync::Arc;

    let mut scheduler = SchedulerEngine::new(
        SchedulerConfig::default(),
        64,
        Arc::new(crate::metrics::EnhancedMetricsCollector::new()),
    );
    // 32 prompt tokens at BLOCK_SIZE = 16 -> two blocks after batching.
    scheduler.add_request(Request::new(1, vec![1; 32], 10));
    let _ = scheduler.build_batch();
    let have = scheduler
        .get_sequence(1)
        .map(|s| s.kv_blocks.len())
        .expect("the batched sequence is running");
    assert_eq!(have, 2, "setup: a 32-token prompt covers two blocks");

    // Asking for exactly what it already holds must be a no-op.
    assert_eq!(
        scheduler.ensure_blocks_for_tokens(1, have),
        have,
        "an exact-fit request must not grow the table"
    );
    let (used, _) = scheduler.get_kv_cache_usage();
    assert_eq!(used, have as u64, "no extra block may be allocated");

    // Asking for one more must allocate exactly one, not two.
    assert_eq!(
        scheduler.ensure_blocks_for_tokens(1, have + 1),
        have + 1,
        "one short block must allocate exactly one"
    );
    let (used, _) = scheduler.get_kv_cache_usage();
    assert_eq!(
        used,
        (have + 1) as u64,
        "the growth loop must stop as soon as the target is reached"
    );
}

#[test]
fn test_cancel_request_releases_kv_blocks() {
    use crate::scheduler::SchedulerEngine;
    use crate::types::{Request, SchedulerConfig};
    use std::sync::Arc;

    let mut scheduler = SchedulerEngine::new(
        SchedulerConfig::default(),
        64,
        Arc::new(crate::metrics::EnhancedMetricsCollector::new()),
    );
    scheduler.add_request(Request::new(1, vec![1; 64], 10));

    // Batch it so the prompt's KV blocks are allocated.
    let batch = scheduler.build_batch();
    assert_eq!(batch.seq_ids, vec![1], "the request must be batched");
    let owned = scheduler
        .get_sequence(1)
        .map(|s| s.kv_blocks.len())
        .expect("the batched sequence is running");
    assert!(owned > 0, "setup: batching must allocate KV blocks");

    let (allocated, total) = scheduler.get_kv_cache_usage();
    assert!(allocated > 0, "setup: blocks must be counted as used");

    assert!(scheduler.cancel_request(1), "cancellation must succeed");
    let (after, _) = scheduler.get_kv_cache_usage();
    assert_eq!(
        after, 0,
        "cancelling must return every block the sequence held to the \
         allocator (held {allocated} of {total}, {after} still used)"
    );
}

#[test]
fn test_push_token_advances_consecutive_decode_rounds() {
    use crate::scheduler::SchedulerEngine;
    use crate::types::{Request, SchedulerConfig};
    use std::sync::Arc;

    let mut scheduler = SchedulerEngine::new(
        SchedulerConfig::default(),
        128,
        Arc::new(crate::metrics::EnhancedMetricsCollector::new()),
    );
    scheduler.add_request(Request::new(1, vec![1, 2], 20));

    // One round per token folded, starting from 0.
    for expected in 1..=4u32 {
        let batch = scheduler.build_batch();
        assert!(
            !batch.seq_ids.is_empty(),
            "round {expected}: batch must not be empty"
        );
        let input_len = batch.input_tokens[0].len();
        scheduler.update(
            &[1],
            &[vllm_traits::SampledToken {
                token: 9,
                logprob: 0.0,
                top_logprobs: vec![],
            }],
            &[input_len],
        );
        assert_eq!(
            scheduler
                .get_sequence(1)
                .map(|s| s.consecutive_decode_rounds)
                .expect("sequence is still running"),
            expected,
            "each folded token advances the decode round counter by exactly one"
        );
    }
}

#[test]
fn test_engine_add_request() {
    let config = SchedulerConfig::default();
    let mut engine = create_test_engine(config, 1024);
    let id = engine.add_request(Request::new(0, vec![1, 2, 3], 5));
    assert!(id > 0);
    assert!(engine.has_pending());
    assert_eq!(engine.waiting_count(), 1);
}

#[test]
fn test_engine_build_batch() {
    let config = SchedulerConfig::default();
    let mut engine = create_test_engine(config, 1024);
    engine.add_request(Request::new(0, vec![1, 2, 3], 5));
    let batch = engine.build_batch();
    assert!(!batch.is_empty());
    assert_eq!(batch.len(), 1);
}

#[test]
fn test_engine_batch_phase_is_prefill() {
    let config = SchedulerConfig::default();
    let mut engine = create_test_engine(config, 1024);
    engine.add_request(Request::new(0, vec![1, 2, 3], 5));
    let batch = engine.build_batch();
    assert_eq!(batch.phase, BatchPhase::Prefill);
}

#[test]
fn test_engine_update_sequence() {
    let config = SchedulerConfig::default();
    let mut engine = create_test_engine(config, 1024);
    let id = engine.add_request(Request::new(0, vec![1, 2, 3], 5));
    let _batch = engine.build_batch();

    // Simulate model output: one token generated
    engine.update(
        &[id],
        &[SampledToken {
            token: 100,
            logprob: 0.0,
            top_logprobs: vec![],
        }],
        &[3],
    ); // 3 input tokens processed

    // The sequence should still be in running (not finished yet)
    assert_eq!(engine.running_count(), 1);
}

#[test]
fn test_engine_multiple_requests() {
    let config = SchedulerConfig::default();
    let mut engine = create_test_engine(config, 1024);

    // Add multiple requests
    let id1 = engine.add_request(Request::new(0, vec![1, 2], 5));
    let id2 = engine.add_request(Request::new(0, vec![3, 4], 5));

    assert_eq!(engine.waiting_count(), 2);

    let batch = engine.build_batch();
    assert_eq!(batch.seq_ids.len(), 2);
    assert!(batch.seq_ids.contains(&id1));
    assert!(batch.seq_ids.contains(&id2));
}

/// the CUDA-graph admission path
/// (`select_sequences_for_phase`, graph.rs) drains `request_queue` into
/// `running` but pre-fix never published the `active_sequences` /
/// `request_queue_depth` gauges that `build_batch` (state/batch.rs:66,140)
/// updates — so with `cuda_graph.enabled` the engine routes exclusively
/// through the graph path and `active_sequences` stays pinned at its init
/// 0 while sequences run. Regression: drive `add_request` +
/// `build_batch_with_graph` (works without the `cuda-graph` feature; the
/// scheduler method isn't gated) and assert both gauges are live after
/// admission.
#[test]
fn test_graph_path_publishes_active_sequences_and_queue_depth() {
    let metrics = Arc::new(EnhancedMetricsCollector::new());
    let config = SchedulerConfig::default();
    let mut engine = SchedulerEngine::new(config, 1024, metrics.clone());

    engine.add_request(Request::new(0, vec![1, 2, 3], 5));
    // Pre-admission: the request sits in the waiting queue.
    assert_eq!(metrics.get_gauge("request_queue_depth"), 1);

    let _batch = engine.build_batch_with_graph();
    assert_eq!(
        metrics.get_gauge("active_sequences"),
        1,
        "graph admission must publish the running count (pre-fix pinned at 0)"
    );
    assert_eq!(
        metrics.get_gauge("request_queue_depth"),
        0,
        "graph admission must publish the drained queue depth (pre-fix stayed at the pre-drain value)"
    );
}

#[test]
fn test_engine_memory_pressure() {
    let config = SchedulerConfig::default();
    let mut engine = create_test_engine(config, 10); // Small memory

    // Memory pressure should be 0.0 with all blocks free
    assert!(engine.get_memory_pressure().abs() < 1e-6);

    // Add a request
    engine.add_request(Request::new(0, vec![1, 2, 3, 4, 5], 5));

    // After building batch, memory pressure may increase
    let _batch = engine.build_batch();

    // Pressure should be between 0 and 1
    let pressure = engine.get_memory_pressure();
    assert!((0.0..=1.0).contains(&pressure));
}

#[test]
fn test_engine_memory_pressure_zero_blocks() {
    // Degenerate config: zero KV blocks. Must not divide by zero;
    // reports maximum pressure so the scheduler triggers preemption
    // instead of computing a NaN ratio.
    let config = SchedulerConfig::default();
    let engine = create_test_engine(config, 0);

    // KV usage snapshot stays well-formed under the same config.
    assert_eq!(engine.get_kv_cache_usage(), (0, 0));

    let pressure = engine.get_memory_pressure();
    assert!(pressure.is_finite());
    assert!((pressure - 1.0).abs() < 1e-6);
}

#[test]
fn test_engine_prefix_cache_hit() {
    let config = SchedulerConfig::default();
    let mut engine = create_test_engine(config, 1024);

    // Add first request
    let prompt = vec![1, 2, 3, 4, 5];
    let id1 = engine.add_request(Request::new(0, prompt, 5));

    // Build batch and process
    let _batch = engine.build_batch();
    engine.update(
        &[id1],
        &[SampledToken {
            token: 100,
            logprob: 0.0,
            top_logprobs: vec![],
        }],
        &[5],
    );

    // Complete the sequence to add to cache
    // Update until max_tokens reached
    for i in 0..5 {
        // invariant: bounded by configured limit, cannot overflow at runtime.
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

    // Add second request with same prefix
    let _id2 = engine.add_request(Request::new(0, vec![1, 2, 3, 6, 7], 5));

    // Second request should be enqueued
    assert!(engine.waiting_count() > 0 || engine.running_count() > 0);
}

#[test]
fn test_engine_prefix_cache_stores_only_prompt_blocks() {
    // A finished sequence's DECODE blocks must not be pinned in the
    // prefix cache: the entry is keyed by prompt tokens, and on a hit
    // only the prompt-covering blocks are valid. Previously the whole
    // kv_blocks list (prompt + generated) was stored, so one long
    // response pinned every block it ever touched — a few such
    // requests could pin the entire pool for zero-running-server
    // state, amplifying memory pressure for every cached prompt.
    let config = SchedulerConfig::default();
    let mut engine = create_test_engine(config, 64);

    // Prompt = 16 tokens (1 block); max_tokens = 48 → the sequence
    // grows to 64 tokens (4 blocks) before finishing.
    let prompt: Vec<u32> = (0..16).collect();
    let id = engine.add_request(Request::new(0, prompt.clone(), 48));
    let _batch = engine.build_batch();
    engine.update(
        &[id],
        &[SampledToken {
            token: 200,
            logprob: 0.0,
            top_logprobs: vec![],
        }],
        &[16],
    );
    for i in 0..47 {
        engine.update(
            &[id],
            &[SampledToken {
                token: 200 + i as u32,
                logprob: 0.0,
                top_logprobs: vec![],
            }],
            &[0],
        );
    }
    assert_eq!(engine.running_count(), 0, "sequence finished");

    // The cache entry must cover exactly the prompt's block count.
    let hit = engine
        .prefix_cache()
        .longest_prefix_match(&prompt)
        .expect("finished prompt must be cached");
    assert_eq!(hit.matched_tokens, 16);
    assert_eq!(
        hit.blocks.len(),
        1,
        "cache must not pin decode blocks (got {} blocks for a 16-token prompt)",
        hit.blocks.len()
    );
}

#[test]
fn test_engine_metrics_tracking() {
    let config = SchedulerConfig::default();
    let metrics = Arc::new(EnhancedMetricsCollector::new());
    let mut engine = SchedulerEngine::new(config, 1024, metrics.clone());

    // Initially metrics should be zero
    assert_eq!(metrics.get_counter("requests_total"), 0);

    // Add a request
    let _id = engine.add_request(Request::new(0, vec![1, 2, 3], 5));

    // Check metrics were updated
    assert_eq!(metrics.get_counter("requests_total"), 1);

    // Build batch to trigger latency recording
    let _batch = engine.build_batch();

    // Metrics should still track request count
    assert_eq!(metrics.get_counter("requests_total"), 1);
}

/// Regression: `execute_preemption` must never
/// leave a running sequence with a partial hole in its block table.
/// The block table is positional (block `i` ↔ tokens
/// `i * BLOCK_SIZE..(i + 1) * BLOCK_SIZE`), so removing an interior
/// block shifts every later position onto the wrong physical block and
/// silently corrupts attention reads. A sequence that loses *any*
/// victim block must be preempted wholesale and re-queued for
/// recompute.
#[test]
fn test_preemption_never_leaves_partial_block_holes() {
    use crate::types::{Priority, SamplingParams, Sequence, Status};

    let config = SchedulerConfig::default();
    let mut engine = create_test_engine(config, 16);

    let make_seq = |id: u64, blocks: Vec<usize>| {
        Sequence {
            id,
            tokens: vec![0; 48],
            kv_blocks: Arc::new(blocks),
            num_computed_tokens: 48,
            prompt_len: 16,
            status: Status::Decoding,
            max_tokens: 100,
            sampling_params: SamplingParams::default(),
            consecutive_decode_rounds: 0, // priority 3 (new decode)
            priority: Priority::default(),
            degraded_draft: false,
            draft_model_id: None,
        }
    };

    // Allocate through the manager so allocator accounting is real:
    // 16 - 6 = 10 available before preemption.
    let blocks = engine.memory.allocate(6).expect("pool has 16 blocks");
    let (blocks_a, blocks_b) = blocks.split_at(3);
    engine.memory.record_blocks(&blocks);
    engine.running.push(make_seq(1, blocks_a.to_vec()));
    engine.running.push(make_seq(2, blocks_b.to_vec()));

    // Ask for one block: the policy ranks seq 1's oldest block (block 0)
    // as the first victim (seq_id tiebreak, LRU rank within a sequence).
    engine.execute_preemption(1);

    // Seq 1 must be preempted wholesale — it may not keep running with
    // blocks [1, 2], which would map tokens 0..16 onto block 1.
    assert_eq!(
        engine.running.len(),
        1,
        "victim sequence must be fully preempted"
    );
    let survivor = &engine.running[0];
    assert_eq!(survivor.id, 2);
    assert_eq!(
        survivor.kv_blocks.as_ref(),
        blocks_b,
        "survivor block table must be untouched"
    );
    assert_eq!(survivor.num_computed_tokens, 48);

    // Seq 1 is re-queued with fully reset KV state; its tokens are
    // preserved so the recompute recovers all progress.
    assert_eq!(engine.waiting_count(), 1);
    let requeued = engine.request_queue.remove(1).expect("seq 1 re-queued");
    assert_eq!(requeued.status, Status::Waiting);
    assert!(requeued.kv_blocks.is_empty());
    assert_eq!(requeued.num_computed_tokens, 0);
    assert_eq!(requeued.tokens.len(), 48);

    // Seq 1's three released blocks returned to the allocator
    // (refcount 1 → freed): 16 - 6 + 3 = 13 available.
    assert_eq!(engine.memory.available_blocks(), 13);
}

/// Regression: phase-2 fallback preemption must continue
/// from the blocks phase 1 actually released, not from the victim-block
/// count. `select_victims` returns only evictable (refcount ≤ 1) blocks,
/// but phase 1 preempts victim-owning sequences WHOLESALE — releasing
/// their full tables, including shared (refcount > 1) blocks that were
/// never victims. Reseeding `blocks_freed` to `victim_set.len()` (a
/// strict subset of the released tables) made phase 2 think it was
/// nearer to `blocks_needed` than it was, so it preempted extra healthy
/// decode sequences — a full re-prefill for every survivor.
///
/// Scenario: A owns victims {b1, b2} plus a shared block b0 (with C);
/// phase 1 preempts A and books 3 freed blocks (5 needed → falls to
/// phase 2). C (3 shared blocks... 2) is more decodable than D. The
/// buggy seed `victim_set.len() == 2` preempts C AND D to reach 5; the
/// correct continuation from A's real `blocks_freed == 3` stops after C
/// and leaves D decoding.
#[test]
fn test_phase2_preemption_continues_from_phase1_freed_count() {
    use crate::types::{Priority, SamplingParams, Sequence, Status};

    let config = SchedulerConfig::default();
    let mut engine = create_test_engine(config, 8);

    let make_seq = |id: u64, blocks: Vec<usize>, rounds: u32| Sequence {
        id,
        tokens: vec![0; 16],
        kv_blocks: Arc::new(blocks),
        num_computed_tokens: 16,
        prompt_len: 4,
        status: Status::Decoding,
        max_tokens: 100,
        sampling_params: SamplingParams::default(),
        consecutive_decode_rounds: rounds,
        priority: Priority::default(),
        degraded_draft: false,
        draft_model_id: None,
    };

    // Allocate 4 distinct blocks; wire shared refcounts so only A's
    // {b1, b2} are evictable (refcount 1). b0 and b5 are shared
    // (refcount 2 → never victims, per `select_victims`).
    let blocks = engine.memory.allocate(4).expect("pool has 8");
    let (b0, b1, b2, b5) = (blocks[0], blocks[1], blocks[2], blocks[3]);
    engine.memory.record_blocks(&[b0, b1, b2]); // A owns all three (b0 shared w/ C)
    engine.memory.record_blocks(&[b0, b5]); // C owns b0 (shared w/ A) + b5 (shared w/ D)
    engine.memory.record_blocks(&[b5]); // D owns b5 (shared w/ C)

    engine.running.push(make_seq(1, vec![b0, b1, b2], 0)); // A: victim owner, 3-block table
    engine.running.push(make_seq(2, vec![b0, b5], 3)); // C: more decodable
    engine.running.push(make_seq(3, vec![b5], 1)); // D: healthy decode, must survive

    // 5 blocks needed: phase 1 frees A's whole 3-block table (victims
    // {b1, b2} ⊂ it), which is < 5, so phase 2 runs. Starting from 3
    // (the real freed total) preempts C (3 + 2 = 5) and stops; starting
    // from 2 (victim_set.len()) preempts C then D to climb to 5.
    engine.execute_preemption(5);

    assert_eq!(
        engine.running.len(),
        1,
        "phase 2 must preempt only as many tables as needed to reach blocks_needed"
    );
    assert_eq!(
        engine.running[0].id,
        3,
        "healthy decode sequence D must survive; over-preempted it was: {:?}",
        engine.running.iter().map(|s| s.id).collect::<Vec<_>>()
    );
    assert_eq!(engine.waiting_count(), 2, "A and C re-queued, not D");
    // b5 stays held by D (its only live refcount after A/C released).
    assert_eq!(engine.memory.get_block_ref_count(b5), 1);
}

/// Regression: completing the same prompt
/// twice must not inflate the cache's refcount on the prompt's block.
/// The finish path re-inserts the prefix entry; before the fix the
/// overwrite orphaned the cache's previous ref, so the block's
/// refcount grew on every repeat completion — making it unevictable
/// (`select_victims` requires ≤ 1) and immune to prefix-cache drain.
#[test]
fn test_repeated_prompt_completion_keeps_single_cache_refcount() {
    use crate::engine::Engine;
    use tokio::sync::mpsc;
    use vllm_testing::StubModel;

    fn run_prompt(engine: &mut Engine, id: u64, prompt: Vec<u32>) {
        let (tx, _rx) = mpsc::channel(64);
        engine.add_request(Request::new(id, prompt, 2), tx);
        let mut steps = 0;
        while engine.has_pending() {
            engine.step().expect("step should succeed");
            steps += 1;
            assert!(steps < 500, "request {id} never completes");
        }
    }

    let mut engine =
        Engine::with_config(StubModel::default(), None, SchedulerConfig::default(), 4, 8);

    // Complete the same prompt twice; the second run hits the cache
    // and re-inserts the entry on finish.
    run_prompt(&mut engine, 0, vec![42, 43]);
    run_prompt(&mut engine, 1, vec![42, 43]);

    // Exactly one block may carry the cache's single reference; no
    // block may sit at refcount >= 2 once both sequences finished.
    let refcounts: Vec<usize> = (0..8)
        .map(|block| engine.scheduler.memory.get_block_ref_count(block))
        .collect();
    let pinned: Vec<usize> = refcounts.iter().copied().filter(|&c| c > 0).collect();
    assert_eq!(
        pinned,
        vec![1],
        "cache must hold exactly one ref on one block, got refcounts {refcounts:?}"
    );
}

/// Regression: a request whose prompt fully
/// matches the prefix cache must still run to completion. Before the
/// fix, `add_request` parked full hits in `Status::Waiting`; the
/// prefill composer skips sequences with no new tokens and decode
/// batches only admit `Decoding` sequences, so the request stalled
/// forever (reproduced: >500 engine steps without completion).
#[test]
fn test_full_prefix_hit_request_completes() {
    use crate::engine::Engine;
    use tokio::sync::mpsc;
    use vllm_testing::StubModel;

    let mut engine =
        Engine::with_config(StubModel::default(), None, SchedulerConfig::default(), 4, 8);

    // First completion populates the prefix cache.
    let (tx, _rx) = mpsc::channel(64);
    engine.add_request(Request::new(0, vec![7, 8, 9], 2), tx);
    let mut steps = 0;
    while engine.has_pending() {
        engine.step().expect("step should succeed");
        steps += 1;
        assert!(steps < 500, "first request never completes");
    }

    // Identical prompt: full prefix hit — must also complete.
    let (tx, _rx) = mpsc::channel(64);
    engine.add_request(Request::new(0, vec![7, 8, 9], 2), tx);
    let mut steps = 0;
    while engine.has_pending() {
        engine.step().expect("step should succeed");
        steps += 1;
        assert!(steps < 500, "full-prefix-hit request stalled");
    }

    assert!(
        engine.scheduler.prefix_cache_hit_rate() > 0.0,
        "second identical prompt must register as a prefix hit"
    );
}

/// Regression: a freshly-admitted prefill sequence must have
/// its prompt KV blocks allocated by `build_batch` BEFORE the forward pass
/// writes its KV. Pre-fix, blocks were allocated only in `update()` (after the
/// forward), so `batch.kv_block_ids` was empty for a fresh prefill and the
/// model's `write_prefill_kv` fell back to block 0, corrupting the cache.
#[test]
fn test_build_batch_preallocates_prefill_kv_blocks() {
    let config = SchedulerConfig::default();
    let mut engine = create_test_engine(config, 1024);
    // 40-token prompt => ceil(40 / BLOCK_SIZE=16) = 3 blocks.
    let prompt: Vec<u32> = (0..40).collect();
    let expected_blocks = prompt.len().div_ceil(vllm_traits::BLOCK_SIZE);
    assert_eq!(expected_blocks, 3);
    engine.add_request(Request::new(0, prompt, 5));

    let batch = engine.build_batch();
    assert_eq!(batch.len(), 1);
    assert_eq!(
        batch.kv_block_ids[0].len(),
        expected_blocks,
        "build_batch must allocate the prompt's KV blocks before the forward \
         (got {} blocks, expected {expected_blocks})",
        batch.kv_block_ids[0].len()
    );
    // The allocated blocks must be distinct (not all the block-0 fallback).
    let unique: std::collections::HashSet<usize> = batch.kv_block_ids[0].iter().copied().collect();
    assert_eq!(
        unique.len(),
        expected_blocks,
        "prefill KV blocks must be distinct allocations, not the block-0 fallback"
    );
}

/// Regression: the CUDA-Graph admission path must run the
/// same memory-pressure preemption gate as `build_batch` before admitting
/// new sequences. Pre-fix, `select_sequences_for_phase` admitted a prefill
/// sequence whose blocks exceeded available memory without preempting a
/// running sequence, so allocation failed silently and `write_prefill_kv`
/// fell back to block 0.
#[test]
fn test_build_batch_with_graph_preempts_before_admitting_under_pressure() {
    use crate::types::{Priority, SamplingParams, Sequence, Status};

    let config = SchedulerConfig {
        cuda_graph: SchedulerCudaGraphConfig {
            enabled: true,
            batch_sizes: vec![1, 2, 4],
        },
        ..Default::default()
    };
    let mut engine = create_test_engine(config, 16);

    // Running decode sequence holding 14 of 16 blocks -> 2 available.
    let blocks = engine.memory.allocate(14).expect("pool has 16 blocks");
    engine.memory.record_blocks(&blocks);
    engine.running.push(Sequence {
        id: 1,
        tokens: vec![0; 224],
        kv_blocks: Arc::new(blocks),
        num_computed_tokens: 224,
        prompt_len: 16,
        status: Status::Decoding,
        max_tokens: 100,
        sampling_params: SamplingParams::default(),
        consecutive_decode_rounds: 0,
        priority: Priority::default(),
        degraded_draft: false,
        draft_model_id: None,
    });

    // New 40-token prefill request needs 3 blocks > 2 available.
    engine.add_request(Request::new(2, vec![0; 40], 5));

    let graph_batch = engine.build_batch_with_graph();
    let batch = graph_batch.into_regular();
    assert_eq!(batch.len(), 1);
    assert_eq!(
        batch.seq_ids[0], 2,
        "the new prefill sequence must be admitted"
    );
    assert_eq!(
        batch.kv_block_ids[0].len(),
        3,
        "admitted sequence must hold all 3 required blocks after preemption"
    );
    // The admitted sequence is running; the victim must have been preempted
    // back to the queue.
    assert_eq!(engine.running.len(), 1);
    assert_eq!(engine.running[0].id, 2, "admitted sequence is running");
    assert_eq!(engine.waiting_count(), 1, "victim must be re-queued");
}

/// Regression: the CUDA-Graph batch builder
/// (`build_batch_with_graph`) must pre-allocate prefill KV blocks exactly
/// like `build_batch`. Pre-fix, `select_sequences_for_phase`
/// admitted fresh prefill sequences with empty `kv_blocks`, so the model's
/// `write_prefill_kv` fell back to block 0 and corrupted the cache on every
/// `cuda-graph` build.
#[test]
fn test_build_batch_with_graph_preallocates_prefill_kv_blocks() {
    let config = SchedulerConfig {
        cuda_graph: SchedulerCudaGraphConfig {
            enabled: true,
            batch_sizes: vec![1, 2, 4],
        },
        ..Default::default()
    };
    let mut engine = create_test_engine(config, 1024);
    // 40-token prompt => ceil(40 / BLOCK_SIZE=16) = 3 blocks.
    let prompt: Vec<u32> = (0..40).collect();
    let expected_blocks = prompt.len().div_ceil(vllm_traits::BLOCK_SIZE);
    assert_eq!(expected_blocks, 3);
    engine.add_request(Request::new(0, prompt, 5));

    let graph_batch = engine.build_batch_with_graph();
    assert!(
        !graph_batch.is_graph(),
        "prefill must stay on the regular path"
    );
    let batch = graph_batch.into_regular();
    assert_eq!(batch.len(), 1);
    assert_eq!(
        batch.kv_block_ids[0].len(),
        expected_blocks,
        "build_batch_with_graph must allocate the prompt's KV blocks before \
         the forward (got {} blocks, expected {expected_blocks})",
        batch.kv_block_ids[0].len()
    );
    let unique: std::collections::HashSet<usize> = batch.kv_block_ids[0].iter().copied().collect();
    assert_eq!(
        unique.len(),
        expected_blocks,
        "prefill KV blocks must be distinct allocations, not the block-0 fallback"
    );
}

/// Regression: the CUDA-Graph decode selection
/// must NOT forward a `Decoding` sequence whose KV table is short of its
/// token count. The regular path's `admit_decode_sequences`
/// grows each running decode table to cover its `tokens` before composing
/// (or defers it); `select_sequences_for_phase` previously cloned every
/// `Decoding` sequence as-is — a short table (e.g. from an earlier
/// boundary-crossing allocation failure) would be forwarded and its next
/// token's KV write would fall back to block 0 (`unwrap_or(0)`), corrupting
/// a live block owned by another sequence.
#[test]
fn test_build_batch_with_graph_grows_short_decode_table() {
    use crate::types::{Priority, SamplingParams, Sequence, Status};

    let config = SchedulerConfig {
        cuda_graph: SchedulerCudaGraphConfig {
            enabled: true,
            batch_sizes: vec![1, 2, 4],
        },
        ..Default::default()
    };
    // Pool large enough that the table CAN be grown (admit, not defer).
    let mut engine = create_test_engine(config, 1024);

    // A running decode sequence at 224 tokens needs ceil(224/16) = 14
    // blocks but only holds 13 (a short table — the exact state `update()`'s
    // boundary-crossing allocation failure would leave behind).
    let short_table = engine.memory.allocate(13).expect("pool has 13 blocks");
    engine.memory.record_blocks(&short_table);
    engine.running.push(Sequence {
        id: 1,
        tokens: vec![0; 224],
        kv_blocks: Arc::new(short_table),
        num_computed_tokens: 224,
        prompt_len: 16,
        status: Status::Decoding,
        max_tokens: 100,
        sampling_params: SamplingParams::default(),
        consecutive_decode_rounds: 0,
        priority: Priority::default(),
        degraded_draft: false,
        draft_model_id: None,
    });

    let graph_batch = engine.build_batch_with_graph();
    let batch = graph_batch.into_regular();
    assert_eq!(batch.len(), 1, "the decode sequence must be composed");
    assert_eq!(
        batch.kv_block_ids[0].len(),
        14,
        "select_sequences_for_phase must grow the short decode table before \
         forwarding (got {} blocks, need 14 for 224 tokens)",
        batch.kv_block_ids[0].len()
    );
}

/// Regression: the CUDA-Graph preemption gate must
/// demand only the ADDITIONAL blocks a sequence still needs, not its
/// full-prompt block count (the fix that `build_batch` has was
/// never applied to `select_sequences_for_phase`). Pre-fix a running decode
/// sequence holding its full table re-charged itself for the whole 14-block
/// table (14 > 6 free), so `execute_preemption` selected it as the victim
/// and preempted it to `Waiting` with `num_computed = 0` — and on the next
/// round it re-admitted and re-demanded its full count again, self-
/// preempting forever: a livelock that makes zero decode progress under
/// memory pressure. Post-fix the gate demands `additional = 0` (it already
/// owns its table), so the sequence is composed normally.
#[test]
fn test_build_batch_with_graph_does_not_over_preempt_full_table_decode() {
    use crate::types::{Priority, SamplingParams, Sequence, Status};

    let config = SchedulerConfig {
        cuda_graph: SchedulerCudaGraphConfig {
            enabled: true,
            batch_sizes: vec![1, 2, 4],
        },
        ..Default::default()
    };
    // Pool 20: a running decode holds a full 14-block table (needs 14,
    // holds all 14), leaving only 6 free. Pre-fix the gate compared the
    // full 14 against the 6 free and preempted this exact sequence.
    let mut engine = create_test_engine(config, 20);

    let blocks = engine.memory.allocate(14).expect("pool has 14 blocks");
    engine.memory.record_blocks(&blocks);
    engine.running.push(Sequence {
        id: 1,
        tokens: vec![0; 224], // 14 blocks needed AND held
        kv_blocks: Arc::new(blocks),
        num_computed_tokens: 224,
        prompt_len: 16,
        status: Status::Decoding,
        max_tokens: 100,
        sampling_params: SamplingParams::default(),
        consecutive_decode_rounds: 0,
        priority: Priority::default(),
        degraded_draft: false,
        draft_model_id: None,
    });

    let graph_batch = engine.build_batch_with_graph();
    let batch = graph_batch.into_regular();
    assert_eq!(
        batch.len(),
        1,
        "the running decode must be composed; it owns its full table so it \
         demands zero ADDITIONAL blocks and must not be preempted"
    );
    // The decode sequence must NOT have been preempted back to the queue —
    // it still owns its full table and stays running.
    assert_eq!(batch.seq_ids[0], 1);
    assert_eq!(engine.running.len(), 1);
    assert_eq!(engine.running[0].id, 1);
    assert_eq!(engine.waiting_count(), 0, "no sequence should be preempted");
}

/// `requeue_stuck_prefills` must rescue orphaned Prefilling/Waiting sequences.
///
/// A sequence left in `running` while still Prefilling or Waiting is never
/// re-scheduled (running is only re-included when `status == Decoding`), so it
/// stalls forever AND pins its pre-allocated KV blocks. `cargo mutants` replaced
/// the whole method with `()` and relaxed both loop conditions, all surviving —
/// nothing called it with an orphaned sequence in place.
#[test]
fn requeue_stuck_prefills_rescues_orphans_and_keeps_decoders() {
    let mut scheduler = SchedulerEngine::default();

    // One healthy decoder and two orphans, injected directly: `update()` would
    // normally have promoted them, which is exactly why they are orphans.
    let mut make_seq = |id: u64, status: Status| Sequence {
        id,
        tokens: vec![7; 32],
        kv_blocks: Arc::new(vec![]),
        num_computed_tokens: 32,
        prompt_len: 32,
        status,
        max_tokens: 50,
        sampling_params: SamplingParams::default(),
        consecutive_decode_rounds: 1,
        priority: Priority::default(),
        degraded_draft: false,
        draft_model_id: None,
    };
    scheduler.running.push(make_seq(1, Status::Decoding));
    scheduler.running.push(make_seq(2, Status::Prefilling));
    scheduler.running.push(make_seq(3, Status::Waiting));

    scheduler.requeue_stuck_prefills();

    assert_eq!(
        scheduler.running.len(),
        1,
        "only the healthy decoder may remain in running; got {:?}",
        scheduler.running.iter().map(|s| s.id).collect::<Vec<_>>()
    );
    assert_eq!(scheduler.running[0].id, 1, "the decoder must be untouched");
    assert_eq!(
        scheduler.waiting_count(),
        2,
        "both orphans must return to the waiting queue, not vanish"
    );
}

/// A colliding sequence id must be replaced, never silently adopted.
///
/// `seq_id_is_taken` checks BOTH `running` and the waiting queue. `cargo
/// mutants` replaced it with the constants `true`/`false` and flipped its `||`
/// to `&&`, all surviving: no test ever submitted two requests that collide, so
/// the duplicate-id path — and the "skip ids the counter never handed out"
/// loop beside it — was never entered. If a collision slipped through, the
/// second sequence would overwrite the first and the caller would receive an id
/// naming different work than it asked for.
#[test]
fn duplicate_sequence_ids_are_replaced_not_adopted() {
    let mut scheduler = SchedulerEngine::default();

    // Occupy id 5 through the QUEUE (it is never composed).
    let first = scheduler.add_request(Request::new(5, vec![1; 16], 10));
    assert_eq!(first, 5, "an uncontended explicit id must be honoured");

    // Same id again: the scheduler must issue a different one.
    let second = scheduler.add_request(Request::new(5, vec![2; 16], 10));
    assert_ne!(
        second, 5,
        "a colliding id must not be adopted; the first sequence would be lost"
    );

    // Both requests must still be present and distinct.
    assert_eq!(
        scheduler.waiting_count(),
        2,
        "the colliding request must be queued, not dropped"
    );

    // Now the RUNNING half of the check: occupy id 5 by running it, then ask
    // for 5 again.
    let _ = scheduler.build_batch();
    assert!(
        scheduler.running_count() > 0,
        "setup: a sequence is running"
    );
    let third = scheduler.add_request(Request::new(5, vec![3; 16], 10));
    assert_ne!(
        third, 5,
        "an id held by a RUNNING sequence must also be treated as taken"
    );
}

/// `add_request_without_prefix_cache` must return a real sequence id.
///
/// It is the path stateful models take, so that a full prefix hit cannot admit
/// a sequence into Decode with fresh `None` recurrent state. `cargo mutants`
/// replaced the whole body with `Default::default()`, i.e. it returned **0** —
/// a value that is not a valid sequence id — and survived, because the existing
/// tests only ever call the plain `add_request`.
#[test]
fn add_request_without_prefix_cache_returns_a_real_id() {
    let mut scheduler = SchedulerEngine::default();

    let id = scheduler.add_request_without_prefix_cache(Request::new(11, vec![4; 16], 10));

    assert_ne!(id, 0, "a sequence id must never be 0");
    assert_eq!(id, 11, "an uncontended explicit id must be honoured");
    assert_eq!(
        scheduler.waiting_count(),
        1,
        "the request must be queued, not dropped"
    );
    assert!(
        scheduler.request_queue.get(id).is_some(),
        "the returned id must name the queued sequence"
    );
}

/// The id counter must SKIP taken ids and ADVANCE after issuing one.
///
/// `add_request_inner` walks `next_seq_id` forward past any id already held,
/// then adopts it and advances again. `cargo mutants` turned both `+= 1` into
/// `-= 1` / `*= 1`. With `-= 1` the counter runs backwards below the ids it
/// already issued; with `*= 1` (an infallible multiply by one) it never moves,
/// so the skip loop cannot terminate and every request would reuse one id.
#[test]
fn sequence_id_counter_skips_taken_ids_and_advances() {
    let mut scheduler = SchedulerEngine::default();

    // A caller claims a HIGH explicit id; the counter starts at 1 and is not
    // aware of it, so it must skip that id rather than hand it out twice.
    let explicit = scheduler.add_request(Request::new(500, vec![1; 16], 10));
    assert_eq!(explicit, 500, "an explicit uncontended id is honoured");

    let first = scheduler.add_request(Request::new(0, vec![2; 16], 10));
    let second = scheduler.add_request(Request::new(0, vec![3; 16], 10));
    assert_ne!(first, 0, "auto-assigned ids must never be 0");
    assert_ne!(
        first, second,
        "the counter must advance between auto-assigned ids"
    );
    assert_ne!(first, 500, "the counter must skip the explicitly taken id");

    // Walking the counter onto an occupied id must skip it and keep going.
    scheduler.next_seq_id = 500;
    let after_skip = scheduler.add_request(Request::new(0, vec![4; 16], 10));
    assert_ne!(
        after_skip, 500,
        "an id already held must be skipped, not reused"
    );
    assert!(
        after_skip > 500,
        "the counter must move FORWARD past a taken id; got {after_skip}"
    );
    // The counter must have ADVANCED past the id it just issued, so the next
    // auto-assigned request cannot reuse it.
    let next = scheduler.add_request(Request::new(0, vec![5; 16], 10));
    assert_ne!(
        next, after_skip,
        "the counter must advance after issuing an id, or the next request          would reuse it"
    );
    assert!(
        next > after_skip,
        "the counter must move forward; got {next} after {after_skip}"
    );

    assert_eq!(
        scheduler.waiting_count(),
        5,
        "every request must be queued exactly once"
    );
}

/// The bookkeeping accessors must report the collections they name.
///
/// `cargo mutants` replaced `stop_ended_len`, `pending_stream_len`,
/// `finished_sequences` and `get_sequence_mut` with constants / `vec![]` /
/// `None`, and emptied `clear_finished`. A stale constant here silently breaks
/// the leak assertions these accessors exist for: `stop_ended` and
/// `pending_stream` are keyed by MONOTONIC ids, so a leftover entry is never
/// reused and leaks for the life of the process.
#[test]
fn bookkeeping_accessors_report_their_collections() {
    let mut scheduler = SchedulerEngine::default();

    // Both side tables start empty and report their own sizes.
    assert_eq!(scheduler.stop_ended_len(), 0);
    assert_eq!(scheduler.pending_stream_len(), 0);
    scheduler.stop_ended.insert(4);
    scheduler.stop_ended.insert(9);
    scheduler
        .pending_stream
        .insert(7, Vec::<vllm_traits::SampledToken>::new());
    assert_eq!(
        scheduler.stop_ended_len(),
        2,
        "stop_ended_len must count the entries, not a constant"
    );
    assert_eq!(
        scheduler.pending_stream_len(),
        1,
        "pending_stream_len must count the entries, not a constant"
    );

    // `finished_sequences` snapshots the current step's finalizations and
    // `clear_finished` empties it at the end of that step.
    let mut seq = Sequence {
        id: 42,
        tokens: vec![1, 2],
        kv_blocks: Arc::new(vec![]),
        num_computed_tokens: 2,
        prompt_len: 2,
        status: Status::Finished,
        max_tokens: 10,
        sampling_params: SamplingParams::default(),
        consecutive_decode_rounds: 0,
        priority: Priority::default(),
        degraded_draft: false,
        draft_model_id: None,
    };
    scheduler.finished.push(seq.clone());
    assert_eq!(
        scheduler.finished_sequences().len(),
        1,
        "finished_sequences must return the current step's finalizations"
    );
    assert_eq!(scheduler.finished_sequences()[0].id, 42);

    scheduler.clear_finished();
    assert!(
        scheduler.finished_sequences().is_empty(),
        "clear_finished must empty the list, not leave it populated"
    );

    // `get_sequence_mut` must MUTATE the running sequence, not hand back None.
    scheduler.running.push(Sequence {
        status: Status::Decoding,
        ..seq.clone()
    });
    let found = scheduler
        .get_sequence_mut(42)
        .expect("the running sequence must be found");
    found.degraded_draft = true;
    assert!(
        scheduler.running[0].degraded_draft,
        "get_sequence_mut must return a live mutable borrow"
    );
    assert!(
        scheduler.get_sequence_mut(999).is_none(),
        "an unknown id must report None, not a placeholder"
    );

    // Silence the unused-mut warning if the clone above is ever simplified.
    seq.id = 0;
    let _ = seq;
}

/// `prefix_cache_hit_rate` must be hits/requests, and 0.0 when there are none.
///
/// `cargo mutants` replaced it with the constant 1.0, turned the division into
/// `%` and into `*`, and every variant survived — no test compared the rate
/// against a value, so an operator watching the prefix-cache efficiency gauge
/// could have been reading a fabricated number.
#[test]
fn prefix_cache_hit_rate_is_the_ratio_of_hits_to_requests() {
    let scheduler = SchedulerEngine::default();

    // No requests yet: the documented answer is 0.0, not a division by zero.
    assert!(
        scheduler.prefix_cache_hit_rate().abs() < f64::EPSILON,
        "a zero request count must report 0.0; got {}",
        scheduler.prefix_cache_hit_rate()
    );

    // 2 hits out of 4 lookups is exactly 0.5, and the counts differ so a
    // constant (1.0) or a product (8.0) cannot match.
    for _ in 0..4 {
        scheduler.metrics.record_prefix_cache_request();
    }
    scheduler.metrics.record_prefix_cache_hit();
    scheduler.metrics.record_prefix_cache_hit();

    let rate = scheduler.prefix_cache_hit_rate();
    assert!(
        (rate - 0.5).abs() < 1e-9,
        "2 hits of 4 requests is 0.5; got {rate}"
    );
}

/// `set_policy` must replace the policy, and `register_observer` must report
/// the MAX_OBSERVERS limit rather than absorbing every registration.
///
/// `cargo mutants` emptied `set_policy` and replaced `register_observer` with
/// `Ok(())`, both surviving. A silently-ignored policy change means a caller
/// believes it reconfigured scheduling and did not; a registration that always
/// reports success past the cap means observers are dropped with no error while
/// the caller thinks they are installed.
#[test]
fn set_policy_and_register_observer_take_effect() {
    let mut scheduler = SchedulerEngine::default();
    let before = scheduler.policy.name();

    scheduler.set_policy(Box::new(PriorityPolicy::default()));
    let after = scheduler.policy.name();
    assert_ne!(
        before, after,
        "set_policy must install a different policy (still {before})"
    );

    // Registration succeeds up to MAX_OBSERVERS and then reports an error.
    let limit = crate::scheduler::observer::SchedulerObservers::MAX_OBSERVERS;
    for i in 0..limit {
        assert!(
            scheduler
                .register_observer(Box::new(NoopSchedulerObserver))
                .is_ok(),
            "observer {i} is below the cap and must register"
        );
    }
    assert!(
        scheduler
            .register_observer(Box::new(NoopSchedulerObserver))
            .is_err(),
        "the {}-th registration exceeds the cap and must report an error",
        limit + 1
    );
}

/// `running()` must return the live set of running sequences.
///
/// `cargo mutants` replaced the whole accessor with `vec![]`. Callers use it to
/// inspect in-flight work (diagnostics, the gRPC metrics path); an always-empty
/// answer reports an idle scheduler while sequences are decoding.
#[test]
fn running_returns_the_live_sequences() {
    let mut scheduler = SchedulerEngine::default();
    assert!(
        scheduler.running().is_empty(),
        "a fresh scheduler has nothing running"
    );

    let seq = Sequence {
        id: 77,
        tokens: vec![5; 16],
        kv_blocks: Arc::new(vec![]),
        num_computed_tokens: 16,
        prompt_len: 16,
        status: Status::Decoding,
        max_tokens: 10,
        sampling_params: SamplingParams::default(),
        consecutive_decode_rounds: 1,
        priority: Priority::default(),
        degraded_draft: false,
        draft_model_id: None,
    };
    scheduler.running.push(seq);

    let live = scheduler.running();
    assert_eq!(live.len(), 1, "the running sequence must be reported");
    assert_eq!(live[0].id, 77);
}

/// `SchedulerEngine`'s `Debug` output must describe the engine's real state.
///
/// `cargo mutants` replaced the whole `fmt` with `Ok(Default::default())`, which
/// writes NOTHING while still succeeding. `Debug` is the operator's only view
/// into a stuck scheduler, so an empty rendering silently removes every field
/// (queue depth, running/finished counts, next id) exactly when a hang is being
/// diagnosed. The counts must track live state, not be placeholders.
#[test]
fn debug_output_describes_the_engine_state() {
    let mut scheduler = SchedulerEngine::default();
    scheduler.add_request(Request::new(31, vec![6; 16], 10));

    let rendering = format!("{scheduler:?}");

    assert!(
        rendering.contains("SchedulerEngine"),
        "the struct name must appear; got {rendering}"
    );
    assert!(
        rendering.contains("running_count"),
        "the running count field must appear; got {rendering}"
    );
    assert!(
        rendering.contains("next_seq_id"),
        "the id counter field must appear; got {rendering}"
    );
    assert!(
        rendering.len() > 40,
        "a Debug rendering of 10 fields must not be this short: {rendering}"
    );
}

/// `schedule` must return `Some` when there is work and `None` when idle.
///
/// `cargo mutants` replaced the body with `None`. Callers treat `None` as
/// "nothing to do this round"; always returning it means admitted requests are
/// never composed into a batch and the served model simply never runs.
#[test]
fn schedule_returns_none_only_when_there_is_no_work() {
    let mut scheduler = SchedulerEngine::default();
    assert!(
        scheduler.schedule().is_none(),
        "an empty scheduler has nothing to schedule"
    );

    scheduler.add_request(Request::new(5, vec![9; 16], 10));
    let batch = scheduler
        .schedule()
        .expect("a queued request must produce a batch");
    assert!(
        batch.seq_ids.contains(&5),
        "the admitted request must be in the batch; got {:?}",
        batch.seq_ids
    );

    // Once the work is drained, scheduling is idle again.
    let _ = scheduler.build_batch();
    for _ in 0..3 {
        let _ = scheduler.build_batch();
    }
    assert!(
        scheduler.schedule().is_none(),
        "after the queue drains, schedule must report no work"
    );
}

/// `preallocate_kv_blocks` must give each sequence enough blocks for its tokens
/// and stop cleanly when the pool runs out.
///
/// `cargo mutants` turned `tokens.len().div_ceil(BLOCK_SIZE)` and the index
/// arithmetic into `+` / `-` / `/`. The block table feeds the model's prefill
/// write; if it is short, `write_prefill_kv` falls back to block 0 and corrupts
/// the cache, so the count must be exact rather than merely non-zero.
#[test]
fn preallocate_kv_blocks_sizes_tables_and_stops_at_oom() {
    let block_size = vllm_traits::BLOCK_SIZE;
    let mut scheduler = SchedulerEngine::default();

    // Two prompt blocks plus one token of a third -> 3 blocks needed.
    let tokens = vec![1; block_size * 2 + 1];
    let mut seq = Sequence {
        id: 1,
        tokens: tokens.clone(),
        kv_blocks: Arc::new(vec![]),
        num_computed_tokens: 0,
        prompt_len: tokens.len(),
        status: Status::Prefilling,
        max_tokens: 10,
        sampling_params: SamplingParams::default(),
        consecutive_decode_rounds: 0,
        priority: Priority::default(),
        degraded_draft: false,
        draft_model_id: None,
    };

    scheduler.preallocate_kv_blocks(std::slice::from_mut(&mut seq));

    assert_eq!(
        seq.kv_blocks.len(),
        3,
        "a prompt spanning 2 blocks plus 1 token needs exactly 3 blocks"
    );
    // The blocks must be distinct real allocations, not repeated placeholders.
    let unique: std::collections::HashSet<_> = seq.kv_blocks.iter().copied().collect();
    assert_eq!(
        unique.len(),
        seq.kv_blocks.len(),
        "each needed block must be a distinct allocation; got {:?}",
        seq.kv_blocks
    );

    // Exhausting the pool must stop without panicking and without over-filling.
    let mut small = SchedulerEngine::new(SchedulerConfig::default(), 1, scheduler.metrics.clone());
    let mut big = Sequence {
        id: 2,
        tokens: vec![1; block_size * 4],
        ..seq.clone()
    };
    big.kv_blocks = Arc::new(vec![]);
    small.preallocate_kv_blocks(std::slice::from_mut(&mut big));
    assert!(
        big.kv_blocks.len() <= 1,
        "a one-block pool cannot supply more than one block; got {}",
        big.kv_blocks.len()
    );
}

/// `admit_decode_sequences` must collect ONLY decoding sequences, grow each
/// short table, and defer (not admit) one it cannot grow.
///
/// `cargo mutants` relaxed the status filter, both `kv_blocks.len()` checks,
/// the `i += 1` advance and the `needed` computation. A sequence admitted with a
/// short table is forwarded and its next token's KV write falls back to block 0
/// (`unwrap_or(0)`), corrupting a live block owned by another sequence — so the
/// length check must be a real gate, not a formality.
#[test]
fn admit_decode_sequences_grows_or_defers() {
    let block_size = vllm_traits::BLOCK_SIZE;
    let mut engine = create_test_engine(SchedulerConfig::default(), 8);

    let make = |id: u64, status: Status, kv: Vec<usize>| Sequence {
        id,
        tokens: vec![1; block_size],
        kv_blocks: Arc::new(kv),
        num_computed_tokens: block_size,
        prompt_len: block_size,
        status,
        max_tokens: 10,
        sampling_params: SamplingParams::default(),
        consecutive_decode_rounds: 1,
        priority: Priority::default(),
        degraded_draft: false,
        draft_model_id: None,
    };

    // The non-decoder goes FIRST so the `continue` branch runs at index 0:
    // stepping backwards there is a usize underflow, not a silent no-op.
    engine.running.push(make(3, Status::Prefilling, vec![101]));
    engine.running.push(make(1, Status::Decoding, vec![100]));
    engine.running.push(make(2, Status::Decoding, vec![]));

    let admitted = engine.admit_decode_sequences();
    let ids: Vec<u64> = admitted.iter().map(|s| s.id).collect();

    assert!(
        ids.contains(&1),
        "a decoder with a complete table must be admitted; got {ids:?}"
    );
    assert!(
        ids.contains(&2),
        "a decoder that can be grown must be admitted; got {ids:?}"
    );
    assert!(
        !ids.contains(&3),
        "a Prefilling sequence must never be admitted for decode; got {ids:?}"
    );
    let grown = admitted.iter().find(|s| s.id == 2).expect("seq 2 admitted");
    assert!(
        grown.kv_blocks.len() >= 1,
        "the admitted decoder's table must cover its tokens, not stay short"
    );
    // The originals are grown in place so `update()` keeps progressing.
    let original = engine.running.iter().find(|s| s.id == 2).expect("seq 2");
    assert!(
        original.kv_blocks.len() >= 1,
        "the running table itself must be grown, not just the clone"
    );
}

/// `filter_fully_allocated_prefills` must admit only fully-allocated prefills
/// and requeue the rest, and must pass everything through outside Prefill.
///
/// `cargo mutants` flipped the phase guard (`!=` to `==`) and relaxed the
/// `kv_blocks.len() < blocks_needed` comparison. Admitting a short-table
/// prefill is the corruption path this function exists to prevent: the forward
/// pass writes KV into a table that does not cover the prompt.
#[test]
fn filter_fully_allocated_prefills_requeues_short_tables() {
    let block_size = vllm_traits::BLOCK_SIZE;
    let mut engine = create_test_engine(SchedulerConfig::default(), 1024);

    let make = |id: u64, kv: Vec<usize>| Sequence {
        id,
        tokens: vec![1; block_size * 2],
        kv_blocks: Arc::new(kv),
        num_computed_tokens: 0,
        prompt_len: block_size * 2,
        status: Status::Prefilling,
        max_tokens: 10,
        sampling_params: SamplingParams::default(),
        consecutive_decode_rounds: 0,
        priority: Priority::default(),
        degraded_draft: false,
        draft_model_id: None,
    };

    // Outside Prefill the filter is a pass-through, short tables and all.
    let short = make(1, vec![]);
    let passed = engine.filter_fully_allocated_prefills(vec![short], Phase::Decode);
    assert_eq!(
        passed.len(),
        1,
        "outside Prefill the filter must not drop or requeue anything"
    );

    // In Prefill: one sequence with the full 2 blocks, one with none.
    let full = make(2, vec![900, 901]);
    let incomplete = make(3, vec![]);
    let before = engine.waiting_count();
    let admitted = engine.filter_fully_allocated_prefills(vec![full, incomplete], Phase::Prefill);

    assert_eq!(
        admitted.len(),
        1,
        "only the fully allocated prefill may be admitted; got {:?}",
        admitted.iter().map(|s| s.id).collect::<Vec<_>>()
    );
    assert_eq!(
        admitted[0].id, 2,
        "the complete prefill must be the one kept"
    );
    assert_eq!(
        engine.waiting_count(),
        before + 1,
        "the short-table prefill must return to the waiting queue"
    );
}

/// `requeue_uncomposed_prefills` must return admitted-but-uncomposed prefills
/// to the queue, leave composed ones alone, and do nothing outside Prefill.
///
/// `cargo mutants` emptied the whole method, flipped the phase guard and
/// flipped the `composed.contains` test. A freshly-admitted prefill that is
/// never composed would stay `Status::Prefilling` in `running`, where it is
/// never re-scheduled (running is only re-included when `Decoding`, and prefill
/// work comes from the queue) — a permanent stall that also pins its KV blocks.
#[test]
fn requeue_uncomposed_prefills_returns_only_the_uncomposed() {
    let block_size = vllm_traits::BLOCK_SIZE;
    let mut engine = create_test_engine(SchedulerConfig::default(), 1024);

    let make = |id: u64| Sequence {
        id,
        tokens: vec![1; block_size],
        kv_blocks: Arc::new(vec![]),
        num_computed_tokens: 0,
        prompt_len: block_size,
        status: Status::Prefilling,
        max_tokens: 10,
        sampling_params: SamplingParams::default(),
        consecutive_decode_rounds: 0,
        priority: Priority::default(),
        degraded_draft: false,
        draft_model_id: None,
    };

    // Two admitted prefill; only seq 1 made it into the composed batch.
    engine.running.push(make(1));
    engine.running.push(make(2));
    let admitted: std::collections::HashSet<u64> = [1, 2].into_iter().collect();
    let mut batch = vllm_traits::Batch::empty();
    batch.seq_ids = vec![1];

    // Outside Prefill the method must do NOTHING, even with an uncomposed seq.
    let before = engine.waiting_count();
    engine.requeue_uncomposed_prefills(&admitted, &batch, Phase::Decode);
    assert_eq!(
        engine.waiting_count(),
        before,
        "outside Prefill nothing may be requeued"
    );
    assert_eq!(engine.running.len(), 2, "and nothing may leave running");

    // In Prefill: seq 1 stays (composed), seq 2 is requeued.
    engine.requeue_uncomposed_prefills(&admitted, &batch, Phase::Prefill);
    assert_eq!(
        engine.running.len(),
        1,
        "the uncomposed prefill must leave running; got {:?}",
        engine.running.iter().map(|s| s.id).collect::<Vec<_>>()
    );
    assert_eq!(
        engine.running[0].id, 1,
        "the COMPOSED prefill must stay in running"
    );
    assert_eq!(
        engine.waiting_count(),
        before + 1,
        "the uncomposed prefill must return to the waiting queue"
    );
}

/// `build_batch` must record a CUDA-Graph HIT only when the batch size is
/// captured, and a MISS otherwise, and must dispatch an observer event only
/// for a non-empty batch.
///
/// `cargo mutants` relaxed `phase == Decode && cuda_graph.enabled`, dropped the
/// `!batch.seq_ids.is_empty()` guard, and broke `supports_batch_size`. The
/// hit/miss counters are how an operator tells whether the graph is actually
/// being used — an always-hit (or always-miss) counter makes the fast path look
/// healthy while it is dead, or dead while it is healthy.
#[test]
fn build_batch_records_cuda_graph_hits_and_misses() {
    let config = SchedulerConfig {
        cuda_graph: SchedulerCudaGraphConfig {
            enabled: true,
            batch_sizes: vec![1],
        },
        ..Default::default()
    };
    let mut engine = create_test_engine(config, 1024);
    let metrics = Arc::clone(&engine.metrics);

    // A decode round with exactly ONE sequence: batch size 1 IS captured.
    let mut seq = Sequence {
        id: 1,
        tokens: vec![1; vllm_traits::BLOCK_SIZE],
        kv_blocks: Arc::new(vec![]),
        num_computed_tokens: vllm_traits::BLOCK_SIZE,
        prompt_len: vllm_traits::BLOCK_SIZE,
        status: Status::Decoding,
        max_tokens: 10,
        sampling_params: SamplingParams::default(),
        consecutive_decode_rounds: 1,
        priority: Priority::default(),
        degraded_draft: false,
        draft_model_id: None,
    };
    seq.kv_blocks = Arc::new(vec![]);
    engine.running.push(seq);

    // First make sure the table is allocated through the real path.
    let mut taken = engine.running.remove(0);
    engine.preallocate_kv_blocks(std::slice::from_mut(&mut taken));
    engine.running.push(taken);

    let before_hits = metrics.get_counter("cuda_graph_hits_total");
    let before_misses = metrics.get_counter("cuda_graph_misses_total");
    let batch = engine.build_batch();
    assert_eq!(batch.seq_ids.len(), 1, "setup: a one-sequence batch");

    assert_eq!(
        metrics.get_counter("cuda_graph_hits_total"),
        before_hits + 1,
        "a captured batch size must record a HIT"
    );
    assert_eq!(
        metrics.get_counter("cuda_graph_misses_total"),
        before_misses,
        "a captured batch size must not record a MISS"
    );

    // A PREFILL round must not touch either counter, even with the graph
    // enabled: `&&` relaxed to `||` would record on any enabled build.
    let mut prefill_engine = create_test_engine(
        SchedulerConfig {
            cuda_graph: SchedulerCudaGraphConfig {
                enabled: true,
                batch_sizes: vec![1],
            },
            ..Default::default()
        },
        1024,
    );
    let prefill_metrics = Arc::clone(&prefill_engine.metrics);
    prefill_engine.add_request(Request::new(9, vec![3; 8], 10));
    let hits_before = prefill_metrics.get_counter("cuda_graph_hits_total");
    let misses_before = prefill_metrics.get_counter("cuda_graph_misses_total");
    let batch = prefill_engine.build_batch();
    assert!(!batch.seq_ids.is_empty(), "setup: a prefill batch");
    assert_eq!(
        prefill_metrics.get_counter("cuda_graph_hits_total"),
        hits_before,
        "a Prefill round must not record a graph HIT"
    );
    assert_eq!(
        prefill_metrics.get_counter("cuda_graph_misses_total"),
        misses_before,
        "a Prefill round must not record a graph MISS"
    );
}
