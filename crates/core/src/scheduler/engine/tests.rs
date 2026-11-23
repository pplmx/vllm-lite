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
use crate::scheduler::observer::{NoopSchedulerObserver, SchedulerObserver};
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
    let make_seq = |id: u64, status: Status| Sequence {
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

/// A build that composes NOTHING must not emit a `BatchScheduled` event.
///
/// `if !batch.seq_ids.is_empty()` guards the observer dispatch. `cargo mutants`
/// deleted the `!`, so an idle round would announce a batch of size 0 to every
/// subscriber — dashboards built on this event would show spurious empty batches
/// and any subscriber that allocates per event would churn once per idle round.
#[test]
fn empty_build_does_not_dispatch_a_batch_event() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Counts `BatchScheduled` callbacks and records the largest size seen.
    struct CountingObserver {
        calls: Arc<AtomicUsize>,
        max_size: Arc<AtomicUsize>,
    }
    impl SchedulerObserver for CountingObserver {
        fn on_request_arrived(&self, _seq_id: u64, _prompt_len: usize) {}
        fn on_batch_scheduled(&self, _seq_ids: &[u64], batch_size: usize) {
            self.calls.fetch_add(1, Ordering::Relaxed);
            self.max_size.fetch_max(batch_size, Ordering::Relaxed);
        }
        fn on_decoding(&self, _seq_id: u64, _token: u32) {}
        fn on_sequence_finished(&self, _seq_id: u64, _total_tokens: usize) {}
        fn on_preemption(&self, _seq_id: u64, _reason: &str) {}
        fn on_memory_pressure(&self, _available_blocks: usize) {}
        fn on_distributed_prefix_matched(&self, _seq_id: u64, _matched_tokens: usize) {}
    }

    let config = SchedulerConfig {
        // Keep the graph OFF so the only observer interaction is the dispatch.
        cuda_graph: SchedulerCudaGraphConfig::default(),
        ..Default::default()
    };
    let mut engine = create_test_engine(config, 1024);

    let calls = Arc::new(AtomicUsize::new(0));
    let max_size = Arc::new(AtomicUsize::new(0));
    engine
        .register_observer(Box::new(CountingObserver {
            calls: Arc::clone(&calls),
            max_size: Arc::clone(&max_size),
        }))
        .expect("observer registration must succeed");

    // An idle engine composes nothing.
    let batch = engine.build_batch();
    assert!(batch.seq_ids.is_empty(), "setup: nothing to schedule");
    assert_eq!(
        calls.load(Ordering::Relaxed),
        0,
        "an empty batch must not be announced to observers"
    );

    // Control: a real batch DOES dispatch, so the assertion above cannot pass
    // by the observer never being wired.
    engine.add_request(Request::new(3, vec![8; 16], 10));
    let batch = engine.build_batch();
    assert!(!batch.seq_ids.is_empty(), "setup: a real batch");
    assert_eq!(
        calls.load(Ordering::Relaxed),
        1,
        "a non-empty batch must dispatch exactly one event"
    );
    assert_eq!(
        max_size.load(Ordering::Relaxed),
        batch.seq_ids.len(),
        "the announced size must match the composed batch"
    );
}

/// The scheduler wait time recorded during a build must be in MILLISECONDS.
///
/// `build_batch` converts the queue→admission delay with
/// `wait.as_secs_f64() * 1000.0`. `cargo mutants` turned that `*` into `+` and
/// `/`, both surviving: nothing read `avg_scheduler_wait_time_ms` back after a
/// real delay. With `+` a 20 ms wait records as ~1000 ms and with `/` as
/// ~0.00002 ms, so the published metric is wrong by three orders of magnitude in
/// either direction.
#[test]
fn scheduler_wait_time_is_recorded_in_milliseconds() {
    let mut engine = create_test_engine(SchedulerConfig::default(), 1024);
    engine.add_request(Request::new(1, vec![5; 16], 10));

    // Let the request sit in the queue so the delay is measurable and bounded.
    std::thread::sleep(std::time::Duration::from_millis(20));

    let batch = engine.build_batch();
    assert!(!batch.seq_ids.is_empty(), "setup: the request was admitted");

    let recorded = engine.metrics.snapshot().avg_scheduler_wait_time_ms;
    assert!(
        (5.0..500.0).contains(&recorded),
        "a ~20ms wait must record as tens of milliseconds; got {recorded}. \
         A `+` mutant would give ~1000 and a `/` mutant ~0.00002."
    );
}

/// An EXACT block fit must not trigger preemption.
///
/// `build_batch` gates preemption on `additional > available_blocks()`. With
/// `>=` an exact fit also calls `execute_preemption`, whose phase 1 selects
/// victims by LRU — and a decode victim is preempted WHOLESALE, its whole table
/// released and its progress reset to a full re-prefill, to free blocks the new
/// sequence did not even need. The batch must be built from the blocks already
/// free.
#[test]
fn exact_block_fit_does_not_preempt() {
    let block_size = vllm_traits::BLOCK_SIZE;
    // Pool of 6 blocks. One decoder holds 4, leaving exactly 2 free.
    let mut engine = create_test_engine(SchedulerConfig::default(), 6);

    let mut decoder = Sequence {
        id: 1,
        tokens: vec![1; block_size * 4],
        kv_blocks: Arc::new(vec![]),
        num_computed_tokens: block_size * 4,
        prompt_len: block_size * 4,
        status: Status::Decoding,
        max_tokens: 10,
        sampling_params: SamplingParams::default(),
        consecutive_decode_rounds: 1,
        priority: Priority::default(),
        degraded_draft: false,
        draft_model_id: None,
    };
    engine.preallocate_kv_blocks(std::slice::from_mut(&mut decoder));
    assert_eq!(decoder.kv_blocks.len(), 4, "setup: decoder holds 4 blocks");
    engine.running.push(decoder);
    assert_eq!(
        engine.memory.available_blocks(),
        2,
        "setup: exactly two blocks free"
    );

    // A new prompt of exactly 2 blocks: an exact fit for the free space.
    engine.add_request(Request::new(2, vec![2; block_size * 2], 10));
    let batch = engine.build_batch();

    assert!(
        batch.seq_ids.contains(&2),
        "the exactly-fitting request must be admitted; got {:?}",
        batch.seq_ids
    );
    assert!(
        engine.running.iter().any(|s| s.id == 1),
        "the running decoder must NOT be preempted for an exact fit; \
         running = {:?}",
        engine.running.iter().map(|s| s.id).collect::<Vec<_>>()
    );
    assert_eq!(
        engine
            .running
            .iter()
            .find(|s| s.id == 1)
            .map(|s| s.kv_blocks.len()),
        Some(4),
        "the decoder's table must be untouched"
    );
}

/// A request needing LESS than the free pool must not preempt anything.
///
/// Companion to `exact_block_fit_does_not_preempt`: `>` relaxed to `<` INVERTS
/// the gate, so the exact fit passes while this case — where the need is
/// smaller than the free space — wrongly preempts a healthy decoder, rolling it
/// back to a full re-prefill to free blocks nobody asked for.
#[test]
fn smaller_than_free_request_does_not_preempt() {
    let block_size = vllm_traits::BLOCK_SIZE;
    let mut engine = create_test_engine(SchedulerConfig::default(), 8);

    let mut holder = Sequence {
        id: 10,
        tokens: vec![1; block_size * 2],
        kv_blocks: Arc::new(vec![]),
        num_computed_tokens: block_size * 2,
        prompt_len: block_size * 2,
        status: Status::Decoding,
        max_tokens: 10,
        sampling_params: SamplingParams::default(),
        consecutive_decode_rounds: 1,
        priority: Priority::default(),
        degraded_draft: false,
        draft_model_id: None,
    };
    engine.preallocate_kv_blocks(std::slice::from_mut(&mut holder));
    engine.running.push(holder);

    let free = engine.memory.available_blocks();
    assert!(free >= 3, "setup: several blocks free, got {free}");

    // One block needed against several free: additional < available.
    engine.add_request(Request::new(11, vec![3; block_size], 10));
    let batch = engine.build_batch();

    assert!(
        batch.seq_ids.contains(&11),
        "the small request must be admitted; got {:?}",
        batch.seq_ids
    );
    assert!(
        engine.running.iter().any(|s| s.id == 10),
        "nothing may be preempted when the need is SMALLER than the free pool; \
         running = {:?}",
        engine.running.iter().map(|s| s.id).collect::<Vec<_>>()
    );
}

/// A Prefill round must not pull leftover Decoding sequences into the batch.
///
/// `if phase == Phase::Decode { sequences = self.admit_decode_sequences(); }`
/// guards the phase. `cargo mutants` flipped it to `!=`, so a Prefill round
/// would ALSO run decode admission and mix decoders into a batch whose
/// `is_prefill` flags and `kv_block_ids` describe prefill work.
///
/// Reaching Prefill WHILE a decoder sits in `running` is the whole difficulty:
/// `PhaseScheduler` starts in Prefill and switches to Decode only once
/// `prefill_queue_len == 0`, so a queued prefill request is what holds the
/// scheduler in the Prefill phase. A first attempt without a queued request saw
/// the selector choose Decode and admitted the decoder legitimately, which is
/// why this fixture queues one.
#[test]
fn prefill_round_does_not_admit_decode_sequences() {
    let block_size = vllm_traits::BLOCK_SIZE;
    let mut engine = create_test_engine(SchedulerConfig::default(), 1024);

    let mut decoder = Sequence {
        id: 1,
        tokens: vec![1; block_size],
        kv_blocks: Arc::new(vec![]),
        num_computed_tokens: block_size,
        prompt_len: block_size,
        status: Status::Decoding,
        max_tokens: 10,
        sampling_params: SamplingParams::default(),
        consecutive_decode_rounds: 1,
        priority: Priority::default(),
        degraded_draft: false,
        draft_model_id: None,
    };
    engine.preallocate_kv_blocks(std::slice::from_mut(&mut decoder));
    engine.running.push(decoder);

    // A QUEUED prefill request keeps the phase selector in Prefill, so the
    // decode-admission branch must not run this round.
    engine.add_request(Request::new(2, vec![4; block_size], 10));
    let batch = engine.build_batch();

    assert!(
        batch.is_prefill.iter().any(|&p| p),
        "setup: this round must be a Prefill round; got phase {:?}",
        batch.phase
    );
    assert!(
        !batch.seq_ids.contains(&1),
        "a Prefill round must not admit a leftover decoder; got {:?}",
        batch.seq_ids
    );
}

/// `update` must grow the KV table when a new token crosses a block boundary.
///
/// `push_token_and_allocate` grows with
/// `while seq.kv_blocks.len() < blocks_needed`. `cargo mutants` relaxed that
/// condition three ways. A loop that stops early (or never runs) leaves the
/// table short of the tokens it must cover, so the next forward's KV write falls
/// back to block 0 and corrupts a live block.
#[test]
fn update_grows_the_kv_table_across_a_block_boundary() {
    let block_size = vllm_traits::BLOCK_SIZE;
    let mut engine = create_test_engine(SchedulerConfig::default(), 64);

    // Exactly one full block of tokens, and a table holding that one block.
    let mut seq = Sequence {
        id: 1,
        tokens: vec![1; block_size],
        kv_blocks: Arc::new(vec![]),
        num_computed_tokens: block_size,
        prompt_len: block_size,
        status: Status::Decoding,
        max_tokens: 10,
        sampling_params: SamplingParams::default(),
        consecutive_decode_rounds: 0,
        priority: Priority::default(),
        degraded_draft: false,
        draft_model_id: None,
    };
    engine.preallocate_kv_blocks(std::slice::from_mut(&mut seq));
    assert_eq!(
        seq.kv_blocks.len(),
        1,
        "setup: one block for one block of tokens"
    );
    engine.running.push(seq);

    // One more token pushes the count to block_size + 1, which needs 2 blocks.
    let sampled = vllm_traits::SampledToken {
        token: 42,
        logprob: 0.0,
        top_logprobs: vec![],
    };
    engine.update(&[1], &[sampled], &[0]);

    let updated = engine.running.iter().find(|s| s.id == 1).expect("seq 1");
    assert_eq!(
        updated.tokens.len(),
        block_size + 1,
        "setup: the new token must be appended"
    );
    assert_eq!(
        updated.kv_blocks.len(),
        2,
        "a token past the block boundary must grow the table to 2 blocks; got {}",
        updated.kv_blocks.len()
    );
    // The two blocks must be distinct real allocations.
    assert_ne!(
        updated.kv_blocks[0], updated.kv_blocks[1],
        "the grown table must not reuse the same block twice"
    );
}

/// `update` must park a partially-computed prefill instead of leaving it stuck.
///
/// At the end of `update`, `requeue_partial_prefills` parks sequences that are
/// still `Prefilling` with `num_computed_tokens < prompt_len` — a chunked
/// prefill resumes next round via `forward_prefill_continue`. `cargo mutants`
/// relaxed that predicate's conjunction to `||` and its comparison to `<=`, and
/// with either change a COMPLETE prefill is parked too, throwing away the whole
/// prompt's prefill work.
#[test]
fn update_parks_a_partial_prefill_and_keeps_its_progress() {
    let block_size = vllm_traits::BLOCK_SIZE;
    let mut engine = create_test_engine(SchedulerConfig::default(), 1024);

    // A prefill that has computed only half its prompt.
    let mut seq = Sequence {
        id: 1,
        tokens: vec![1; block_size * 2],
        kv_blocks: Arc::new(vec![]),
        num_computed_tokens: block_size,
        prompt_len: block_size * 2,
        status: Status::Prefilling,
        max_tokens: 10,
        sampling_params: SamplingParams::default(),
        consecutive_decode_rounds: 0,
        priority: Priority::default(),
        degraded_draft: false,
        draft_model_id: None,
    };
    engine.preallocate_kv_blocks(std::slice::from_mut(&mut seq));
    engine.running.push(seq);

    let sampled = vllm_traits::SampledToken {
        token: 7,
        logprob: 0.0,
        top_logprobs: vec![],
    };
    // Advance by only one block, leaving the prompt still incomplete, so the
    // sequence never reaches `Decoding` this round.
    let before = engine.waiting_count();
    engine.update(&[1], &[sampled], &[8]);

    assert!(
        !engine.running.iter().any(|s| s.id == 1),
        "the partial prefill must leave running; running = {:?}",
        engine.running.iter().map(|s| s.id).collect::<Vec<_>>()
    );
    assert_eq!(
        engine.waiting_count(),
        before + 1,
        "it must be parked in the waiting queue, not dropped"
    );
    let parked = engine
        .request_queue
        .get(1)
        .expect("the partial prefill is queued for the next chunk");
    assert_eq!(
        parked.num_computed_tokens,
        block_size + 8,
        "a parked chunk must keep its computed progress"
    );
    assert_eq!(parked.status, Status::Waiting);
}

/// A `Decoding` sequence must never be parked as a partial prefill, however few
/// tokens it has computed.
///
/// The predicate is `status == Prefilling && num_computed < prompt_len`.
/// Relaxing `&&` to `||` parks any `Decoding` sequence whose
/// `num_computed < prompt_len` — which is every decoder mid-prompt-boundary,
/// including one that is legitimately decoding ahead of its prompt length.
/// This is the case the partial-prefill test alone cannot catch.
#[test]
fn decoding_sequences_are_never_parked_as_partial_prefills() {
    let block_size = vllm_traits::BLOCK_SIZE;
    let mut engine = create_test_engine(SchedulerConfig::default(), 1024);

    let mut seq = Sequence {
        id: 1,
        tokens: vec![1; block_size],
        kv_blocks: Arc::new(vec![]),
        num_computed_tokens: block_size / 2, // < prompt_len
        prompt_len: block_size,
        status: Status::Decoding,
        max_tokens: 10,
        sampling_params: SamplingParams::default(),
        consecutive_decode_rounds: 1,
        priority: Priority::default(),
        degraded_draft: false,
        draft_model_id: None,
    };
    engine.preallocate_kv_blocks(std::slice::from_mut(&mut seq));
    engine.running.push(seq);

    let sampled = vllm_traits::SampledToken {
        token: 3,
        logprob: 0.0,
        top_logprobs: vec![],
    };
    let before = engine.waiting_count();
    engine.update(&[1], &[sampled], &[1]);

    assert!(
        engine.running.iter().any(|s| s.id == 1),
        "a Decoding sequence must stay in running; running = {:?}",
        engine.running.iter().map(|s| s.id).collect::<Vec<_>>()
    );
    assert_eq!(
        engine.waiting_count(),
        before,
        "nothing may be requeued from a decode round"
    );
}

/// `update_speculative` must fold tokens only for sequences that are NOT a
/// mid-chunk prefill, while still advancing the computed frontier for both.
///
/// A chunked prefill left `Prefilling` after the advance predicted a stale
/// next-prompt token that is re-derived on the final chunk, so its entries must
/// NOT enter `seq.tokens`, dispatch decode events, or allocate blocks — only
/// `num_computed_tokens` moves. `cargo mutants` emptied the whole method and
/// relaxed both the `.is_some_and(...)` chain and the `status != Prefilling`
/// test, none of which had a test.
#[test]
fn update_speculative_folds_only_non_prefill_sequences() {
    let block_size = vllm_traits::BLOCK_SIZE;
    let mut engine = create_test_engine(SchedulerConfig::default(), 1024);

    let make = |id: u64, status: Status, computed: usize, prompt: usize| Sequence {
        id,
        tokens: vec![1; prompt],
        kv_blocks: Arc::new(vec![]),
        num_computed_tokens: computed,
        prompt_len: prompt,
        status,
        max_tokens: 10,
        sampling_params: SamplingParams::default(),
        consecutive_decode_rounds: 1,
        priority: Priority::default(),
        degraded_draft: false,
        draft_model_id: None,
    };

    // Seq 1 is a mid-chunk prefill; seq 2 is a healthy decoder.
    engine
        .running
        .push(make(1, Status::Prefilling, block_size, block_size * 4));
    engine
        .running
        .push(make(2, Status::Decoding, block_size, block_size));
    // Allocate each table through the real path, one at a time.
    let mut all = std::mem::take(&mut engine.running);
    engine.preallocate_kv_blocks(&mut all);
    engine.running = all;

    let before_tokens_1 = engine.running[0].tokens.len();
    let before_blocks_1 = engine.running[0].kv_blocks.len();
    let before_tokens_2 = engine.running[1].tokens.len();

    let tok = |t: u32| vllm_traits::SampledToken {
        token: t,
        logprob: 0.0,
        top_logprobs: vec![],
    };
    engine.update_speculative(
        &[1, 2],
        &[vec![tok(100), tok(101)], vec![tok(200)]],
        &[8, 1],
    );

    // Seq 1 was parked by `requeue_partial_prefills` (it is still Prefilling
    // with computed < prompt), so read it from the queue.
    let seq1 = engine.request_queue.get(1).expect("seq 1 is parked");
    assert_eq!(
        seq1.num_computed_tokens,
        block_size + 8,
        "a mid-chunk prefill must still advance its computed frontier"
    );
    assert_eq!(
        seq1.tokens.len(),
        before_tokens_1,
        "a mid-chunk prefill must NOT fold its stale predicted tokens"
    );
    assert_eq!(
        seq1.kv_blocks.len(),
        before_blocks_1,
        "and must NOT allocate blocks for them"
    );

    // Seq 2: the token IS folded.
    let seq2 = engine.running.iter().find(|s| s.id == 2).expect("seq 2");
    assert_eq!(
        seq2.tokens.len(),
        before_tokens_2 + 1,
        "a real decoder must fold its emitted token"
    );
    assert_eq!(
        seq2.tokens.last().copied(),
        Some(200),
        "the folded token must be the one emitted"
    );
}

/// `update` must advance the frontier for `Prefilling`/`Waiting` sequences and
/// promote to `Decoding` once the prompt is covered — but leave `Decoding`
/// alone.
///
/// The guard is `status == Waiting || status == Prefilling`.
/// `cargo mutants` flipped that `==` to `!=`, which advances the DECODERS and
/// skips every prefill: a decoder's `num_computed_tokens` would be incremented
/// again on top of the frontier `push_token_and_allocate` already moved, and a
/// prefill would never reach `Decoding`, stalling the request forever.
#[test]
fn update_advances_prefills_and_leaves_decoders_alone() {
    let block_size = vllm_traits::BLOCK_SIZE;
    let mut engine = create_test_engine(SchedulerConfig::default(), 1024);

    let make = |id: u64, status: Status, computed: usize, prompt: usize| Sequence {
        id,
        tokens: vec![1; prompt],
        kv_blocks: Arc::new(vec![]),
        num_computed_tokens: computed,
        prompt_len: prompt,
        status,
        max_tokens: 10,
        sampling_params: SamplingParams::default(),
        consecutive_decode_rounds: 1,
        priority: Priority::default(),
        degraded_draft: false,
        draft_model_id: None,
    };

    // Seq 1: a prefill that this round's input COMPLETES the prompt.
    // Seq 2: a decoder whose frontier must not move again.
    engine
        .running
        .push(make(1, Status::Prefilling, block_size, block_size));
    engine
        .running
        .push(make(2, Status::Decoding, block_size, block_size));
    let mut all = std::mem::take(&mut engine.running);
    engine.preallocate_kv_blocks(&mut all);
    engine.running = all;

    let decoder_before = engine.running[1].num_computed_tokens;

    let sampled = vllm_traits::SampledToken {
        token: 5,
        logprob: 0.0,
        top_logprobs: vec![],
    };
    engine.update(&[1, 2], &[sampled.clone(), sampled], &[block_size, 4]);

    let seq1 = engine
        .running
        .iter()
        .chain(engine.finished.iter())
        .find(|s| s.id == 1)
        .expect("seq 1 exists");
    assert_eq!(
        seq1.num_computed_tokens,
        block_size + block_size,
        "a Prefilling sequence must advance by the input count"
    );
    assert_ne!(
        seq1.status,
        Status::Prefilling,
        "covering the prompt must promote the sequence out of Prefilling"
    );

    let seq2 = engine
        .running
        .iter()
        .chain(engine.finished.iter())
        .find(|s| s.id == 2)
        .expect("seq 2 exists");
    assert_eq!(
        seq2.num_computed_tokens, decoder_before,
        "a Decoding sequence's frontier must NOT be advanced by input_count"
    );
}

/// `finish_sequence` must move a running sequence to `finished`, mark it
/// `Finished`, and return its KV blocks to the pool.
///
/// `cargo mutants` replaced the whole method with `()`, which survived — no test
/// aborted a sequence this way. A no-op leaves the sequence in `running` where
/// it is re-included in every subsequent batch, burning compute on tokens the
/// client will never see and pinning its KV blocks until `max_tokens` is
/// reached naturally.
#[test]
fn finish_sequence_moves_and_releases() {
    let block_size = vllm_traits::BLOCK_SIZE;
    let mut engine = create_test_engine(SchedulerConfig::default(), 32);

    let mut seq = Sequence {
        id: 9,
        tokens: vec![1; block_size * 2],
        kv_blocks: Arc::new(vec![]),
        num_computed_tokens: block_size * 2,
        prompt_len: block_size * 2,
        status: Status::Decoding,
        max_tokens: 10,
        sampling_params: SamplingParams::default(),
        consecutive_decode_rounds: 1,
        priority: Priority::default(),
        degraded_draft: false,
        draft_model_id: None,
    };
    engine.preallocate_kv_blocks(std::slice::from_mut(&mut seq));
    assert_eq!(seq.kv_blocks.len(), 2, "setup: two blocks held");
    engine.running.push(seq);
    let available_before = engine.memory.available_blocks();

    engine.finish_sequence(9);

    assert!(
        !engine.running.iter().any(|s| s.id == 9),
        "the sequence must leave running; running = {:?}",
        engine.running.iter().map(|s| s.id).collect::<Vec<_>>()
    );
    let finished = engine
        .finished
        .iter()
        .find(|s| s.id == 9)
        .expect("the sequence must land in finished");
    assert_eq!(finished.status, Status::Finished);
    assert_eq!(
        engine.memory.available_blocks(),
        available_before + 2,
        "its two blocks must return to the pool"
    );

    // An unknown id must be a no-op, not a panic.
    engine.finish_sequence(9999);
}

/// A `Finished` sequence must be moved to `finished` and have its blocks
/// released, while a still-running one is untouched.
///
/// `finalize_finished_sequences` filters `running` by
/// `status == Status::Finished`; `cargo mutants` flipped that to `!=`, which
/// moves every RUNNING sequence into `finished` and releases blocks it still
/// needs — silent live-state destruction. Reached through the public `update`
/// by letting one sequence hit `max_tokens`.
#[test]
fn finalize_moves_only_finished_sequences() {
    let mut engine = create_test_engine(SchedulerConfig::default(), 64);

    let make = |id: u64, tokens: usize, max_tokens: usize| Sequence {
        id,
        tokens: vec![1; tokens],
        kv_blocks: Arc::new(vec![]),
        num_computed_tokens: vllm_traits::BLOCK_SIZE,
        prompt_len: vllm_traits::BLOCK_SIZE,
        status: Status::Decoding,
        max_tokens,
        sampling_params: SamplingParams::default(),
        consecutive_decode_rounds: 1,
        priority: Priority::default(),
        degraded_draft: false,
        draft_model_id: None,
    };

    // Seq 1 already emitted its max_tokens (1) so it finishes this round.
    // Seq 2 has a long budget and must keep running.
    engine.running.push(make(1, vllm_traits::BLOCK_SIZE + 1, 1));
    engine.running.push(make(2, vllm_traits::BLOCK_SIZE, 50));
    let mut all = std::mem::take(&mut engine.running);
    engine.preallocate_kv_blocks(&mut all);
    engine.running = all;

    let sampled = vllm_traits::SampledToken {
        token: 6,
        logprob: 0.0,
        top_logprobs: vec![],
    };
    engine.update(&[1, 2], &[sampled.clone(), sampled], &[1, 1]);

    assert!(
        engine.finished.iter().any(|s| s.id == 1),
        "the completed sequence must be finalized; finished = {:?}",
        engine.finished.iter().map(|s| s.id).collect::<Vec<_>>()
    );
    assert!(
        engine.running.iter().any(|s| s.id == 2),
        "a sequence still under its budget must remain running; running = {:?}",
        engine.running.iter().map(|s| s.id).collect::<Vec<_>>()
    );
    assert!(
        !engine.finished.iter().any(|s| s.id == 2),
        "an unfinished sequence must NEVER be moved to finished"
    );
}

/// `cancel_request` must release a QUEUED sequence's blocks too.
///
/// The queue branch has its own `if !seq.kv_blocks.is_empty()` guard
/// (`memory.rs:251`). `cargo mutants` deleted that `!`, which INVERTS the guard:
/// blocks are released only when the table is empty, so a queued sequence
/// holding real blocks leaks every one of them. The existing
/// `test_cancel_request_releases_kv_blocks` covers the RUNNING branch, where the
/// other copy of the guard lives — which is why the mutation survived.
#[test]
fn cancel_request_releases_blocks_of_a_queued_sequence() {
    use crate::scheduler::policy::FcfsPolicy;
    use crate::scheduler::policy::trait_def::SchedulingContext;

    let block_size = vllm_traits::BLOCK_SIZE;
    let mut engine = create_test_engine(SchedulerConfig::default(), 32);
    let block_size_tokens = vec![1; block_size * 2];

    // Allocate first, then park the sequence in the QUEUE while still holding
    // its blocks — the state a re-queued advanced prefill actually has.
    let mut seq = Sequence {
        id: 7,
        tokens: block_size_tokens,
        kv_blocks: Arc::new(vec![]),
        num_computed_tokens: 0,
        prompt_len: block_size * 2,
        status: Status::Waiting,
        max_tokens: 10,
        sampling_params: SamplingParams::default(),
        consecutive_decode_rounds: 0,
        priority: Priority::default(),
        degraded_draft: false,
        draft_model_id: None,
    };
    engine.preallocate_kv_blocks(std::slice::from_mut(&mut seq));
    assert_eq!(seq.kv_blocks.len(), 2, "setup: two blocks held");
    let ctx = SchedulingContext {
        current_time: std::time::Instant::now(),
        queue_length: 0,
        running_count: 0,
        memory_pressure: 0.0,
    };
    engine.request_queue.enqueue(seq, &FcfsPolicy, &ctx);
    let available_before = engine.memory.available_blocks();
    assert!(engine.request_queue.get(7).is_some(), "setup: queued");

    assert!(
        engine.cancel_request(7),
        "cancelling a queued request works"
    );

    assert!(
        engine.request_queue.get(7).is_none(),
        "the cancelled sequence must leave the queue"
    );
    assert_eq!(
        engine.memory.available_blocks(),
        available_before + 2,
        "its two blocks must return to the pool, not leak"
    );
}

/// `ensure_blocks_for_tokens` must grow a running sequence's table to at least
/// `min_blocks`, and report the count it actually holds.
///
/// The speculative step reserves the whole verification span BEFORE the target
/// writes draft KV; without the extra blocks the write's
/// `block_ids.get(block_idx).unwrap_or(0)` fallback puts the overflow into
/// block 0 and corrupts the prompt's first-block KV. `cargo mutants` relaxed the
/// `while len < min_blocks` condition three ways, and a short table is exactly
/// what this function exists to prevent.
#[test]
fn ensure_blocks_for_tokens_grows_to_the_requested_minimum() {
    let block_size = vllm_traits::BLOCK_SIZE;
    let mut engine = create_test_engine(SchedulerConfig::default(), 64);

    let mut seq = Sequence {
        id: 4,
        tokens: vec![1; block_size],
        kv_blocks: Arc::new(vec![]),
        num_computed_tokens: 0,
        prompt_len: block_size,
        status: Status::Decoding,
        max_tokens: 10,
        sampling_params: SamplingParams::default(),
        consecutive_decode_rounds: 1,
        priority: Priority::default(),
        degraded_draft: false,
        draft_model_id: None,
    };
    engine.preallocate_kv_blocks(std::slice::from_mut(&mut seq));
    assert_eq!(seq.kv_blocks.len(), 1, "setup: one block");
    engine.running.push(seq);

    let held = engine.ensure_blocks_for_tokens(4, 5);

    assert_eq!(held, 5, "the reported count must be the grown table size");
    let after = engine.running.iter().find(|s| s.id == 4).expect("seq 4");
    assert_eq!(
        after.kv_blocks.len(),
        5,
        "the table must hold at least min_blocks; got {}",
        after.kv_blocks.len()
    );
    let unique: std::collections::HashSet<_> = after.kv_blocks.iter().copied().collect();
    assert_eq!(
        unique.len(),
        5,
        "all five blocks must be distinct allocations, not repeats"
    );

    // Already large enough: no further allocation.
    let available_before = engine.memory.available_blocks();
    assert_eq!(engine.ensure_blocks_for_tokens(4, 3), 5);
    assert_eq!(
        engine.memory.available_blocks(),
        available_before,
        "asking for FEWER blocks than held must allocate nothing"
    );

    // Unknown sequence reports 0.
    assert_eq!(engine.ensure_blocks_for_tokens(9999, 5), 0);
}

#[cfg(feature = "multi-node")]
#[test]
fn set_distributed_kv_on_the_engine_reaches_the_memory_manager() {
    use vllm_dist::{CacheConfig, DistributedKVCache, NodeId};

    let mut engine = create_test_engine(SchedulerConfig::default(), 8);
    let cache = Arc::new(DistributedKVCache::new(CacheConfig::new(NodeId(0), 4)));

    // Nothing is wired yet, so allocation must not touch the cache.
    let _ = engine.memory_mut().allocate(1);
    assert_eq!(cache.stats().updates, 0, "no cache wired yet");

    engine.set_distributed_kv(Arc::clone(&cache));
    let _ = engine.memory_mut().allocate(2);

    assert_eq!(
        cache.stats().updates,
        2,
        "set_distributed_kv must reach the MemoryManager so later allocations \
         register in the cache"
    );
}

/// Phase-1 preemption must preempt a running decoder that OWNS a victim block.
///
/// `execute_preemption` selects victims block-wise, then preempts every
/// `Decoding`/`Prefilling` sequence owning one, WHOLE. `cargo mutants` flipped
/// that status test to `!=`, which matches neither status, so the victim set
/// frees nothing and phase 2 then falls back to a different rule — losing the
/// block-level targeting entirely. A probe confirmed this fixture reaches phase 1
/// with a NON-EMPTY victim set, which earlier attempts never did.
#[test]
fn phase1_preempts_the_decoder_that_owns_a_victim_block() {
    let block_size = vllm_traits::BLOCK_SIZE;
    let make = |id: u64| Sequence {
        id,
        tokens: vec![1; block_size * 2],
        kv_blocks: Arc::new(vec![]),
        num_computed_tokens: block_size * 2,
        prompt_len: block_size * 2,
        status: Status::Decoding,
        max_tokens: 50,
        sampling_params: SamplingParams::default(),
        consecutive_decode_rounds: 1,
        priority: Priority::default(),
        degraded_draft: false,
        draft_model_id: None,
    };

    // Pool of 4: two decoders hold 2 blocks each, leaving none free.
    let mut engine = create_test_engine(SchedulerConfig::default(), 4);
    for id in [1u64, 2] {
        let mut one = [make(id)];
        engine.preallocate_kv_blocks(&mut one);
        engine.running.push(one[0].clone());
    }
    assert_eq!(engine.memory.available_blocks(), 0, "setup: pool exhausted");
    assert!(
        !engine.memory.select_victims(&engine.running, 1).is_empty(),
        "setup: the fixture must reach phase 1 with a victim"
    );

    // A 1-block request forces preemption of exactly one 2-block decoder.
    engine.add_request(Request::new(9, vec![2; block_size], 10));
    let batch = engine.build_batch();

    assert!(
        batch.seq_ids.contains(&9),
        "the new request must be admitted; batched={:?}",
        batch.seq_ids
    );
    assert_eq!(
        engine.running.iter().filter(|s| s.id != 9).count(),
        1,
        "exactly ONE decoder must be preempted; running={:?}",
        engine.running.iter().map(|s| s.id).collect::<Vec<_>>()
    );
    assert_eq!(
        engine.waiting_count(),
        1,
        "the preempted decoder must be re-queued, not dropped"
    );
}

/// Phase 2 must continue from the blocks phase 1 ALREADY freed.
///
/// `blocks_freed` is accumulated across both phases. `cargo mutants` turned the
/// phase-2 `blocks_freed += block_count` into `*=` and `-=`: with `*=` the total
/// stays 0, so the `blocks_freed >= blocks_needed` guard never fires and phase 2
/// preempts EVERY decodable sequence rather than stopping at the demand —
/// over-evicting sequences that could have kept running.
///
/// The fixture makes phase 1 free SOME blocks but not enough: block A holds 4
/// eligible blocks and block B holds 2 whose refcount is 2 (ineligible), against
/// a 6-block demand.
#[test]
fn phase2_continues_from_the_blocks_phase1_freed() {
    let block_size = vllm_traits::BLOCK_SIZE;
    let make = |id: u64, blocks: usize| Sequence {
        id,
        tokens: vec![1; block_size * blocks],
        kv_blocks: Arc::new(vec![]),
        num_computed_tokens: block_size * blocks,
        prompt_len: block_size * blocks,
        status: Status::Decoding,
        max_tokens: 50,
        sampling_params: SamplingParams::default(),
        consecutive_decode_rounds: 1,
        priority: Priority::default(),
        degraded_draft: false,
        draft_model_id: None,
    };

    // Pool of 12 so a second, healthy decoder can also exist.
    let mut engine = create_test_engine(SchedulerConfig::default(), 12);
    let mut a = [make(1, 4)];
    engine.preallocate_kv_blocks(&mut a);
    let mut b = [make(2, 2)];
    engine.preallocate_kv_blocks(&mut b);
    // B's blocks get an EXTRA reference, making them ineligible for phase 1.
    engine.memory.record_blocks(b[0].kv_blocks.as_ref());
    // A third, plain decoder that must NOT be preempted if phase 2 stops early.
    let mut c = [make(3, 2)];
    engine.preallocate_kv_blocks(&mut c);
    for s in [&a[0], &b[0], &c[0]] {
        engine.running.push(s.clone());
    }

    // Phase 1 can offer A's 4 blocks; the demand of 6 must be completed by
    // phase 2, which should stop after ONE sequence rather than draining all.
    engine.execute_preemption(6);

    let still_running = engine.running.len();
    assert!(
        still_running >= 1,
        "phase 2 must STOP once the demand is met, not evict every decoder; \
         running={:?}",
        engine.running.iter().map(|s| s.id).collect::<Vec<_>>()
    );
}

/// A Prefill round on the CUDA-Graph path must not select running decoders,
/// and a Decode round must select only `Decoding` sequences.
///
/// `select_sequences_for_phase` opens with `if phase == Phase::Decode {
/// running.filter(|s| s.status == Status::Decoding) } else { Vec::new() }`.
/// `cargo mutants` flipped BOTH comparisons to `!=` and both survived.
/// Observing this through `build_batch_with_graph` is not enough: the composer
/// SKIPS a decoding sequence on a Prefill round (its remaining-token count is
/// zero), so an inverted gate still produces a clean batch and only the
/// sequence SELECTION is wrong. The tests therefore call
/// `select_sequences_for_phase` directly, which is `pub(super)` and so visible
/// from this module.
#[test]
fn graph_prefill_round_excludes_running_decoders_and_decode_round_admits_only_decoders() {
    let config = SchedulerConfig {
        cuda_graph: SchedulerCudaGraphConfig {
            enabled: true,
            batch_sizes: vec![1, 2, 4],
        },
        ..Default::default()
    };
    let block_size = vllm_traits::BLOCK_SIZE;
    let mut engine = create_test_engine(config, 64);

    // A healthy decoder that must only ever be forwarded on a Decode round.
    let mut decoder = Sequence {
        id: 100,
        tokens: vec![1; block_size * 2],
        kv_blocks: Arc::new(vec![]),
        num_computed_tokens: block_size * 2,
        prompt_len: block_size * 2,
        status: Status::Decoding,
        max_tokens: 50,
        sampling_params: SamplingParams::default(),
        consecutive_decode_rounds: 1,
        priority: Priority::default(),
        degraded_draft: false,
        draft_model_id: None,
    };
    engine.preallocate_kv_blocks(std::slice::from_mut(&mut decoder));
    engine.running.push(decoder);

    // An orphaned mid-prefill sequence sitting in `running`: it is NOT a
    // decoder, so a Decode round must leave it out.
    let orphan = Sequence {
        id: 200,
        tokens: vec![2; block_size * 4],
        kv_blocks: Arc::new(vec![]),
        num_computed_tokens: block_size,
        prompt_len: block_size * 4,
        status: Status::Prefilling,
        max_tokens: 50,
        sampling_params: SamplingParams::default(),
        consecutive_decode_rounds: 0,
        priority: Priority::default(),
        degraded_draft: false,
        draft_model_id: None,
    };
    engine.running.push(orphan);

    // Ask for a PREFILL round explicitly with a queued prefill request.
    engine.add_request(Request::new(300, vec![3; block_size], 5));
    let (selected, _admitted) = engine.select_sequences_for_phase(Phase::Prefill);
    let prefill_ids: Vec<_> = selected.iter().map(|s| s.id).collect();
    assert!(
        prefill_ids.contains(&300),
        "setup: the queued prefill must be admitted, got {prefill_ids:?}"
    );
    assert!(
        !prefill_ids.contains(&100),
        "a Prefill round must NOT forward the running decoder; got {prefill_ids:?}"
    );
    assert!(
        !prefill_ids.contains(&200),
        "a Prefill round must not forward the orphan either; got {prefill_ids:?}"
    );

    // Drain the prefill queue, then ask for a DECODE round.
    while engine.request_queue.phase_len(Phase::Prefill) > 0 {
        let _ = engine.build_batch_with_graph();
    }
    let (selected, _admitted) = engine.select_sequences_for_phase(Phase::Decode);
    let decode_ids: Vec<_> = selected.iter().map(|s| s.id).collect();
    assert!(
        decode_ids.contains(&100),
        "a Decode round must forward the real decoder; got {decode_ids:?}"
    );
    assert!(
        !decode_ids.contains(&200),
        "a Decode round must NOT forward a Prefilling sequence; got {decode_ids:?}"
    );
}

/// The graph path's preemption gate must be charged for the DECODERS' growth.
///
/// `select_sequences_for_phase` fills `sequences` from
/// `running.filter(|s| s.status == Status::Decoding)` and then runs the
/// memory-pressure gate over it — so a decoder that needs one more block is
/// charged that block and the engine preempts to make room. `cargo mutants`
/// flipped that status filter to `!=` and it survived, because the filter's
/// OUTPUT is replaced by `admit_decode_sequences()` a few lines later: the
/// composed batch looks identical either way. What changes is the GATE, which
/// is fed the wrong set — with `!=` it sees no decoders at all, so nothing is
/// preempted, and `admit_decode_sequences` then has to defer the very decoder
/// the gate was supposed to make room for.
///
/// Fixture: a 4-block pool, one decoder holding 3 blocks with a complete table,
/// and one decoder holding 1 block whose next token needs a second. Nothing is
/// free, so only preemption can admit the growing decoder.
#[test]
fn graph_decode_gate_charges_decoders_for_growth() {
    let block_size = vllm_traits::BLOCK_SIZE;
    let mut engine = create_test_engine(SchedulerConfig::default(), 4);

    let mut holder = Sequence {
        id: 2,
        tokens: vec![2; block_size * 3],
        kv_blocks: Arc::new(vec![]),
        num_computed_tokens: block_size * 3,
        prompt_len: block_size * 3,
        status: Status::Decoding,
        max_tokens: 10,
        sampling_params: SamplingParams::default(),
        // Young decoder -> eviction priority 3 (evicted first).
        consecutive_decode_rounds: 1,
        priority: Priority::default(),
        degraded_draft: false,
        draft_model_id: None,
    };
    engine.preallocate_kv_blocks(std::slice::from_mut(&mut holder));
    assert_eq!(holder.kv_blocks.len(), 3, "setup: holder owns 3 blocks");
    engine.running.push(holder);

    // A decoder whose token count has crossed a block boundary: 17 tokens need
    // 2 blocks but only 1 is held.
    let mut growing = Sequence {
        id: 1,
        tokens: vec![1; block_size + 1],
        kv_blocks: Arc::new(vec![]),
        num_computed_tokens: block_size,
        prompt_len: block_size,
        status: Status::Decoding,
        max_tokens: 10,
        sampling_params: SamplingParams::default(),
        // Long-running decoder -> eviction priority 1, so the victim is the
        // holder above rather than this sequence itself.
        consecutive_decode_rounds: 6,
        priority: Priority::default(),
        degraded_draft: false,
        draft_model_id: None,
    };
    engine.preallocate_kv_blocks(std::slice::from_mut(&mut growing));
    assert_eq!(
        growing.kv_blocks.len(),
        1,
        "setup: the growing decoder holds 1 of the 2 blocks its tokens need"
    );
    engine.running.push(growing);
    assert_eq!(
        engine.memory.available_blocks(),
        0,
        "setup: the pool is fully held, so growth REQUIRES preemption"
    );

    let (selected, _admitted) = engine.select_sequences_for_phase(Phase::Decode);
    let selected_ids: Vec<_> = selected.iter().map(|s| s.id).collect();
    assert!(
        selected_ids.contains(&1),
        "the gate must charge decoder 1 for its growth block and preempt to \
         make room; selected = {selected_ids:?}"
    );
    assert!(
        selected
            .iter()
            .find(|s| s.id == 1)
            .is_some_and(|s| s.kv_blocks.len() >= 2),
        "the admitted decoder's table must cover its tokens"
    );
}

/// The graph path must record the admission wait in MILLISECONDS too.
///
/// `select_sequences_for_phase` has its own `wait.as_secs_f64() * 1000.0`,
/// separate from `build_batch`'s. `cargo mutants` turned that `*` into `+` and
/// `/` and both survived, because the existing wait-time test drives
/// `build_batch`, not the graph path — so on a `cuda-graph` build (where the
/// engine routes exclusively through here) the published metric would be wrong
/// by three orders of magnitude in either direction.
#[test]
fn graph_path_records_scheduler_wait_in_milliseconds() {
    let config = SchedulerConfig {
        cuda_graph: SchedulerCudaGraphConfig {
            enabled: true,
            batch_sizes: vec![1, 2, 4],
        },
        ..Default::default()
    };
    let mut engine = create_test_engine(config, 1024);
    engine.add_request(Request::new(1, vec![5; vllm_traits::BLOCK_SIZE], 10));

    std::thread::sleep(std::time::Duration::from_millis(20));

    let batch = engine.build_batch_with_graph().into_regular();
    assert!(!batch.seq_ids.is_empty(), "setup: the request was admitted");

    let recorded = engine.metrics.snapshot().avg_scheduler_wait_time_ms;
    assert!(
        (5.0..500.0).contains(&recorded),
        "a ~20ms wait must record as tens of milliseconds on the graph path; \
         got {recorded}. A `+` mutant gives ~1000 and a `/` mutant ~0.00002."
    );
}

/// An EXACT block fit must not preempt on the CUDA-Graph path either.
///
/// `select_sequences_for_phase` gates preemption on
/// `additional > self.memory.available_blocks()`. With `>=` an exact fit also
/// calls `execute_preemption`, whose phase 1 evicts a decode victim WHOLESALE —
/// its whole table released and its progress reset to a full re-prefill — to
/// free blocks the new sequence did not need.
#[test]
fn graph_exact_block_fit_does_not_preempt() {
    let config = SchedulerConfig {
        cuda_graph: SchedulerCudaGraphConfig {
            enabled: true,
            batch_sizes: vec![1, 2, 4],
        },
        ..Default::default()
    };
    let block_size = vllm_traits::BLOCK_SIZE;
    // Pool of 6 blocks: the decoder holds 4, leaving exactly 2 free.
    let mut engine = create_test_engine(config, 6);

    let mut decoder = Sequence {
        id: 1,
        tokens: vec![1; block_size * 4],
        kv_blocks: Arc::new(vec![]),
        num_computed_tokens: block_size * 4,
        prompt_len: block_size * 4,
        status: Status::Decoding,
        max_tokens: 10,
        sampling_params: SamplingParams::default(),
        consecutive_decode_rounds: 1,
        priority: Priority::default(),
        degraded_draft: false,
        draft_model_id: None,
    };
    engine.preallocate_kv_blocks(std::slice::from_mut(&mut decoder));
    assert_eq!(decoder.kv_blocks.len(), 4, "setup: decoder holds 4 blocks");
    engine.running.push(decoder);
    assert_eq!(
        engine.memory.available_blocks(),
        2,
        "setup: exactly two blocks free"
    );

    engine.add_request(Request::new(2, vec![2; block_size * 2], 10));
    let batch = engine.build_batch_with_graph().into_regular();

    assert!(
        batch.seq_ids.contains(&2),
        "the exactly-fitting request must be admitted; got {:?}",
        batch.seq_ids
    );
    assert!(
        engine.running.iter().any(|s| s.id == 1),
        "the running decoder must NOT be preempted for an exact fit; running = {:?}",
        engine.running.iter().map(|s| s.id).collect::<Vec<_>>()
    );
    assert_eq!(
        engine
            .running
            .iter()
            .find(|s| s.id == 1)
            .map(|s| s.kv_blocks.len()),
        Some(4),
        "the decoder's table must be untouched"
    );
}

/// Graph routing needs BOTH `enabled` and a captured batch size.
///
/// `build_batch_with_graph` ends with
/// `if self.cuda_graph.enabled && self.cuda_graph.supports_batch_size(n)`.
/// `cargo mutants` relaxed the `&&` to `||`, which survived: nothing exercised
/// the two halves independently. With `||` a DISABLED graph still routes
/// through `GraphBatch::Graph` whenever the batch size happens to be captured,
/// replaying a captured graph the operator explicitly turned off.
#[test]
fn graph_routing_requires_both_enabled_and_a_captured_batch_size() {
    let block_size = vllm_traits::BLOCK_SIZE;

    // Helper: one decode round with `batch_sizes` captured and `enabled`
    // controlled, returning (is_graph, cuda_graph_hits, cuda_graph_misses).
    let run = |enabled: bool, batch_sizes: Vec<usize>| {
        let config = SchedulerConfig {
            cuda_graph: SchedulerCudaGraphConfig {
                enabled,
                batch_sizes,
            },
            ..Default::default()
        };
        let mut engine = create_test_engine(config, 64);
        let mut decoder = Sequence {
            id: 7,
            tokens: vec![1; block_size],
            kv_blocks: Arc::new(vec![]),
            num_computed_tokens: block_size,
            prompt_len: block_size,
            status: Status::Decoding,
            max_tokens: 10,
            sampling_params: SamplingParams::default(),
            consecutive_decode_rounds: 1,
            priority: Priority::default(),
            degraded_draft: false,
            draft_model_id: None,
        };
        engine.preallocate_kv_blocks(std::slice::from_mut(&mut decoder));
        engine.running.push(decoder);
        // The phase selector starts in Prefill and completes it on an empty
        // prefill queue, so this single round is already a Decode round.
        let batch = engine.build_batch_with_graph();
        (
            batch.is_graph(),
            engine.metrics.get_counter("cuda_graph_hits_total"),
            engine.metrics.get_counter("cuda_graph_misses_total"),
        )
    };

    let (is_graph, hits, misses) = run(true, vec![1]);
    assert!(
        is_graph,
        "enabled + captured size must route through the graph"
    );
    assert_eq!(hits, 1, "a graph-routed round records exactly one hit");
    assert_eq!(misses, 0, "a graph-routed round records no miss");

    let (is_graph, hits, misses) = run(true, vec![2]);
    assert!(
        !is_graph,
        "an enabled graph that never captured this batch size must fall back"
    );
    assert_eq!(hits, 0, "no hit without a captured size");
    assert_eq!(misses, 1, "the miss must be counted");

    let (is_graph, hits, misses) = run(false, vec![1]);
    assert!(
        !is_graph,
        "a DISABLED graph must never route through it, even with a captured size"
    );
    assert_eq!(hits, 0, "a disabled graph must record no hit");
    assert_eq!(misses, 1, "a disabled graph records the miss");
}

/// Under `multi-node`, each block's chain hash must cover THAT block's tokens.
///
/// `preallocate_kv_blocks` and `grow_running_table` derive the token slice from
/// the block index with `start = block_idx * BLOCK_SIZE` and
/// `end = (start + BLOCK_SIZE).min(tokens.len())`, inside
/// `#[cfg(feature = "multi-node")]` blocks. A default-feature scan cannot see
/// those mutants at all; with the feature ON, `cargo mutants` still relaxed the
/// multiplication to `/` and both `+`s to `*`, all surviving — the suite never
/// ran a multi-node engine over a sequence spanning MORE THAN ONE block, where
/// those expressions stop coinciding with the first block's slice.
///
/// A wrong slice silently poisons the distributed prefix cache: peers compute
/// their hashes from the real block contents, so a node that hashed block 0's
/// tokens for block 1 can never match a peer's entry and `TransferKVBlock`
/// resolves to nothing.
#[cfg(feature = "multi-node")]
#[test]
fn multi_node_chain_hashes_cover_each_blocks_own_tokens() {
    let block_size = vllm_traits::BLOCK_SIZE;
    // Two full blocks plus one token, so block 1 has both a non-zero start and
    // a partial end.
    let prompt: Vec<u32> = (0..block_size * 2 + 1).map(|t| t as u32 + 1).collect();
    let mut engine = create_test_engine(SchedulerConfig::default(), 64);
    let seq_id = engine.add_request(Request::new(0, prompt.clone(), 5));

    let _batch = engine.build_batch();

    // Expected chain, computed with the manager's own hasher: block 0 chains
    // from 0, each later block from its predecessor. The prompt spans THREE
    // blocks (16 + 16 + 1 tokens), so the final cursor is block 2's hash —
    // computed from the one-token slice `[32..33]`. A `block_idx / BLOCK_SIZE`
    // mutant collapses every start back to 0 and re-hashes block 0's tokens.
    let hasher = engine.memory.hasher();
    let mut expected = 0u64;
    let mut seen = Vec::new();
    for start in (0..prompt.len()).step_by(block_size) {
        let end = (start + block_size).min(prompt.len());
        expected = hasher.hash_block(expected, &prompt[start..end]);
        seen.push(expected);
    }
    assert_eq!(seen.len(), 3, "fixture: the prompt spans three blocks");
    assert_ne!(
        seen[0], seen[1],
        "fixture: distinct slices hash differently"
    );
    assert_ne!(
        seen[1], seen[2],
        "fixture: distinct slices hash differently"
    );

    assert_eq!(
        engine.chain_cursors.get(&seq_id).copied(),
        Some(expected),
        "the per-sequence cursor must be the LAST block's hash, chained over \
         that block's own token slice"
    );
}

/// The decode-growth path must chain each NEW block from the previous hash.
///
/// `grow_running_table` records each freshly-added block's tokens with
/// `start = (have + offset) * BLOCK_SIZE`, `end = (start + BLOCK_SIZE).min(..)`.
/// `cargo mutants` relaxed both `+`s to `*` and they survived because no
/// multi-node test grew a decode table across a block boundary. A wrong start
/// re-feeds the previous block's tokens, so the new block's hash — and every
/// hash chained after it — diverges from what peers compute, making the
/// distributed prefix cache miss forever.
///
/// The fixture grows TWO blocks at once (a 48-token decoder holding a single
/// block): with only one growth block the `.min(tokens.len())` clamp hides the
/// `end` mutant, because at that point the sequence has no tokens past the
/// block being filled.
#[cfg(feature = "multi-node")]
#[test]
fn multi_node_decode_growth_chains_the_new_block_from_the_previous_hash() {
    let block_size = vllm_traits::BLOCK_SIZE;
    let mut engine = create_test_engine(SchedulerConfig::default(), 64);

    // 48 tokens span three blocks but only one is held, so admission grows two.
    let tokens: Vec<u32> = (0..block_size * 3).map(|t| t as u32 + 1).collect();
    let mut decoder = Sequence {
        id: 5,
        tokens: tokens.clone(),
        kv_blocks: Arc::new(vec![]),
        num_computed_tokens: tokens.len(),
        prompt_len: tokens.len(),
        status: Status::Decoding,
        max_tokens: 10,
        sampling_params: SamplingParams::default(),
        consecutive_decode_rounds: 1,
        priority: Priority::default(),
        degraded_draft: false,
        draft_model_id: None,
    };
    // Seed exactly one block by hand: preallocate only the first block's worth.
    let mut head = Sequence {
        tokens: tokens[..block_size].to_vec(),
        ..decoder.clone()
    };
    engine.preallocate_kv_blocks(std::slice::from_mut(&mut head));
    decoder.kv_blocks = head.kv_blocks.clone();
    assert_eq!(decoder.kv_blocks.len(), 1, "setup: one block held");
    engine.running.push(decoder);

    let _batch = engine.build_batch();
    assert!(
        engine.running.iter().any(|s| s.id == 5),
        "setup: the decoder was admitted (table grown), not deferred"
    );

    let hasher = engine.memory.hasher();
    let mut expected = 0u64;
    for start in (0..tokens.len()).step_by(block_size) {
        let end = (start + block_size).min(tokens.len());
        expected = hasher.hash_block(expected, &tokens[start..end]);
    }
    assert_eq!(
        engine.chain_cursors.get(&5).copied(),
        Some(expected),
        "the grown blocks must chain over their OWN slices: block 1 from \
         tokens[{block_size}..{}], block 2 from tokens[{}..{}]",
        block_size * 2,
        block_size * 2,
        block_size * 3
    );
}
