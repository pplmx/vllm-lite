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
use crate::types::{Priority, Request, SamplingParams, SchedulerConfig, Sequence, Status};

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
