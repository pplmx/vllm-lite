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
use crate::scheduler::engine::SchedulerEngine;
use crate::types::{Request, SchedulerConfig};

fn create_test_engine(config: SchedulerConfig, num_kv_blocks: usize) -> SchedulerEngine {
    let metrics = Arc::new(EnhancedMetricsCollector::new());
    SchedulerEngine::new(config, num_kv_blocks, metrics)
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

/// Regression (RIL TASK-001 / ISS-001): `execute_preemption` must never
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

/// Regression (RIL TASK-003 / ISS-003): completing the same prompt
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

/// Regression (RIL TASK-005 / ISS-005): a request whose prompt fully
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
