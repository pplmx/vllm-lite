//! `build_batch` admission regressions (RIL TASK-046, ISS-041/ISS-042).
//!
//! Two failure modes guarded here:
//! - **ISS-041** — `build_batch` drains *every* waiting prefill into `running`,
//!   but the composer only serves `max_batch_size` / breaks on the token
//!   budget. Without the fix the un-composed overflow stays `Status::Prefilling`
//!   in `running` forever (running is only re-included for `Decoding`), pinning
//!   its pre-allocated KV blocks. The fix releases and re-queues it.
//! - **ISS-042** — when `preallocate_kv_blocks` cannot obtain the full KV
//!   table (pool exhausted even after the preemption gate), the under-allocated
//!   prefill was still admitted, so `write_prefill_kv` falls back to block 0 and
//!   corrupts the cache (ISS-026/028 elsewhere). The fix re-queues such
//!   sequences instead of admitting them.

use std::sync::Arc;
use vllm_core::metrics::EnhancedMetricsCollector;
use vllm_core::scheduler::SchedulerEngine;
use vllm_core::types::{Request, SchedulerConfig, SequencePackingConfig, Status};
use vllm_traits::BLOCK_SIZE;

fn create_test_engine(config: SchedulerConfig, num_kv_blocks: usize) -> SchedulerEngine {
    let metrics = Arc::new(EnhancedMetricsCollector::new());
    SchedulerEngine::new(config, num_kv_blocks, metrics)
}

#[test]
fn prefill_overflow_is_requeued_not_stranded_in_running() {
    // max_num_seqs drives the composer's batch-size cap (SchedulerEngine::new
    // wires BatchCompositionConfig.max_batch_size from config.max_num_seqs), so
    // two prefills fit per round and the remaining three must be deferred.
    let config = SchedulerConfig::builder()
        .with_max_num_seqs(2)
        .with_packing(SequencePackingConfig {
            enabled: false,
            ..Default::default()
        })
        .build();
    let mut engine = create_test_engine(config, 1024);

    for id in 1..=5u64 {
        engine.add_request(Request::new(id, vec![1; 20], 10));
    }

    let batch = engine.build_batch();

    // Only max_batch_size sequences are composed this round...
    assert_eq!(
        batch.seq_ids.len(),
        2,
        "batch must be capped at max_batch_size"
    );
    // ...and the overflow is back in the waiting queue, not stranded in
    // `running` as Prefilling (which would never be re-scheduled).
    assert_eq!(engine.waiting_count(), 3, "overflow must be requeued");
    assert_eq!(
        engine.running_count(),
        2,
        "only the composed sequences stay running"
    );

    // Every composed running sequence holds its full KV table (no partial
    // allocation leaked into the batch) and all running entries belong to the
    // composed set — the requeue must not leave half-allocated blocks pinned.
    let running = engine.running();
    assert_eq!(running.len(), 2);
    for seq in &running {
        assert_eq!(seq.status, Status::Prefilling);
        let blocks_needed = seq.tokens.len().div_ceil(BLOCK_SIZE);
        assert!(
            seq.kv_blocks.len() >= blocks_needed,
            "composed sequence {} must hold a full KV table",
            seq.id
        );
        assert!(
            batch.seq_ids.contains(&seq.id),
            "running sequence {} must be in the composed batch",
            seq.id
        );
    }

    // The deferred sequences are retried on later rounds instead of stalling
    // forever: every request must eventually be composed.
    // `batch` is not used after this point, so moving its seq ids is fine.
    let mut served: Vec<u64> = batch.seq_ids;
    for _ in 0..6 {
        if engine.waiting_count() == 0 {
            break;
        }
        let b = engine.build_batch();
        served.extend(b.seq_ids.iter().copied());
    }
    served.sort_unstable();
    served.dedup();
    assert_eq!(
        served,
        vec![1, 2, 3, 4, 5],
        "every request must eventually be composed"
    );
    assert_eq!(engine.waiting_count(), 0, "the queue must fully drain");
}

#[test]
fn under_allocated_prefill_is_requeued_not_admitted() {
    // 2 KV blocks but the 100-token prompt needs ceil(100/16) = 7, so
    // pre-allocation can only obtain 2 before the pool is exhausted.
    let config = SchedulerConfig::builder()
        .with_packing(SequencePackingConfig {
            enabled: false,
            ..Default::default()
        })
        .build();
    let mut engine = create_test_engine(config.clone(), 2);

    engine.add_request(Request::new(9, vec![1; 100], 10));

    let batch = engine.build_batch();

    // The partially-allocated sequence must NOT be admitted (that would let
    // `write_prefill_kv` fall back to block 0); it goes back to the queue.
    assert!(
        batch.seq_ids.is_empty(),
        "under-allocated prefill must not be served"
    );
    assert_eq!(engine.waiting_count(), 1, "sequence must be requeued");
    assert_eq!(
        engine.running_count(),
        0,
        "nothing may be admitted with a partial KV table"
    );

    // Once memory is available, the same request is admitted with a full table.
    let mut engine = create_test_engine(config, 64);
    engine.add_request(Request::new(9, vec![1; 100], 10));
    let batch = engine.build_batch();
    assert_eq!(
        batch.seq_ids,
        vec![9],
        "with enough blocks the prefill is served"
    );
    let running = engine.running();
    assert_eq!(running.len(), 1);
    assert_eq!(
        running[0].kv_blocks.len(),
        100usize.div_ceil(BLOCK_SIZE),
        "admitted sequence must hold its full KV table"
    );
}
