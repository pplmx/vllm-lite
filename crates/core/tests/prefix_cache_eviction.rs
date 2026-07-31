//! Regression: prefix-cache entries must not pin KV blocks forever.
//!
//! Finished sequences insert their KV blocks into the radix-tree prefix
//! cache, and the cache holds a refcount on each block. Without
//! eviction, those blocks never return to the allocator — a long-running
//! server exhausts the block pool even with zero running sequences, and
//! new requests cannot obtain KV cache (verified pre-fix: a 16-block
//! pool stayed at 16/16 used after 16 distinct finished prompts).
//!
//! `execute_preemption` now clears the prefix cache (dropping its
//! refcounts) before evicting live sequences, so memory pressure
//! releases cached blocks back to the allocator.

use tokio::sync::mpsc;
use vllm_core::engine::Engine;
use vllm_core::types::{Request, SchedulerConfig, TokenId};
use vllm_testing::StubModel;

fn run_prompt(engine: &mut Engine, id: u64, prompt: Vec<TokenId>) {
    let (tx, _rx) = mpsc::channel(64);
    engine.add_request(Request::new(id, prompt, 2), tx);
    let mut steps = 0;
    while engine.has_pending() {
        engine.step().expect("step should succeed");
        steps += 1;
        assert!(steps < 500, "request {id} never completes");
    }
}

#[test]
fn prefix_cache_eviction_frees_blocks_for_new_requests() {
    // 16-block pool. 16 distinct 2-token prompts each need one block on
    // finish (2 tokens <= BLOCK_SIZE), so without eviction the pool
    // would be fully pinned by the cache.
    let mut engine = Engine::with_config(
        StubModel::default(),
        None,
        SchedulerConfig::default(),
        4,
        16,
    );

    for i in 0..16u64 {
        run_prompt(
            &mut engine,
            i,
            vec![i as TokenId + 100, (i + 1) as TokenId + 100],
        );
    }

    // New request with a fresh prompt must still be able to run to
    // completion — the preemption path evicts the pinned cache blocks.
    let (tx, _rx) = mpsc::channel(64);
    engine.add_request(Request::new(999, vec![7, 8, 9, 10], 2), tx);
    let mut steps = 0;
    while engine.has_pending() {
        engine.step().expect("step should succeed");
        steps += 1;
        assert!(steps < 500, "new request never completes");
    }

    // The allocator must have reclaimed blocks for the new request
    // (usage below the full pool), not run it KV-less.
    let (used, total) = engine.scheduler.get_kv_cache_usage();
    assert!(
        used < total,
        "prefix cache must release pinned blocks under pressure: used {used}/{total}"
    );
    assert_eq!(total, 16);
}
