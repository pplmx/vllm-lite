// crates/core/tests/distributed_kv_integration.rs
//
// Verifies that Engine owns a `DistributedKVCache` when constructed via
// `EngineBuilder::with_distributed_kv`, that the status / stats
// accessors reflect the cache's state, and that the cache is wired
// through to the scheduler's `MemoryManager` so block allocate / free
// round-trips through it.

#![cfg(feature = "multi-node")]

use std::sync::Arc;
use vllm_core::engine::EngineBuilder;
use vllm_dist::distributed_kv::protocol::NodeId;
use vllm_dist::{CacheConfig, DistributedKVCache};
use vllm_traits::StubModelBackend;

fn make_cache() -> Arc<DistributedKVCache> {
    let config = CacheConfig::new(NodeId(0), 4);
    Arc::new(DistributedKVCache::new(config))
}

#[test]
fn engine_without_distributed_kv_reports_disabled() {
    // Default EngineBuilder path: no cache installed.
    let engine: vllm_core::engine::Engine =
        EngineBuilder::new(Box::new(StubModelBackend::default())).build();
    assert!(
        !engine.distributed_kv_enabled(),
        "default EngineBuilder must not install a distributed KV cache"
    );
    assert!(
        engine.distributed_kv_stats().is_none(),
        "stats accessor must return None when no cache is installed"
    );
}

#[test]
fn engine_with_distributed_kv_reports_enabled() {
    // Builder path: cache installed via with_distributed_kv.
    let cache = make_cache();
    let engine = EngineBuilder::new(Box::new(StubModelBackend::default()))
        .with_distributed_kv(cache)
        .build();
    assert!(
        engine.distributed_kv_enabled(),
        "EngineBuilder::with_distributed_kv must flip the enabled flag"
    );
}

#[test]
fn engine_distributed_kv_stats_reflect_cache_state() {
    // The engine returns the cache's own stats snapshot. We mutate the
    // cache directly and confirm the engine reports the new values.
    let cache = make_cache();
    cache.put(42, 0xCAFE);
    cache.put(43, 0xBEEF);
    let miss = cache.get(99);
    assert!(miss.is_none(), "miss path bumps the misses counter");

    let engine = EngineBuilder::new(Box::new(StubModelBackend::default()))
        .with_distributed_kv(Arc::clone(&cache))
        .build();
    let stats = engine
        .distributed_kv_stats()
        .expect("cache installed → stats must be Some");
    assert_eq!(stats.updates, 2, "two put() calls → two updates");
    assert_eq!(stats.misses, 1, "one miss → one miss");
}

#[test]
fn multiple_engines_can_share_a_cache_via_arc() {
    // The cache is wrapped in Arc inside the builder so two engines can
    // point at the same cache (e.g. for cross-engine coherence in a
    // server that hosts multiple model instances on the same node).
    let cache = make_cache();
    let cache_for_engine_2 = Arc::clone(&cache);

    let engine_1 = EngineBuilder::new(Box::new(StubModelBackend::default()))
        .with_distributed_kv(Arc::clone(&cache))
        .build();
    let engine_2 = EngineBuilder::new(Box::new(StubModelBackend::default()))
        .with_distributed_kv(cache_for_engine_2)
        .build();

    cache.put(1, 100);
    let stats_1 = engine_1.distributed_kv_stats().unwrap();
    let stats_2 = engine_2.distributed_kv_stats().unwrap();
    assert_eq!(stats_1.updates, stats_2.updates);
    assert_eq!(stats_1.updates, 1);
}

#[test]
fn engine_propagates_distributed_kv_to_scheduler_memory_manager() {
    // When the engine is built with a distributed cache, the cache is
    // propagated into the scheduler's MemoryManager so every block
    // allocate / free round-trips through the cache. We exercise this
    // end-to-end by:
    //   1. Building an engine with a cache.
    //   2. Asking the scheduler to allocate blocks (the same call the
    //      step loop makes for every prefill).
    //   3. Asserting the cache's `updates` counter moved.
    let cache = make_cache();
    let mut engine = EngineBuilder::new(Box::new(StubModelBackend::default()))
        .with_num_kv_blocks(64)
        .with_distributed_kv(Arc::clone(&cache))
        .build();

    // The scheduler owns the MemoryManager. We grab a mutable handle
    // and ask it to allocate blocks — this is what step() does under
    // the hood during prefill.
    let scheduler = &mut engine.scheduler;
    let blocks = scheduler
        .memory_mut()
        .allocate(2)
        .expect("prefill-time allocation must succeed");
    assert_eq!(blocks.len(), 2);

    let stats = engine
        .distributed_kv_stats()
        .expect("cache installed → stats must be Some");
    assert_eq!(
        stats.updates, 2,
        "two block allocations should register two puts in the cache"
    );
}

#[test]
fn engine_scheduler_lookup_distributed_prefix_round_trip() {
    // Phase 19 OPS-05b3 — verify the scheduler-level
    // `lookup_distributed_prefix` end-to-end through the engine.
    //
    // 1. Build an engine with a distributed cache.
    // 2. Allocate blocks via the MemoryManager and record tokens for
    //    each — this publishes content-hash-keyed entries into the
    //    cache via `record_block_tokens`.
    // 3. Call `scheduler.lookup_distributed_prefix(prompt)` and
    //    assert it returns a `DistributedPrefixMatch` covering the
    //    same blocks.
    use std::sync::Arc;
    use vllm_core::scheduler::memory::DistributedPrefixMatch;
    use vllm_traits::{TokenId, XorShiftHasher};

    let cache = make_cache();
    let mut engine = EngineBuilder::new(Box::new(StubModelBackend::default()))
        .with_num_kv_blocks(64)
        .with_distributed_kv(Arc::clone(&cache))
        .build();

    let scheduler = &mut engine.scheduler;
    scheduler
        .memory_mut()
        .set_block_hasher(Arc::new(XorShiftHasher));

    // Build a 2-block prompt (BLOCK_SIZE = 16).
    const BLOCK_SIZE: usize = 16;
    let prompt: Vec<TokenId> = (0..(2 * BLOCK_SIZE))
        .map(|i| (i + 200) as TokenId)
        .collect();

    // Allocate 2 blocks, record tokens with a real chain cursor so
    // the cache entries are content-hash-keyed (the path
    // `lookup_distributed_prefix` queries).
    let blocks = scheduler
        .memory_mut()
        .allocate(2)
        .expect("allocation must succeed");
    let h0 = scheduler
        .memory_mut()
        .record_block_tokens(blocks[0], 0, &prompt[..BLOCK_SIZE]);
    let h1 = scheduler
        .memory_mut()
        .record_block_tokens(blocks[1], h0, &prompt[BLOCK_SIZE..]);
    let _ = h1;

    // Lookup the same prompt — both chain hashes are present, so
    // the lookup returns a full match.
    let result = scheduler
        .lookup_distributed_prefix(&prompt)
        .expect("all chain hashes recorded → lookup must hit");
    let expected = DistributedPrefixMatch {
        matched_blocks: 2,
        matched_tokens: 2 * BLOCK_SIZE,
        hasher_name: "xorshift",
    };
    assert_eq!(result, expected);
}

#[test]
fn engine_scheduler_lookup_distributed_prefix_partial_match() {
    // Record tokens for 1 block; lookup a 2-block prompt — only the
    // first block should match (the 2nd block's chain hash is unknown).
    use std::sync::Arc;
    use vllm_traits::{TokenId, XorShiftHasher};

    let cache = make_cache();
    let mut engine = EngineBuilder::new(Box::new(StubModelBackend::default()))
        .with_num_kv_blocks(64)
        .with_distributed_kv(Arc::clone(&cache))
        .build();

    let scheduler = &mut engine.scheduler;
    scheduler
        .memory_mut()
        .set_block_hasher(Arc::new(XorShiftHasher));

    const BLOCK_SIZE: usize = 16;
    let prompt: Vec<TokenId> = (0..(2 * BLOCK_SIZE))
        .map(|i| (i + 300) as TokenId)
        .collect();

    let blocks = scheduler
        .memory_mut()
        .allocate(2)
        .expect("allocation must succeed");
    scheduler
        .memory_mut()
        .record_block_tokens(blocks[0], 0, &prompt[..BLOCK_SIZE]);
    // Note: do NOT record blocks[1] — its chain hash must remain
    // absent from the cache for the partial-match assertion below.

    let result = scheduler
        .lookup_distributed_prefix(&prompt)
        .expect("first block recorded → partial hit expected");
    assert_eq!(result.matched_blocks, 1);
    assert_eq!(result.matched_tokens, BLOCK_SIZE);
    assert_eq!(result.hasher_name, "xorshift");
}
