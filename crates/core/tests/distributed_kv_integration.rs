// crates/core/tests/distributed_kv_integration.rs
//
// Phase 19 OPS-05a — verifies that Engine owns a `DistributedKVCache`
// when constructed via `EngineBuilder::with_distributed_kv`, and that the
// status / stats accessors reflect the cache's state. The allocator
// hooks that wire block allocate/free into the cache are OPS-05b; this
// file only exercises the *seam*.

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
    // cache directly (the allocator hooks are OPS-05b) and confirm the
    // engine reports the new values.
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
