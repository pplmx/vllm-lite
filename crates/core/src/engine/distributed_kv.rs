//! Engine distributed-KV helpers: status query, stats accessor, and the
//! post-construction setter used by [`crate::engine::EngineBuilder`].
//!
//! Exposes the *presence* of the cache (so callers can check it's wired
//! up) and the cache's own stats snapshot (so callers can introspect the
//! cache's view of activity). The actual write-through happens inside
//! [`crate::scheduler::memory::MemoryManager`] — see that module for
//! how `allocate` / `free` round-trip through the cache.

#[cfg(feature = "multi-node")]
use std::sync::Arc;

// Sub-module for distributed-KV accessor methods on Engine.
// See mod.rs for the Engine struct definition.

impl crate::engine::Engine {
    /// Install a distributed KV-cache after construction.
    ///
    /// Used by [`crate::engine::EngineBuilder::with_distributed_kv`] to
    /// install the cache. Crate-internal because the cache type lives
    /// below the `core → dist` boundary; embedders go through the builder.
    ///
    /// Also propagates the cache into the scheduler's
    /// [`crate::scheduler::engine::SchedulerEngine`] so every subsequent
    /// block allocate / free round-trips through the cache.
    #[cfg(feature = "multi-node")]
    pub(crate) fn set_distributed_kv(&mut self, cache: Arc<vllm_dist::DistributedKVCache>) {
        self.scheduler.set_distributed_kv(Arc::clone(&cache));
        self.distributed_kv = Some(cache);
    }

    /// Whether a [`vllm_dist::DistributedKVCache`] is installed.
    ///
    /// Returns `true` when the engine was built (or later configured)
    /// with a distributed KV cache; `false` on single-node builds. Use
    /// this before calling [`Self::distributed_kv_stats`] when the
    /// caller can't statically prove the multi-node feature is on.
    #[cfg(feature = "multi-node")]
    #[must_use]
    pub const fn distributed_kv_enabled(&self) -> bool {
        self.distributed_kv.is_some()
    }

    /// Always-`false` stub for non-`multi-node` builds. Mirrors the
    /// `cuda_graph_enabled` / `cuda_graph.rs` pattern so call sites
    /// compile unchanged regardless of feature flags.
    #[cfg(not(feature = "multi-node"))]
    #[must_use]
    pub const fn distributed_kv_enabled(&self) -> bool {
        false
    }

    /// Snapshot the cache's internal statistics (hits / misses /
    /// invalidations / updates).
    ///
    /// Returns `None` when no cache is installed. The returned snapshot
    /// is a cheap copy — safe to call from metrics exporters on the
    /// hot path.
    #[cfg(feature = "multi-node")]
    #[must_use]
    pub fn distributed_kv_stats(&self) -> Option<vllm_dist::distributed_kv::cache::CacheStats> {
        self.distributed_kv.as_ref().map(|c| c.stats())
    }
}
