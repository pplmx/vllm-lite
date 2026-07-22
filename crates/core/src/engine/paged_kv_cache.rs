//! Engine `PagedKvCache` helpers: setter + wrapper getter, gated behind
//! `multi-node` per ADR-008. The actual wrapper lives in
//! `vllm_model::paged_tensor::PagedKvCacheWrapper` (P40) — this module
//! just installs the wrapper on the engine's `MemoryManager` and exposes
//! the wrapper to the server's gRPC bootstrap.

#[cfg(feature = "multi-node")]
use std::sync::Arc;

#[cfg(feature = "multi-node")]
use parking_lot::Mutex;

#[cfg(feature = "multi-node")]
use vllm_dist::BlockDataSource;

#[cfg(feature = "multi-node")]
use vllm_model::paged_tensor::PagedKvCache;

impl crate::engine::Engine {
    /// Install a `PagedKvCache` after construction (Phase 41 OPS-32a
    /// second-half; refined for P42 receiver-side sink).
    ///
    /// Constructs a [`vllm_model::paged_tensor::PagedKvCacheWrapper`]
    /// internally and propagates it to
    /// [`crate::scheduler::engine::SchedulerEngine::set_block_data_source`]
    /// so every subsequent gRPC `TransferKVBlock` call resolves to the
    /// wrapper.
    ///
    /// The cache is wrapped in `Arc<Mutex<PagedKvCache>>` so both the
    /// engine (for diagnostics) and the wrapper (for the BlockDataSource
    /// + BlockSink implementations) share the same underlying data.
    /// The `Mutex` is required because `PagedKvCache::write_block_bytes`
    /// takes `&mut self` for `slice_assign` on the candle tensors.
    ///
    /// Crate-internal — embedders go through
    /// [`crate::engine::EngineBuilder::with_paged_kv_cache`].
    #[cfg(feature = "multi-node")]
    pub(crate) fn set_paged_kv_cache(&mut self, cache: Arc<PagedKvCache>) {
        // Wrap the cache in a Mutex. Both the engine and the wrapper
        // will hold an `Arc<Mutex<PagedKvCache>>` pointing at the same
        // data; either can lock to read/write. This requires the input
        // Arc to be uniquely owned (which the builder ensures by
        // taking the cache from `EngineBuilder::paged_kv_cache`).
        let cache_lock = Arc::new(Mutex::new(
            Arc::try_unwrap(cache)
                .expect("PagedKvCache must be uniquely owned at Engine::set_paged_kv_cache time"),
        ));
        let wrapper: Arc<dyn BlockDataSource + Send + Sync> = Arc::new(
            vllm_model::paged_tensor::PagedKvCacheWrapper::from_arc_mutex(Arc::clone(&cache_lock)),
        );
        self.scheduler.set_block_data_source(Arc::clone(&wrapper));
        self.paged_kv_cache = Some(cache_lock);
        self.paged_kv_cache_wrapper = Some(wrapper);
    }

    /// Returns the `BlockDataSource` wrapper if a `PagedKvCache` is wired in.
    ///
    /// The server bootstrap (`crates/server/src/bootstrap/engine.rs`) uses
    /// this to populate the gRPC server's `start_grpc_server_with_listener`
    /// `block_data_source` parameter without re-constructing the wrapper.
    #[cfg(feature = "multi-node")]
    #[must_use]
    pub fn paged_kv_cache_wrapper(&self) -> Option<Arc<dyn BlockDataSource + Send + Sync>> {
        self.paged_kv_cache_wrapper.as_ref().map(Arc::clone)
    }

    /// Returns the wired `PagedKvCache` (or `None`).
    ///
    /// Useful for diagnostics / tests that need to read K/V bytes
    /// directly without going through the gRPC layer. Returns an
    /// `Arc<Mutex<PagedKvCache>>` (P42) since the wrapper and the
    /// engine share the same underlying data via a Mutex.
    #[cfg(feature = "multi-node")]
    #[must_use]
    pub fn paged_kv_cache(&self) -> Option<Arc<parking_lot::Mutex<PagedKvCache>>> {
        self.paged_kv_cache.as_ref().map(Arc::clone)
    }

    /// Always-`None` stub for non-`multi-node` builds. Mirrors
    /// `distributed_kv_enabled` so call sites compile unchanged.
    #[cfg(not(feature = "multi-node"))]
    #[must_use]
    pub const fn paged_kv_cache_wrapper(
        &self,
    ) -> Option<Arc<dyn vllm_dist::BlockDataSource + Send + Sync>> {
        None
    }
}
