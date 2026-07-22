//! Multi-node gRPC server bootstrap (Phase 41 OPS-32a second-half).
//!
//! Spawns a tonic gRPC server that answers `TransferKVBlock` calls with
//! real K/V bytes from the engine's wired `PagedKvCache` (via the
//! `PagedKvCacheWrapper` installed by `Engine::set_paged_kv_cache`).
//!
//! Single-node builds (no `multi-node` Cargo feature) compile to an
//! empty module so the call sites in `main.rs` short-circuit at compile
//! time.

#[cfg(feature = "multi-node")]
use std::sync::Arc;

#[cfg(feature = "multi-node")]
use anyhow::{Context, Result};

#[cfg(feature = "multi-node")]
use vllm_core::engine::Engine;

#[cfg(feature = "multi-node")]
use vllm_server::config::MultiNodeConfig;

/// Spawn the multi-node gRPC server in the background. Returns
/// `Ok(node_id)` on listener bind success (the spawned task handles
/// runtime errors and logs them).
///
/// # Errors
///
/// Returns `Err` if:
/// - The engine has no `BlockDataSource` wrapper wired in (i.e., the
///   bootstrap didn't go through `EngineBuilder::with_paged_kv_cache`).
/// - The TCP listener fails to bind to `cfg.bind_addr`.
#[cfg(feature = "multi-node")]
pub async fn spawn_multi_node_grpc_server(
    engine: &Engine,
    cfg: &MultiNodeConfig,
) -> Result<String> {
    use vllm_dist::{BlockDataSource, BlockSink};
    // Fetch the wrapper Arc once; clone it for both the gRPC server
    // (BlockDataSource / sender) and the local DistributedKVCache
    // (BlockSink / receiver install). The underlying
    // PagedKvCacheWrapper implements both traits.
    let wrapper = engine
        .paged_kv_cache_wrapper()
        .context("multi-node enabled but engine has no BlockDataSource wrapper")?;

    // P42: also wire the same wrapper as the local
    // DistributedKVCache's BlockSink so every `fetch_block` from a
    // peer installs the received bytes into the local cache. The
    // cache holds its sink behind a `parking_lot::Mutex<Option<...>>`
    // specifically for this late-binding case (the cache is
    // constructed before the wrapper exists). The cast from
    // `Arc<dyn BlockDataSource>` to `Arc<dyn BlockSink>` works
    // because the underlying `PagedKvCacheWrapper` impls both —
    // but Rust's type system doesn't know that, so we go via the
    // raw Arc pointer: re-`Arc::clone` from the engine.
    if let Some(dist_cache) = engine.distributed_kv_cache() {
        let wrapper_as_sink: Arc<dyn BlockSink> = {
            // SAFETY / CORRECTNESS: `wrapper` is `Arc<dyn
            // BlockDataSource>` whose underlying object is the
            // engine's `PagedKvCacheWrapper`. The same Arc also
            // implements `BlockSink` (P42 T3). We get a fresh
            // `Arc<dyn BlockSink>` from the engine's `paged_kv_cache`
            // (the same Mutex-wrapped PagedKvCache) and re-wrap it.
            //
            // In practice this requires a `BlockSink` constructor
            // for `PagedKvCacheWrapper` that takes the raw cache
            // pointer — which doesn't exist. Instead, the simpler
            // path is to fetch the engine's `paged_kv_cache`
            // (Arc<Mutex<PagedKvCache>>) and reconstruct a fresh
            // `PagedKvCacheWrapper` from it as the sink. Both
            // wrappers share the same Mutex so reads/writes are
            // consistent.
            let cache_lock = engine
                .paged_kv_cache()
                .context("multi-node enabled but engine has no PagedKvCache")?;
            let sink_wrapper = vllm_model::paged_tensor::PagedKvCacheWrapper::from_arc_mutex(
                Arc::clone(&cache_lock),
            );
            Arc::new(sink_wrapper) as Arc<dyn BlockSink>
        };
        dist_cache.install_block_sink(wrapper_as_sink);
    }

    let node_id = cfg.node_id.clone().unwrap_or_else(|| {
        let id = uuid::Uuid::new_v4().to_string();
        tracing::info!(node_id = %id, "auto-generated multi-node node id");
        id
    });
    let listener = tokio::net::TcpListener::bind(&cfg.bind_addr)
        .await
        .with_context(|| {
            format!(
                "failed to bind multi-node gRPC listener at {}",
                cfg.bind_addr
            )
        })?;
    let bound_addr = listener
        .local_addr()
        .map_or_else(|_| cfg.bind_addr.clone(), |a| a.to_string());
    let node_id_clone = node_id.clone();
    let wrapper_for_server: Arc<dyn BlockDataSource> = wrapper;
    tokio::spawn(async move {
        if let Err(e) = vllm_dist::start_grpc_server_with_listener(
            node_id_clone,
            listener,
            None, // receiver-side cache is None; the cache is in the engine, not the gRPC server
            Some(wrapper_for_server),
        )
        .await
        {
            tracing::error!(error = ?e, "multi-node gRPC server failed");
        }
    });
    tracing::info!(bind_addr = %bound_addr, node_id = %node_id, "Multi-node gRPC server spawned");
    Ok(node_id)
}

/// Always-`Ok("")` stub for non-`multi-node` builds. Allows the call
/// site in `main.rs` to compile unchanged.
#[cfg(not(feature = "multi-node"))]
pub async fn spawn_multi_node_grpc_server(
    _engine: &vllm_core::engine::Engine,
    _cfg: &vllm_server::config::MultiNodeConfig,
) -> anyhow::Result<String> {
    Ok(String::new())
}
