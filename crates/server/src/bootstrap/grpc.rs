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
    use vllm_dist::BlockDataSource;
    let wrapper = engine
        .paged_kv_cache_wrapper()
        .context("multi-node enabled but engine has no BlockDataSource wrapper")?;
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
    let wrapper_clone: Arc<dyn BlockDataSource> = wrapper;
    tokio::spawn(async move {
        if let Err(e) = vllm_dist::start_grpc_server_with_listener(
            node_id_clone,
            listener,
            None, // receiver-side cache is None in P41; P42 wires it
            Some(wrapper_clone),
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
