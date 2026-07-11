//! Tensor-parallel and pipeline-parallel primitives for multi-node inference.
//!
//! Activated by `--features multi-node`. Pulls in `tonic` for gRPC transport
//! and exposes cooperative tensor + pipeline parallel APIs to `vllm-core`.
#![allow(clippy::module_name_repetitions)]
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tonic::{Request, Response, Status};
use tracing::{info, warn};

use crate::distributed_kv::DistributedKVCache;

// Lints are disabled for the generated proto module because tonic_build output
// is not under our control. See crates/dist/proto/node.proto for the source.
#[allow(
    clippy::derive_partial_eq_without_eq,
    clippy::doc_markdown,
    clippy::missing_const_for_fn,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::default_trait_access,
    clippy::too_many_lines
)]
mod generated_proto {
    tonic::include_proto!("vllm.distributed");
}
pub use generated_proto::*;

/// Internal state of Grpc. Mutated under a lock; read via accessor methods on the parent type.
#[derive(Debug, Clone)]
pub struct GrpcState {
    pub node_id: String,
    pub peers: Arc<RwLock<Vec<String>>>,
    /// Optional reference to the local [`DistributedKVCache`] that
    /// `PutKVCache` / `InvalidateKVCache` RPCs replicate into.
    /// Phase 19 OPS-05c.
    pub distributed_kv: Option<Arc<DistributedKVCache>>,
    /// Legacy: byte-string KV store used by the `GetKVCache` RPC
    /// handler. Kept for backward compatibility with the existing
    /// `GetKVCache` RPC and is **not** the same as
    /// [`Self::distributed_kv`].
    pub kv_cache: Arc<RwLock<HashMap<String, Vec<u8>>>>,
}

impl GrpcState {
    #[must_use]
    pub fn new(node_id: String) -> Self {
        Self {
            node_id,
            peers: Arc::new(RwLock::new(Vec::new())),
            distributed_kv: None,
            kv_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Attach a [`DistributedKVCache`] so `PutKVCache` /
    /// `InvalidateKVCache` RPCs from peers replicate into it.
    #[must_use]
    pub fn with_distributed_kv(mut self, cache: Arc<DistributedKVCache>) -> Self {
        self.distributed_kv = Some(cache);
        self
    }

    pub async fn add_peer(&self, peer: String) {
        let mut peers = self.peers.write().await;
        if !peers.contains(&peer) {
            peers.push(peer);
            info!("Added peer to cluster");
        }
        drop(peers);
    }

    pub async fn remove_peer(&self, peer: &str) {
        let mut peers = self.peers.write().await;
        peers.retain(|p| p != peer);
        info!(peer = %peer, "Removed peer from cluster");
        drop(peers);
    }
}

/// `NodeServiceImpl`. See the type definition for fields and behavior.
#[derive(Debug)]
pub(crate) struct NodeServiceImpl {
    state: GrpcState,
}

impl NodeServiceImpl {
    pub const fn new(state: GrpcState) -> Self {
        Self { state }
    }

    pub fn into_service(self) -> node_service_server::NodeServiceServer<Self> {
        node_service_server::NodeServiceServer::new(self)
    }
}

#[tonic::async_trait]
impl node_service_server::NodeService for NodeServiceImpl {
    async fn ping(&self, request: Request<PingRequest>) -> Result<Response<PingResponse>, Status> {
        let req = request.into_inner();
        info!(node_id = %req.node_id, timestamp = req.timestamp, "Ping received");

        Ok(Response::new(PingResponse {
            success: true,
            node_id: self.state.node_id.clone(),
            // invariant: monotonic clock is always >= UNIX_EPOCH.
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                // invariant: pre-conditions make this infallible at this call site.
                .unwrap()
                .as_secs(),
        }))
    }

    async fn all_reduce(
        &self,
        request: Request<AllReduceRequest>,
    ) -> Result<Response<AllReduceResponse>, Status> {
        let req = request.into_inner();
        warn!("AllReduce called - NCCL multi-node not yet implemented");
        Ok(Response::new(AllReduceResponse {
            success: true,
            result: req.data,
        }))
    }

    async fn get_kv_cache(
        &self,
        request: Request<GetKvCacheRequest>,
    ) -> Result<Response<GetKvCacheResponse>, Status> {
        let req = request.into_inner();
        let cache = self.state.kv_cache.read().await;

        #[allow(clippy::option_if_let_else)]
        if let Some(data) = cache.get(&req.block_hash) {
            Ok(Response::new(GetKvCacheResponse {
                found: true,
                data: data.clone(),
                num_tokens: 0,
            }))
        } else {
            Ok(Response::new(GetKvCacheResponse {
                found: false,
                data: Vec::new(),
                num_tokens: 0,
            }))
        }
    }

    async fn get_peers(
        &self,
        _request: Request<GetPeersRequest>,
    ) -> Result<Response<GetPeersResponse>, Status> {
        let peers = self.state.peers.read().await;
        Ok(Response::new(GetPeersResponse {
            peer_addresses: peers.clone(),
        }))
    }

    async fn put_kv_cache(
        &self,
        request: Request<PutKvCacheRequest>,
    ) -> Result<Response<PutKvCacheResponse>, Status> {
        let req = request.into_inner();
        // Replicate into the local cache. `put` is sync and bumps
        // stats; we treat any failure here as a server-side bug
        // (not a network error) so always return success.
        if let Some(cache) = self.state.distributed_kv.as_ref() {
            cache.put(req.block_id, req.value_hash);
        } else {
            // No cache wired — log and accept the message anyway
            // (peer may be sending for an outdated config). The
            // alternative (failing the RPC) would create retry
            // storms.
            warn!(
                block_id = req.block_id,
                "PutKVCache received but no DistributedKVCache wired in; dropping"
            );
        }
        Ok(Response::new(PutKvCacheResponse { success: true }))
    }

    async fn invalidate_kv_cache(
        &self,
        request: Request<InvalidateKvCacheRequest>,
    ) -> Result<Response<InvalidateKvCacheResponse>, Status> {
        let req = request.into_inner();
        if let Some(cache) = self.state.distributed_kv.as_ref() {
            cache.invalidate(req.block_id);
        } else {
            warn!(
                block_id = req.block_id,
                "InvalidateKVCache received but no DistributedKVCache wired in; dropping"
            );
        }
        Ok(Response::new(InvalidateKvCacheResponse { success: true }))
    }
}

/// Bind to the configured address and start accepting gRPC connections.
///
/// `cache` is the local [`DistributedKVCache`] that `PutKVCache` /
/// `InvalidateKVCache` RPCs replicate into. Pass `None` if the server
/// is not part of a multi-node cluster (e.g., a standalone server
/// for tensor-parallel only).
/// # Errors
///
/// Returns [`crate::error::GrpcError::Bind`] if `listen_addr` cannot
/// be bound, or any error propagated from
/// [`start_grpc_server_with_listener`].
pub async fn start_grpc_server(
    node_id: String,
    listen_addr: &str,
    cache: Option<Arc<DistributedKVCache>>,
) -> Result<(), crate::error::GrpcError> {
    let listener = tokio::net::TcpListener::bind(listen_addr).await?;
    info!(addr = %listener.local_addr()?, "Starting gRPC server");
    start_grpc_server_with_listener(node_id, listener, cache).await
}

/// Same as [`start_grpc_server`] but takes a pre-bound listener.
///
/// Useful for tests that need to bind to port `0` (let the OS pick a
/// free port) and then read back the chosen port from
/// `listener.local_addr()` before starting the server.
///
/// # Errors
///
/// Returns [`crate::error::GrpcError`] from the tonic transport if
/// the server fails to start serving (e.g., shutdown mid-flight).
pub async fn start_grpc_server_with_listener(
    node_id: String,
    listener: tokio::net::TcpListener,
    cache: Option<Arc<DistributedKVCache>>,
) -> Result<(), crate::error::GrpcError> {
    let state = GrpcState {
        node_id,
        peers: Arc::new(RwLock::new(Vec::new())),
        distributed_kv: cache,
        kv_cache: Arc::new(RwLock::new(HashMap::new())),
    };
    let service = NodeServiceImpl::new(state).into_service();

    tonic::transport::Server::builder()
        .add_service(service)
        .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener))
        .await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_grpc_state_add_peer() {
        let state = GrpcState::new("node-0".to_string());
        state.add_peer("node-1:50051".to_string()).await;

        let peers = state.peers.read().await;
        assert_eq!(peers.len(), 1);
        assert_eq!(peers[0], "node-1:50051");
        drop(peers);
    }

    #[tokio::test]
    async fn test_grpc_state_remove_peer() {
        let state = GrpcState::new("node-0".to_string());
        state.add_peer("node-1:50051".to_string()).await;
        state.add_peer("node-2:50051".to_string()).await;

        state.remove_peer("node-1:50051").await;

        let peers = state.peers.read().await;
        assert_eq!(peers.len(), 1);
        assert_eq!(peers[0], "node-2:50051");
        drop(peers);
    }

    #[tokio::test]
    async fn test_grpc_state_with_distributed_kv() {
        use crate::distributed_kv::CacheConfig;
        use crate::distributed_kv::protocol::NodeId;
        let cache = Arc::new(DistributedKVCache::new(CacheConfig::new(NodeId(0), 2)));
        let state = GrpcState::new("node-0".to_string()).with_distributed_kv(Arc::clone(&cache));
        assert!(state.distributed_kv.is_some());
        // The same Arc should be returned.
        assert!(Arc::ptr_eq(state.distributed_kv.as_ref().unwrap(), &cache,));
    }
}
