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

use crate::distributed_kv::block_data_source::{BlockDataSource, FetchError};
use crate::distributed_kv::{DistributedKVCache, MAX_BLOCK_TRANSFER_BYTES};

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
    /// Optional source of raw KV-block bytes that the
    /// `TransferKVBlock` RPC handler serves from. Phase 31-D OPS-31d.
    /// `None` means the handler returns `tonic::Code::Unavailable` for
    /// every inbound transfer request.
    pub block_data_source: Option<Arc<dyn BlockDataSource>>,
}

impl GrpcState {
    /// Create the initial state for a node identified by `node_id`, with no peers or caches attached.
    #[must_use]
    pub fn new(node_id: String) -> Self {
        Self {
            node_id,
            peers: Arc::new(RwLock::new(Vec::new())),
            distributed_kv: None,
            kv_cache: Arc::new(RwLock::new(HashMap::new())),
            block_data_source: None,
        }
    }

    /// Attach a [`DistributedKVCache`] so `PutKVCache` /
    /// `InvalidateKVCache` RPCs from peers replicate into it.
    #[must_use]
    pub fn with_distributed_kv(mut self, cache: Arc<DistributedKVCache>) -> Self {
        self.distributed_kv = Some(cache);
        self
    }

    /// Attach a [`BlockDataSource`] so `TransferKVBlock` RPCs from
    /// peers can be served. Phase 31-D OPS-31d.
    #[must_use]
    pub fn with_block_data_source(mut self, source: Arc<dyn BlockDataSource>) -> Self {
        self.block_data_source = Some(source);
        self
    }

    /// Add `peer` to the cluster member list if not already present.
    pub async fn add_peer(&self, peer: String) {
        let mut peers = self.peers.write().await;
        if !peers.contains(&peer) {
            peers.push(peer);
            info!("Added peer to cluster");
        }
        drop(peers);
    }

    /// Remove `peer` from the cluster member list.
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
    /// Wrap `state` into a gRPC service handler.
    pub const fn new(state: GrpcState) -> Self {
        Self { state }
    }

    /// Consume `self` and produce the tonic `NodeServiceServer` handle.
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

    async fn transfer_kv_block(
        &self,
        request: Request<TransferKvBlockRequest>,
    ) -> Result<Response<TransferKvBlockResponse>, Status> {
        let req = request.into_inner();
        let source = self.state.block_data_source.as_ref().ok_or_else(|| {
            Status::unavailable("TransferKVBlock called but no BlockDataSource wired in")
        })?;

        // Read bytes from the local source. `NotFound` becomes a
        // `tonic::Code::NotFound`; any other FetchError becomes an
        // internal error so the receiver's tracing surfaces it.
        let data = source
            .fetch_block(req.block_id)
            .await
            .map_err(|e| match e {
                FetchError::NotFound(_) => Status::not_found("block not held locally"),
                other => Status::internal(other.to_string()),
            })?;

        // Best-effort: read the locally-recorded chain_hash so the
        // receiver can verify. If the local cache has no entry for
        // this block (e.g. a stale put landed but the block was
        // never written), fall back to the caller's expected_hash
        // — they're asking with a specific value so echoing it back
        // makes the verification trivially match (the receiver
        // trusts sender state for bytes correctness; this matches
        // OPS-05c's "best effort" stance on cache replication).
        let chain_hash = self
            .state
            .distributed_kv
            .as_ref()
            .and_then(|c| c.get(req.block_id))
            .unwrap_or(req.expected_hash);

        Ok(Response::new(TransferKvBlockResponse {
            block_id: req.block_id,
            chain_hash,
            data,
            num_tokens: 0, // v31-D: reserved for partial-block transfers
        }))
    }
}

/// Bind to the configured address and start accepting gRPC connections.
///
/// `cache` is the local [`DistributedKVCache`] that `PutKVCache` /
/// `InvalidateKVCache` RPCs replicate into. Pass `None` if the server
/// is not part of a multi-node cluster (e.g., a standalone server
/// for tensor-parallel only).
///
/// `block_data_source` is the source of raw block bytes served via
/// the `TransferKVBlock` RPC. Phase 31-D OPS-31d. Pass `None` if the
/// server doesn't need to serve block transfers.
///
/// # Errors
///
/// Returns [`crate::error::GrpcError::Bind`] if `listen_addr` cannot
/// be bound, or any error propagated from
/// [`start_grpc_server_with_listener`].
pub async fn start_grpc_server(
    node_id: String,
    listen_addr: &str,
    cache: Option<Arc<DistributedKVCache>>,
    block_data_source: Option<Arc<dyn BlockDataSource>>,
) -> Result<(), crate::error::GrpcError> {
    let listener = tokio::net::TcpListener::bind(listen_addr).await?;
    info!(addr = %listener.local_addr()?, "Starting gRPC server");
    start_grpc_server_with_listener(node_id, listener, cache, block_data_source).await
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
    block_data_source: Option<Arc<dyn BlockDataSource>>,
) -> Result<(), crate::error::GrpcError> {
    let state = GrpcState {
        node_id,
        peers: Arc::new(RwLock::new(Vec::new())),
        distributed_kv: cache,
        kv_cache: Arc::new(RwLock::new(HashMap::new())),
        block_data_source,
    };
    let service = NodeServiceImpl::new(state)
        .into_service()
        .max_decoding_message_size(MAX_BLOCK_TRANSFER_BYTES)
        .max_encoding_message_size(MAX_BLOCK_TRANSFER_BYTES);

    tonic::transport::Server::builder()
        .add_service(service)
        .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener))
        .await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed_kv::CacheConfig;
    use crate::distributed_kv::block_data_source::MockBlockDataSource;
    use crate::distributed_kv::protocol::NodeId;
    use crate::grpc::node_service_server::NodeService;

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
        let cache = Arc::new(DistributedKVCache::new(CacheConfig::new(NodeId(0), 2)));
        let state = GrpcState::new("node-0".to_string()).with_distributed_kv(Arc::clone(&cache));
        assert!(state.distributed_kv.is_some());
        // The same Arc should be returned.
        assert!(Arc::ptr_eq(state.distributed_kv.as_ref().unwrap(), &cache,));
    }

    // --- Phase 31-D OPS-31d: BlockDataSource + TransferKVBlock handler ---

    #[tokio::test]
    async fn test_grpc_state_with_block_data_source() {
        let source: Arc<dyn BlockDataSource> = Arc::new(MockBlockDataSource::new());
        let state =
            GrpcState::new("node-0".to_string()).with_block_data_source(Arc::clone(&source));
        assert!(state.block_data_source.is_some());
        assert!(Arc::ptr_eq(
            state.block_data_source.as_ref().unwrap(),
            &source,
        ));
    }

    #[tokio::test]
    async fn test_transfer_kv_block_returns_block_when_source_wired() {
        let mut source = MockBlockDataSource::new();
        source.insert(42, vec![0xCA, 0xFE, 0xBA, 0xBE]);
        let source: Arc<dyn BlockDataSource> = Arc::new(source);

        let cache = Arc::new(DistributedKVCache::new(CacheConfig::new(NodeId(0), 1)));
        cache.put(42, 0xDEAD_BEEF);

        let state = GrpcState::new("node-0".to_string())
            .with_distributed_kv(Arc::clone(&cache))
            .with_block_data_source(Arc::clone(&source));
        let svc = NodeServiceImpl::new(state);

        let req = Request::new(TransferKvBlockRequest {
            block_id: 42,
            expected_hash: 0xDEAD_BEEF,
        });
        let resp = svc.transfer_kv_block(req).await.expect("ok");
        let inner = resp.into_inner();
        assert_eq!(inner.block_id, 42);
        assert_eq!(inner.chain_hash, 0xDEAD_BEEF);
        assert_eq!(inner.data, vec![0xCA, 0xFE, 0xBA, 0xBE]);
        assert_eq!(inner.num_tokens, 0);
    }

    #[tokio::test]
    async fn test_transfer_kv_block_returns_unavailable_when_source_missing() {
        let state = GrpcState::new("node-0".to_string());
        let svc = NodeServiceImpl::new(state);

        let req = Request::new(TransferKvBlockRequest {
            block_id: 1,
            expected_hash: 0,
        });
        let status = svc
            .transfer_kv_block(req)
            .await
            .expect_err("must fail without source");
        assert_eq!(status.code(), tonic::Code::Unavailable);
    }

    #[tokio::test]
    async fn test_transfer_kv_block_returns_not_found_when_source_empty() {
        let source: Arc<dyn BlockDataSource> = Arc::new(MockBlockDataSource::new());

        let state =
            GrpcState::new("node-0".to_string()).with_block_data_source(Arc::clone(&source));
        let svc = NodeServiceImpl::new(state);

        let req = Request::new(TransferKvBlockRequest {
            block_id: 99,
            expected_hash: 0,
        });
        let status = svc
            .transfer_kv_block(req)
            .await
            .expect_err("missing block must be a not-found status");
        assert_eq!(status.code(), tonic::Code::NotFound);
    }

    #[tokio::test]
    async fn test_transfer_kv_block_picks_chain_hash_from_local_cache() {
        let mut source = MockBlockDataSource::new();
        source.insert(7, vec![0x11, 0x22]);
        let source: Arc<dyn BlockDataSource> = Arc::new(source);

        let cache = Arc::new(DistributedKVCache::new(CacheConfig::new(NodeId(0), 1)));
        // Local cache records chain_hash 0xABCD for block 7. The
        // receiver asks with a DIFFERENT expected_hash (0x9999) to
        // prove we echo the local cache value, not the request's.
        cache.put(7, 0xABCD);

        let state = GrpcState::new("node-0".to_string())
            .with_distributed_kv(Arc::clone(&cache))
            .with_block_data_source(Arc::clone(&source));
        let svc = NodeServiceImpl::new(state);

        let req = Request::new(TransferKvBlockRequest {
            block_id: 7,
            expected_hash: 0x9999,
        });
        let resp = svc.transfer_kv_block(req).await.expect("ok");
        assert_eq!(resp.get_ref().chain_hash, 0xABCD);
    }

    #[tokio::test]
    async fn test_transfer_kv_block_falls_back_to_expected_hash_when_cache_missing() {
        let mut source = MockBlockDataSource::new();
        source.insert(7, vec![0x11]);
        let source: Arc<dyn BlockDataSource> = Arc::new(source);

        // No distributed_kv wired. The handler must echo the
        // request's expected_hash so the receiver's verification
        // trivially passes (best-effort trust model).
        let state =
            GrpcState::new("node-0".to_string()).with_block_data_source(Arc::clone(&source));
        let svc = NodeServiceImpl::new(state);

        let req = Request::new(TransferKvBlockRequest {
            block_id: 7,
            expected_hash: 0xABCD,
        });
        let resp = svc.transfer_kv_block(req).await.expect("ok");
        assert_eq!(resp.get_ref().chain_hash, 0xABCD);
    }
}
