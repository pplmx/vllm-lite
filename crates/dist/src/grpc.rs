use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tonic::{Request, Response, Status};
use tracing::{info, warn};

tonic::include_proto!("vllm.distributed");

#[derive(Debug, Clone)]
pub struct GrpcState {
    pub node_id: String,
    pub peers: Arc<RwLock<Vec<String>>>,
    pub kv_cache: Arc<RwLock<HashMap<String, Vec<u8>>>>,
}

impl GrpcState {
    pub fn new(node_id: String) -> Self {
        Self {
            node_id,
            peers: Arc::new(RwLock::new(Vec::new())),
            kv_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn add_peer(&self, peer: String) {
        let mut peers = self.peers.write().await;
        if !peers.contains(&peer) {
            peers.push(peer);
            info!("Added peer to cluster");
        }
    }

    pub async fn remove_peer(&self, peer: &str) {
        let mut peers = self.peers.write().await;
        peers.retain(|p| p != peer);
        info!(peer = %peer, "Removed peer from cluster");
    }
}

#[derive(Debug)]
pub struct NodeServiceImpl {
    state: GrpcState,
}

impl NodeServiceImpl {
    pub fn new(state: GrpcState) -> Self {
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
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
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
}

pub async fn start_grpc_server(
    node_id: String,
    listen_addr: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let state = GrpcState::new(node_id);
    let service = NodeServiceImpl::new(state).into_service();

    info!(addr = %listen_addr, "Starting gRPC server");

    let listener = tokio::net::TcpListener::bind(listen_addr).await?;
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
    }
}
