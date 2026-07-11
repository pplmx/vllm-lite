//! Client wrapper for cross-node distributed-KV replication.
//!
//! [`PeerClient`] wraps the tonic-generated [`NodeServiceClient`] and
//! exposes just the RPCs that the local
//! [`crate::distributed_kv::DistributedKVCache`] needs to talk to
//! peers: `put`, `invalidate`, and `fetch_block`. The underlying
//! [`Channel`] is lazily connected on first call and reused for
//! subsequent calls — cheap because tonic's [`Channel`] is internally
//! `Arc`-backed.
//!
//! [`NodeServiceClient`]: crate::grpc::node_service_client::NodeServiceClient

use std::sync::Arc;

use tokio::sync::Mutex;
use tonic::transport::{Channel, Endpoint};

use crate::distributed_kv::MAX_BLOCK_TRANSFER_BYTES;
use crate::error::GrpcError;
use crate::grpc::node_service_client::NodeServiceClient;

/// Cloneable, cheaply-shared handle to a peer node's gRPC service.
///
/// Internally holds an [`Endpoint`] (cheap, no connection yet) and
/// the lazily-connected [`Channel`] behind an async [`Mutex`].
/// Subsequent calls reuse the channel, so only the first RPC pays
/// the TCP+HTTP/2 handshake cost.
///
/// All RPC failures are surfaced to the caller as
/// [`tonic::Status`]. The caller (typically
/// [`crate::distributed_kv::DistributedKVCache`]'s fire-and-forget
/// broadcast task) decides whether to retry or drop.
#[derive(Clone, Debug)]
pub struct PeerClient {
    url: String,
    endpoint: Endpoint,
    /// Lazy-connected channel. `None` until the first RPC succeeds;
    /// subsequent calls clone it (cheap — `Channel` is `Clone` and
    /// internally `Arc`-backed).
    channel: Arc<Mutex<Option<Channel>>>,
}

impl PeerClient {
    /// Build a [`PeerClient`] targeting `url` (e.g. `"http://node-1:50051"`).
    ///
    /// # Errors
    ///
    /// Returns [`GrpcError::Transport`] if `url` is not a valid
    /// [`Endpoint`] (e.g. missing scheme).
    pub fn new(url: impl Into<String>) -> Result<Self, GrpcError> {
        let url: String = url.into();
        let endpoint = Endpoint::from_shared(url.clone()).map_err(GrpcError::Transport)?;
        Ok(Self {
            url,
            endpoint,
            channel: Arc::new(Mutex::new(None)),
        })
    }

    /// The peer URL this client was constructed for (debug / logs).
    #[must_use]
    pub fn url(&self) -> &str {
        &self.url
    }

    /// Whether the underlying [`Channel`] has been lazily connected.
    /// False until the first RPC has succeeded; true thereafter.
    pub async fn is_connected(&self) -> bool {
        self.channel.lock().await.is_some()
    }

    /// Replicate `DistributedKVCache::put(block_id, value_hash)` to the peer.
    ///
    /// # Errors
    ///
    /// Returns the [`tonic::Status`] from the underlying RPC or a
    /// transport-level error during connect.
    pub async fn put(&self, block_id: u64, value_hash: u64) -> Result<(), tonic::Status> {
        let channel = self.ensure_channel().await?;
        let mut client = NodeServiceClient::new(channel);
        let req = crate::grpc::PutKvCacheRequest {
            block_id,
            value_hash,
        };
        client.put_kv_cache(req).await.map(|_| ())
    }

    /// Replicate `DistributedKVCache::invalidate(block_id)` to the peer.
    ///
    /// # Errors
    ///
    /// Returns the [`tonic::Status`] from the underlying RPC or a
    /// transport-level error during connect.
    pub async fn invalidate(&self, block_id: u64) -> Result<(), tonic::Status> {
        let channel = self.ensure_channel().await?;
        let mut client = NodeServiceClient::new(channel);
        let req = crate::grpc::InvalidateKvCacheRequest { block_id };
        client.invalidate_kv_cache(req).await.map(|_| ())
    }

    /// Request a block-bytes transfer from this peer.
    ///
    /// Phase 31-D OPS-31d. The caller is responsible for verifying
    /// `response.chain_hash == expected_hash` (the caller typically
    /// also has a `DistributedKVCache::get(block_id)` result to
    /// compare against). We do not pre-validate here so the caller
    /// can distinguish "peer doesn't have the block" (response with
    /// mismatched hash or a `Status::not_found`) from "peer has the
    /// block but it disagrees with our local view" (mismatched hash).
    ///
    /// The generated client is configured with
    /// [`MAX_BLOCK_TRANSFER_BYTES`] on both decode and encode so
    /// production-sized blocks (≈14 MiB for Qwen3-7B at F32) fit.
    ///
    /// # Errors
    ///
    /// Returns the [`tonic::Status`] from the underlying RPC or a
    /// transport-level error during connect.
    pub async fn fetch_block(
        &self,
        block_id: u64,
        expected_hash: u64,
    ) -> Result<crate::grpc::TransferKvBlockResponse, tonic::Status> {
        let channel = self.ensure_channel().await?;
        let mut client = NodeServiceClient::new(channel)
            .max_decoding_message_size(MAX_BLOCK_TRANSFER_BYTES)
            .max_encoding_message_size(MAX_BLOCK_TRANSFER_BYTES);
        let req = crate::grpc::TransferKvBlockRequest {
            block_id,
            expected_hash,
        };
        client
            .transfer_kv_block(req)
            .await
            .map(tonic::Response::into_inner)
    }

    /// Ensure a [`Channel`] exists in the cache, connecting if not.
    async fn ensure_channel(&self) -> Result<Channel, tonic::Status> {
        // Fast path: someone already connected. Return their channel
        // clone and drop the guard before the await so other callers
        // can take the lock instead of waiting on us.
        {
            let guard = self.channel.lock().await;
            if let Some(c) = guard.as_ref() {
                return Ok(c.clone());
            }
        }

        // Slow path: take the lock briefly to insert, then drop it
        // before returning so the channel lock isn't held across
        // `Ok` propagation.
        let channel = self
            .endpoint
            .connect()
            .await
            .map_err(|e| tonic::Status::unavailable(format!("connect to {}: {e}", self.url)))?;
        let clone = channel.clone();
        let mut guard = self.channel.lock().await;
        *guard = Some(clone);
        drop(guard);
        Ok(channel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn peer_client_new_accepts_valid_url() {
        let client = PeerClient::new("http://localhost:50051").unwrap();
        assert_eq!(client.url(), "http://localhost:50051");
        assert!(!client.is_connected().await);
    }

    #[test]
    fn peer_client_new_rejects_invalid_url() {
        // Empty URL → Endpoint::from_shared fails.
        let client = PeerClient::new("");
        assert!(client.is_err(), "empty URL must fail Endpoint::from_shared");
    }

    #[test]
    fn peer_client_is_clone() {
        let client = PeerClient::new("http://localhost:50051").unwrap();
        let clone = client.clone();
        assert_eq!(clone.url(), client.url());
        assert_eq!(
            Arc::strong_count(&client.channel),
            2,
            "clones should share the inner Arc<Mutex>"
        );
    }

    #[tokio::test]
    async fn peer_client_fetch_block_returns_unavailable_on_dead_url() {
        // Build a client pointing at a port that's almost certainly
        // unbound on a CI host. The connect attempt must surface as
        // `tonic::Status::unavailable` (or a similar transport
        // status), not panic.
        let client = PeerClient::new("http://127.0.0.1:1").expect("valid url");
        let result = client.fetch_block(0, 0).await;
        let status = result.expect_err("connect to closed port must fail");
        assert_eq!(
            status.code(),
            tonic::Code::Unavailable,
            "expected Unavailable on dead URL; got {status:?}"
        );
    }
}
