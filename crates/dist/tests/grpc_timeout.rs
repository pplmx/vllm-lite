//! RIL ISS-088: bounded timeouts on distributed-KV gRPC calls.
//!
//! A peer that completes the HTTP/2 handshake but never answers a `fetch`
//! request must not stall `DistributedKVCache::fetch_block` forever. Pre-fix,
//! the `PeerClient` Endpoint had no request/connect deadline and
//! `fetch_from_peers` awaited every peer in a `JoinSet` with no aggregate
//! deadline, so a wedged peer hung the fetch for the lifetime of the process.

use std::sync::Arc;
use std::time::Duration;

use tokio::time::sleep;
use vllm_dist::distributed_kv::block_data_source::{BlockDataSource, FetchError};
use vllm_dist::distributed_kv::protocol::NodeId;
use vllm_dist::{
    CacheConfig, DistributedKVCache, FetchError as CacheFetchError, start_grpc_server_with_listener,
};

/// A `BlockDataSource` whose `fetch_block` never resolves. Used as the
/// server-side source so the `TransferKVBlock` gRPC handler hangs: the
/// server completes the HTTP/2 handshake, receives the request, then awaits
/// this forever — exactly the "handshake done, request unanswered" profile
/// that the missing client timeouts would have let stall indefinitely.
#[derive(Debug)]
struct HangingSource;

#[async_trait::async_trait]
impl BlockDataSource for HangingSource {
    async fn fetch_block(&self, _block_id: u64) -> Result<Vec<u8>, FetchError> {
        // Never resolves within any realistic deadline.
        tokio::time::sleep(Duration::from_secs(3600)).await;
        Err(FetchError::NotFound(0))
    }
}

/// Spawn a real gRPC server whose `TransferKVBlock` handler hangs (its source
/// never resolves). Returns the server task and the URL to reach it.
async fn spawn_server_with_hanging_source() -> (tokio::task::JoinHandle<()>, String) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local_addr on bound listener");
    let url = format!("http://{addr}");

    let cache = Arc::new(DistributedKVCache::new(CacheConfig::new(NodeId(0), 4)));
    let handle = tokio::spawn(async move {
        let source: Arc<dyn BlockDataSource> = Arc::new(HangingSource);
        let _ = start_grpc_server_with_listener(
            "hanging-node".to_string(),
            listener,
            Some(cache),
            Some(source),
        )
        .await;
    });

    // Let tonic's serve machinery bind and become ready.
    sleep(Duration::from_millis(50)).await;
    (handle, url)
}

#[tokio::test]
async fn fetch_block_is_bounded_against_a_hanging_peer() {
    let (server_handle, url) = spawn_server_with_hanging_source().await;

    let mut cache =
        DistributedKVCache::new(CacheConfig::new(NodeId(9), 4).with_peer_urls(vec![url]));
    cache.connect_peers().expect("connect_peers ok");
    // Local entry so fetch_block has an expected_hash precheck and fans out.
    cache.put(1, 0xDEAD_BEEF);

    // The peer fetch must resolve (peer skipped, then local-source fallback /
    // final error) well inside this outer bound — pre-fix it hung until the
    // wrapper tripped because neither the RPC nor the fan-out had a deadline.
    let bounded = tokio::time::timeout(Duration::from_secs(20), cache.fetch_block(1)).await;
    server_handle.abort();

    assert!(
        bounded.is_ok(),
        "fetch_block must not hang on a peer that never answers (RIL ISS-088); \
         the peer RPC / fan-out needs a deadline"
    );
    match bounded.unwrap() {
        Err(CacheFetchError::AllPeersFailed(_)) => (),
        other => panic!("unexpected fetch outcome with a hanging peer: {other:?}"),
    }
}
