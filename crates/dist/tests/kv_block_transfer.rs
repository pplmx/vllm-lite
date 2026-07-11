//! 2-node integration tests for Phase 31-D OPS-31d: KV block transfer.
//!
//! Spins up real gRPC servers on ephemeral ports, wires a
//! [`BlockDataSource`] into each, and verifies that
//! `DistributedKVCache::fetch_block` correctly transfers bytes
//! across the wire with chain-hash verification.

use std::sync::Arc;
use std::time::Duration;

use tokio::time::sleep;
use vllm_dist::distributed_kv::block_data_source::{BlockDataSource, MockBlockDataSource};
use vllm_dist::distributed_kv::protocol::NodeId;
use vllm_dist::{CacheConfig, DistributedKVCache, FetchError, start_grpc_server_with_listener};

/// Spawn a gRPC server on a free port (OS-assigned). Returns the
/// server task and the URL clients should use to reach it.
///
/// Wires both the cache (for Put/Invalidate replication) AND the
/// optional `BlockDataSource` (for `TransferKVBlock` serving).
async fn spawn_server(
    cache: Arc<DistributedKVCache>,
    block_source: Option<Arc<dyn BlockDataSource>>,
) -> (tokio::task::JoinHandle<()>, String) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local_addr on bound listener");
    let url = format!("http://{addr}");

    let handle = tokio::spawn(async move {
        let _ = start_grpc_server_with_listener(
            "test-node".to_string(),
            listener,
            Some(cache),
            block_source,
        )
        .await;
    });

    sleep(Duration::from_millis(50)).await;
    (handle, url)
}

/// Build a [`DistributedKVCache`] that owns a (separately-wrapped)
/// `MockBlockDataSource`. Returns both the cache and a clone of the
/// Arc to the source so callers can also wire the same source into
/// the gRPC server state.
fn make_cache_with_source(
    config: CacheConfig,
    source: Arc<MockBlockDataSource>,
) -> DistributedKVCache {
    DistributedKVCache::new(config).with_block_data_source(source)
}

#[tokio::test]
async fn peer_serves_block_bytes_via_transfer_kv_block() {
    // Node A: server has the source AND the cache entry.
    let mut source_a_mut = MockBlockDataSource::new();
    source_a_mut.insert(42, vec![0xCA, 0xFE, 0xBA, 0xBE]);
    let source_a: Arc<MockBlockDataSource> = Arc::new(source_a_mut);
    let cache_a = Arc::new(make_cache_with_source(
        CacheConfig::new(NodeId(0), 2),
        Arc::clone(&source_a),
    ));
    cache_a.put(42, 0xDEAD_BEEF);

    let (_server_a, url_a) = spawn_server(
        Arc::clone(&cache_a),
        Some(Arc::clone(&source_a) as Arc<dyn BlockDataSource>),
    )
    .await;

    // Node B: client cache, no local source.
    let mut cache_b =
        DistributedKVCache::new(CacheConfig::new(NodeId(1), 2).with_peer_urls(vec![url_a]));
    cache_b.connect_peers().expect("connect_peers ok");
    cache_b.put(42, 0xDEAD_BEEF);

    let bytes = cache_b
        .fetch_block(42)
        .await
        .expect("peer should serve bytes");
    assert_eq!(bytes, vec![0xCA, 0xFE, 0xBA, 0xBE]);
}

#[tokio::test]
async fn fetch_block_falls_back_to_local_source_when_no_peers() {
    // Single-node: no peers configured. Local source has the bytes.
    let mut source = MockBlockDataSource::new();
    source.insert(7, vec![0x11, 0x22, 0x33]);
    let source = Arc::new(source);

    let cache = DistributedKVCache::new(CacheConfig::new(NodeId(0), 1))
        .with_block_data_source(Arc::clone(&source) as Arc<dyn BlockDataSource>);
    cache.put(7, 0xABCD);

    let bytes = cache.fetch_block(7).await.expect("local fallback fetch ok");
    assert_eq!(bytes, vec![0x11, 0x22, 0x33]);
}

#[tokio::test]
async fn fetch_block_returns_not_found_when_block_unknown_everywhere() {
    // B has no local cache entry → precheck fails → NotFound
    // immediately, regardless of peers or source.
    let cache_a = Arc::new(DistributedKVCache::new(CacheConfig::new(NodeId(0), 2)));
    let (_server_a, url_a) = spawn_server(Arc::clone(&cache_a), None).await;

    let mut cache_b =
        DistributedKVCache::new(CacheConfig::new(NodeId(1), 2).with_peer_urls(vec![url_a]));
    cache_b.connect_peers().expect("connect_peers ok");

    let result = cache_b.fetch_block(99).await;
    assert!(
        matches!(result, Err(FetchError::NotFound(99))),
        "expected NotFound(99); got {result:?}"
    );
}

#[tokio::test]
async fn fetch_block_returns_all_peers_failed_when_server_has_no_source() {
    // Server has cache but NO source; client's local precheck passes
    // (we put the entry on B too) but the server's handler returns
    // Status::unavailable, so fan-out fails and B has no local
    // source. Final result: AllPeersFailed(1).
    let cache_a = Arc::new(DistributedKVCache::new(CacheConfig::new(NodeId(0), 2)));
    let (_server_a, url_a) = spawn_server(Arc::clone(&cache_a), None).await;

    let mut cache_b =
        DistributedKVCache::new(CacheConfig::new(NodeId(1), 2).with_peer_urls(vec![url_a]));
    cache_b.connect_peers().expect("connect_peers ok");
    cache_b.put(42, 0xCAFE);

    let result = cache_b.fetch_block(42).await;
    assert!(
        matches!(result, Err(FetchError::AllPeersFailed(1))),
        "expected AllPeersFailed(1); got {result:?}"
    );
}

#[tokio::test]
async fn fetch_block_succeeds_when_peer_has_source_and_block() {
    // Server has BOTH a cache and a source; client's local put
    // mirrors the chain_hash. Successful fetch.
    let mut source_a_mut = MockBlockDataSource::new();
    source_a_mut.insert(99, vec![0xAA, 0xBB, 0xCC, 0xDD]);
    let source_a: Arc<MockBlockDataSource> = Arc::new(source_a_mut);

    let cache_a = Arc::new(make_cache_with_source(
        CacheConfig::new(NodeId(0), 2),
        Arc::clone(&source_a),
    ));
    cache_a.put(99, 0x1234);

    let (_server_a, url_a) = spawn_server(
        Arc::clone(&cache_a),
        Some(Arc::clone(&source_a) as Arc<dyn BlockDataSource>),
    )
    .await;

    let mut cache_b =
        DistributedKVCache::new(CacheConfig::new(NodeId(1), 2).with_peer_urls(vec![url_a]));
    cache_b.connect_peers().expect("connect_peers ok");
    cache_b.put(99, 0x1234);

    let bytes = cache_b
        .fetch_block(99)
        .await
        .expect("server's source should serve the block");
    assert_eq!(bytes, vec![0xAA, 0xBB, 0xCC, 0xDD]);
}

#[tokio::test]
async fn fan_out_returns_first_successful_peer() {
    // Two servers A and C, both have the block. Client has both as
    // peers; fan-out returns the first successful response.
    let mut source_a_mut = MockBlockDataSource::new();
    source_a_mut.insert(7, vec![0xAA]);
    let source_a: Arc<MockBlockDataSource> = Arc::new(source_a_mut);
    let mut source_c_mut = MockBlockDataSource::new();
    source_c_mut.insert(7, vec![0xCC]);
    let source_c: Arc<MockBlockDataSource> = Arc::new(source_c_mut);

    let cache_a = Arc::new(make_cache_with_source(
        CacheConfig::new(NodeId(0), 3),
        Arc::clone(&source_a),
    ));
    let cache_c = Arc::new(make_cache_with_source(
        CacheConfig::new(NodeId(2), 3),
        Arc::clone(&source_c),
    ));
    cache_a.put(7, 0xBEEF);
    cache_c.put(7, 0xBEEF);

    let (_server_a, url_a) = spawn_server(
        Arc::clone(&cache_a),
        Some(Arc::clone(&source_a) as Arc<dyn BlockDataSource>),
    )
    .await;
    let (_server_c, url_c) = spawn_server(
        Arc::clone(&cache_c),
        Some(Arc::clone(&source_c) as Arc<dyn BlockDataSource>),
    )
    .await;

    let mut cache_b =
        DistributedKVCache::new(CacheConfig::new(NodeId(1), 3).with_peer_urls(vec![url_a, url_c]));
    cache_b.connect_peers().expect("connect_peers ok");
    cache_b.put(7, 0xBEEF);

    let bytes = cache_b.fetch_block(7).await.expect("fan-out fetch ok");
    assert!(
        bytes == vec![0xAA] || bytes == vec![0xCC],
        "got unexpected bytes from fan-out: {bytes:?}"
    );
}

#[tokio::test]
async fn fetch_block_works_above_default_message_limit() {
    // Verifies that the 64 MiB message limit is applied symmetrically
    // on server AND client: a 5 MiB block (above tonic's 4 MiB default)
    // transfers successfully end-to-end. Uses a fresh server with
    // that block pre-loaded.
    let big_block: Vec<u8> = (0..5 * 1024 * 1024).map(|i| (i & 0xFF) as u8).collect();
    let mut source_a_mut = MockBlockDataSource::new();
    source_a_mut.insert(7, big_block.clone());
    let source_a: Arc<MockBlockDataSource> = Arc::new(source_a_mut);

    let cache_a = Arc::new(make_cache_with_source(
        CacheConfig::new(NodeId(0), 2),
        Arc::clone(&source_a),
    ));
    cache_a.put(7, 0x00C0_FFEE);

    let (_server_a, url_a) = spawn_server(
        Arc::clone(&cache_a),
        Some(Arc::clone(&source_a) as Arc<dyn BlockDataSource>),
    )
    .await;

    let mut cache_b =
        DistributedKVCache::new(CacheConfig::new(NodeId(1), 2).with_peer_urls(vec![url_a]));
    cache_b.connect_peers().expect("connect_peers ok");
    cache_b.put(7, 0x00C0_FFEE);

    let bytes = cache_b
        .fetch_block(7)
        .await
        .expect("5 MiB block must round-trip");
    assert_eq!(bytes.len(), big_block.len(), "byte counts must match");
    assert_eq!(bytes, big_block, "block contents must match");
}
