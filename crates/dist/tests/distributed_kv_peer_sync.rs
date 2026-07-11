//! 2-node integration tests for distributed KV-cache peer sync.
//!
//! Spins up a real gRPC server with a [`DistributedKVCache`]
//! attached, then creates a second [`DistributedKVCache`] whose
//! `peer_urls` point at the server. Verifies that `put` /
//! `invalidate` on the second cache round-trip through gRPC and
//! land in the first cache's local map.
//!
//! Phase 19 OPS-05c.

use std::sync::Arc;
use std::time::Duration;

use tokio::time::sleep;
use vllm_dist::distributed_kv::protocol::NodeId;
use vllm_dist::{CacheConfig, DistributedKVCache, PeerClient, start_grpc_server_with_listener};

/// Spawn a gRPC server on a free port (OS-assigned). Returns the
/// server task and the URL clients should use to reach it.
///
/// The server is shut down when the returned `JoinHandle` is dropped
/// (the `tokio::spawn`'d future returns).
async fn spawn_server(cache: Arc<DistributedKVCache>) -> (tokio::task::JoinHandle<()>, String) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local_addr on bound listener");
    let url = format!("http://{addr}");

    let handle = tokio::spawn(async move {
        // On any error from the server, just log; the test has its
        // own assertions and shouldn't depend on the server's
        // shutdown semantics.
        let _ =
            start_grpc_server_with_listener("test-node".to_string(), listener, Some(cache)).await;
    });

    // Give the server a moment to actually bind + register with
    // tonic's internal machinery. The listener is bound above, but
    // tonic's `serve_with_incoming` does additional setup before
    // it's ready to accept.
    sleep(Duration::from_millis(50)).await;

    (handle, url)
}

#[tokio::test]
async fn put_broadcasts_to_peer_via_grpc() {
    // Node A: the gRPC server, with its own cache.
    let cache_a = Arc::new(DistributedKVCache::new(CacheConfig::new(NodeId(0), 2)));
    let (_server_handle, server_url) = spawn_server(Arc::clone(&cache_a)).await;

    // Node B: a separate cache that knows about node A.
    let mut cache_b = DistributedKVCache::new(
        CacheConfig::new(NodeId(1), 2).with_peer_urls(vec![server_url.clone()]),
    );
    cache_b.connect_peers().expect("connect_peers ok");
    assert_eq!(cache_b.peer_client_count(), 1);

    // Put something on B. Local cache + broadcast to A.
    cache_b.put(42, 0xCAFE_BABE);

    // Wait for the broadcast (fire-and-forget) to land.
    // A small retry loop avoids flakiness on slow CI without making
    // the test slow on fast machines.
    let mut found = false;
    for _ in 0..50 {
        if cache_a.get(42) == Some(0xCAFE_BABE) {
            found = true;
            break;
        }
        sleep(Duration::from_millis(20)).await;
    }
    assert!(
        found,
        "node A's cache should observe node B's put via gRPC broadcast"
    );

    // B's local cache should also have the entry.
    assert_eq!(cache_b.get(42), Some(0xCAFE_BABE));
}

#[tokio::test]
async fn invalidate_broadcasts_to_peer_via_grpc() {
    let cache_a = Arc::new(DistributedKVCache::new(CacheConfig::new(NodeId(0), 2)));
    let (_server_handle, server_url) = spawn_server(Arc::clone(&cache_a)).await;

    let mut cache_b = DistributedKVCache::new(
        CacheConfig::new(NodeId(1), 2).with_peer_urls(vec![server_url.clone()]),
    );
    cache_b.connect_peers().expect("connect_peers ok");

    // Seed both caches with the same key.
    cache_b.put(7, 0xABCD);
    // Wait for A to observe the put.
    for _ in 0..50 {
        if cache_a.get(7) == Some(0xABCD) {
            break;
        }
        sleep(Duration::from_millis(20)).await;
    }
    assert_eq!(cache_a.get(7), Some(0xABCD), "seed put must reach A");

    // Now invalidate on B.
    cache_b.invalidate(7);
    for _ in 0..50 {
        if cache_a.get(7).is_none() {
            break;
        }
        sleep(Duration::from_millis(20)).await;
    }
    assert!(
        cache_a.get(7).is_none(),
        "node A's cache should observe node B's invalidate via gRPC broadcast"
    );
    assert!(
        cache_b.get(7).is_none(),
        "node B's local cache should also be invalidated"
    );
}

#[tokio::test]
async fn multi_peer_broadcast() {
    // Two servers (nodes A and C); one client (B) broadcasts to both.
    let cache_a = Arc::new(DistributedKVCache::new(CacheConfig::new(NodeId(0), 3)));
    let cache_c = Arc::new(DistributedKVCache::new(CacheConfig::new(NodeId(2), 3)));

    let (_h_a, url_a) = spawn_server(Arc::clone(&cache_a)).await;
    let (_h_c, url_c) = spawn_server(Arc::clone(&cache_c)).await;

    let mut cache_b = DistributedKVCache::new(
        CacheConfig::new(NodeId(1), 3).with_peer_urls(vec![url_a.clone(), url_c.clone()]),
    );
    cache_b.connect_peers().expect("connect_peers ok");
    assert_eq!(cache_b.peer_client_count(), 2);

    cache_b.put(99, 0xDEAD);

    let mut found_a = false;
    let mut found_c = false;
    for _ in 0..100 {
        found_a = cache_a.get(99) == Some(0xDEAD);
        found_c = cache_c.get(99) == Some(0xDEAD);
        if found_a && found_c {
            break;
        }
        sleep(Duration::from_millis(20)).await;
    }
    assert!(found_a, "node A should observe B's put");
    assert!(found_c, "node C should observe B's put");
}

#[tokio::test]
async fn single_node_cache_does_not_broadcast() {
    // No peer URLs configured — broadcast helpers must be no-ops.
    let cache = DistributedKVCache::new(CacheConfig::new(NodeId(0), 1));
    // put/invalidate must not panic without a runtime either, but
    // for safety run inside one.
    cache.put(1, 0xAA);
    cache.invalidate(1);
    // peer_client_count stays at 0 (never connected).
    assert_eq!(cache.peer_client_count(), 0);
}

#[tokio::test]
async fn peer_client_lazy_connection() {
    // A `PeerClient` shouldn't connect until the first RPC.
    let (_server_handle, server_url) = spawn_server(Arc::new(DistributedKVCache::new(
        CacheConfig::new(NodeId(0), 1),
    )))
    .await;

    let client = PeerClient::new(server_url).expect("valid url");
    assert!(!client.is_connected().await);

    // The first RPC triggers the connection.
    let _ = client.put(1, 0xBB).await;
    assert!(client.is_connected().await);

    // Subsequent calls reuse the same channel.
    let _ = client.invalidate(1).await;
    assert!(client.is_connected().await);
}
