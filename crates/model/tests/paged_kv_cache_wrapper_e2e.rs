//! End-to-end test: real `PagedKvCache` → `PagedKvCacheWrapper` →
//! `TransferKVBlock` gRPC → bytes back on the receiver.
//!
//! Mirrors the OPS-31d in-process pair pattern from
//! `crates/dist/tests/kv_block_transfer.rs`. Gated by
//! `#[cfg(feature = "multi-node")]` — the test imports
//! `vllm_dist::*` which is itself feature-gated.
//!
//! This file closes the **first half** of OPS-32a (ADR-020 §5):
//! it proves the production `BlockDataSource` impl serves real KV
//! tensor bytes through the gRPC layer end-to-end. Engine-level
//! plumbing (`Arc<PagedKvCache>` → `MemoryManager` → `EngineBuilder`)
//! is the second half — deferred to P41+ per the spec.

#![cfg(feature = "multi-node")]

use std::sync::Arc;
use std::time::Duration;

use candle_core::{Device, Tensor};
use tokio::net::TcpListener;
use tokio::time::sleep;
use vllm_dist::distributed_kv::block_data_source::BlockDataSource;
use vllm_dist::distributed_kv::protocol::NodeId;
use vllm_dist::{CacheConfig, DistributedKVCache, start_grpc_server_with_listener};
use vllm_model::paged_tensor::PagedKvCacheWrapper;
use vllm_traits::BLOCK_SIZE;

const SENTINEL_HASH: u64 = 0xDEAD_BEEF_CAFE_BABE;
const TARGET_BLOCK_ID: u64 = 1;

fn small_cache() -> Arc<vllm_model::paged_tensor::PagedKvCache> {
    Arc::new(
        vllm_model::paged_tensor::PagedKvCache::new(2, 2, 4, 4, Device::Cpu, false).expect("cache"),
    )
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn real_paged_kv_cache_bytes_round_trip_via_wrapper() {
    // Sender: small PagedKvCache with a known write at (layer 0, block 1, token 0).
    let sender_kv_arc = small_cache();
    let sender_kv: Arc<vllm_model::paged_tensor::PagedKvCache> = {
        // Arc::try_unwrap requires a unique owner; small_cache() returns a fresh Arc
        // so we can unwrap for the write_kv mutation.
        let mut cache_mut = Arc::try_unwrap(sender_kv_arc).expect("unique Arc owner for write");
        let k = Tensor::from_slice(
            &[42.0f32, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0],
            (1, 2, 4),
            &Device::Cpu,
        )
        .expect("k");
        let v = Tensor::zeros((1, 2, 4), candle_core::DType::F32, &Device::Cpu).expect("v");
        cache_mut
            .write_kv(0, TARGET_BLOCK_ID as usize, 0, &k, &v)
            .expect("write");
        Arc::new(cache_mut)
    };
    let wrapper: Arc<dyn BlockDataSource> = Arc::new(PagedKvCacheWrapper::new(sender_kv));

    // Sender's DistributedKVCache owns the chain_hash for the wrapper's served blocks.
    // `transfer_kv_block` reads `distributed_kv.get(block_id)` to populate the wire
    // `chain_hash` field; receiver verifies it against its local entry.
    let sender_dist = Arc::new(
        DistributedKVCache::new(CacheConfig::new(NodeId(0), 2))
            .with_block_data_source(Arc::clone(&wrapper) as Arc<dyn BlockDataSource>),
    );
    sender_dist.put(TARGET_BLOCK_ID, SENTINEL_HASH);

    // Spawn sender gRPC server on ephemeral port.
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local_addr on bound listener");
    let url = format!("http://{addr}");

    let sender_dist_for_server = Arc::clone(&sender_dist);
    let wrapper_for_server = Arc::clone(&wrapper);
    let _server_handle = tokio::spawn(async move {
        let _ = start_grpc_server_with_listener(
            "sender-node".to_string(),
            listener,
            Some(sender_dist_for_server),
            Some(wrapper_for_server),
        )
        .await;
    });

    // Wait briefly for server to come up (mirrors OPS-31d test pattern).
    sleep(Duration::from_millis(50)).await;

    // Receiver: connects to sender, mirrors the chain_hash, then fetches.
    let mut receiver_dist =
        DistributedKVCache::new(CacheConfig::new(NodeId(1), 2).with_peer_urls(vec![url]));
    receiver_dist.connect_peers().expect("connect_peers ok");
    receiver_dist.put(TARGET_BLOCK_ID, SENTINEL_HASH);

    let bytes = receiver_dist
        .fetch_block(TARGET_BLOCK_ID)
        .await
        .expect("peer should serve real PagedKvCache bytes");

    // Total bytes: 2 layers * 2 (K + V) * (num_heads * BLOCK_SIZE * head_dim) * 4 bytes/f32
    let expected_len = 2 * 2 * (2 * BLOCK_SIZE * 4) * 4;
    assert_eq!(
        bytes.len(),
        expected_len,
        "byte length must match expected K+V layout"
    );

    // The first 16 bytes are head 0 / token 0 / 4 dims of layer 0's K block.
    // `write_kv` populated head 0 with [42.0, 43.0, 44.0, 45.0] at token 0.
    let as_f32: &[f32] = bytemuck::cast_slice(&bytes[..16]);
    assert_eq!(
        as_f32,
        &[42.0, 43.0, 44.0, 45.0],
        "head 0 / token 0 of layer 0 K must round-trip"
    );
}
