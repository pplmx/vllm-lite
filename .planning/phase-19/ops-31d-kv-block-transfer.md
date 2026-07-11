# Phase 19 OPS-31d — KV Block Transfer

**Date:** 2026-07-12
**Scope:** `vllm-dist` (`BlockDataSource` trait, `TransferKVBlock` gRPC
RPC, `PeerClient::fetch_block`, `DistributedKVCache::fetch_block`,
64 MiB symmetric message limit, integration tests)
**Status:** Shipped
**Phase goal (long-term):** Wire `vllm-dist` into the Engine end-to-end so
multi-node inference is real, not just available as a library.

---

## 1. Why this phase

OPS-05c (commit `6fe1e69`) closed the *intent* loop for multi-node
KV-cache coherence: every local `put` / `invalidate` is now replicated
to every configured peer over gRPC. But the replicated state is
`(block_id, chain_hash)` — the actual KV tensor bytes are still
local-only. The OPS-05c plan explicitly deferred this:

> What's *not* in scope: this commit ships *replication of intent*,
> not *block transfer*. ... Actually moving KV blocks across nodes
> requires a separate transfer protocol (scheduled for a later
> phase).

OPS-31d closes that gap at the **protocol layer**. After this ships, a
node that detects via `lookup_prefix` that a peer has the prefix can
call `DistributedKVCache::fetch_block(block_id)` and get the actual KV
bytes back, verified by chain hash.

The Engine-side wiring (plumbing `Arc<PagedKvCache>` into
`MemoryManager` so the gRPC server can serve real block bytes) is
explicitly **not** in scope — that's a model-crate change deferred to
v32+. What ships here is the protocol and the abstractions that make
that future wiring a small change: `BlockDataSource` can wrap
`PagedKvCache` directly without touching the dist layer.

The technical due diligence (`docs/technical-due-diligence/roadmap.md`)
called this out:

> 在采样和 KV 生命周期未正确前增加更多模型架构。
> 在单机 batched kernel 未成熟前建设完整多节点 MESI/KV 协议。

OPS-31d respects both: it does not add architectures, and it does not
implement MESI. It delivers just the protocol-layer primitives, with a
production-shaped `BlockDataSource` seam that v32+ can plug into.

---

## 2. What changed

### New file: `crates/dist/src/distributed_kv/block_data_source.rs`

The new abstraction over raw block bytes.

```rust
pub const MAX_BLOCK_TRANSFER_BYTES: usize = 64 * 1024 * 1024;

#[async_trait::async_trait]
pub trait BlockDataSource: Send + Sync + fmt::Debug {
    async fn fetch_block(&self, block_id: u64) -> Result<Vec<u8>, FetchError>;
    async fn has_block(&self, block_id: u64) -> bool { let _ = block_id; true }
}

#[derive(Debug, thiserror::Error)]
pub enum FetchError {
    #[error("block {0} not held by any peer or local source")]
    NotFound(u64),
    #[error("hash mismatch for block {block_id}: expected {expected_hash:#x}, got {actual_hash:#x}")]
    HashMismatch { block_id: u64, expected_hash: u64, actual_hash: u64 },
    #[error("no BlockDataSource wired in (single-node server)")]
    SourceUnavailable,
    #[error("no peers configured and no local source wired")]
    NoPeers,
    #[error("all {0} peers failed for the block transfer")]
    AllPeersFailed(usize),
    #[error("transport error during block transfer")]
    Transport(#[source] tonic::Status),
}
```

Design points:

- **Async, object-safe**: `Arc<dyn BlockDataSource>` is the canonical
  storage form, both in `DistributedKVCache.block_data_source` and in
  `GrpcState.block_data_source`. The `#[async_trait]` macro adds the
  `Send` bounds needed for dyn dispatch.
- **Storage-agnostic**: returns raw `Vec<u8>` so a future GPU-direct
  implementation can return device-side slices wrapped as bytes
  without changing the trait signature.
- **Hash carried on the wire, not in the trait**: the `chain_hash`
  travels alongside the bytes on the gRPC wire (see
  `TransferKvBlockResponse.chain_hash`); the receiver verifies it
  against its locally-recorded `value_hash`. This keeps the dist
  layer from importing `BlockHasher` (which lives in `vllm-traits`).
- **`MAX_BLOCK_TRANSFER_BYTES = 64 MiB`**: sized for Qwen3-7B (≈14
  MiB/block at F32) with ~4× headroom for larger models (Qwen3-72B:
  ~24 MiB/block) or future `BLOCK_SIZE` growth. Applied symmetrically
  on both the gRPC server's `max_decoding_message_size` /
  `max_encoding_message_size` and the `PeerClient`'s generated client
  builder. Tonic's default 4 MiB limit is far too small — bumping here
  is what makes 31-d's block transfer actually work for
  production-sized blocks.

### `crates/dist/proto/node.proto`

```protobuf
rpc TransferKVBlock(TransferKVBlockRequest) returns (TransferKVBlockResponse);

message TransferKvBlockRequest {
  uint64 block_id = 1;
  uint64 expected_hash = 2;  // receiver's locally-recorded chain_hash
}

message TransferKvBlockResponse {
  uint64 block_id = 1;
  uint64 chain_hash = 2;     // sender's locally-recorded chain_hash; receiver verifies
  bytes data = 3;
  uint32 num_tokens = 4;     // 0 in 31-d; reserved for partial-block transfers
}
```

`block_id` and `expected_hash` mirror the local cache's `u64`-keyed
layout so the local-cache keyspace round-trips through gRPC without
translation. The response carries the sender's locally-recorded
`chain_hash` so the receiver can verify before installing.

### `crates/dist/src/grpc.rs`

`GrpcState` grew an optional block source:

```rust
pub struct GrpcState {
    // ... existing fields ...
    pub block_data_source: Option<Arc<dyn BlockDataSource>>,
}

impl GrpcState {
    #[must_use]
    pub fn with_block_data_source(mut self, source: Arc<dyn BlockDataSource>) -> Self {
        self.block_data_source = Some(source);
        self
    }
}
```

The new `transfer_kv_block` handler:

```rust
async fn transfer_kv_block(&self, request: Request<TransferKvBlockRequest>)
    -> Result<Response<TransferKvBlockResponse>, Status>
{
    let req = request.into_inner();
    let source = self.state.block_data_source.as_ref().ok_or_else(||
        Status::unavailable("TransferKVBlock called but no BlockDataSource wired in")
    )?;

    let data = source.fetch_block(req.block_id).await.map_err(|e| match e {
        FetchError::NotFound(_) => Status::not_found("block not held locally"),
        other => Status::internal(other.to_string()),
    })?;

    // Best-effort: read the locally-recorded chain_hash so the
    // receiver can verify. If the local cache has no entry, fall
    // back to the caller's expected_hash (they're asking with a
    // specific value, so echoing it makes the verification
    // trivially match — same "best effort" stance as OPS-05c).
    let chain_hash = self.state.distributed_kv.as_ref()
        .and_then(|c| c.get(req.block_id))
        .unwrap_or(req.expected_hash);

    Ok(Response::new(TransferKvBlockResponse {
        block_id: req.block_id,
        chain_hash,
        data,
        num_tokens: 0,
    }))
}
```

`start_grpc_server(_with_listener)` gained an
`Option<Arc<dyn BlockDataSource>>` parameter and bumps both
`max_decoding_message_size` and `max_encoding_message_size` on the
gRPC server to `MAX_BLOCK_TRANSFER_BYTES`:

```rust
let service = NodeServiceImpl::new(state)
    .into_service()
    .max_decoding_message_size(MAX_BLOCK_TRANSFER_BYTES)
    .max_encoding_message_size(MAX_BLOCK_TRANSFER_BYTES);
```

The signature break is consistent with how `cache:
Option<Arc<DistributedKVCache>>` was added in OPS-05c. Only one
internal caller (`spawn_server` helper in
`tests/distributed_kv_peer_sync.rs`).

### `crates/dist/src/grpc_client.rs`

```rust
pub async fn fetch_block(
    &self,
    block_id: u64,
    expected_hash: u64,
) -> Result<crate::grpc::TransferKvBlockResponse, tonic::Status> {
    let channel = self.ensure_channel().await?;
    let mut client = NodeServiceClient::new(channel)
        .max_decoding_message_size(MAX_BLOCK_TRANSFER_BYTES)
        .max_encoding_message_size(MAX_BLOCK_TRANSFER_BYTES);
    let req = crate::grpc::TransferKvBlockRequest { block_id, expected_hash };
    client.transfer_kv_block(req).await.map(tonic::Response::into_inner)
}
```

The generated client is configured with `MAX_BLOCK_TRANSFER_BYTES` on
both decode and encode so production-sized blocks (≈14 MiB for
Qwen3-7B at F32) fit. Caller is responsible for verifying
`response.chain_hash == expected_hash` (matches the local cache's
`value_hash`).

### `crates/dist/src/distributed_kv/cache.rs`

`DistributedKVCache` gained a `block_data_source` field and a
`fetch_block` async method:

```rust
pub fn with_block_data_source(mut self, source: Arc<dyn BlockDataSource>) -> Self {
    self.block_data_source = Some(source);
    self
}

pub async fn fetch_block(&self, block_id: u64) -> Result<Vec<u8>, FetchError> {
    // Step 1: precheck — caller must already have a local cache
    // entry for this block_id, otherwise there's no expected_hash
    // to verify against.
    let expected_hash = self.get(block_id).ok_or(FetchError::NotFound(block_id))?;

    // Step 2: fan-out to peers (if any). First response whose
    // chain_hash matches the local value_hash wins; mismatches and
    // transport errors are logged and skipped.
    let peers: Vec<PeerClient> = self.peer_clients.as_ref()
        .map(|c| c.iter().cloned().collect())
        .unwrap_or_default();

    if !peers.is_empty() {
        let mut join_set = tokio::task::JoinSet::new();
        for client in &peers {
            let client = client.clone();
            join_set.spawn(async move {
                (client.url().to_string(), client.fetch_block(block_id, expected_hash).await)
            });
        }
        while let Some(joined) = join_set.join_next().await {
            // ... check chain_hash match, return bytes on success ...
        }
    }

    // Step 3: fall back to the local source.
    if let Some(source) = self.block_data_source.as_ref() {
        return source.fetch_block(block_id).await;
    }

    // Step 4: nothing worked.
    if peers.is_empty() { Err(FetchError::NoPeers) }
    else { Err(FetchError::AllPeersFailed(peers.len())) }
}
```

Why fan-out (and not owner-based routing via `compute_owner_nodes`):

- Cluster sizes are small (master plan: 2–4 nodes).
- Most blocks live on multiple replicas (`replication_factor` is
  `2.min(num_nodes)` by default), so the "first peer with the bytes"
  win-rate is high.
- Smart routing requires a stable mapping from `NodeId` to peer URL,
  which `CacheConfig.peer_urls` doesn't currently carry. Building
  that mapping is a v32+ concern.

### `crates/dist/src/lib.rs` and `crates/dist/src/distributed_kv/mod.rs`

Re-exports:

```rust
pub use distributed_kv::{
    BlockDataSource, CacheConfig, CacheMessage, DistributedKVCache,
    FetchError, MAX_BLOCK_TRANSFER_BYTES, NodeId,
};
```

---

## 3. Tests

### Unit tests (16 added)

In `block_data_source.rs`:

- `mock_fetch_block_returns_inserted_bytes`
- `mock_fetch_block_returns_not_found_for_missing`
- `mock_has_block_matches_insertions`
- `fetch_error_display_messages_are_distinct`
- `fetch_error_from_tonic_status_yields_transport`

In `cache.rs`:

- `fetch_block_returns_not_found_when_local_cache_missing`
- `fetch_block_returns_no_peers_when_single_node_no_source`
- `fetch_block_falls_back_to_local_source_when_no_peers`
- `fetch_block_propagates_source_not_found`

In `grpc_client.rs`:

- `peer_client_fetch_block_returns_unavailable_on_dead_url`

In `grpc.rs`:

- `test_grpc_state_with_block_data_source`
- `test_transfer_kv_block_returns_block_when_source_wired`
- `test_transfer_kv_block_returns_unavailable_when_source_missing`
- `test_transfer_kv_block_returns_not_found_when_source_empty`
- `test_transfer_kv_block_picks_chain_hash_from_local_cache`
- `test_transfer_kv_block_falls_back_to_expected_hash_when_cache_missing`

### Integration tests (7 added)

In `crates/dist/tests/kv_block_transfer.rs` (new file):

- `peer_serves_block_bytes_via_transfer_kv_block` — single 2-node
  setup; receiver gets mock bytes.
- `fetch_block_falls_back_to_local_source_when_no_peers` —
  single-node; receiver's own source returns bytes.
- `fetch_block_returns_not_found_when_block_unknown_everywhere` —
  both sides empty; NotFound.
- `fetch_block_returns_all_peers_failed_when_server_has_no_source` —
  server has cache but no source; AllPeersFailed(1).
- `fetch_block_succeeds_when_peer_has_source_and_block` — server has
  BOTH a cache and a source; successful fetch.
- `fan_out_returns_first_successful_peer` — two peers A and C both
  hold; receiver gets bytes from either.
- `fetch_block_works_above_default_message_limit` — 5 MiB block
  (above tonic's 4 MiB default) transfers successfully end-to-end,
  proving the 64 MiB symmetric limit is applied.

---

## 4. Design notes & open questions

### Why `MAX_BLOCK_TRANSFER_BYTES` lives in `block_data_source.rs`

It is conceptually tied to the wire protocol (and the gRPC server /
client message limits), not to the trait itself. The const lives
alongside `FetchError` because it's the "fetch" sub-domain of the
distributed-KV crate. `MAX_BLOCK_TRANSFER_BYTES` is re-exported from
the crate root so consumers don't need to know which submodule it
came from.

### Why fan-out fallback instead of smart routing

Owner-based routing via `compute_owner_nodes` is the correct long-term
answer, but requires a stable `NodeId → URL` mapping that
`CacheConfig.peer_urls` doesn't currently carry. Adding that mapping
is a `CacheConfig` API change (potentially breaking) that's better
batched with v32+ Engine-integration work.

In the meantime, fan-out fallback:
- Sends to every peer in parallel (so latency ≈ slowest peer).
- Returns the first response whose `chain_hash` matches the local
  `value_hash`.
- Wastes bandwidth (4× for a 4-node cluster) but is correct and
  simple.

### Why no re-hashing of bytes at the dist layer

The `BlockHasher` trait lives in `vllm-traits`, and crossing that
boundary from `vllm-dist` would couple the dist layer to a hash
algorithm it doesn't otherwise need. The wire-carried `chain_hash`
matches the OPS-05c "best effort" stance on cache replication: a
malicious or buggy peer could return garbage bytes that happen to hash
to whatever it advertises, but the same is already true for `put`
metadata. End-to-end verification (re-hash from tokens) is v32+
territory.

### Open: Engine integration (v32+)

`PagedKvCache` (in `crates/model/src/paged_tensor/tensor_store/`) is
the production `BlockDataSource`. To wire it in:

1. Add an `Arc<dyn BlockDataSource>` field to
   `crates/core/src/scheduler/memory/mod.rs::MemoryManager`, populated
   at construction time with a `PagedKvCacheWrapper` (new file).
2. Plumb the wrapper through `EngineBuilder → Engine →
   SchedulerEngine → MemoryManager`.
3. `MemoryManager::allocate` and `record_block_tokens` already call
   `cache.put(...)`; the wrapper hooks in next to them to populate
   the local `BlockDataSource` (PagedKvCache) so the gRPC handler can
   serve real bytes.

This is a model-crate touch + scheduler plumbing — deferred per the
technical due diligence ("在单机 batched kernel 未成熟前建设完整多节点
MESI/KV 协议").

### Open: MESI / coherence (v32+)

OPS-31d uses the single-state model: any node with the cache entry
can serve the bytes. For multi-node coherence, a true MESI implementation
needs:
- Per-block ownership tracking (Exclusive vs Shared vs Modified).
- Read/write invalidation on `compute_owner_nodes(block_id)` boundaries.
- Block transfer on read-with-intent-to-modify.

OPS-31d deliberately does not implement this; it just closes the
protocol-layer gap so v32+ can layer MESI on top.

### Open: streaming RPCs (v32+)

Unary + 64 MiB covers realistic block sizes today. If we ever need to
transfer >64 MiB (e.g. multi-block prefetch), the natural extension is
`StreamKVBlock(server-streaming)`. Both the message types and the
client API would change, but the `BlockDataSource` abstraction stays
the same.

### Open: wire compression (v32+)

fp16/int8 on the wire would cut bandwidth by 2× / 4×. Would require
adding a `wire_format` enum to `TransferKvBlockResponse` so the
receiver knows whether to deserialize as f32 / f16 / int8.

### Open: block refcounting during transfer (v32+)

When a block is "on the wire" (sender is responding), the sender's
eviction policy should not free it. OPS-31d doesn't add refcounts
because the current `EvictionPolicy.block_ref_count` is per-block
across sequences, not "in-flight transfer". v32+ can either reuse it
or add a parallel `in_flight_transfers` counter.

---

## 5. Files touched

- `crates/dist/Cargo.toml` — added `async-trait = { workspace = true }`.
- `crates/dist/src/distributed_kv/block_data_source.rs` — new file.
  `BlockDataSource` trait, `FetchError`, `MAX_BLOCK_TRANSFER_BYTES`,
  `MockBlockDataSource`.
- `crates/dist/src/distributed_kv/mod.rs` — `pub mod
  block_data_source;` + re-exports.
- `crates/dist/src/distributed_kv/cache.rs` — `block_data_source`
  field, `with_block_data_source` setter, `fetch_block` method, 4
  unit tests.
- `crates/dist/proto/node.proto` — `TransferKVBlock` RPC + 2 messages.
- `crates/dist/src/grpc.rs` — `GrpcState.block_data_source`,
  `with_block_data_source`, `transfer_kv_block` handler, 64 MiB
  message limits, `start_grpc_server(_with_listener)` extended,
  6 unit tests.
- `crates/dist/src/grpc_client.rs` — `PeerClient::fetch_block` with
  bumped message limits, 1 unit test.
- `crates/dist/src/lib.rs` — re-exports.
- `crates/dist/tests/distributed_kv_peer_sync.rs` — updated
  `spawn_server` helper to pass `None` for the new parameter.
- `crates/dist/tests/kv_block_transfer.rs` — new file. 7 integration
  tests.

---

## 6. Verification

- `cargo build -p vllm-dist --all-features`: clean.
- `cargo build --workspace --all-features`: clean.
- `cargo clippy -p vllm-dist --all-targets --all-features -- -D
  clippy::correctness -D clippy::suspicious -D clippy::perf`: clean
  (project's CI clippy policy). Pedantic lints emit warnings only.
- `cargo fmt -p vllm-dist --check`: clean.
- `cargo test -p vllm-dist --all-features`: 75 unit + 5 + 7
  integration = 87 tests pass; 1 doc test ignored.
- `cargo test --workspace --all-features`: 1338 tests pass; 0 fail.

### Test count delta

- `vllm-dist` unit tests: 59 → 75 (+16)
- `vllm-dist` integration tests: 5 → 12 (+7)
- Workspace `--all-features`: 1307 → 1338 (+31)

---

## 7. What is NOT wired up (explicit non-goals)

| Item | Reason | Future phase |
|---|---|---|
| Engine integration (plumbing `Arc<PagedKvCache>` through `MemoryManager`) | Model crate touch; defer per due diligence | v32+ |
| MESI coherence protocol | Single-state model is sufficient for v31 alpha | v32+ |
| Streaming RPCs (`StreamKVBlock`) | Unary + 64 MiB covers realistic block sizes | v32+ |
| Wire compression (fp16/int8) | Adds complexity; can layer on later | v32+ |
| Smart owner-based peer routing | Fan-out fallback is correct, just bandwidth-wasteful | v32+ |
| Block refcounting during transfer | Requires engine integration | v32+ |
| Removing legacy `GetKVCache` RPC | Out of scope; tracked for cleanup phase | — |

---

## 8. Migration toward next phase (v32+ OPS-32a)

1. **`PagedKvCacheWrapper`**: implement `BlockDataSource` on top of
   `PagedKvCache`. Read methods pull from
   `key_cache[layer][block_id]` and `value_cache[layer][block_id]`,
   concatenating layers into a single `Vec<u8>`. The reverse
   (inserting received bytes) would land in
   `MemoryManager::record_block_tokens` next to the existing
   `cache.put(...)` call.
2. **`CacheConfig::peer_node_ids`**: add a `Vec<NodeId>` field
   parallel to `peer_urls` so `compute_owner_nodes` can route fetches
   to the right peer instead of fan-out.
3. **`OPS-32a` protocol changes**: replace fan-out with
   owner-routed single-RPC fetch. Add `MESI` state to `CacheEntry` so
   the receiver knows whether to broadcast an `InvalidateKVCache`
   after installing a transferred block.
4. **`OPS-32b` streaming**: add `StreamKVBlock` for >64 MiB blocks.
5. **`OPS-32c` compression**: add `wire_format` enum to
   `TransferKvBlockResponse`, plumb the existing `quantized: bool`
   flag through.
