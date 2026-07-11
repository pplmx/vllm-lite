# Phase 19 OPS-05c — gRPC peer-sync for `DistributedKVCache`

**Date:** 2026-07-12
**Scope:** `vllm-dist` (`PeerClient`, `DistributedKVCache::connect_peers` / `put` /
`invalidate` broadcast, `NodeService` Put/InvalidateKVCache RPCs, integration tests)
**Status:** Shipped
**Phase goal (long-term):** Wire vllm-dist into the Engine end-to-end so
multi-node inference is real, not just available as a library.

---

## 1. Why this phase

OPS-05b3 (commit `aa768ae`) established the lookup path: when a request
arrives with a token prefix, the scheduler walks the chain hash and asks
the cache whether the prefix is present *somewhere in the cluster*. But
the cache was purely local — a peer's `put` never reached this node.

OPS-05c closes that loop by replicating every local `put` / `invalidate`
out to every configured peer over gRPC. After this ships, two nodes
sharing `peer_urls` will see the same KV-cache state from the local
cache's perspective (modulo broadcast latency).

What's *not* in scope: this commit ships *replication of intent*, not
*block transfer*. The peer still has a `value_hash` (the content hash
from OPS-05b2); it doesn't have the KV blocks themselves. Actually
moving KV blocks across nodes requires a separate transfer protocol
(scheduled for a later phase). Until then the operational value of this
change is in *prefix-match telemetry*: with both sides of the chain in
sync, `lookup_prefix` (OPS-05b3) returns real data about how much of a
remote request's prompt is already cached anywhere in the cluster.

---

## 2. What changed

### New file: `crates/dist/src/grpc_client.rs`

`PeerClient` is a small wrapper over the tonic-generated
`NodeServiceClient`. It holds:

- The peer URL (for logs).
- A pre-built `Endpoint` (cheap; no connection yet).
- An `Arc<Mutex<Option<Channel>>>` for lazy connect.

```rust
pub struct PeerClient {
    url: String,
    endpoint: Endpoint,
    channel: Arc<Mutex<Option<Channel>>>,
}
```

The two RPCs exposed mirror the local cache's mutating methods:

```rust
pub async fn put(&self, block_id: u64, value_hash: u64) -> Result<(), tonic::Status>;
pub async fn invalidate(&self, block_id: u64) -> Result<(), tonic::Status>;
```

Channel creation is `async fn ensure_channel`, with two paths:

1. **Fast path:** someone already connected — return their `Channel`
   clone, drop the mutex guard before the `Ok` propagates (per
   `significant_drop_tightening`).
2. **Slow path:** `endpoint.connect().await`, then briefly re-take the
   lock to cache, then drop it before returning.

Only the first RPC pays the TCP+HTTP/2 handshake cost; subsequent
calls reuse the channel (tonic's `Channel` is internally `Arc`-backed,
so cloning is cheap).

### `DistributedKVCache::connect_peers` (`crates/dist/src/distributed_kv/cache.rs`)

```rust
pub fn connect_peers(&mut self) -> Result<(), GrpcError>;
```

Builds a `PeerClient` for every URL in [`CacheConfig::peer_urls`]. Sync
because `PeerClient::new` only validates URLs (`Endpoint::from_shared`)
— actual TCP happens later. Idempotent: re-calling replaces the prior
client set; calling with empty `peer_urls` clears back to single-node
mode.

Three observation helpers (`peer_urls()`, `peers_connected()`,
`peer_client_count()`) let tests and operators verify configuration
without poking at the `peer_clients` field directly.

### `DistributedKVCache::put` / `invalidate` peer broadcast

Both methods gained a tail-call to a new private helper:

```rust
pub fn put(&self, key: u64, value_hash: u64) {
    // ... existing local-cache mutation + stats bump ...

    self.broadcast_put(key, value_hash);
}

fn broadcast_put(&self, key: u64, value_hash: u64) {
    let Some(clients) = self.peer_clients.clone() else { return; };
    if clients.is_empty() { return; }
    let Ok(handle) = tokio::runtime::Handle::try_current() else {
        tracing::debug!(block_id = key, "put: no tokio runtime; skipping peer broadcast");
        return;
    };
    handle.spawn(async move {
        for client in clients.iter() {
            if let Err(e) = client.put(key, value_hash).await {
                tracing::warn!(peer = %client.url(), block_id = key, error = %e,
                    "peer put failed; local update stands");
            }
        }
    });
}
```

`broadcast_invalidate` mirrors the same pattern with the
`invalidate_kv_cache` RPC.

Design points worth flagging:

- **Local first, broadcast second.** The local put/invalidate runs
  synchronously and updates stats before the broadcast is even
  spawned. A broadcast failure can never roll back a local change.
- **Fire-and-forget.** The spawned task's `JoinHandle` is dropped; we
  don't wait for completion and don't surface errors to the caller
  beyond a `tracing::warn!`. The local put has already happened; the
  caller's response latency is unaffected by peer RTT.
- **No retry.** A peer that's down at put-time will simply miss this
  update. Subsequent `lookup_prefix` calls on the peer will miss the
  block, and the prefix-match telemetry will reflect that. This is the
  right tradeoff for cache replication (vs. e.g. RAFT-style
  coordination) — KV-cache state is recoverable from any node that has
  the source blocks, and the cost of strong consistency is too high for
  this hot path.
- **Runtime-aware skip.** When there's no tokio runtime in scope (e.g.,
  a unit test calling `put` outside `#[tokio::test]`), the broadcast is
  silently dropped. The local put still happens. This keeps the
  existing unit tests in `cache.rs` working without modification.

### `CacheConfig::peer_urls` (`crates/dist/src/distributed_kv/mod.rs`)

```rust
pub struct CacheConfig {
    // ... existing fields ...
    pub peer_urls: Vec<String>,
}
```

Empty by default — single-node mode, no replication, no network.
Builder setter:

```rust
impl CacheConfig {
    pub fn with_peer_urls(mut self, peer_urls: Vec<String>) -> Self { ... }
}
```

### `NodeService` Put/InvalidateKVCache RPCs (`crates/dist/proto/node.proto` + `crates/dist/src/grpc.rs`)

Two new RPCs:

```protobuf
rpc PutKVCache(PutKVCacheRequest) returns (PutKVCacheResponse);
rpc InvalidateKVCache(InvalidateKVCacheRequest) returns (InvalidateKVCacheResponse);

message PutKVCacheRequest { uint64 block_id = 1; uint64 value_hash = 2; }
message PutKVCacheResponse { bool success = 1; }
message InvalidateKVCacheRequest { uint64 block_id = 1; }
message InvalidateKVCacheResponse { bool success = 1; }
```

`block_id` and `value_hash` are both `u64`, matching the local cache's
key/value layout — no translation needed on the wire.

`GrpcState` grew an optional cache handle:

```rust
pub struct GrpcState {
    pub node_id: String,
    pub peers: Arc<RwLock<Vec<String>>>,
    pub distributed_kv: Option<Arc<DistributedKVCache>>,
    pub kv_cache: Arc<RwLock<HashMap<String, Vec<u8>>>>,
}

impl GrpcState {
    pub fn with_distributed_kv(mut self, cache: Arc<DistributedKVCache>) -> Self { ... }
}
```

The new RPC handlers fold inbound calls into the local cache:

```rust
async fn put_kv_cache(&self, request: Request<PutKvCacheRequest>)
    -> Result<Response<PutKvCacheResponse>, Status>
{
    let req = request.into_inner();
    if let Some(cache) = self.state.distributed_kv.as_ref() {
        cache.put(req.block_id, req.value_hash);
    } else {
        warn!(block_id = req.block_id,
            "PutKVCache received but no DistributedKVCache wired in; dropping");
    }
    Ok(Response::new(PutKvCacheResponse { success: true }))
}
```

The `None` branch is intentional: failing the RPC would cause client
retries. The server is configured at boot — if it doesn't have a
cache wired, that's the operator's problem to fix, and dropping the
message is correct.

### `start_grpc_server_with_listener` (`crates/dist/src/grpc.rs`)

`start_grpc_server` retained its `(node_id, listen_addr, cache)` shape
but now delegates to a new `start_grpc_server_with_listener` that takes
a pre-bound `TcpListener`. Tests bind to port `0` to get an
OS-assigned port and read it back from `listener.local_addr()` before
starting the server — the only reliable way to test multi-node
behavior on a single host without flakes from port collisions.

### Re-exports (`crates/dist/src/lib.rs`)

```rust
pub mod grpc_client;
pub use grpc::{GrpcState, start_grpc_server_with_listener};
pub use grpc_client::PeerClient;
```

---

## 3. Tests

### Unit tests (existing in `cache.rs`)

All pre-existing tests continue to pass unchanged — they don't call
`connect_peers`, so broadcast is silently skipped at the
`Handle::try_current` check. `peer_clients` is `None`, the `else` branch
fires, debug-log fires, the local put still completes.

### Unit tests (new in `grpc_client.rs`)

- `peer_client_new_accepts_valid_url` — `new("http://localhost:50051")`
  succeeds; `is_connected()` returns false (no RPC has happened).
- `peer_client_new_rejects_invalid_url` — `new("")` returns
  `Err(GrpcError::Transport)` because `Endpoint::from_shared` rejects
  empty strings.
- `peer_client_is_clone` — clones share the inner `Arc<Mutex>` (verified
  via `Arc::strong_count`).

### Unit test (new in `grpc.rs`)

- `test_grpc_state_with_distributed_kv` — `with_distributed_kv` stores
  the same `Arc` instance (verified via `Arc::ptr_eq`).

### Integration tests (`crates/dist/tests/distributed_kv_peer_sync.rs`)

Five tests, each spins up one or more real gRPC servers on ephemeral
ports:

1. **`put_broadcasts_to_peer_via_grpc`** — node B's `put(42, hash)`
   shows up in node A's local cache within 1s (50 × 20ms poll).
2. **`invalidate_broadcasts_to_peer_via_grpc`** — seeds both sides,
   then node B's `invalidate(7)` removes the entry on A.
3. **`multi_peer_broadcast`** — node B broadcasts to two peers (A and
   C); both observe the put.
4. **`single_node_cache_does_not_broadcast`** — no peer URLs
   configured; `put` / `invalidate` are no-ops, no panic, no
   `peer_clients` allocated.
5. **`peer_client_lazy_connection`** — first RPC triggers channel
   connect; subsequent calls reuse the channel.

The polling loop (`for _ in 0..50 { sleep(20ms) }`) avoids CI flakiness
on slow hosts without making the test slow on fast ones — at 20ms
intervals, 50 iterations is 1s, plenty of headroom for local-loopback
RPCs that complete in microseconds.

---

## 4. Design notes & open questions

### Why fire-and-forget instead of request-response

Strong-consistency replication (synchronous, request-response, with
retries) would be required if the local put was conditional on the
peers' ack. But the local put is unconditional — we already have the
block locally; the broadcast is purely informational. So:

- Local put latency is unaffected by peer RTT or availability.
- A slow peer can't stall the hot path.
- A failed peer just misses this update; the next `lookup_prefix` on
  that peer will simply not find the block (and prefix-match telemetry
  reflects reality).

The cost: peers can briefly disagree about cache state. That's
acceptable because the cache is best-effort by design — it's a hint,
not a source of truth. The actual KV blocks are the source of truth,
and they're stored in this node's block allocator regardless.

### Why `Option<Arc<Vec<PeerClient>>>` instead of `Vec<PeerClient>`

The `Option` distinguishes three states:

| `peer_clients`           | Meaning                                                |
|--------------------------|--------------------------------------------------------|
| `None`                   | `connect_peers` was never called — single-node by construction. |
| `Some(empty vec)`        | `connect_peers` was called with no peer URLs — explicit single-node mode. |
| `Some(non-empty vec)`    | Multi-node — broadcasts fire on `put` / `invalidate`. |

The broadcast helpers (`broadcast_put`, `broadcast_invalidate`) early-
return on `None` and on empty `Vec`, but they *behave* the same way
(no-op). The distinction matters only for the observation helpers
(`peers_connected()`) and for the tests that want to assert
"`connect_peers` was called".

### Why `connect_peers` is sync

It only validates URLs and constructs endpoints. The actual TCP+HTTP/2
handshake is lazy in `PeerClient::ensure_channel` on the first RPC.
This means `connect_peers` doesn't need a tokio runtime, which keeps
the construction-time API surface small (no `async fn new`, no
runtime-in-scope requirement for the simple cases).

Originally `connect_peers` was `async`. Clippy caught that
(`unused_async`); the sync version is the same code minus one keyword.

### Open: block transfer

The replicated state is `(block_id, value_hash)`, not the KV blocks
themselves. A node that receives a peer's `put` knows that *some
content* with hash `value_hash` exists at *some block_id* on the peer
— but it can't reconstruct the KV blocks from that. For real
cross-node inference, we need a block-transfer protocol: when node A
detects via `lookup_prefix` that node B has the prefix, A needs to
fetch the actual KV blocks from B. That's a separate phase.

---

## 5. Files touched

- `crates/dist/proto/node.proto` — added `PutKVCache` /
  `InvalidateKVCache` RPCs + 4 message types.
- `crates/dist/src/grpc.rs` — `GrpcState.distributed_kv`, `with_distributed_kv`,
  `put_kv_cache` / `invalidate_kv_cache` handlers,
  `start_grpc_server_with_listener`.
- `crates/dist/src/distributed_kv/mod.rs` — `CacheConfig.peer_urls`,
  `with_peer_urls`.
- `crates/dist/src/distributed_kv/cache.rs` — `peer_clients` field,
  `connect_peers`, broadcast helpers wired into `put` / `invalidate`.
- `crates/dist/src/grpc_client.rs` — new file. `PeerClient`.
- `crates/dist/src/lib.rs` — re-export `PeerClient` and
  `start_grpc_server_with_listener`.
- `crates/dist/tests/distributed_kv_peer_sync.rs` — new file. 5
  integration tests.

---

## 6. Verification

- `cargo build -p vllm-dist`: clean.
- `cargo build --workspace --all-features`: clean.
- `cargo clippy --all-targets --workspace --all-features -- -D
  clippy::correctness -D clippy::suspicious -D clippy::perf`: clean
  (project's CI clippy policy).
- `cargo test -p vllm-dist --all-features`: 59 unit tests + 5
  integration tests pass; 1 doc test ignored (no network in doctests).
- `cargo test --workspace --all-features`: 1307 tests pass; 0 fail.
- `cargo fmt --all -- --check`: clean.
