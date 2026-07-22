# ADR-020: Multi-Node KV Block Transfer Architecture

**Date:** 2026-07-15
**Status:** Accepted (protocol layer shipped in Phase 31-D / OPS-31d; wrapper shipped in v31.0 P40; engine plumbing shipped in v31.0 P41; receiver-side `write_kv_batch` deferred to P42+)
**Context version:** v31.0

## Context

OPS-05c (commit `6fe1e69`) closed the *intent* loop for multi-node
KV-cache coherence: every local `put(block_id, value_hash)` /
`invalidate(block_id)` is replicated to every configured peer over
gRPC. But the replicated state is `(block_id, chain_hash)` — the
actual KV tensor bytes are still local-only. A node that detects
via `lookup_prefix` that a peer has the prefix still cannot pull
the prefix's bytes without a manual copy.

OPS-05c explicitly deferred this:

> What's *not* in scope: this commit ships *replication of intent*,
> not *block transfer*. ... Actually moving KV blocks across nodes
> requires a separate transfer protocol (scheduled for a later
> phase).

The technical due diligence (`docs/technical-due-diligence/roadmap.md`)
reinforced the sequencing:

> 在采样和 KV 生命周期未正确前增加更多模型架构。
> 在单机 batched kernel 未成熟前建设完整多节点 MESI/KV 协议。

By v31.0 Phase 31-A (chunked prefill correctness) and Phase 31-A
follow-up (ARCH-02 sampling seam), the prerequisites are met — the
single-node KV lifecycle is correct, and sampling honors per-sequence
params. OPS-31d closes the *protocol layer* gap. Engine-side wiring
(plumbing `Arc<PagedKvCache>` through `MemoryManager`) is explicitly
**not** in scope — that's a model-crate touch deferred to v32+.

## Decision

### 1. `BlockDataSource` trait

Add a storage-agnostic abstraction over raw block bytes:

```rust
#[async_trait::async_trait]
pub trait BlockDataSource: Send + Sync + fmt::Debug {
    async fn fetch_block(&self, block_id: u64) -> Result<Vec<u8>, FetchError>;
    async fn has_block(&self, block_id: u64) -> bool { let _ = block_id; true }
}
```

- **Storage-agnostic** — returns `Vec<u8>` so a future GPU-direct
  path (mmap / cudaMemcpyAsync into `bytes`) doesn't touch the dist
  layer.
- **`async`** — required because future wrappers (e.g., a
  `PagedKvCacheWrapper` reading from GPU memory) will block on
  device→host transfers.
- **Object-safe** — implementations can be `Arc<dyn BlockDataSource>`,
  which is how `GrpcState` wires it.

Production wrapper (`PagedKvCacheWrapper`) is **not** shipped —
the v31.0 engine does not expose `Arc<PagedKvCache>`. Integration
tests use a `MockBlockDataSource` against an in-process gRPC pair.

### 2. Unary `TransferKVBlock` gRPC RPC

```protobuf
rpc TransferKVBlock(TransferKvBlockRequest)
    returns (TransferKvBlockResponse);

message TransferKvBlockRequest {
  uint64 block_id       = 1;
  uint64 expected_hash  = 2;  // receiver's local chain_hash
}

message TransferKvBlockResponse {
  uint64 block_id    = 1;     // echo of requested block_id
  uint64 chain_hash  = 2;     // sender's local chain_hash
                              // (MUST equal expected_hash)
  bytes  data        = 3;     // opaque block bytes
  uint32 num_tokens  = 4;     // reserved; always 0 in v0.1
}
```

- **Unary, not streaming** — at `BLOCK_SIZE=16` and Qwen3-7B F32, a
  block is ≈14 MiB. A 64 MiB message limit comfortably fits. Streaming
  RPCs add complexity (`tonic::Stream`, back-pressure, partial-block
  reassembly) that is not justified at this size. Deferred to v32+
  OPS-32b for blocks >64 MiB.
- **Hash verification in the receiver** — the receiver compares
  `response.chain_hash` against its `expected_hash` BEFORE installing
  bytes. Mismatch is fatal (no retry). This prevents stale-block
  installation when a peer's local cache has been invalidated between
  `lookup_prefix` and `fetch_block`. Reliability recovery (retry with
  backoff on `HashMismatch`) is v32+ — see "Failure recovery" below.
- **`num_tokens` reserved** — always `0` in v0.1. The field is
  pre-declared so a future partial-block transfer can ship without a
  proto revision.

### 3. Fan-out fallback for peer routing

`DistributedKVCache::fetch_block` asks every peer, accepts the first
response whose `chain_hash` matches the local `value_hash`, and falls
back to the local `BlockDataSource` if every peer fails.

- **Owner-routed alternative** — track which peer holds each block
  (via the directory-coherence protocol) and route fetches to that
  peer. Lower bandwidth but requires (a) the directory protocol
  working (currently `None` → eventual consistency only) and (b)
  consistent ownership across the cluster.
- **Why fan-out wins for v0.1** — bandwidth is not the bottleneck for
  2-4 nodes, and fan-out is structurally correct regardless of who
  owns what. When the directory protocol lands (v32+), owner-routed
  routing is a small `compute_owner_nodes` change inside
  `fetch_block` — no protocol change required.

### 4. 64 MiB symmetric message limit

Both server and client bump `max_decoding_message_size` and
`max_encoding_message_size` to `MAX_BLOCK_TRANSFER_BYTES = 64 MiB`
(`crates/dist/src/distributed_kv/block_data_source.rs`).

- **Why explicit** — Tonic's default 4 MiB silently fails any
  production-sized block (Qwen3-7B F32 ≈14 MiB, fp8 ≈7 MiB). The
  integration test `fetch_block_works_above_default_message_limit`
  verifies a 5 MiB block round-trips end-to-end — guards against
  accidental revert to defaults.
- **Why 64 MiB specifically** — 4× headroom over the largest realistic
  block (14 MiB Qwen3-7B F32). Larger blocks in the future (BLOCK_SIZE
  bump, fp8→fp4 transitions) stay under this for at least one more
  doubling. The `MAX_BLOCK_TRANSFER_BYTES` constant is the single
  knob to bump if/when 64 MiB becomes tight.

### 5. Engine integration deferred (v32+ / OPS-32a)

The wiring that closes the loop end-to-end:

```
PagedKvCache ─► PagedKvCacheWrapper: BlockDataSource
       │
       ▼
GrpcState.block_data_source ─► transfer_kv_block handler
       │
       ▼
PeerClient.fetch_block ─► DistributedKVCache::fetch_block
```

is **not** shipped in v31.0. The `PagedKvCacheWrapper` requires
plumbing `Arc<PagedKvCache>` from `crates/model/src/paged_tensor/`
through `crates/core/src/scheduler/memory/MemoryManager` — a
model-crate touch that the technical due diligence deferred. Without
the wrapper, the gRPC server answers
`Status::unavailable("TransferKVBlock called but no BlockDataSource
wired in")` for every block transfer, so multi-node replication
works for `(block_id, chain_hash)` *intent* but actual block bytes
stay local-only in the default engine build. OPS-32a is the next
phase for this work.

### 6. Removal of the legacy `GetKVCache` RPC

The pre-existing `GetKVCache(block_hash) -> bytes` RPC was an
experimental stub from earlier multi-node iterations that used
content-hash lookup rather than block-id lookup. `TransferKVBlock`
supersedes it. The legacy RPC is NOT removed in OPS-31d (it's marked
as "Out of scope; tracked for cleanup phase") — removing it would
break embedders that depended on it during the v0.x window. Removal
lands in a dedicated cleanup phase before 1.0.

## Consequences

**Positive:**
- The protocol layer is complete and tested. Future work is mechanical
  (engine wiring) rather than architectural.
- The `BlockDataSource` seam is small, async, and object-safe — a
  production wrapper is a few hundred lines.
- Hash verification prevents the most likely silent-data-corruption
  failure mode (stale peer cache).
- 64 MiB symmetric limit is documented as a hard requirement for
  embedders building custom gRPC clients/servers.

**Negative / known limitations (v32+ candidates):**
- **MESI / Directory coherence** — the protocol enum exists but the
  active implementation is `None` (eventual consistency only).
  Switching to `MESI` or `Directory` currently returns `unimplemented!()`.
- **Smart owner-based routing** — fan-out is correct but
  bandwidth-wasteful for clusters > 4 nodes. v32+ OPS-32a replaces
  fan-out with owner-routed single-RPC fetch.
- **Failure recovery** — `HashMismatch` is treated as a hard error.
  v32+ adds retry with backoff and a transient-error class.
- **Streaming RPCs** — needed for blocks >64 MiB (BLOCK_SIZE growth
  or future larger model parallelism). v32+ OPS-32b.
- **Wire compression** — fp16/int8 quantization over the wire would
  cut bandwidth ~50%. The `quantized: bool` flag is plumbed in the
  proto's `num_tokens` reservation but not implemented. v32+ OPS-32c.
- **Block refcounting during transfer** — current transfer model
  treats each fetch as a fresh copy; refcount-based dedup requires
  engine integration (OPS-32a).
- **Engine integration** — `PagedKvCacheWrapper` is the load-bearing
  piece; without it the gRPC server returns `unavailable` for every
  block transfer. OPS-32a work.

## Alternatives considered

- **Owner-routed routing from day 1** — rejected. Requires the
  directory protocol to be working, which requires MESI or
  Directory coherence, which the technical due diligence explicitly
  deferred until the single-node KV lifecycle is correct (Phase 31-A).
  Fan-out is correct *today* without those prerequisites.
- **Streaming RPCs from day 1** — rejected. Adds tonic::Stream
  complexity for a size (14 MiB) that comfortably fits unary at
  64 MiB. Reassess when blocks exceed 64 MiB.
- **Put block bytes directly in `PutKVCache`** — rejected. Would
  conflate intent (replication metadata) with bulk transfer and
  force every `PutKVCache` to carry a 14 MiB payload. Separating
  intent from transfer keeps `PutKVCache` cheap (8 + 8 = 16 bytes).
- **Use `GetKVCache(block_hash)` instead of `TransferKVBlock(block_id)`**
  — rejected. `GetKVCache` uses content-hash lookup which loses the
  block-id semantics needed for chain-hash verification on the
  receiver. Block-id also matches the local cache keyspace (u64),
  so no translation is needed.
- **Remove `GetKVCache` in OPS-31d** — rejected. Breaks embedders
  that depend on it during the v0.x window. Tracked for a dedicated
  cleanup phase before 1.0.

## See also

- Phase plan: `.planning/phase-19/ops-31d-kv-block-transfer.md`
- Protocol definition: `crates/dist/proto/node.proto`
- Operator quickstart: `OPERATIONS.md` §"Multi-Node (Experimental)"
- Architecture context: `docs/architecture.md` §"Crate Responsibilities"
- ADR-008 (`vllm-dist` feature-gated) — the prerequisite workspace
  decision that kept the multi-node crate compilable in isolation.
- ADR-015 (`vllm-dist` investment decision) — the decision to resume
  multi-node work in v31.0 rather than delete the crate.
- Phase 31-D master plan: `.planning/v31.0-MASTER-PLAN.md` §31-D
