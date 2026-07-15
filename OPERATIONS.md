# vLLM-lite Operations Guide

Deployment, monitoring, and troubleshooting for production operators.

## Quick Deploy

```bash
# Build release binary
cargo build --release -p vllm-server

# Start with config
cargo run -p vllm-server -- --config config.yaml

# Docker
docker compose up -d
```

## Health Checks

| Endpoint | Purpose | Expected |
|----------|---------|----------|
| `GET /health` | Liveness | `200 OK` |
| `GET /ready` | Readiness (model loaded) | `200 OK` when ready |
| `GET /health/details` | Structured status | JSON with engine state |
| `GET /metrics` | Prometheus scrape | Text exposition format |

## Logging

```bash
# Console (formatted)
RUST_LOG=info cargo run -p vllm-server

# Debug scheduling / batching
RUST_LOG=debug cargo run -p vllm-server

# File output (JSON)
cargo run -p vllm-server -- --log-dir ./logs
```

Standard log fields: `request_id`, `seq_id`, `batch_size`, `phase`, `duration_ms`.

## Monitoring

Key Prometheus metrics (via `/metrics`):

- Request throughput and latency histograms
- KV cache utilization
- Prefix cache hit rate
- Speculative decoding acceptance rate
- Queue depth and active sequences

Grafana dashboard templates: see `docs/` and deployment manifests in `k8s/`.

## Configuration

Priority order: CLI flags > environment variables > YAML config file.

| Variable | Default | Notes |
|----------|---------|-------|
| `VLLM_HOST` | `0.0.0.0` | Bind address |
| `VLLM_PORT` | `8000` | API port |
| `VLLM_KV_BLOCKS` | `1024` | GPU memory sizing |
| `VLLM_API_KEY` | — | Bearer token auth |

See [README.md](./README.md) for full scheduler and engine config.

## Security Checklist

- [ ] Set `VLLM_API_KEY` or configure `auth.api_keys` in YAML
- [ ] Enable TLS termination (reverse proxy or `security/tls.rs`)
- [ ] Configure rate limiting (`rate_limit_requests`)
- [ ] Restrict `/debug/*` endpoints in production
- [ ] Run as non-root in containers

## Troubleshooting

### Model fails to load

1. Verify `config.json` + weights exist in model directory
2. Check `RUST_LOG=error` for `LoadError` variants
3. For GGUF: ensure `--features gguf` or `full`

### OOM / KV cache exhaustion

1. Reduce `num_kv_blocks` or `max_num_seqs`
2. Enable chunked prefill (`prefill_chunk_size`)
3. Check `/debug/kv-cache` for block utilization

### Slow first token (TTFT)

1. Check prefix cache hit rate in metrics
2. Reduce prompt length or enable prefix caching
3. Profile with `RUST_LOG=debug` on scheduler

### CUDA Graph disabled warning

Expected on CPU builds. For GPU: build with `--features cuda,cuda-graph`.

### Checkpoint integration tests

```bash
export VLLM_TEST_MODEL_DIR=/path/to/Qwen3-0.6B
just nextest-checkpoint
```

## Graceful Shutdown

```bash
curl -H "Authorization: Bearer $VLLM_API_KEY" http://localhost:8000/shutdown
# or SIGTERM / SIGINT to the process
```

Both paths run the production-readiness §7 sequence:

1. `readiness=false` is published (so the next `/health/ready` probe
   returns `503 not_ready` and Kubernetes stops routing new traffic
   to this pod).
2. SIGTERM path waits `server.shutdown_drain_grace_secs` (default
   `5` s, capped at `300`) to give the orchestrator time to detect
   the failed probe and remove the pod from the Service endpoints
   list. `/shutdown` does **not** wait — the operator owns the
   rolling-update cadence.
3. axum stops accepting new connections and drains in-flight HTTP
   requests.
4. Engine receives `EngineMessage::Shutdown`; the worker thread
   is joined with a `10` s deadline so a stuck engine can't pin
   the process forever.

Tune `shutdown_drain_grace_secs` in `config.yaml` to match your K8s
probe `failureThreshold * periodSeconds`:

```yaml
server:
  shutdown_drain_grace_secs: 5   # default; raise for slower probes
```

## Multi-Node (Experimental)

Multi-node setup is **library-level only** — there is no CLI flag
or `VLLM_*` environment variable for `peer_urls`. Embedders
construct a `vllm_dist::CacheConfig` and pass it into the engine
init; the binary does not currently expose it. If you just want to
ship a single-node binary, **stop reading here** — the rest of this
section only matters for embedders building multi-node deployments.

### What works (Phase 31-D / OPS-31d, v0.1+)

- Cross-node `(block_id, chain_hash)` replication over gRPC
  (`CacheMessage::Put` / `Invalidate`).
- On-demand `TransferKVBlock` RPC for KV tensor bytes
  (`MAX_BLOCK_TRANSFER_BYTES = 64 MiB` per direction — production
  Qwen3-7B blocks are ≈14 MiB at F32, so this is comfortable
  headroom).
- Fan-out fallback in `DistributedKVCache::fetch_block`: ask
  every peer, accept the first response whose `chain_hash`
  matches the local `value_hash`, fall back to the local
  `BlockDataSource` if every peer fails.

### What is **not** yet production-ready

- **Engine integration (the load-bearing piece)** — the
  `BlockDataSource` trait and `TransferKVBlock` handler exist,
  but no `PagedKvCacheWrapper` wires them through
  `MemoryManager` yet. Without that wrapper, the gRPC server
  answers `Status::unavailable("TransferKVBlock called but no
  BlockDataSource wired in")` for every block transfer. The
  integration tests in `crates/dist/tests/kv_block_transfer.rs`
  use an in-memory `MockBlockDataSource`; production wiring
  lands in v32+ (OPS-32a). Until then, multi-node replication
  works for `(block_id, chain_hash)` *intent* but actual block
  bytes stay local-only in the default engine build.
- **Smart owner-based routing** — fan-out is fine for 2–4 nodes
  but degrades quadratically; v32+ will track the block owner via
  the directory-coherence protocol and route the `TransferKVBlock`
  request to a single peer.
- **Failure recovery** — a peer that returns `HashMismatch` is
  treated as a hard error rather than a transient; v32+ will add
  retry with backoff.
- **MESI / Directory protocol enforcement** — the protocol enum
  exists but the active implementation is `None` (eventual
  consistency); switching to `MESI` or `Directory` currently
  returns `unimplemented!`.

### Minimum viable cluster (2 nodes, library API)

This is the canonical 2-node setup. The library API is the same
whether you run 2 or 32 nodes; only `peer_urls` changes.

```rust
use vllm_dist::{CacheConfig, NodeId, DistributedKVCache};

// On node 0:
let cfg = CacheConfig::new(NodeId(0), 2)
    .with_peer_urls(vec!["http://node-1:50051".to_string()]);
let cache = DistributedKVCache::new(cfg);
cache.connect_peers().expect("connect_peers ok");
```

For 3 nodes, pass both peer URLs — fan-out broadcasts to each:

```rust
use vllm_dist::{CacheConfig, NodeId, DistributedKVCache};

// On node 1 (middle of A=0, B=1, C=2):
let cfg = CacheConfig::new(NodeId(1), 3)
    .with_peer_urls(vec![
        "http://node-0:50051".to_string(),
        "http://node-2:50051".to_string(),
    ]);
let cache = DistributedKVCache::new(cfg);
cache.connect_peers().expect("connect_peers ok");
assert_eq!(cache.peer_client_count(), 2);
```

### Verify it works

The integration tests in `crates/dist/tests/` exercise both
the intent loop and the block-transfer loop end-to-end on a
local in-process gRPC pair (no real network needed):

```bash
# Peer-sync (intent loop) — 2-node + 3-node + single-node
cargo test -p vllm-dist --test distributed_kv_peer_sync

# Block transfer — fan-out fallback, hash verification,
# above-default message sizes (5 MiB round-trip)
cargo test -p vllm-dist --test kv_block_transfer
```

`multi_peer_broadcast` is the 3-node test: it spawns two servers
(A, C) and one client (B), calls `cache_b.put(99, 0xDEAD)`,
and asserts both A and C observe `block_id=99` within 100
polls × 20 ms. If that test passes, your `peer_urls` wiring is
correct. The block-transfer tests use a `MockBlockDataSource`
to verify `TransferKVBlock` round-trips including a
deliberately-large 5 MiB block (above Tonic's default 4 MiB
message limit).

### Wire protocol (TransferKVBlock, Phase 31-D)

The KV-block-transfer protocol is defined in
[`crates/dist/proto/node.proto`](./crates/dist/proto/node.proto).
The wire shape is intentionally minimal — the dist layer treats
block bytes as an opaque `Vec<u8>`:

```protobuf
service NodeService {
  // ... Ping, AllReduce, GetKVCache, GetPeers,
  //     PutKVCache, InvalidateKVCache ...
  rpc TransferKVBlock(TransferKVBlockRequest)
      returns (TransferKVBlockResponse);
}

message TransferKVBlockRequest {
  uint64 block_id       = 1;
  uint64 expected_hash  = 2;  // receiver's local chain_hash
}

message TransferKVBlockResponse {
  uint64 block_id    = 1;     // echo of requested block_id
  uint64 chain_hash  = 2;     // sender's local chain_hash
                              // (MUST equal expected_hash)
  bytes  data        = 3;     // opaque block bytes
  uint32 num_tokens  = 4;     // reserved; always 0 in v0.1
}
```

**Message-size limit:** both server and client bump
`max_decoding_message_size` and `max_encoding_message_size`
to `MAX_BLOCK_TRANSFER_BYTES = 64 MiB`
(`crates/dist/src/distributed_kv/block_data_source.rs`).
Tonic's default 4 MiB would silently fail any production-sized
block. If you embed vllm-dist in a custom server / client,
**bump the same limits** or block transfers will return
`tonic::Status::out_of_range_error`.

**Hash verification:** the receiver compares
`response.chain_hash` against its `expected_hash` before
installing bytes. Mismatch is fatal (no retry — see
"Failure recovery" under "What is not yet production-ready"
above). This prevents stale-block installation when a peer's
local cache has been invalidated between `lookup_prefix` and
`fetch_block`.

For architecture context see [docs/architecture.md §Crate Responsibilities](./docs/architecture.md#crate-responsibilities).
The phase plan for OPS-31d (the work that shipped this section)
is at `.planning/phase-19/ops-31d-kv-block-transfer.md`.

## CI Verification (Developers)

```bash
just ci          # fmt + clippy + doc + nextest + public-api-check
just ci-all      # ci + security audit + cargo-deny
just nextest-all # include #[ignore] slow tests (real checkpoints)
just public-api-check   # fail if any crate's public API grew without a CHANGELOG bullet
```

The public-API check enforces the GOV-01 contract that every
intentional API addition carries a `public-api: vllm-<crate>
added ...` bullet under `[Unreleased]`. Shrinking the API is
allowed without a CHANGELOG entry (the baseline IS the record
of what was removed).
