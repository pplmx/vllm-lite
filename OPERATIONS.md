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
init; the binary does not currently expose it.

What works (Phase 31-D / OPS-31d, v0.1+):

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

What is **not** yet production-ready:

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

Minimum viable cluster (2 nodes, library API):

```rust
use vllm_dist::{CacheConfig, NodeId, DistributedKVCache};

let cfg = CacheConfig::new(NodeId(0), 2)
    .with_peer_urls(vec!["http://node-1:50051".to_string()]);
let cache = DistributedKVCache::new(cfg);
// see crates/dist/src/distributed_kv for the full API surface
```

For architecture context see [docs/architecture.md §vllm-dist](./docs/architecture.md#crate-map).

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
