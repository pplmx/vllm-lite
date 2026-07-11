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
curl http://localhost:8000/shutdown
# or SIGTERM to process
```

Engine drains in-flight requests before exit.

## Multi-Node (Experimental)

Requires `--features multi-node` on all nodes. Configure `peer_urls` for
distributed KV metadata sync. KV block transfer is not yet production-ready —
see [docs/architecture.md](./docs/architecture.md).

## CI Verification (Developers)

```bash
just ci          # fmt + clippy + doc + nextest + public-api-check
just ci-all      # ci + security audit
just nextest-all # include #[ignore] slow tests
```
