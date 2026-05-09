# External Integrations

**Last updated:** 2026-05-09
**Focus:** Tech

## Model File Formats

### Safetensors (Primary)
- Library: `safetensors` 0.7.0
- Supports sharded checkpoints (e.g., `model-00001-of-00002.safetensors`)
- Auto-detected by file extension in `crates/model/src/loader/format.rs`
- Used for loading transformer model weights

### GGUF (Optional, feature-gated)
- Library: `gguf` 0.1 (optional, feature `gguf`)
- Supports Q4_K_M quantization (dequantizes to FP16)
- File-based model format, single-file loading
- Implementation in `crates/model/src/quantize/gguf.rs`

## Inference Backend

### Candle (Candle-core / Candle-nn)
- Version: 0.10.2
- All model architectures use Candle tensors
- Optional CUDA support via `cuda` feature flag (`candle-core/cuda`, `candle-nn/cuda`)
- CPU fallback when CUDA is unavailable

### Flash Attention
- Custom CUDA kernel in `crates/model/src/kernels/flash_attention.rs`
- Tiled attention in `crates/model/src/components/attention/mod.rs`
- Flash Attention v3 in `crates/model/src/components/attention/flash_v3.rs`

## Tokenization

### tiktoken
- Version: 3
- OpenAI-compatible BPE tokenizer
- Used for models with OpenAI-style tokenizers (e.g., Qwen3)

### tokenizers
- Version: 0.22
- HuggingFace tokenizers library
- General-purpose tokenizer for most model architectures

## Networking & APIs

### HTTP API (OpenAI Compatible)
- Framework: `axum` 0.7
- Endpoints in `crates/server/src/openai/`:
  - `POST /chat/completions` — Chat completions (streaming + non-streaming)
  - `POST /completions` — Text completions
  - `POST /embeddings` — Embeddings
  - `GET /models` — List available models
  - `POST /v1/chat/completions` — OpenAI-compatible chat
  - `POST /v1/completions` — OpenAI-compatible completions
  - `POST /v1/embeddings` — OpenAI-compatible embeddings
  - `GET /v1/models` — OpenAI-compatible model listing
- SSE streaming support for chat/completions
- Batch processing support via `crates/server/src/openai/batch/`

### Prometheus Metrics
- Endpoint: `GET /metrics`
- Exposes engine metrics (throughput, latency, batch size, KV cache usage)
- Implemented via `metrics-exporter-prometheus`

### gRPC (Distributed)
- Framework: `tonic` 0.12 + `prost` 0.13
- Used for distributed tensor parallelism in `crates/dist/`
- Service definition in `crates/dist/build.rs` (compiled via `tonic-build`)
- Generated code in `crates/dist/src/generated/vllm.distributed.rs`

## Authentication & Security

| Feature | Location | Description |
|---------|----------|-------------|
| JWT validation | `crates/server/src/security/jwt.rs` | Token-based auth |
| RBAC | `crates/server/src/security/rbac.rs` | Role-based access control |
| TLS | `crates/server/src/security/tls.rs` | HTTPS listener support |
| Auth middleware | `crates/server/src/auth.rs` | API key and token auth |
| Audit logging | `crates/server/src/security/audit.rs` | Request audit trail |
| Correlation IDs | `crates/server/src/security/correlation.rs` | Request tracing headers |

## Request API Types

### EngineMessage (Internal IPC)
Defined in `crates/core/src/types.rs`:
- `AddRequest` — Submit new inference request
- `GetMetrics` — Query performance metrics
- `GetEmbeddings` — Request embeddings
- `Shutdown` — Graceful shutdown

## Crate Dependency Graph

```
vllm-traits
├── vllm-core (optional: vllm-model for cuda-graph)
│   ├── vllm-server
│   └── vllm-testing
├── vllm-model
│   ├── vllm-dist
│   └── vllm-testing
├── vllm-dist (build: tonic-build)
└── vllm-lite-benchmarks
```

## Observability Export

- **Prometheus**: Scrape endpoint at `/metrics` (default port 8000)
- **OpenTelemetry**: Optional OTLP export for distributed tracing
- **File logging**: JSON-formatted logs to files via `tracing-appender`
- **Console logging**: Human-readable format via `tracing-subscriber`
