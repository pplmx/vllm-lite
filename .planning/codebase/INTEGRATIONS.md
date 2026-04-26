# External Integrations

**Analysis Date:** 2026-04-26

## APIs & External Services

### OpenAI-Compatible API

The server implements OpenAI-compatible endpoints for drop-in replacement:

| Endpoint | Method | Description | Location |
|----------|--------|-------------|----------|
| `/v1/chat/completions` | POST | Chat completion | `crates/server/src/openai/chat.rs` |
| `/v1/completions` | POST | Text completion | `crates/server/src/openai/completions.rs` |
| `/v1/embeddings` | POST | Vector embeddings | `crates/server/src/openai/embeddings.rs` |
| `/v1/batches` | POST/GET | Batch requests | `crates/server/src/openai/batch/` |
| `/v1/models` | GET | List available models | `crates/server/src/openai/models.rs` |

**API Compatibility:**
- OpenAI API v1 response format
- Bearer token authentication (optional)
- Server-Sent Events (SSE) streaming

### Health & Metrics Endpoints

| Endpoint | Method | Description | Auth Required |
|----------|--------|-------------|---------------|
| `/health` | GET | Liveness probe | No |
| `/ready` | GET | Readiness probe | No |
| `/metrics` | GET | Prometheus metrics | No |

## Model Format Support

### Supported Formats

| Format | Extension | Status | Loader |
|--------|-----------|--------|--------|
| **Safetensors** | `.safetensors` | ✅ Stable | `crates/model/src/loader/checkpoint.rs` |
| **GGUF** | `.gguf` | ✅ With `gguf` feature | `crates/model/src/quantize/gguf.rs` |

### Safetensors Loading

**Features:**
- Single file: `model.safetensors`
- Sharded files: `model-00001-of-00002.safetensors`
- Automatic directory detection

**Implementation:** `crates/model/src/loader/io.rs`
- `find_safetensors_files()` - discovers shard files
- `load_safetensors()` - deserializes weights
- `convert_tensor()` - maps safetensors dtypes to Candle

### GGUF Loading

**Supported Quantization:**
- Q4_K_M (primary)
- Others: designed for extensibility

**Implementation:** `crates/model/src/quantize/gguf.rs`
```rust
pub fn load_gguf_tensors(path: &Path, device: &Device) -> Result<HashMap<String, StorageTensor>>
```

**Storage Tensors:**
```rust
pub enum StorageTensor {
    Quantized(QuantizedTensor),  // Memory efficient
    Fp16(Tensor),                // Balanced
    Fp32(Tensor),                // Highest precision
}
```

## Hardware Acceleration

### CUDA Support

**Status:** Optional (via `cuda` feature)

**Features:**
- GPU tensor operations via Candle CUDA backend
- CUDA Graph optimization for decode phase
- Memory-efficient KV cache on GPU

**Configuration:**
```bash
# Enable CUDA
cargo run -p vllm-server --features cuda -- -m /model

# Environment variables
VLLM_CUDA_GRAPH_ENABLED=true    # CUDA Graph optimization
VLLM_CUDA_GRAPH_BATCH_SIZES=... # Batch sizes to capture
```

**CUDA Graph Integration:**
- `crates/core/src/scheduler/cuda_graph.rs` - scheduler integration
- `crates/model/src/kernels/cuda_graph.rs` - graph capture/execute
- `crates/model/src/kernels/cuda_graph/executor.rs` - execution manager

**Fallback:** CPU execution when CUDA unavailable or disabled

### CPU Support

**Status:** Always available

**Use cases:**
- Development/testing
- Fallback when GPU unavailable
- Small models (Qwen2.5-0.5B runs on CPU)

## Deployment Integrations

### Docker

**Image:** Multi-stage build for minimal runtime image
- Build stage: `rust:1.82-bookworm` with full build tools
- Runtime stage: `debian:bookworm-slim` with minimal deps

**Dockerfile:** `Dockerfile` (76 lines)
- Non-root user (`vllm:vllm`)
- Health check configured
- Port 8000 exposed

**Docker Compose:** `docker-compose.yml` (101 lines)
```yaml
services:
  vllm-server:
    image: vllm-lite:latest
    ports: ["8000:8000"]
    environment:
      - RUST_LOG=info
      - VLLM_MAX_NUM_SEQS=256
      - VLLM_MAX_BATCHED_TOKENS=4096
    volumes:
      - ./models:/app/models:ro
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
```

### Kubernetes

**Manifests:** `k8s/` directory

| File | Purpose |
|------|---------|
| `namespace.yaml` | vllm namespace |
| `deployment.yaml` | Pod spec with resource limits |
| `service.yaml` | ClusterIP service |
| `configmap.yaml` | Config volume |
| `hpa.yaml` | Horizontal pod autoscaling |

**Resource Configuration:**
```yaml
resources:
  requests:
    memory: "4Gi"
    cpu: "2"
  limits:
    memory: "8Gi"
    cpu: "4"
```

**HPA Configuration:**
```yaml
minReplicas: 1
maxReplicas: 10
metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        averageUtilization: 70
```

## Observability Integrations

### Prometheus Metrics

**Endpoint:** `GET /metrics`

**Key Metrics:**
```
vllm_tokens_total
vllm_requests_total
vllm_avg_latency_ms
vllm_p50_latency_ms
vllm_p90_latency_ms
vllm_p99_latency_ms
vllm_avg_batch_size
vllm_current_batch_size
vllm_requests_in_flight
vllm_kv_cache_usage_percent
vllm_prefix_cache_hit_rate
vllm_prefill_throughput
vllm_decode_throughput
vllm_cuda_graph_hits
vllm_cuda_graph_misses
vllm_cuda_graph_hit_rate
```

**Prometheus Config:** `config/prometheus.yml`
```yaml
scrape_configs:
  - job_name: 'vllm'
    static_configs:
      - targets: ['vllm:9090']
```

### OpenTelemetry (Optional)

**Feature:** Via `opentelemetry` feature flag

**Traces:**
- Request lifecycle
- Model forward passes
- Scheduler decisions

**Metrics:**
- Standard metrics with trace context

### Structured Logging

**Framework:** `tracing` crate

**Outputs:**
- Console: Pretty-printed with colors
- File: JSON format (via `tracing-appender`)

**Log Levels:**
- ERROR, WARN, INFO, DEBUG, TRACE
- Configurable via `RUST_LOG` env var or CLI

**Log Fields Standard:**
```rust
request_id: String
prompt_tokens: usize
output_tokens: usize
duration_ms: u64
seq_id: SeqId
batch_size: usize
phase: Phase  // Prefill/Decode
```

## Authentication

### API Key Authentication

**Methods:**
- CLI flag: `--api-key`
- Environment: `VLLM_API_KEY`
- File: `--api-key-file`

**Middleware:** `crates/server/src/auth.rs`
- Bearer token validation
- Per-endpoint auth requirements

## Tokenizer Integrations

### HuggingFace Tokenizers

**Package:** `tokenizers` 0.22

**Features:**
- BPE tokenization
- Special token handling
- Chat template support

**Implementation:** `crates/model/src/tokenizer.rs`
```rust
pub struct Tokenizer {
    inner: Option<Box<HFTokenizer>>,
    vocab_size: usize,
    special_tokens: Vec<String>,
}
```

### Tiktoken (OpenAI-compatible)

**Package:** `tiktoken` 3

**Use case:** Alternative tokenizer for OpenAI-compatible models

## Model Registry

### Supported Architectures

| Architecture | Location | Key Features |
|--------------|----------|--------------|
| **Llama** | `crates/model/src/llama/` | RMSNorm, RoPE, SwiGLU |
| **Mistral** | `crates/model/src/mistral/` | Sliding Window, GQA |
| **Qwen2/3** | `crates/model/src/qwen3/` | GQA, MLA, RoPE, QK-Norm |
| **Qwen3.5** | `crates/model/src/qwen3_5/` | Mamba SSM Hybrid |
| **Gemma4** | `crates/model/src/gemma4/` | Hybrid Attention |
| **Mixtral** | `crates/model/src/mixtral/` | Sparse MoE (8 experts) |

### Architecture Registry System

**Pattern:** Plugin architecture for extensibility
- `crates/model/src/arch/registry.rs` - registry management
- Each arch implements `Architecture` trait
- Automatic model config detection

## Environment Configuration

### Required Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_HOST` | `0.0.0.0` | Server bind address |
| `VLLM_PORT` | `8000` | Server port |
| `VLLM_MODEL` | (required) | Model path |

### Engine Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_KV_BLOCKS` | `1024` | KV cache blocks |
| `VLLM_MAX_BATCH_SIZE` | `256` | Max batch size |
| `VLLM_MAX_DRAFT_TOKENS` | `8` | Speculative decoding |
| `VLLM_TENSOR_PARALLEL_SIZE` | `1` | GPU count |
| `VLLM_KV_QUANTIZATION` | `false` | KV cache quantization |
| `VLLM_ADAPTIVE_SPECULATIVE` | `false` | Adaptive draft tokens |

### CUDA Variables

| Variable | Description |
|----------|-------------|
| `VLLM_CUDA_GRAPH_ENABLED` | Enable CUDA Graph |
| `VLLM_CUDA_GRAPH_BATCH_SIZES` | Batch sizes to capture |

---

*Integration audit: 2026-04-26*
