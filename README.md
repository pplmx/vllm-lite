# vLLM-lite

A lightweight LLM inference engine written in Rust, implementing key vLLM innovations.

## Supported Models

| Model | Architecture | Status |
|-------|--------------|--------|
| Qwen3 | GQA + RoPE | ✅ |
| Llama | GQA + RMSNorm | ✅ |
| Mistral | Sliding Window + GQA | ✅ |
| Gemma4 | Hybrid Attention + GeGLU | ✅ |
| Mixtral | Sparse MoE (8 experts) | ✅ |

## Features

- **Continuous Batching** - Dynamic batch scheduling with fairness
- **Paged KV Cache** - Memory-efficient cache management with pool
- **Prefix Caching** - Block hash-based cache reuse
- **Speculative Decoding** - Accelerated token generation
- **OpenAI-compatible API** - `/v1/completions`, `/v1/chat/completions`
- **Flash Attention** - Dynamic tile size selection (64/128/256)
- **Fused Kernels** - Optimized attention + MLP layers
- **Authentication** - API key support
- **Rate Limiting** - Request rate control

## Quick Start

```bash
# Build
cargo build --workspace

# Run server (default: /models/Qwen2.5-0.5B-Instruct)
cargo run -p vllm-server

# Or specify model path
cargo run -p vllm-server -- --model /path/to/your/model

# Test
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, how are you?", "max_tokens": 50, "stream": true}'
```

## Configuration

Configuration can be provided via YAML file or environment variables:

```yaml
# config.yaml
server:
  host: "0.0.0.0"
  port: 8000
  log_level: "info"

engine:
  max_draft_tokens: 8
  num_kv_blocks: 1024
  max_batch_size: 256
  tensor_parallel_size: 1
  kv_quantization: false  # Enable INT8 KV cache quantization

auth:
  api_keys: []
  rate_limit_requests: 100
  rate_limit_window_secs: 60
```

### Environment Variables

| Variable                    | Description                       | Default   |
| --------------------------- | --------------------------------- | --------- |
| `VLLM_HOST`                 | Server host                       | `0.0.0.0` |
| `VLLM_PORT`                 | Server port                       | `8000`    |
| `VLLM_LOG_LEVEL`            | Log level                         | `info`    |
| `VLLM_MAX_DRAFT_TOKENS`     | Max speculative tokens            | `8`       |
| `VLLM_KV_BLOCKS`            | Number of KV blocks               | `1024`    |
| `VLLM_TENSOR_PARALLEL_SIZE` | Tensor parallel size              | `1`       |
| `VLLM_KV_QUANTIZATION`      | Enable INT8 KV cache quantization | `false`   |

### CLI Options

```bash
cargo run -p vllm-server -- --help
```

## API Endpoints

| Endpoint               | Method   | Description        |
| ---------------------- | -------- | ------------------ |
| `/v1/chat/completions` | POST     | Chat completion    |
| `/v1/completions`      | POST     | Text completion    |
| `/v1/embeddings`       | POST     | Get embeddings     |
| `/v1/batches`          | POST/GET | Batch requests     |
| `/metrics`             | GET      | Prometheus metrics |
| `/health`              | GET      | Health check       |
| `/shutdown`            | GET      | Shutdown server    |

## Examples

### Chat Completion

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is Rust?"}
    ],
    "max_tokens": 100
  }'
```

### Streaming

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Once upon a time",
    "max_tokens": 50,
    "stream": true
  }'
```

### With Authentication

```bash
# Set API key
export VLLM_API_KEY=your-secret-key

# Use with requests
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-key" \
  -d '{"prompt": "Hello", "max_tokens": 10}'
```

## Features

- 🚀 Fast Rust implementation
- 🎯 Continuous batching with decode-priority scheduling
- 💾 Paged KV cache with LRU eviction + memory pool
- 🔍 Block hash-based prefix caching
- ⚡ Flash Attention with dynamic tile selection
- 🔗 Fused attention and MLP kernels
- 🔄 Streaming token generation (SSE)
- 📡 OpenAI-compatible HTTP API
- 🖥️ CUDA GPU support (via Candle)
- 📊 Real-time metrics collection
- 🔐 API key authentication
- ⏱️ Rate limiting

## Documentation

| File                           | Description                           |
| ------------------------------ | ------------------------------------- |
| [ROADMAP.md](./ROADMAP.md)     | Development roadmap and milestones    |
| [CHANGELOG.md](./CHANGELOG.md) | Version history and changes           |
| [AGENTS.md](./AGENTS.md)       | Developer guide and conventions       |
| [docs/](./docs/)               | Design specs and implementation plans |

## Architecture

```
vllm-lite/
├── crates/
│   ├── traits/      # Interface definitions (ModelBackend trait)
│   ├── core/        # Engine, Scheduler, KV Cache, Metrics
│   ├── model/       # Qwen3, Llama, Mistral, Gemma4, Mixtral
│   ├── dist/        # Tensor Parallel support
│   └── server/      # HTTP API (OpenAI compatible)
```

## Tech Stack

- **Runtime**: tokio
- **ML Backend**: Candle
- **HTTP**: axum
- **Weights**: SafeTensors

## License

MIT

## Links

- [GitHub](https://github.com/pplmx/vllm-lite)
- [Documentation](https://pplmx.github.io/vllm-lite)
