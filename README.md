# vLLM-lite

A lightweight LLM inference engine written in Rust, implementing key vLLM innovations:

- **Continuous Batching** - Dynamic batch scheduling with fairness
- **Paged KV Cache** - Memory-efficient cache management  
- **Prefix Caching** - Cache repeated prompts
- **Speculative Decoding** - Accelerated token generation
- **OpenAI-compatible API** - `/v1/completions`, `/v1/chat/completions`

**Latest:** Continuous batching implemented (commit: d22d3e3)

## Roadmap

See [ROADMAP.md](./ROADMAP.md) for detailed development plan.

## Features

- 🚀 Fast Rust implementation
- 🎯 Continuous batching with decode-priority scheduling
- 💾 Paged KV cache with LRU eviction
- 🔄 Streaming token generation (SSE)
- 📡 OpenAI-compatible HTTP API
- 🖥️ CUDA GPU support (via Candle)

## Quick Start

```bash
# Build
cargo build --workspace

# Run server
cargo run -p vllm-server

# Test
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, how are you?", "max_tokens": 50, "stream": true}'
```

## Architecture

```
vllm-lite/
├── crates/
│   ├── core/       # Scheduler, Engine, KV Cache, Types
│   ├── model/      # Qwen3, Attention, MLP
│   └── server/     # HTTP API (axum)
├── docs/           # Design docs and plans
└── tests/          # Integration tests
```

## Project Phases

| Phase | Feature | Status |
|-------|---------|--------|
| 1 | MVP (single request, fake model) | ✅ Done |
| 2-3 | Continuous Batching | ✅ Done |
| 4 | Paged KV Cache | ✅ Done |
| 5 | Qwen3 Model (real weights) | ✅ Done |
| 6 | Prefix Caching | ✅ Done |
| 7 | Speculative Decoding | ✅ Done |

## Implemented Features

- [x] Continuous batching with decode-priority scheduling
- [x] Fairness-aware scheduling (max_consecutive_decode)
- [x] Chunked prefill processing
- [x] Paged KV Cache (GPU memory management)
- [x] Prefix Caching (exact match + prefix hit)
- [x] Speculative Decoding (draft-target verification)
- [x] Temperature and top-p sampling
- [x] Error tracking in Engine
- [x] Graceful shutdown
- [x] Fake model for testing

## Development

```bash
# Run tests
cargo test --workspace

# Run clippy
cargo clippy --workspace -- -D warnings

# Run specific crate tests
cargo test -p vllm-core
cargo test -p vllm-model
cargo test -p vllm-server
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