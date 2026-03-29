# vLLM-lite

A lightweight LLM inference engine written in Rust, implementing key vLLM innovations:

- **Continuous Batching** - Dynamic batch scheduling
- **Paged KV Cache** - Memory-efficient cache management  
- **Prefix Caching** - Cache repeated prompts
- **Speculative Decoding** - Accelerated token generation
- **OpenAI-compatible API** - `/v1/completions`, `/v1/chat/completions`

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
| 1 | MVP (single request, fake model) | ✅ |
| 2-3 | Continuous Batching | ✅ |
| 4 | Paged KV Cache | ✅ |
| 5 | Qwen3 Model | ✅ |
| 6 | Prefix Caching | ✅ |
| 7 | Speculative Decoding | ✅ |

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