# vLLM-lite

A lightweight LLM inference engine written in Rust, implementing key vLLM innovations:

- **Continuous Batching** - Dynamic batch scheduling with fairness
- **Paged KV Cache** - Memory-efficient cache management  
- **Prefix Caching** - Cache repeated prompts
- **Speculative Decoding** - Accelerated token generation
- **OpenAI-compatible API** - `/v1/completions`, `/v1/chat/completions`

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

## Features

- 🚀 Fast Rust implementation
- 🎯 Continuous batching with decode-priority scheduling
- 💾 Paged KV cache with LRU eviction
- 🔄 Streaming token generation (SSE)
- 📡 OpenAI-compatible HTTP API
- 🖥️ CUDA GPU support (via Candle)
- 📊 Real-time metrics collection

## Documentation

| File | Description |
|------|-------------|
| [ROADMAP.md](./ROADMAP.md) | Development roadmap and milestones |
| [AGENTS.md](./AGENTS.md) | Developer guide and conventions |
| [docs/](./docs/) | Design specs and implementation plans |

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