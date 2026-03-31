# Changelog

All notable changes to vLLM-lite will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Real-time metrics collection with `/v1/stats` and `/metrics` endpoints
- Quantization utilities (`crates/model/src/quantize.rs`)
- Tiled Attention for memory optimization
- INT8 quantization support in KV Cache
- Forward pass with tiled attention strategy
- Comprehensive test suite for tiled attention

### Changed
- Improved documentation structure (README.md, docs/README.md, ROADMAP.md)
- Added detailed development roadmap

### Fixed
- Clippy warnings and code quality improvements
- Test compatibility with new AttentionConfig

## [0.1.0] - 2026-03-31

### Added
- **Continuous Batching** - Dynamic batch scheduling with decode-priority
- **Paged KV Cache** - Memory-efficient cache management with LRU eviction
- **Prefix Caching** - Exact match and prefix hit support
- **Speculative Decoding** - Draft-target verification architecture
- **Qwen3 Model Integration** - Support for Qwen2.5-0.5B model with real weights
- **OpenAI-compatible API** - `/v1/completions`, `/v1/chat/completions`
- **Streaming (SSE)** - Real-time token streaming
- **Sampling** - Temperature, Top-P support
- **Chunked Prefill** - Process long prompts in chunks

### Architecture
- **3-Crate Structure**:
  - `vllm-core`: Scheduler, Engine, KV Cache, Types
  - `vllm-model`: Qwen3, Attention, MLP
  - `vllm-server`: HTTP API (axum)

### Dependencies
- Rust (edition 2021)
- Candle (ML backend)
- Axum (HTTP)
- Tokio (async runtime)
- SafeTensors (weight loading)

---

## Migration Guides

### Upgrading to 0.1.0

No migration needed - initial release.

---

## Known Issues

- Limited model support (Qwen3 only)
- No multi-GPU support
- Quantization in progress

---

## Credits

Thanks to all contributors and the vLLM project for inspiration.