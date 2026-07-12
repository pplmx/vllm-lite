# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

vLLM-lite is a lightweight LLM inference engine written in Rust, implementing key vLLM innovations: Continuous Batching, Paged KV Cache, Prefix Caching, Speculative Decoding, and an OpenAI-compatible API.

## Development Workflow

1. Make changes
2. Run verification: `just ci` (fmt-check → clippy → doc-check → nextest)
3. Commit with format: `<type>(<scope>): <subject>` - types: feat, fix, refactor, test, docs, chore

## Key Design Decisions

- Single GPU worker thread (avoid GPU contention)
- Block size: 16 tokens per KV block
- Max batched tokens: 4096 default
- Max concurrent sequences: 256 default
- Uses `ModelBackend` trait (`Box<dyn ModelBackend>`) for model abstraction
- CUDA support via Candle
- `parking_lot::Mutex` for all scheduler/engine paths (no `std::sync::Mutex` poison checks)
- `std::sync::LazyLock` for lazy initialization (Rust 1.80+)
- All public errors are typed `thiserror::Error` enums; no `Box<dyn Error>` in public APIs
