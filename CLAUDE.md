# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

vLLM-lite is a lightweight LLM inference engine written in Rust, implementing key vLLM innovations: Continuous Batching, Paged KV Cache, Prefix Caching, Speculative Decoding, and an OpenAI-compatible API.

## Common Commands

```bash
# Build the workspace
cargo build --workspace

# Run tests (all features)
just test
# or: cargo test --all-features --workspace

# Run a single test
cargo test -p vllm-core -- <test_name>
cargo test -p vllm-model -- <test_name>

# Clippy linting (required before commit)
just clippy

# Format code
just fmt

# Run the server
cargo run -p vllm-server

# Fast check (faster than build)
cargo check -p vllm-model

# Run all CI checks locally
just ci
```

## Architecture

The project is organized as a Rust workspace with 4 crates:

- **vllm-traits** (`crates/traits/`): Interface definitions - `ModelBackend` trait, shared types (SeqId, TokenId, Batch)
- **vllm-core** (`crates/core/`): Core inference engine - Engine, Scheduler, KV cache management, metrics
- **vllm-model** (`crates/model/`): ML model implementations - Qwen3, attention layers, PagedKvCache, SafeTensors loader
- **vllm-server** (`crates/server/`): HTTP server - OpenAI-compatible API endpoints, config, auth

### Key Components

**Engine** (`crates/core/src/engine.rs`): Main orchestrator managing scheduler, target/draft models, and metrics
```rust
pub struct Engine<M: ModelBackend> {
    pub scheduler: Scheduler,
    pub target_model: M,
    pub draft_model: M,
    pub max_draft_tokens: usize,
    pub metrics: MetricsCollector,
}
```

**Scheduler** (`crates/core/src/scheduler.rs`): Batch scheduling with waiting/running/finished queues, prefix cache, block allocator

**GqaAttention** (`crates/model/src/qwen3/attention.rs`): Grouped-query attention with `forward_prefill` and `forward_decode` methods

## Development Workflow

1. Make changes
2. Run verification: `just ci` (fmt-check → clippy → doc-check → nextest)
3. Commit with format: `<type>(<scope>): <subject>` - types: feat, fix, refactor, test, docs, chore

## Key Design Decisions

- Single GPU worker thread (avoid GPU contention)
- Block size: 16 tokens per KV block
- Max batched tokens: 4096 default
- Max concurrent sequences: 256 default
- Uses `ModelBackend` trait for model abstraction
- CUDA support via Candle