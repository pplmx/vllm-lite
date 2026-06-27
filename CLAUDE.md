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

The project is organized as a Rust workspace with **6 crates**:

- **vllm-traits** (`crates/traits/`): Interface definitions - `ModelBackend` trait, shared types (SeqId, TokenId, Batch)
- **vllm-core** (`crates/core/`): Core inference engine - Engine, Scheduler, KV cache management, metrics
- **vllm-model** (`crates/model/`): ML model implementations - Qwen3, Llama, Mistral, attention layers, PagedKvCache, SafeTensors loader
- **vllm-server** (`crates/server/`): HTTP server - OpenAI-compatible API endpoints, config, auth, security middleware
- **vllm-dist** (`crates/dist/`): Distributed primitives - tensor parallel, distributed KV, pipeline parallel, gRPC server (feature-gated: `--features multi-node`)
- **vllm-testing** (`crates/testing/`): Test harness, request factories, slow-model stubs

### Key Components

**Engine** (`crates/core/src/engine.rs`): Main orchestrator managing scheduler, target/draft models, and metrics. Non-generic; uses `Box<dyn ModelBackend>` for model abstraction.

```rust
pub struct Engine {
    pub target_model: Box<dyn ModelBackend>,
    pub draft_model: Option<Box<dyn ModelBackend>>,
    pub scheduler: Scheduler,
    pub max_draft_tokens: usize,
    pub metrics: Arc<EnhancedMetricsCollector>,
    // ...
}
```

Construction: `Engine::with_config(target_model, draft_model, config, max_draft_tokens, num_kv_blocks)`.

**Scheduler** (`crates/core/src/scheduler/`): Split into focused sub-modules:
- `queue/` - waiting/running/finished queues
- `preemption/` - preemption logic
- `eviction/` - KV cache eviction policies
- `batch/` - batch composition
- `policy/` - scheduling policies (FCFS, SJF, Priority)
- `memory/` - block allocator + prefix cache

**GqaAttention** (`crates/model/src/components/attention/gqa.rs`): Grouped-query attention with `forward_prefill` and `forward_decode` methods. Shared component used across all model architectures (Qwen3, Llama, Mistral).

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
