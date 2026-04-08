# vLLM-lite Development Guide

This guide helps AI agents work effectively with the vllm-lite codebase.

## ⚠️ IMPORTANT: No Worktrees

**DO NOT use git worktrees for local development.** 

Single-developer local worktrees cause unnecessary merge conflicts and add complexity without benefit. Just work directly on the current branch.

## Quick Commands

```bash
# Build
cargo build --workspace

# Run all tests
cargo test --workspace

# Run single test (by name)
cargo test -p vllm-core test_engine_streaming
cargo test -p vllm-model -- attention

# Run clippy (required before commit)
cargo clippy --workspace -- -D warnings

# Format check
cargo fmt --all --check

# Full CI check
just ci
```

## Project Structure

```text
vllm-lite/
├── Cargo.toml              # Workspace root (5 crates: traits, core, model, server, dist)
├── justfile                # Build automation
├── crates/
│   ├── traits/             # Interface definitions (ModelBackend trait)
│   ├── core/               # Engine, Scheduler, KV cache, Metrics
│   │   └── src/
│   │       ├── scheduler/  # Scheduler modules (queue, preemption, eviction, batch)
│   │       └── kv_cache/   # Logical KV cache (block_allocator, prefix_cache)
│   ├── model/              # Model implementations
│   │   └── src/
│   │       ├── kernels/    # GPU kernels (flash_attention, fused_mlp, cuda_graph)
│   │       ├── paged_tensor/ # Physical KV cache (tensor_store, quantization)
│   │       └── components/ # Model components (attention, mlp, norm, positional)
│   ├── dist/               # Tensor Parallel support
│   └── server/             # HTTP API (OpenAI compatible)
└── tests/                  # Integration tests
```

## Code Style Guidelines

### Imports

- Use absolute imports: `use crate::types::Request;`
- External crates: `use vllm_traits::{ModelBackend, SeqId};`
- Group std, external, then crate imports
- Use `super` for sibling module access

### Formatting

- Run `cargo fmt` before committing
- 4-space indentation (Rust standard)
- Max line length: 100 characters (soft limit)
- Use `#![allow(clippy::too_many_arguments)]` when needed

### Naming Conventions

- **Types**: `PascalCase` (structs, enums, traits)
- **Functions/Variables**: `snake_case`
- **Constants**: `SCREAMING_SNAKE_CASE`
- **Modules**: `snake_case`
- **Crate names**: `kebab-case` (e.g., `vllm-core`)

### Types

- Use `usize` for sizes/lengths, `u64` for IDs (`SeqId`, `TokenId`)
- Prefer explicit type annotations in function signatures
- Use `&T` for read-only, `&mut T` for mutable references
- Use `Option<T>` for nullable values, not `null`

### Error Handling

- Use `thiserror` for error enums with `#[derive(Debug, thiserror::Error)]`
- Always implement `Display` and use `#[error(...)]` macro
- Return `Result<T>` from fallible functions
- Use `?` operator for error propagation
- Example (`crates/core/src/error.rs`):

```rust
#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("sequence {id} not found")]
    SeqNotFound { id: u64 },
    #[error("model forward failed: {0}")]
    ModelError(String),
}
pub type Result<T> = std::result::Result<T, EngineError>;
```

### Tests

- Add to `#[cfg(test)]` module in the same file
- Use `FakeModel` or `StubModel` for mocking
- Name: `test_<function>_<expected_behavior>`
- Run: `cargo test -p vllm-core -- test_name`

### Documentation

- Document public APIs with `///` doc comments
- Add examples for complex functions
- Use `//` for implementation comments, avoid `/**`

## Commit Message Format

```text
<type>(<scope>): <subject>
```

**Types**: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`

**Examples**:

```bash
git commit -m "feat(model): add forward_prefill to GqaAttention"
git commit -m "fix(core): resolve prefix cache eviction bug"
git commit -m "test(core): add prefix cache hit test"
```

## Verification (Required Before Commit)

```bash
just fmt-check    # Format validation
just clippy       # Code quality
just test         # Run tests
just ci           # Full CI (fmt-check → clippy → doc-check → test)
```

## Key Design Patterns

- **ModelBackend trait**: Abstracts ML model implementations
- **Paged KV Cache**: Block-based KV memory management
- **Prefix Caching**: Reuse KV for repeated prompts
- **Speculative Decoding**: Draft-then-verify token generation

## Crate Dependencies

- `vllm-traits` → (no deps)
- `vllm-core` → `vllm-traits`
- `vllm-model` → `vllm-traits`, candle
- `vllm-server` → `vllm-core`, `vllm-model`, tokio
