# vLLM-lite Development Guide

## Project Structure

```
vllm-lite/
├── Cargo.toml              # Workspace root
├── crates/
│   ├── core/            # Core engine (scheduler, engine, kv_cache, types)
│   ├── model/          # ML models (Qwen3, attention, mlp)
│   └── server/         # HTTP server
├── docs/
│   ├── superpowers/
│   │   ├── specs/      # Design documents
│   │   └── plans/      # Implementation plans
└── tests/              # Integration tests
```

## Key Commands

```bash
# Build all crates
cargo build --workspace

# Run tests
cargo test --workspace

# Run clippy
cargo clippy --workspace -- -D warnings

# Run specific crate
cargo test -p vllm-core
cargo test -p vllm-model  
cargo test -p vllm-server

# Run server
cargo run -p vllm-server
```

## Development Workflow

1. **New Feature**: Use brainstorming skill to design, then writing-plans skill for implementation plan
2. **Implementation**: Use subagent-driven-development skill for task execution
3. **Code Review**: Two-stage review (spec compliance → code quality)
4. **Commit**: Follow conventional commits format (see below)

## Commit Message Format

Use [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>(<scope>): <subject>

<body>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `refactor`: Code refactoring
- `test`: Adding tests
- `docs`: Documentation (specs, plans, README, etc.)
- `chore`: Maintenance

**Examples:**
```bash
git commit -m "feat(model): add forward_prefill to GqaAttention"
git commit -m "fix(core): resolve prefix cache eviction bug"
git commit -m "docs(prefix-caching): add Phase 2 prefix hit design"
git commit -m "docs(paged-attention): add implementation plan"
```

**Rules:**
- Use lowercase for subject
- No period at end of subject
- Scope is optional but recommended (e.g., model, core, server)
- Body explaining what and why (not how)

## Crate Responsibilities

### core
- `types.rs`: Core types (Request, Sequence, Batch, etc.)
- `scheduler.rs`: Batch scheduling, decode-priority
- `kv_cache.rs`: BlockAllocator, PrefixCache
- `engine.rs`: Engine loop, ModelBackend trait
- `sampling.rs`: Greedy, top-k sampling

### model
- `qwen3/`: Qwen3 model implementation
- `kv_cache.rs`: PagedKvCache (Candle tensors)
- `loader.rs`: SafeTensors weight loading
- `fake.rs`: FakeModel for testing

### server
- `main.rs`: Server entry point
- `api.rs`: HTTP handlers

## Key Design Decisions

- Single GPU worker thread (avoid GPU contention)
- Workspace structure: 3 crates (core/model/server)
- Block size: 16 tokens per KV block
- Max batched tokens: 4096 default
- Max concurrent sequences: 256 default

## Common Patterns

### Adding a new feature
1. Add types to `crates/core/src/types.rs`
2. Implement in appropriate module
3. Add tests
4. Integrate with Engine if needed

### Running tests
```bash
# All tests
cargo test --workspace

# With output
cargo test --workspace -- --nocapture

# Specific test
cargo test -p vllm-core -- scheduler
```

## Notes

- Uses Rust edition 2021
- Requires nightly features (if any)
- CUDA support via Candle