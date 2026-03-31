# vLLM-lite Development Guide

## Project Structure

```
vllm-lite/
├── Cargo.toml              # Workspace root
├── ROADMAP.md              # Development roadmap
├── CHANGELOG.md            # Version history
├── AGENTS.md               # This guide
├── crates/
│   ├── core/               # Core engine
│   │   ├── src/
│   │   │   ├── engine.rs   # Engine loop, ModelBackend trait
│   │   │   ├── scheduler.rs # Batch scheduling
│   │   │   ├── kv_cache.rs # BlockAllocator, PrefixCache
│   │   │   ├── metrics.rs  # MetricsCollector
│   │   │   ├── types.rs    # Core types
│   │   │   └── sampling.rs # Sampling strategies
│   ├── model/              # ML models
│   │   └── src/
│   │       ├── qwen3/      # Qwen3 implementation
│   │       ├── kv_cache.rs # PagedKvCache (GPU)
│   │       ├── quantize.rs # INT8 quantization
│   │       ├── loader.rs   # SafeTensors loading
│   │       └── fake.rs     # FakeModel for testing
│   └── server/             # HTTP server
│       ├── src/main.rs     # Entry point
│       └── src/api.rs      # HTTP handlers
├── docs/
│   ├── README.md           # Documentation index
│   └── superpowers/
│       ├── specs/          # Design documents
│       └── plans/          # Implementation plans
└── tests/                  # Integration tests
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
5. **Update**: Update CHANGELOG.md for significant changes

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
- `docs`: Documentation (specs, plans, README, CHANGELOG, etc.)
- `chore`: Maintenance

**Examples:**
```bash
git commit -m "feat(model): add forward_prefill to GqaAttention"
git commit -m "fix(core): resolve prefix cache eviction bug"
git commit -m "docs(spec): add Phase 2 prefix hit design"
git commit -m "docs(plan): add paged attention implementation plan"
git commit -m "chore: update dependencies"
```

**Rules:**
- Use lowercase for subject
- No period at end of subject
- Scope is optional but recommended (e.g., model, core, server)
- Body explaining what and why (not how)

## Crate Responsibilities

### core
- `engine.rs`: Engine loop, ModelBackend trait
- `scheduler.rs`: Batch scheduling, decode-priority
- `kv_cache.rs`: BlockAllocator, PrefixCache
- `metrics.rs`: MetricsCollector, MetricsSnapshot
- `types.rs`: Core types (Request, Sequence, Batch, etc.)
- `sampling.rs`: Sampling strategies

### model
- `qwen3/`: Qwen3 model implementation
- `kv_cache.rs`: PagedKvCache (GPU tensors)
- `quantize.rs`: INT8 quantization utilities
- `loader.rs`: SafeTensors weight loading
- `fake.rs`: FakeModel for testing

### server
- `main.rs`: Server entry point
- `api.rs`: HTTP handlers, OpenAI-compatible endpoints

## Key Design Decisions

- Single GPU worker thread (avoid GPU contention)
- Workspace structure: 3 crates (core/model/server)
- Block size: 16 tokens per KV block
- Max batched tokens: 4096 default
- Max concurrent sequences: 256 default

## Common Patterns

### Adding a new feature
1. Write spec to `docs/superpowers/specs/YYYY-MM-DD-feature-name.md`
2. Write plan to `docs/superpowers/plans/YYYY-MM-DD-feature-name.md`
3. Implement in appropriate module
4. Add tests
5. Update CHANGELOG.md if significant

### Running tests
```bash
# All tests
cargo test --workspace

# With output
cargo test --workspace -- --nocapture

# Specific test
cargo test -p vllm-core -- scheduler
```

## Documentation

| File | Purpose |
|------|---------|
| ROADMAP.md | Long-term development plan |
| CHANGELOG.md | Version history |
| AGENTS.md | Developer guide (this file) |
| docs/README.md | Documentation index |
| docs/specs/ | Feature specifications |
| docs/plans/ | Implementation plans |

## Notes

- Uses Rust edition 2021
- CUDA support via Candle
- Follow TDD pattern where possible