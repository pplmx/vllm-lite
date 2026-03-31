# vLLM-lite Development Guide

This guide helps AI agents and developers work effectively with the vLLM-lite codebase.

## Quick Reference

| Task | Skill | Command |
|------|-------|---------|
| Design new feature | `brainstorming` | Load skill → discuss → get approval |
| Write implementation plan | `writing-plans` | Load skill → spec approved → write plan |
| Execute implementation | `subagent-driven-development` | Load skill → task list → execute tasks |
| Debug issues | `systematic-debugging` | Load skill → reproduce → diagnose → fix |

## Project Structure

```
vllm-lite/
├── Cargo.toml              # Workspace root
├── ROADMAP.md              # Development roadmap
├── CHANGELOG.md            # Version history
├── AGENTS.md               # This guide
├── crates/
│   ├── core/               # Core engine
│   │   └── src/
│   │       ├── engine.rs   # Engine loop, ModelBackend trait
│   │       ├── scheduler.rs # Batch scheduling
│   │       ├── kv_cache.rs # BlockAllocator, PrefixCache
│   │       ├── metrics.rs  # MetricsCollector
│   │       ├── types.rs    # Core types
│   │       └── sampling.rs # Sampling strategies
│   ├── model/              # ML models
│   │   └── src/
│   │       ├── qwen3/      # Qwen3 implementation
│   │       │   ├── attention.rs # GqaAttention, tiled attention
│   │       │   ├── block.rs    # TransformerBlock
│   │       │   └── model.rs    # Qwen3Model
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
│       ├── specs/          # Design specs (YYYY-MM-DD-*.md)
│       └── plans/          # Implementation plans (YYYY-MM-DD-*.md)
└── tests/                  # Integration tests
```

## Key Commands

```bash
# Build
cargo build --workspace

# Test
cargo test --workspace

# Clippy (required before commit)
cargo clippy --workspace -- -D warnings

# Specific crate
cargo test -p vllm-core
cargo test -p vllm-model  
cargo test -p vllm-server

# Run server
cargo run -p vllm-server

# Check only (faster than build)
cargo check -p vllm-model
```

## Development Workflow

### 1. New Feature
```
1. Load brainstorming skill
2. Explore project context (files, docs, recent commits)
3. Ask clarifying questions (one at a time)
4. Propose approaches with trade-offs
5. Get user approval on design
6. Write spec to docs/superpowers/specs/YYYY-MM-DD-feature-name.md
7. Commit with: git commit -m "docs(spec): add <feature> design"
8. Load writing-plans skill
9. Write plan to docs/superpowers/plans/YYYY-MM-DD-feature-name.md
10. Commit with: git commit -m "docs(plan): add <feature> implementation plan"
11. Load subagent-driven-development skill
12. Execute tasks one by one with reviews
```

### 2. Bug Fix
```
1. Load systematic-debugging skill
2. Reproduce the issue
3. Identify root cause
4. Fix and test
5. Commit with: git commit -m "fix(<scope>): <description>"
```

## Commit Message Format

```
<type>(<scope>): <subject>

<body>
```

**Types:** `feat`, `fix`, `refactor`, `test`, `docs`, `chore`

**Examples:**
```bash
git commit -m "feat(model): add forward_prefill to GqaAttention"
git commit -m "fix(core): resolve prefix cache eviction bug"
git commit -m "docs(spec): add Phase 2 prefix hit design"
git commit -m "docs(plan): add paged attention implementation plan"
git commit -m "chore: update dependencies"
git commit -m "test(core): add prefix cache hit test"
```

## Crate API Reference

### core/engine.rs
```rust
pub struct Engine<M: ModelBackend> {
    pub scheduler: Scheduler,
    pub target_model: M,
    pub draft_model: M,
    pub max_draft_tokens: usize,
    pub metrics: MetricsCollector,
}

// Methods
impl<M: ModelBackend> Engine<M> {
    pub fn new(model: M, draft_model: M) -> Self
    pub fn with_config(model: M, draft_model: M, config: SchedulerConfig, max_draft_tokens: usize, num_kv_blocks: usize) -> Self
    pub fn add_request(&mut self, request: Request, response_tx: mpsc::UnboundedSender<TokenId>) -> SeqId
    pub fn step(&mut self) -> Result<BatchOutput>
    pub fn has_pending(&self) -> bool
}
```

### core/scheduler.rs
```rust
pub struct Scheduler {
    pub config: SchedulerConfig,
    pub waiting: VecDeque<Sequence>,
    pub running: VecDeque<Sequence>,
    pub finished: VecDeque<Sequence>,
    pub prefix_cache: PrefixCache,
    pub kv_allocator: BlockAllocator,
}

impl Scheduler {
    pub fn add_request(&mut self, request: Request) -> SeqId
    pub fn build_batch(&mut self) -> Option<Batch>
    pub fn get_sequence(&self, id: SeqId) -> Option<&Sequence>
}
```

### model/kv_cache.rs (PagedKvCache)
```rust
impl PagedKvCache {
    pub fn new(num_layers: usize, num_kv_heads: usize, head_dim: usize, num_blocks: usize, device: Device) -> Result<Self>
    pub fn write_kv(&mut self, layer_idx: usize, block_id: usize, token_offset: usize, k: &Tensor, v: &Tensor) -> Result<()>
    pub fn read_kv(&self, layer_idx: usize, block_ids: &[usize], seq_len: usize) -> Result<(Tensor, Tensor)>
}
```

### model/qwen3/attention.rs
```rust
pub struct GqaAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    config: AttentionConfig,
}

impl GqaAttention {
    pub fn new(hidden_size: usize, num_heads: usize, num_kv_heads: usize, head_dim: usize, vb: Option<VarBuilder>, config: AttentionConfig) -> Result<Self>
    pub fn forward_prefill(&self, x: &Tensor, kv_cache: &mut PagedKvCache, layer_idx: usize, block_ids: &[usize]) -> Result<Tensor>
    pub fn forward_decode(&self, x: &Tensor, kv_cache: &PagedKvCache, layer_idx: usize, block_ids: &[usize], num_computed_tokens: usize) -> Result<Tensor>
}
```

## Testing Guidelines

### Unit Tests
- Add to `#[cfg(test)]` module in the same file
- Use `FakeModel` or create minimal test fixtures
- Name: `test_<function>_<expected_behavior>`

### Integration Tests
- Add to `crates/*/tests/*.rs`
- Use `Engine::with_config()` with `StubModel` or `FakeModel`
- Test full workflows (add_request → step → verify)

### Running Tests
```bash
# All tests
cargo test --workspace

# With output
cargo test --workspace -- --nocapture

# Specific
cargo test -p vllm-core -- prefix_cache
cargo test -p vllm-model -- attention
```

## Common Patterns

### Adding a new type
1. Add to `crates/core/src/types.rs`
2. Export in `crates/core/src/lib.rs`

### Adding a new model
1. Create directory `crates/model/src/<model_name>/`
2. Implement `ModelBackend` trait
3. Add to model `lib.rs`

### Adding API endpoint
1. Add handler to `crates/server/src/api.rs`
2. Register route in `crates/server/src/main.rs`

## Key Design Decisions

- Single GPU worker thread (avoid GPU contention)
- 3 crates: core (engine), model (ML), server (HTTP)
- Block size: 16 tokens per KV block
- Max batched tokens: 4096 default
- Max concurrent sequences: 256 default
- Use `ModelBackend` trait for model abstraction

## Documentation Standards

| Document | Location | When to Update |
|----------|----------|----------------|
| Spec | `docs/superpowers/specs/` | Before implementation |
| Plan | `docs/superpowers/plans/` | After spec approved |
| CHANGELOG.md | Root | For any user-facing changes |
| README.md | Root | For major features |

## Notes

- Uses Rust edition 2021
- CUDA support via Candle
- Follow TDD pattern where possible
- Always run clippy before commit
- Use `cargo check` for fast validation