# Architecture Refactor Design

**Date**: 2026-04-02  
**Status**: Approved  
**Goal**: Decouple modules - move ModelBackend trait to server, remove model→core dependency, split Engine

## Current Problems

1. **Circular dependency risk**: `model` crate depends on `core` crate (ModelBackend trait defined in core)
2. **KV cache duplication**:
    - `core/src/kv_cache.rs`: BlockAllocator + PrefixCache (CPU-side block management)
    - `model/src/kv_cache.rs`: PagedKvCache (GPU tensor storage)
3. **ModelBackend in wrong place**: Trait defined in core, but actual implementations are in model
4. **Engine too large**: 600+ lines in single file with mixed concerns

## Target Architecture

```text
server/
├── src/
│   ├── main.rs              # Entry point
│   ├── api.rs               # HTTP handlers
│   ├── auth.rs              # Auth middleware
│   ├── config.rs            # Config
│   ├── logging.rs           # Logging
│   └── openai/              # OpenAI API
│       ├── mod.rs
│       ├── chat.rs
│       ├── completions.rs
│       ├── embeddings.rs
│       └── batch/
│
├── interfaces/              # NEW: Interface definitions
│   ├── mod.rs
│   ├── types.rs             # SeqId, TokenId, BatchOutput, Batch
│   └── model.rs             # ModelBackend trait
│
└── engine/                  # NEW: Engine components (组装)
    ├── mod.rs
    ├── engine.rs            # Main Engine struct
    ├── batch.rs             # Batch building logic (extracted)
    └── speculative.rs       # Speculative decoding (extracted)

model/
├── Cargo.toml               # REMOVE vllm-core dependency
└── src/
    ├── mod.rs
    ├── qwen3/
    ├── qwen3_5/
    ├── fake.rs              # Implement ModelBackend
    ├── kv_cache.rs          # PagedKvCache (GPU)
    ├── loader.rs
    ├── tokenizer.rs
    └── ...

core/
└── src/
    ├── mod.rs
    ├── scheduler.rs         #调度逻辑
    ├── kv_cache.rs          # BlockAllocator + PrefixCache
    ├── types.rs             # Keep scheduler-specific types
    ├── engine.rs            # Generic Engine<M: ModelBackend>
    ├── metrics.rs
    ├── sampling.rs
    └── ...
```

## Key Changes

### 1. Move ModelBackend to server/interfaces

Current (in `core/src/engine.rs`):

```rust
pub trait ModelBackend: Send + Sync {
    fn forward(&self, seq_ids: &[SeqId], input_tokens: &[Vec<TokenId>], ...) -> Result<BatchOutput>;
    fn forward_logits(...) -> Result<Vec<Vec<f32>>>;
}
```

After (in `server/interfaces/model.rs`):

```rust
use crate::interfaces::types::{BatchOutput, SeqId, TokenId};

pub trait ModelBackend: Send + Sync {
    fn forward(&self, seq_ids: &[SeqId], input_tokens: &[Vec<TokenId>], ...) -> Result<BatchOutput>;
    fn forward_logits(...) -> Result<Vec<Vec<f32>>>;
}
```

### 2. Remove model → core dependency

In `model/Cargo.toml`:

```toml
[dependencies]
# REMOVE: vllm-core = { path = "../core" }
candle-core = "0.10.1"
candle-nn = "0.10.1"
# ... other deps
```

In `model/src/qwen3/model.rs`, `model/src/fake.rs`, etc.:

- Remove `use vllm_core::engine::ModelBackend`
- Implement `server::interfaces::model::ModelBackend` instead

### 3. Split Engine

Split `core/src/engine.rs` (506 lines) into:

**engine.rs** (main struct + run loop, ~150 lines):

```rust
pub struct Engine<M: ModelBackend> {
    pub scheduler: Scheduler,
    pub target_model: Arc<M>,
    pub draft_model: Arc<M>,
    pub max_draft_tokens: usize,
    pub metrics: MetricsCollector,
    response_txs: HashMap<SeqId, mpsc::UnboundedSender<TokenId>>,
}

impl<M: ModelBackend> Engine<M> {
    pub fn new(...) -> Self
    pub fn add_request(...) -> SeqId
    pub fn step(...) -> Result<Vec<(SeqId, TokenId)>>
    pub fn run(...)  // main loop
}
```

**batch.rs** (batch building, ~200 lines):

```rust
pub struct BatchBuilder;
impl BatchBuilder {
    pub fn build(scheduler: &mut Scheduler) -> Batch
    pub fn update(scheduler: &mut Scheduler, ...)
}
```

**speculative.rs** (speculative decoding, ~150 lines):

```rust
impl<M: ModelBackend> Engine<M> {
    pub fn step_speculative(...) -> Result<Vec<(SeqId, TokenId)>>
    fn generate_draft_tokens(...) -> Result<Vec<Vec<TokenId>>>
    fn verify_draft_tokens(...) -> Result<Vec<(SeqId, TokenId)>>
}
```

### 4. Define shared types in server/interfaces

Create `server/interfaces/types.rs`:

```rust
pub type TokenId = u32;
pub type SeqId = u64;

pub struct BatchOutput {
    pub seq_ids: Vec<SeqId>,
    pub next_tokens: Vec<TokenId>,
}

pub struct Batch {
    pub seq_ids: Vec<SeqId>,
    pub input_tokens: Vec<Vec<TokenId>>,
    pub positions: Vec<Vec<usize>>,
}
```

## Implementation Order

1. Create `server/interfaces/` module with types + ModelBackend trait
2. Update core to depend on `server/interfaces` (or share types differently)
3. Remove model → core dependency, update imports
4. Split engine.rs into engine.rs + batch.rs + speculative.rs
5. Run tests and fix issues
6. Run clippy and fix warnings

## Testing Strategy

- Run `cargo test --workspace` after each major change
- Ensure `FakeModel` still works as test stub
- Verify scheduler tests pass without model dependency

## Notes

- This refactor maintains backward compatibility for HTTP API
- No changes to OpenAI API endpoints
- Scheduler continues to work as before
- Prefixes (BlockId, BLOCK_SIZE) stay in core for now
