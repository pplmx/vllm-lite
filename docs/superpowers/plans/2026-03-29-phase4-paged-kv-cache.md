# vLLM-lite Phase 4: Paged KV Cache

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development

**Goal:** Implement paged KV cache with block allocation, enabling memory-efficient inference and laying groundwork for vLLM's core innovation.

**Architecture:** BlockAllocator in core crate (logical allocation), PagedKvCache in model crate (Candle Tensor storage). Each sequence maps to a list of block IDs.

**Tech Stack:** Rust, Candle, this plan

---

## Key Design: Paged KV Cache

```
KV Cache Layout (per layer):

block_0   [seq1_k, seq1_v, seq2_k, seq2_v, ...]  # 16 tokens
block_1   [seq1_k, seq1_v, ...]                  # continues
block_2   [seq3_k, seq3_v, ...]                  # different seq
```

- BLOCK_SIZE = 16 tokens per block
- Each sequence has `Vec<BlockId>` mapping to physical blocks
- BlockAllocator manages free list of block IDs
- PagedKvCache stores actual Tensor data

---

### Task P4-1: BlockAllocator in core crate

**Files:**
- Create: `crates/core/src/kv_cache.rs`

- [ ] **Step 1: Add BlockAllocator**

`crates/core/src/kv_cache.rs`:
```rust
use crate::types::BlockId;

pub const BLOCK_SIZE: usize = 16;

pub struct BlockAllocator {
    num_blocks: usize,
    free_list: Vec<BlockId>,
}

impl BlockAllocator {
    pub fn new(num_blocks: usize) -> Self {
        let free_list: Vec<BlockId> = (0..num_blocks).collect();
        Self { num_blocks, free_list }
    }

    pub fn allocate(&mut self, num_blocks: usize) -> Option<Vec<BlockId>> {
        if self.free_list.len() >= num_blocks {
            Some((0..num_blocks).map(|_| self.free_list.pop().unwrap()).collect())
        } else {
            None
        }
    }

    pub fn free(&mut self, blocks: &[BlockId]) {
        for &block in blocks {
            self.free_list.push(block);
        }
    }

    pub fn available(&self) -> usize {
        self.free_list.len()
    }

    pub fn total(&self) -> usize {
        self.num_blocks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocate_and_free() {
        let mut alloc = BlockAllocator::new(10);
        
        let blocks = alloc.allocate(3).unwrap();
        assert_eq!(blocks.len(), 3);
        assert_eq!(alloc.available(), 7);

        alloc.free(&blocks);
        assert_eq!(alloc.available(), 10);
    }

    #[test]
    fn test_oom() {
        let mut alloc = BlockAllocator::new(2);
        alloc.allocate(2).unwrap();
        assert!(alloc.allocate(1).is_none());
    }
}
```

- [ ] **Step 2: Update lib.rs**

`crates/core/src/lib.rs`:
```rust
pub mod error;
pub mod types;
pub mod scheduler;
pub mod sampling;
pub mod engine;
pub mod kv_cache;
```

- [ ] **Step 3: Run tests**

```bash
cargo test -p vllm-core -- kv_cache
```

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "feat(core): add BlockAllocator for KV cache blocks"
```

---

### Task P4-2: PagedKvCache in model crate

**Files:**
- Modify: `crates/model/Cargo.toml` (add candle dependencies)
- Create: `crates/model/src/kv_cache.rs`

- [ ] **Step 1: Add candle dependencies**

`crates/model/Cargo.toml`:
```toml
[package]
name = "vllm-model"
version = "0.1.0"
edition = "2021"

[dependencies]
vllm-core = { path = "../core" }
rand = "0.10"
candle-core = "0.8"
candle-nn = "0.8"
thiserror = "2"
```

- [ ] **Step 2: Implement PagedKvCache**

`crates/model/src/kv_cache.rs`:
```rust
use candle_core::{Device, Tensor, Result, DType};
use vllm_core::types::BlockId;

pub const BLOCK_SIZE: usize = 16;

pub struct PagedKvCache {
    key_cache: Vec<Tensor>,
    value_cache: Vec<Tensor>,
    num_layers: usize,
    num_heads: usize,
    head_dim: usize,
    block_size: usize,
    device: Device,
}

impl PagedKvCache {
    pub fn new(
        num_layers: usize,
        num_heads: usize,
        head_dim: usize,
        num_blocks: usize,
        device: Device,
    ) -> Result<Self> {
        let key_cache = Vec::with_capacity(num_layers);
        let value_cache = Vec::with_capacity(num_layers);

        Ok(Self {
            key_cache,
            value_cache,
            num_layers,
            num_heads,
            head_dim,
            block_size: BLOCK_SIZE,
            device,
        })
    }

    pub fn num_blocks(&self) -> usize {
        self.key_cache.len()
    }
}
```

Wait - the above is incomplete. Let me rewrite with actual Tensor allocation:

```rust
use candle_core::{Device, Tensor, Result, DType};
use vllm_core::types::BlockId;

pub const BLOCK_SIZE: usize = 16;

pub struct PagedKvCache {
    key_cache: Vec<Tensor>,
    value_cache: Vec<Tensor>,
    num_layers: usize,
    num_heads: usize,
    head_dim: usize,
    block_size: usize,
    device: Device,
}

impl PagedKvCache {
    pub fn new(
        num_layers: usize,
        num_heads: usize,
        head_dim: usize,
        num_blocks: usize,
        device: Device,
    ) -> Result<Self> {
        let mut key_cache = Vec::with_capacity(num_layers);
        let mut value_cache = Vec::with_capacity(num_layers);

        for _ in 0..num_layers {
            let shape = (num_blocks, num_heads, BLOCK_SIZE, head_dim);
            let key = Tensor::zeros(shape, DType::F32, &device)?;
            let value = Tensor::zeros(shape, DType::F32, &device)?;
            key_cache.push(key);
            value_cache.push(value);
        }

        Ok(Self {
            key_cache,
            value_cache,
            num_layers,
            num_heads,
            head_dim,
            block_size: BLOCK_SIZE,
            device,
        })
    }

    pub fn num_blocks(&self) -> usize {
        self.key_cache.first()
            .map(|t| t.shape().dims()[0])
            .unwrap_or(0)
    }
}
```

- [ ] **Step 3: Update model lib.rs**

`crates/model/src/lib.rs`:
```rust
pub mod fake;
pub mod kv_cache;
```

- [ ] **Step 4: Verify compiles**

```bash
cargo check --workspace
```

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(model): add PagedKvCache with Candle Tensor storage"
```

---

### Task P4-3: Integrate with Scheduler

**Files:**
- Modify: `crates/core/src/types.rs` (Sequence has kv_blocks field)
- Modify: `crates/core/src/scheduler.rs` (allocate/free blocks)

- [ ] **Step 1: Update Sequence to include kv_blocks**

In `crates/core/src/types.rs`, add to Sequence struct:
```rust
pub struct Sequence {
    pub id: SeqId,
    pub tokens: Vec<TokenId>,
    pub num_computed_tokens: usize,
    pub kv_blocks: Vec<BlockId>,  // ADD THIS
    pub status: Status,
    pub max_tokens: usize,
    pub sampling_params: SamplingParams,
}
```

- [ ] **Step 2: Update Scheduler to manage blocks**

Add field to Scheduler:
```rust
pub struct Scheduler {
    // ... existing fields
    pub kv_allocator: BlockAllocator,  // ADD THIS
}
```

Update `add_request` to pre-allocate blocks:
```rust
pub fn add_request(&mut self, req: Request) -> SeqId {
    // ... existing code ...
    
    // Pre-allocate blocks for prompt tokens
    let num_blocks_needed = (seq.tokens.len() + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let blocks = self.kv_allocator.allocate(num_blocks_needed).unwrap_or_default();
    seq.kv_blocks = blocks;
    
    // ... rest
}
```

Update `update` to allocate new blocks for decoded tokens:
```rust
pub fn update(&mut self, seq_ids: &[SeqId], next_tokens: &[TokenId], input_counts: &[usize]) {
    for ((seq_id, token), &input_count) in seq_ids.iter().zip(next_tokens).zip(input_counts) {
        if let Some(seq) = self.running.iter_mut().find(|s| s.id == *seq_id) {
            // ... existing logic ...
            
            // Allocate new block if needed for the new token
            let new_total = seq.tokens.len();
            let blocks_needed = (new_total + BLOCK_SIZE - 1) / BLOCK_SIZE;
            while seq.kv_blocks.len() < blocks_needed {
                if let Some(new_block) = self.kv_allocator.allocate(1) {
                    seq.kv_blocks.extend(new_block);
                } else {
                    // OOM - could implement eviction here
                    break;
                }
            }
        }
    }
    
    // ... rest
}
```

Also update Scheduler constructor:
```rust
pub fn new() -> Self {
    Self::with_config(SchedulerConfig::default(), 1024) // 1024 blocks default
}

pub fn with_config(config: SchedulerConfig, num_kv_blocks: usize) -> Self {
    Self {
        // ... existing fields
        kv_allocator: BlockAllocator::new(num_kv_blocks),
    }
}
```

- [ ] **Step 3: Run tests**

```bash
cargo test -p vllm-core
```

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "feat(core): integrate BlockAllocator with Scheduler"
```

---

### Task P4-4: End-to-end verification

**Files:**
- Run full test suite
- Manual test

- [ ] **Step 1: Run tests**

```bash
cargo test --workspace
```

- [ ] **Step 2: Manual test**

```bash
cargo run -p vllm-server

curl -N -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "hello world", "max_tokens": 5, "stream": true}'
```

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "chore: Phase 4 complete - paged KV cache integrated"
```