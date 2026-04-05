# Architecture Refactoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor vllm-lite architecture into maintainable, focused modules across three phases

**Architecture:** Split monolithic scheduler, separate KV cache logical/physical layers, extract kernel layer

**Tech Stack:** Pure Rust refactoring, no new dependencies

---

## Overview

This plan executes three-phase architecture refactoring:

1. **Phase 1**: Split 1340-line scheduler.rs into focused submodules
2. **Phase 2**: Separate KV cache logical (core) and physical (model) layers
3. **Phase 3**: Extract kernel layer from model components

**Files to modify:**
- `crates/core/src/scheduler.rs` - split into directory
- `crates/core/src/engine/batch.rs` - move to scheduler
- `crates/core/src/kv_cache.rs` - split into directory
- `crates/model/src/kv_cache.rs` - rename to paged_tensor.rs
- `crates/model/src/flash_attention.rs` - move to kernels
- `crates/core/src/cuda_graph.rs` - move to model/kernels
- Multiple lib.rs files - update exports

---

# Phase 1: Scheduler Split

## Task 1: Create scheduler directory structure

**Files:**
- Create: `crates/core/src/scheduler/mod.rs`
- Create: `crates/core/src/scheduler/queue.rs`
- Create: `crates/core/src/scheduler/preemption.rs`
- Create: `crates/core/src/scheduler/eviction.rs`
- Modify: `crates/core/src/scheduler/batch.rs` (move from engine)

- [ ] **Step 1: Create scheduler/mod.rs skeleton**

Run: Create file with module declarations:

```rust
// crates/core/src/scheduler/mod.rs

pub mod queue;
pub mod preemption;
pub mod eviction;
// batch.rs will be moved from engine

mod scheduler;
pub use scheduler::Scheduler;
```

- [ ] **Step 2: Create scheduler/queue.rs**

Run: Extract queue management from scheduler.rs
- Move waiting/running/finished VecDeque/Vec to new struct
- Keep Sequence management logic
- About 150 lines

```rust
// skeleton
pub struct SequenceQueue {
    waiting: VecDeque<Sequence>,
    running: Vec<Sequence>,
    finished: Vec<Sequence>,
    next_seq_id: u64,
}

impl SequenceQueue {
    pub fn push_waiting(&mut self, seq: Sequence) { ... }
    pub fn pop_running(&mut self) -> Option<Sequence> { ... }
    pub fn move_to_finished(&mut self, seq_id: u64) { ... }
    // ... other queue operations
}
```

- [ ] **Step 3: Create scheduler/preemption.rs**

Run: Extract preemption logic from scheduler.rs
- Move preemption decision logic
- About 150 lines

```rust
pub struct PreemptionManager { ... }

impl PreemptionManager {
    pub fn should_preempt(&self, running: &[Sequence], waiting: &Sequence) -> bool { ... }
    pub fn select_victim(&self, running: &[Sequence]) -> Option<Sequence> { ... }
    pub fn preempt(&mut self, seq: Sequence) { ... }
}
```

- [ ] **Step 4: Create scheduler/eviction.rs**

Run: Extract eviction logic from scheduler.rs
- Move block eviction policy logic
- About 100 lines

```rust
pub struct EvictionPolicy { ... }

impl EvictionPolicy {
    pub fn select_victim(&self, block_refs: &HashMap<BlockId, usize>) -> Option<BlockId> { ... }
    pub fn evict(&mut self, blocks: &[BlockId], allocator: &mut BlockAllocator) { ... }
}
```

- [ ] **Step 5: Move engine/batch.rs to scheduler/batch.rs**

Run: 
```bash
mv crates/core/src/engine/batch.rs crates/core/src/scheduler/batch.rs
```

Update imports in batch.rs (likely needs `super` instead of `crate::`)

- [ ] **Step 6: Rewrite scheduler/mod.rs with new structure**

Run: Replace monolithic scheduler.rs with modular version
- Use queue, preemption, eviction modules
- Keep public API identical
- Reduce to ~250 lines

```rust
pub struct Scheduler {
    queue: SequenceQueue,
    preemption: PreemptionManager,
    eviction: EvictionPolicy,
    kv_allocator: BlockAllocator,
    prefix_cache: PrefixCache,
    config: SchedulerConfig,
}

impl Scheduler {
    pub fn new() -> Self { ... }
    pub fn add_request(&mut self, req: Request) -> SeqId { ... }
    pub fn schedule(&mut self) -> Batch { ... }
    // ... other methods delegating to submodules
}
```

- [ ] **Step 7: Update core/src/lib.rs exports**

Run: Modify lib.rs to point to new module structure

```rust
pub mod scheduler;
// Remove old scheduler.rs reference
```

- [ ] **Step 8: Run tests**

Run: `cargo test -p vllm-core`
Expected: All tests pass

---

## Task 2: Verify Phase 1

- [ ] **Step 1: Run clippy**

Run: `cargo clippy -p vllm-core -- -D warnings`
Expected: No warnings

- [ ] **Step 2: Verify no API breakage**

Run: Check any external usage of Scheduler
Expected: No changes needed to other crates

- [ ] **Step 3: Count scheduler/mod.rs lines**

Run: `wc -l crates/core/src/scheduler/mod.rs`
Expected: < 300 lines

---

# Phase 2: KV Cache Layer Separation

## Task 3: Split core/kv_cache.rs

**Files:**
- Create: `crates/core/src/kv_cache/block_allocator.rs`
- Create: `crates/core/src/kv_cache/prefix_cache.rs`
- Modify: `crates/core/src/kv_cache.rs` (becomes mod.rs)
- Create: `crates/model/src/paged_tensor/` (new directory)
- Move: `crates/model/src/kv_cache.rs` → `crates/model/src/paged_tensor/tensor_store.rs`
- Move: Extract quantization → `crates/model/src/paged_tensor/quantization.rs`

- [ ] **Step 1: Create kv_cache directory**

Run: Create directory structure:
```bash
mkdir -p crates/core/src/kv_cache
```

- [ ] **Step 2: Create kv_cache/block_allocator.rs**

Run: Move BlockAllocator from core/kv_cache.rs
- About 150 lines
- Keep all allocation logic

- [ ] **Step 3: Create kv_cache/prefix_cache.rs**

Run: Move PrefixCache from core/kv_cache.rs
- About 200 lines
- Keep hash → block mapping

- [ ] **Step 4: Rewrite kv_cache/mod.rs**

Run: Create re-export file

```rust
pub mod block_allocator;
pub mod prefix_cache;

pub use block_allocator::BlockAllocator;
pub use prefix_cache::{PrefixCache, CachedEntry, hash_tokens};
```

- [ ] **Step 5: Create paged_tensor directory**

Run:
```bash
mkdir -p crates/model/src/paged_tensor
mv crates/model/src/kv_cache.rs crates/model/src/paged_tensor/tensor_store.rs
```

- [ ] **Step 6: Extract quantization to separate file**

Run: Move quantization struct/functions from tensor_store.rs to paged_tensor/quantization.rs

- [ ] **Step 7: Create paged_tensor/mod.rs**

Run:
```rust
pub mod tensor_store;
pub mod quantization;

pub use tensor_store::PagedKvCache;
pub use quantization::{quantize, dequantize, QuantizedTensor};
```

- [ ] **Step 8: Update model/src/lib.rs**

Run: Update module declarations

```rust
pub mod paged_tensor;
// Remove: pub mod kv_cache
```

- [ ] **Step 9: Add deprecation alias (optional, for compatibility)**

Run: In model/src/, create kv_cache.rs alias:

```rust
#[deprecated(since = "0.2.0", note = "Use paged_tensor")]
pub mod kv_cache {
    pub use crate::paged_tensor::*;
}
```

- [ ] **Step 10: Run tests**

Run: `cargo test --workspace`
Expected: All tests pass

---

## Task 4: Verify Phase 2

- [ ] **Step 1: Run clippy**

Run: `cargo clippy --workspace -- -D warnings`
Expected: No warnings

- [ ] **Step 2: Verify imports work**

Run: Check model crate compiles
Expected: No import errors

---

# Phase 3: Kernel Layer Extraction

## Task 5: Create model/kernels directory

**Files:**
- Create: `crates/model/src/kernels/mod.rs`
- Create: `crates/model/src/kernels/flash_attention.rs`
- Create: `crates/model/src/kernels/fused_mlp.rs`
- Create: `crates/model/src/kernels/cuda_graph.rs`
- Modify: `crates/model/src/components/` (update to use kernels)
- Modify: `crates/core/src/lib.rs` (remove cuda_graph)

- [ ] **Step 1: Create kernels directory**

Run:
```bash
mkdir -p crates/model/src/kernels
```

- [ ] **Step 2: Create kernels/mod.rs**

Run: Create with exports

```rust
pub mod flash_attention;
pub mod fused_mlp;
pub mod cuda_graph;

pub use flash_attention::{FlashAttention, FlashAttentionConfig, ...};
pub use fused_mlp::FusedMlp;
pub use cuda_graph::CudaGraph;
```

- [ ] **Step 3: Move flash_attention.rs to kernels**

Run:
```bash
mv crates/model/src/flash_attention.rs crates/model/src/kernels/flash_attention.rs
```

Update internal imports (relative paths)

- [ ] **Step 4: Move fused_kernel.rs to kernels**

Run:
```bash
mv crates/model/src/components/fused_kernel.rs crates/model/src/kernels/fused_mlp.rs
```

Rename module and update internal imports

- [ ] **Step 5: Move cuda_graph.rs from core to model/kernels**

Run:
```bash
mv crates/core/src/cuda_graph.rs crates/model/src/kernels/cuda_graph.rs
```

- [ ] **Step 6: Update core/src/lib.rs**

Run: Remove cuda_graph module

```rust
// Remove: pub mod cuda_graph;
// Remove: pub use cuda_graph::CudaGraph;
```

- [ ] **Step 7: Update components to use kernels**

Run: Modify components/attention.rs

```rust
use crate::kernels::flash_attention::FlashAttention;

pub struct Attention {
    kernel: FlashAttention,
    // ...
}
```

Run: Modify components/mlp.rs

```rust
use crate::kernels::fused_mlp::FusedMlp;

pub struct Mlp {
    kernel: FusedMlp,
    // ...
}
```

- [ ] **Step 8: Update model/src/lib.rs**

Run: Add kernels module

```rust
pub mod kernels;
// ... existing modules
```

- [ ] **Step 9: Run tests**

Run: `cargo test --workspace`
Expected: All tests pass

---

## Task 6: Final Verification

- [ ] **Step 1: Run full CI**

Run: `just ci`
Expected: All checks pass

- [ ] **Step 2: Verify no API breakage**

Run: Check server compiles and works
Expected: No changes needed

- [ ] **Step 3: Update AGENTS.md (optional)**

Run: If needed, update architecture section in AGENTS.md

- [ ] **Step 4: Commit changes**

Run: Create commit with all refactoring

---

## Summary

| Phase | Tasks | Files Changed | Lines Restructured |
|-------|-------|---------------|-------------------|
| 1 | 2 | 8 | ~1000 |
| 2 | 4 | 6 | ~500 |
| 3 | 6 | 10 | ~800 |
| **Total** | **12** | **~24** | **~2300** |

**Risk Level**: Medium - all changes are refactoring, no functional changes
**Estimated Time**: 3-5 sessions
