# Architecture Refactoring Specification

**Date**: 2026-04-05
**Goal**: Refactor vllm-lite architecture into maintainable, focused modules

---

## 1. Overview

### 1.1 Motivation

Current architecture issues:
- `scheduler/scheduler.rs` (1379 lines) - split into modules, still large but maintainable
- KV cache responsibilities split across `core/kv_cache.rs` and `model/kv_cache.rs`
- Kernel implementations mixed with model code
- Unclear boundaries between logical (allocation) and physical (tensor) KV cache layers

### 1.2 Scope

Three-phase refactoring:
- **Phase 1**: Split scheduler into focused submodules
- **Phase 2**: Separate KV cache logical/physical layers
- **Phase 3**: Extract kernel layer from model

### 1.3 Principles

- Each module: single responsibility, clear interface, independently testable
- Preserve existing public APIs with deprecation path
- No functional changes, only structural refactoring

---

## 2. Phase 1: Scheduler Split

### 2.1 Current State

```
core/src/scheduler/
├── scheduler.rs       (1379 lines - main scheduler with all logic)
├── queue.rs           (142 lines - RequestQueue, used by scheduler)
├── preemption.rs      (169 lines - PreemptionManager, fully integrated)
├── eviction.rs        (237 lines - EvictionPolicy, fully integrated)
└── batch.rs           (51 lines - batch building)
```

### 2.2 Target Structure

```
core/src/scheduler/
├── mod.rs              # Scheduler struct, entry points (~200 lines)
├── queue.rs            # waiting/running/finished management
├── batch.rs            # batch building strategies (from engine/batch.rs)
├── preemption.rs       # preemption and rejection logic
└── eviction.rs         # block eviction policies
```

### 2.3 Module Responsibilities

| Module | Responsibility | Public API |
|--------|---------------|------------|
| `scheduler.rs` | Main scheduler orchestration | `Scheduler::new()`, `add_request()`, `build_batch()`, `update()` |
| `queue.rs` | Sequence queue management | `RequestQueue::waiting_push_back()`, `running_push()`, etc. |
| `batch.rs` | Batch construction algorithms | `BatchBuilder::build()`, `BatchConfig` |
| `preemption.rs` | Preemption decisions | `PreemptionManager::should_preempt()`, `select_victim()`, fully integrated |
| `eviction.rs` | KV block eviction | `EvictionPolicy::select_victims()`, `record_blocks()`, fully integrated |

### 2.4 Breaking Changes

- Move `engine/batch.rs` to `scheduler/batch.rs`
- Move some scheduler methods to submodules
- Update all internal imports

### 2.5 Migration Path

```rust
// Before
use crate::scheduler::Scheduler;

// After (unchanged)
use crate::scheduler::Scheduler;
```

---

## 3. Phase 2: KV Cache Layer Separation

### 3.1 Current State

Two kv_cache files with unclear boundaries:
- `core/kv_cache/block_allocator.rs` (87 lines): BlockAllocator
- `core/kv_cache/prefix_cache.rs` (251 lines): PrefixCache
- `model/paged_tensor/tensor_store.rs` (680 lines): PagedKvCache (tensor storage)
- `model/paged_tensor/quantization.rs` (76 lines): KV cache quantization

### 3.2 Target Structure (Implemented)

```
core/src/kv_cache/
├── mod.rs              # Re-exports
├── block_allocator.rs  # BlockAllocator (87 lines)
└── prefix_cache.rs     # PrefixCache (251 lines)

model/src/paged_tensor/      # Directory (replaces kv_cache.rs)
├── mod.rs                   # Re-exports
├── tensor_store.rs          # GPU KV tensor management (680 lines)
└── quantization.rs          # INT8/FP8 quantization (76 lines)
```

### 3.3 Responsibilities (Implemented)

| Module | Layer | Responsibility | Status |
|--------|-------|---------------|--------|
| `core/kv_cache/block_allocator.rs` | Logical | Block allocation/deallocation | ✓ |
| `core/kv_cache/prefix_cache.rs` | Logical | Hash → block mapping, LRU | ✓ |
| `model/paged_tensor/tensor_store.rs` | Physical | GPU memory KV tensors | ✓ |
| `model/paged_tensor/quantization.rs` | Physical | KV cache quantization | ✓ |

### 3.4 Naming Rationale

- `model/kv_cache.rs` → `paged_tensor.rs`: Clearly indicates tensor storage
- Avoids naming conflict with logical KV cache
- "PagedTensor" aligns with PagedAttention paper terminology

### 3.5 Deprecation Path

```rust
// Deprecated (Phase 2, remove in Phase 3)
#[deprecated(since = "0.2.0", note = "Use paged_tensor::PagedKvTensor")]
pub mod kv_cache {
    pub use crate::paged_tensor::*;
}
```

---

## 4. Phase 3: Kernel Layer Extraction

### 4.1 Current State (After Refactoring)

- `model/kernels/flash_attention.rs` (488 lines): kernel config + selection + execution
- `model/kernels/fused_mlp.rs` (100 lines): fused MLP
- `model/kernels/cuda_graph.rs` (313 lines): moved from core

### 4.2 Target Structure (Implemented)

```
model/src/kernels/
├── mod.rs              # Re-exports
├── flash_attention.rs  # Kernel implementations (488 lines)
├── fused_mlp.rs        # Fused MLP kernel (100 lines)
└── cuda_graph.rs       # CUDA graph capture/replay (313 lines)

model/src/components/
├── mod.rs              # Uses kernels (line 6: pub use super::kernels::...)
├── attention.rs        # Attention wrapper (uses kernels)
├── mlp.rs              # MLP wrapper (uses kernels)
├── norm.rs             # Normalization
└── positional.rs       # Position encoding
```

### 4.3 Responsibilities (Implemented)

| Module | Responsibility | Status |
|--------|---------------|--------|
| `kernels/flash_attention.rs` | Attention kernel dispatch (FlashDecoding, etc.) | ✓ |
| `kernels/fused_mlp` | Fused MLP kernel (GeGLU, etc.) | ✓ |
| `kernels/cuda_graph` | CUDA graph capture/replay | ✓ |
| `components/attention.rs` | Attention interface, calls kernels | ✓ |

### 4.4 Design Rationale

- **Separation of concerns**: Component uses kernel, doesn't implement it
- **Future-proofing**: Easy to add new backends (TRT-LLM, xFormers)
- **Testability**: Kernels can be mocked independently

---

## 5. Files to Modify

### Phase 1

| File | Action |
|------|--------|
| `crates/core/src/scheduler.rs` | Rewrite as `scheduler/mod.rs` |
| `crates/core/src/engine/batch.rs` | Move to `scheduler/batch.rs` |
| `crates/core/src/scheduler/queue.rs` | New file |
| `crates/core/src/scheduler/preemption.rs` | New file |
| `crates/core/src/scheduler/eviction.rs` | New file |
| `crates/core/src/lib.rs` | Update exports |

### Phase 2

| File | Action |
|------|--------|
| `crates/core/src/kv_cache.rs` | Split into `kv_cache/block_allocator.rs`, `kv_cache/prefix_cache.rs` |
| `crates/model/src/kv_cache.rs` | Split into `paged_tensor/tensor_store.rs`, `paged_tensor/quantization.rs` |
| `crates/core/src/lib.rs` | Update exports |
| `crates/model/src/lib.rs` | Update exports |

### Phase 3

| File | Action |
|------|--------|
| `crates/model/src/flash_attention.rs` | Move to `kernels/flash_attention.rs` |
| `crates/model/src/components/fused_kernel.rs` | Move to `kernels/fused_mlp.rs` |
| `crates/core/src/cuda_graph.rs` | Move to `model/kernels/cuda_graph.rs` |
| `crates/model/src/components/attention.rs` | Update to use kernels |
| `crates/model/src/lib.rs` | Add `kernels` module |
| `crates/core/src/lib.rs` | Remove cuda_graph, update exports |

---

## 6. Dependency Changes

### After All Phases

```
vllm-traits
    ↑
    ├── vllm-core (logical: scheduler, kv_cache)
    │       └── vllm-dist
    └── vllm-model (physical: kernels, paged_tensor)
            └── vllm-dist
```

No new dependencies introduced. Core and model remain peer crates.

---

## 7. Testing Strategy

### Phase 1 Tests
- `scheduler::queue`: Unit test queue operations
- `scheduler::batch`: Test batch building logic
- `scheduler::preemption`: Test preemption decisions

### Phase 2 Tests
- `kv_cache::block_allocator`: Test allocation/deallocation
- `kv_cache::prefix_cache`: Test cache hit/miss, eviction
- `paged_tensor`: Test tensor operations (can use fake device)

### Phase 3 Tests
- `kernels::flash_attention`: Test kernel selection
- `kernels::cuda_graph`: Test capture/replay
- Components still work with new kernel layer

---

## 8. Acceptance Criteria

### Phase 1
- [x] `Scheduler` struct works exactly as before
- [x] All existing scheduler tests pass
- [x] No public API changes
- [ ] `scheduler/scheduler.rs` line count (current: 1379, target: < 300) - modularized but still large

### Phase 2
- [x] Block allocation works identically
- [x] Prefix cache hit/miss behavior unchanged
- [x] Paged tensor operations unchanged
- [x] Clear module boundaries in docs

### Phase 3
- [x] Attention produces same outputs
- [x] CUDA graph capture/replay works
- [x] Can add new kernel implementations without touching components

### Overall
- [x] `cargo test --workspace` passes
- [x] `cargo clippy --workspace` passes with no new warnings
- [ ] Documentation updated for new module structure (in progress)
