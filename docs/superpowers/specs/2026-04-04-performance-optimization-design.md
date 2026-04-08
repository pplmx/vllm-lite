# Performance Optimization Design

**Date**: 2026-04-04
**Status**: Approved
**Goal:** Improve memory usage and kernel performance

---

## 1. Prefix Caching Enhancement

**Current State:** KV cache already exists but no prefix caching.

**Improvement:** Add block hash tracking to enable prefix reuse:

```rust
// In PagedKvCache
pub struct PagedKvCache {
    // ... existing fields ...
    pub block_hashes: Vec<HashMap<u64, usize>>,  // NEW: hash -> block_id
}

impl PagedKvCache {
    pub fn compute_block_hash(&self, block: &Tensor) -> u64 {
        // Compute hash of block content for prefix matching
    }

    pub fn find_matching_blocks(&self, prompt_hash: u64) -> Vec<usize> {
        // Find cached blocks matching the prompt
    }
}
```

---

## 2. Memory Pool Optimization

**Current:** Each block allocates separately.

**Improvement:** Use memory pool for better allocation:

```rust
pub struct KvCachePool {
    blocks: Vec<CacheBlock>,
    free_list: Vec<usize>,
}

impl KvCachePool {
    pub fn allocate(&mut self) -> Option<usize> {
        self.free_list.pop()
    }

    pub fn deallocate(&mut self, block_id: usize) {
        self.free_list.push(block_id);
    }
}
```

---

## 3. Flash Attention Improvements

**Current:** Basic Flash Attention implementation exists.

**Improvement:**

- Add more tile sizes for different seq lengths
- Optimize memory access patterns
- Add sliding window support in Flash Attention

```rust
pub struct FlashAttention {
    pub tile_sizes: Vec<usize>,  // Support [64, 128, 256]
    pub use_fused: bool,
}
```

---

## 4. Fused Kernel Support

Fuse multiple operations into single kernel:

```rust
// Fuse: layernorm + attention + residual
pub fn fused_attention_layer(
    x: &Tensor,
    layernorm_weight: &Tensor,
    qkv_proj: &Linear,
    // ... parameters
) -> Result<Tensor>
```

---

## Implementation Order

1. **Phase 1: KV Cache Pool** - Memory pool for blocks
2. **Phase 2: Prefix Hash** - Block hash computation
3. **Phase 3: Flash Attention** - Improve tile sizes
4. **Phase 4: Fused Kernels** - Add fused layer kernel
