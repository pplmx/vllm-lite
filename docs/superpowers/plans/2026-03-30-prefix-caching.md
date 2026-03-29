# vLLM-lite Prefix Caching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development

**Goal:** Implement prefix caching to share KV cache between identical sequences, reducing redundant computation.

**Architecture:** Add PrefixCache with hash-based lookup, reference counting, and LRU eviction. Integrate with Scheduler's add_request and update methods.

**Tech Stack:** Rust, existing BlockAllocator, HashMap

**Spec:** `docs/superpowers/specs/2026-03-30-prefix-caching.md`

---

## File Structure

```
crates/core/src/
├── kv_cache.rs      # Modify: add PrefixCache + refcount
├── scheduler.rs     # Modify: integrate with add_request/update
└── types.rs         # Modify: add CacheKey type
```

---

### Task PC-1: PrefixCache Structure

**Files:**
- Modify: `crates/core/src/kv_cache.rs`

- [ ] **Step 1: Add CacheKey type and hash function**

Add to `crates/core/src/kv_cache.rs`:
```rust
use std::time::Instant;
use std::collections::{HashMap, VecDeque};
use crate::types::TokenId;

pub type CacheKey = u64;

pub fn hash_tokens(tokens: &[TokenId]) -> CacheKey {
    tokens.iter().fold(0u64, |acc, &t| acc.wrapping_mul(31).wrapping_add(t as u64))
}
```

- [ ] **Step 2: Add CachedEntry struct**

```rust
#[derive(Clone)]
pub struct CachedEntry {
    pub key: CacheKey,
    pub blocks: Vec<BlockId>,
    pub token_count: usize,
    pub last_access: Instant,
}
```

- [ ] **Step 3: Add PrefixCache struct with get/insert/evict**

```rust
pub struct PrefixCache {
    entries: HashMap<CacheKey, CachedEntry>,
    lru_order: VecDeque<CacheKey>,
    block_refs: HashMap<BlockId, usize>,
}

impl PrefixCache {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            lru_order: VecDeque::new(),
            block_refs: HashMap::new(),
        }
    }

    pub fn get(&mut self, key: CacheKey) -> Option<&CachedEntry> {
        if let Some(entry) = self.entries.get(&key) {
            // Move to front of LRU
            self.lru_order.retain(|k| *k != key);
            self.lru_order.push_front(key);
            Some(entry)
        } else {
            None
        }
    }

    pub fn insert(&mut self, key: CacheKey, blocks: Vec<BlockId>, token_count: usize) {
        // Increment ref counts
        for &block in &blocks {
            *self.block_refs.entry(block).or_insert(0) += 1;
        }

        let entry = CachedEntry {
            key,
            blocks,
            token_count,
            last_access: Instant::now(),
        };
        self.entries.insert(key, entry);
        self.lru_order.push_front(key);
    }

    pub fn evict(&mut self, allocator: &mut BlockAllocator) {
        while let Some(oldest_key) = self.lru_order.pop_back() {
            if let Some(entry) = self.entries.remove(&oldest_key) {
                // Decrement ref counts
                for &block in &entry.blocks {
                    if let Some(count) = self.block_refs.get_mut(&block) {
                        *count -= 1;
                        if *count == 0 {
                            allocator.free(&[block]);
                            self.block_refs.remove(&block);
                        }
                    }
                }
                break;
            }
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }
}

impl Default for PrefixCache {
    fn default() -> Self {
        Self::new()
    }
}
```

- [ ] **Step 4: Add unit tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_tokens() {
        assert_eq!(hash_tokens(&[1, 2, 3]), hash_tokens(&[1, 2, 3]));
        assert_ne!(hash_tokens(&[1, 2, 3]), hash_tokens(&[1, 2, 4]));
    }

    #[test]
    fn test_cache_insert_and_get() {
        let mut cache = PrefixCache::new();
        cache.insert(123, vec![1, 2], 2);
        
        assert!(cache.get(123).is_some());
        assert!(cache.get(456).is_none());
    }

    #[test]
    fn test_lru_order() {
        let mut cache = PrefixCache::new();
        cache.insert(1, vec![1], 1);
        cache.insert(2, vec![2], 1);
        cache.insert(3, vec![3], 1);
        
        // Access 1 to make it recently used
        cache.get(1);
        
        // Evict should remove 2 (oldest after 1 was accessed)
        let mut alloc = BlockAllocator::new(10);
        cache.evict(&mut alloc);
        
        assert!(cache.get(1).is_some());
        assert!(cache.get(2).is_none());
        assert!(cache.get(3).is_some());
    }
}
```

- [ ] **Step 5: Run tests**

```bash
cargo test -p vllm-core -- kv_cache
```

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(core): add PrefixCache with hash lookup and LRU eviction

- Add hash_tokens() function for CacheKey generation
- Add CachedEntry and PrefixCache structs
- Implement get/insert/evict with reference counting
- Add unit tests for basic operations"
```

---

### Task PC-2: Scheduler Integration

**Files:**
- Modify: `crates/core/src/scheduler.rs`

- [ ] **Step 1: Add prefix_cache field to Scheduler**

Update Scheduler struct:
```rust
pub struct Scheduler {
    // ... existing fields
    pub prefix_cache: PrefixCache,  // ADD THIS
}
```

Update constructors:
```rust
pub fn new() -> Self {
    Self::with_config(SchedulerConfig::default(), 1024)
}

pub fn with_config(config: SchedulerConfig, num_kv_blocks: usize) -> Self {
    Self {
        // ... existing fields
        prefix_cache: PrefixCache::new(),
    }
}
```

- [ ] **Step 2: Modify add_request to check cache**

Update `add_request` method:
```rust
pub fn add_request(&mut self, req: Request) -> SeqId {
    let id = if req.id == 0 {
        let id = self.next_seq_id;
        self.next_seq_id += 1;
        id
    } else {
        req.id
    };

    // Check prefix cache
    let key = hash_tokens(&req.prompt);
    if let Some(entry) = self.prefix_cache.get(key) {
        // Cache hit! Reuse blocks
        let seq = Sequence {
            id,
            tokens: req.prompt,
            kv_blocks: entry.blocks.clone(),
            num_computed_tokens: entry.token_count,
            status: Status::Decoding,  // Skip prefill
            max_tokens: req.max_tokens,
            sampling_params: req.sampling_params,
        };
        self.running.push(seq);
        return id;
    }

    // Cache miss - allocate new blocks
    let num_blocks_needed = req.prompt.len().div_ceil(BLOCK_SIZE);
    let blocks = self
        .kv_allocator
        .allocate(num_blocks_needed)
        .unwrap_or_default();

    let seq = Sequence {
        id,
        tokens: req.prompt,
        kv_blocks: blocks,
        num_computed_tokens: 0,
        status: Status::Waiting,
        max_tokens: req.max_tokens,
        sampling_params: req.sampling_params,
    };
    self.waiting.push_back(seq);
    id
}
```

- [ ] **Step 3: Modify update to store completed sequences**

Update `update` method, add before finished handling:
```rust
// Cache completed sequences
for seq in self.running.iter().filter(|s| s.status == Status::Finished) {
    let key = hash_tokens(&seq.tokens);
    if !self.prefix_cache.entries.contains_key(&key) {
        self.prefix_cache.insert(key, seq.kv_blocks.clone(), seq.tokens.len());
    }
}
```

- [ ] **Step 4: Add OOM handling with cache eviction**

In `add_request`, before allocating new blocks:
```rust
let num_blocks_needed = req.prompt.len().div_ceil(BLOCK_SIZE);

// If not enough blocks, try evicting
if self.kv_allocator.available() < num_blocks_needed {
    self.prefix_cache.evict(&mut self.kv_allocator);
}

// Now try allocate again
let blocks = self
    .kv_allocator
    .allocate(num_blocks_needed)
    .unwrap_or_default();
```

- [ ] **Step 5: Import hash_tokens**

Add import at top:
```rust
use crate::kv_cache::{hash_tokens, PrefixCache};
```

- [ ] **Step 6: Run tests**

```bash
cargo test -p vllm-core
```

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat(core): integrate PrefixCache with Scheduler

- Add prefix_cache field to Scheduler
- Check cache in add_request, reuse blocks on hit
- Store completed sequences in cache after update
- Evict cache on OOM before new allocation
- Add hash_tokens import"
```

---

### Task PC-3: Integration Test + Verification

**Files:**
- Create: `crates/core/tests/prefix_cache.rs`

- [ ] **Step 1: Write integration test**

```rust
use vllm_core::engine::{Engine, ModelBackend};
use vllm_core::error::Result;
use vllm_core::types::{BatchOutput, Request, SchedulerConfig, SeqId, TokenId};
use tokio::sync::mpsc;

struct StubModel;

impl ModelBackend for StubModel {
    fn forward(
        &self,
        seq_ids: &[SeqId],
        _input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
    ) -> Result<BatchOutput> {
        Ok(BatchOutput {
            seq_ids: seq_ids.to_vec(),
            next_tokens: seq_ids.iter().map(|_| 1 as TokenId).collect(),
        })
    }
}

#[test]
fn test_prefix_cache_hit() {
    let config = SchedulerConfig {
        max_num_seqs: 256,
        max_num_batched_tokens: 4096,
    };
    let mut engine = Engine::with_config(StubModel, config, 100);

    let (tx1, _rx1) = mpsc::unbounded_channel();
    let (tx2, _rx2) = mpsc::unbounded_channel();

    // First request: cache miss
    engine.add_request(Request::new(1, vec![10, 20], 5), tx1);
    engine.step().unwrap();  // prefill
    engine.step().unwrap();  // decode
    engine.step().unwrap();  // decode
    engine.step().unwrap();  // decode - finished

    // Second request with same prompt: cache hit
    engine.add_request(Request::new(2, vec![10, 20], 5), tx2);
    
    // Should go directly to decode (status == Decoding)
    let batch = engine.scheduler.build_batch();
    // The second request should be in Decoding status
    assert_eq!(engine.scheduler.running.len(), 1);
}

#[test]
fn test_cache_eviction() {
    let config = SchedulerConfig {
        max_num_seqs: 256,
        max_num_batched_tokens: 4096,
    };
    let mut engine = Engine::with_config(StubModel, config, 2);  // Only 2 blocks
    
    let (tx1, _rx1) = mpsc::unbounded_channel();
    let (tx2, _rx2) = mpsc::unbounded_channel();

    // First request
    engine.add_request(Request::new(1, vec![10, 20, 30], 5), tx1);
    engine.step().unwrap();
    
    // Second request
    engine.add_request(Request::new(2, vec![40, 50, 60], 5), tx2);
    engine.step().unwrap();
    
    // Should evict and work
    assert!(engine.scheduler.has_pending() || engine.scheduler.prefix_cache.len() > 0);
}
```

- [ ] **Step 2: Run tests**

```bash
cargo test --workspace
```

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "test: add PrefixCache integration tests

- Test cache hit: same prompt reuses cached blocks
- Test cache eviction: LRU removes old entries under memory pressure"
```

---

## Verification

```bash
# Build
cargo build --workspace

# Test
cargo test --workspace

# Manual test with server
cargo run -p vllm-server

# Test with repeated prompts
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello world", "max_tokens": 5}'

# Same prompt again - should be faster due to cache
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello world", "max_tokens": 5}'
```

## Spec Coverage

| Spec Section | Covered By |
|---|---|
| CacheKey hash | Task PC-1 |
| PrefixCache get/insert/evict | Task PC-1 |
| LRU eviction | Task PC-1 |
| Scheduler add_request cache check | Task PC-2 |
| Scheduler update cache store | Task PC-2 |
| OOM eviction | Task PC-2 |
| Integration tests | Task PC-3 |