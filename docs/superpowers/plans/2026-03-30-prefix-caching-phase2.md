# Prefix Caching Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现前缀命中 - 当新请求的 prompt 是已缓存序列的超集时，复用部分 KV cache

**Architecture:**
- PrefixCache 添加 find_prefix_match 方法查找最长前缀匹配
- Scheduler add_request 在完整 hash 未命中时尝试前缀匹配
- 设置正确的 num_computed_tokens 状态

**Tech Stack:** Rust, vllm-core

---

## Task 1: 添加 find_prefix_match 方法

**Files:**
- Modify: `crates/core/src/kv_cache.rs`

- [ ] **Step 1: 查看当前 PrefixCache 结构**

```bash
cat crates/core/src/kv_cache.rs | head -100
```

- [ ] **Step 2: 添加 find_prefix_match 方法**

在 impl PrefixCache 中添加:

```rust
/// Find the longest prefix match in cache
/// Returns the cached entry if the prompt starts with cached sequence
pub fn find_prefix_match(&self, tokens: &[TokenId]) -> Option<&CachedEntry> {
    // Need to find entries where cached tokens are a prefix of input tokens
    // Since we store hash, we need to either:
    // 1. Store tokens along with hash (space tradeoff)
    // 2. Try hashing different prefix lengths (CPU intensive)
    
    // Approach: iterate over entries, find longest match
    // This is O(cache_size * avg_cached_len), acceptable for small cache
    
    // For now, iterate all entries and find longest prefix match
    // A cached entry matches if input starts with cached tokens
    // We need to reconstruct tokens from hash or store them
    
    // Simpler approach: for each cached entry, try to match
    // But we don't have the actual tokens stored
    
    // Alternative: store a mapping from prefix_hash to entry
    // For MVP: just do linear search, try different prefix lengths
    
    None  // TODO: implement
}
```

**Wait - the current design has an issue:** We store only the hash, not the actual tokens. To check prefix match, we need either:
1. Store tokens with each entry (memory cost: O(total cached tokens))
2. Try hashing all possible prefix lengths (CPU: O(seq_len) per check)

**Recommended approach for MVP:**
- When checking cache, also compute hashes for all prefix lengths
- If any prefix hash matches a cached entry, we have a hit

- [ ] **Step 3: 实现改进的 find_prefix_match**

```rust
/// Find prefix match by trying all prefix hashes
pub fn find_prefix_match(&self, tokens: &[TokenId]) -> Option<&CachedEntry> {
    if tokens.is_empty() {
        return None;
    }
    
    // Try from longest to shortest - find the longest prefix that's cached
    for prefix_len in (1..=tokens.len()).rev() {
        let prefix = &tokens[..prefix_len];
        let key = hash_tokens(prefix);
        if let Some(entry) = self.entries.get(&key) {
            // Found a match - prefix_len tokens are cached
            return Some(entry);
        }
    }
    None
}
```

- [ ] **Step 4: 添加测试**

```rust
#[test]
fn test_prefix_match() {
    let mut cache = PrefixCache::new();
    
    // Insert "Hello"
    cache.insert(hash_tokens(&[1, 2]), vec![1], 2);
    
    // Find match for [1, 2, 3]
    let result = cache.find_prefix_match(&[1, 2, 3]);
    assert!(result.is_some());
    assert_eq!(result.unwrap().token_count, 2);
    
    // No match for [3, 4]
    let result = cache.find_prefix_match(&[3, 4]);
    assert!(result.is_none());
}
```

- [ ] **Step 5: 测试并提交**

```bash
cd /home/mystvio/repos/vllm-lite && cargo test -p vllm-core kv_cache -- --nocapture
git add crates/core/src/kv_cache.rs
git commit -m "feat(core): add find_prefix_match to PrefixCache"
```

---

## Task 2: 修改 Scheduler 支持前缀命中

**Files:**
- Modify: `crates/core/src/scheduler.rs`

- [ ] **Step 1: 查看当前 add_request**

```bash
cat crates/core/src/scheduler.rs | head -95
```

- [ ] **Step 2: 修改 add_request 添加前缀匹配**

在 add_request 方法中，完整 hash 检查后添加:

```rust
// Check prefix cache - full match first
let key = hash_tokens(&req.prompt);
if let Some(entry) = self.prefix_cache.get(key) {
    // Full cache hit - direct to decode
    // ... existing code
}

// NEW: Check for prefix match
// Try to find a cached prefix of the prompt
if let Some(entry) = self.prefix_cache.find_prefix_match(&req.prompt) {
    // Partial cache hit - need to prefill remaining tokens
    let cached_len = entry.token_count;
    let remaining_len = req.prompt.len() - cached_len;
    
    // Allocate blocks for remaining tokens
    let num_blocks_needed = remaining_len.div_ceil(BLOCK_SIZE);
    if self.kv_allocator.available() < num_blocks_needed {
        self.prefix_cache.evict(&mut self.kv_allocator);
    }
    
    let blocks = self.kv_allocator
        .allocate(num_blocks_needed)
        .unwrap_or_default();
    
    // Combine cached blocks + new blocks
    let mut all_blocks = entry.blocks.clone();
    all_blocks.extend(blocks);
    
    let seq = Sequence {
        id,
        tokens: req.prompt,
        kv_blocks: all_blocks,
        num_computed_tokens: cached_len,  // These tokens are already computed
        prompt_len: req.prompt.len(),
        status: Status::Prefilling,  // Still need to prefill remaining
        max_tokens: req.max_tokens,
        sampling_params: req.sampling_params,
        consecutive_decode_rounds: 0,
    };
    self.waiting.push_back(seq);
    return id;
}
```

- [ ] **Step 3: 运行测试**

```bash
cd /home/mystvio/repos/vllm-lite && cargo test -p vllm-core -- --nocapture
```

- [ ] **Step 4: 提交**

```bash
git add crates/core/src/scheduler.rs
git commit -m "feat(core): support prefix hit in scheduler"
```

---

## Task 3: 集成测试

**Files:**
- Modify: `crates/core/tests/prefix_cache.rs`

- [ ] **Step 1: 添加前缀命中测试**

```rust
#[test]
fn test_prefix_hit_partial_prefill() {
    let config = SchedulerConfig {
        max_num_seqs: 256,
        max_num_batched_tokens: 4096,
        max_consecutive_decode: 10,
    };
    let mut engine = Engine::with_config(StubModel, StubModel, config, 4, 100);

    let (tx1, _rx1) = mpsc::unbounded_channel();

    // First request: complete it to populate cache
    engine.add_request(Request::new(1, vec![10, 20], 3), tx1);
    while engine.has_pending() {
        engine.step().unwrap();
    }

    // Second request: longer prompt starting with same tokens
    let (tx2, _rx2) = mpsc::unbounded_channel();
    engine.add_request(Request::new(2, vec![10, 20, 30, 40, 50], 3), tx2);
    engine.step().unwrap();

    // Should be in prefilling state with num_computed_tokens = 2
    assert_eq!(engine.scheduler.running().len(), 1);
    let seq = &engine.scheduler.running()[0];
    assert_eq!(seq.status, vllm_core::types::Status::Prefilling);
    assert_eq!(seq.num_computed_tokens, 2);  // [10, 20] are cached
}
```

- [ ] **Step 2: 运行测试**

```bash
cd /home/mystvio/repos/vllm-lite && cargo test -p vllm-core prefix_cache -- --nocapture
```

- [ ] **Step 3: 提交**

```bash
git add crates/core/tests/prefix_cache.rs
git commit -m "test(core): add prefix hit integration test"
```

---

## Verification Checklist

- [ ] find_prefix_match correctly finds longest cached prefix
- [ ] Scheduler uses prefix match when full hash misses
- [ ] num_computed_tokens set correctly for partial cache
- [ ] Sequence status set to Prefilling for remaining tokens
- [ ] All tests pass
- [ ] Clippy clean

---

**Plan complete and saved to `docs/superpowers/plans/2026-03-30-prefix-caching-phase2.md`.**

Two execution options:

1. **Subagent-Driven (recommended)** - Dispatch subagent per task
2. **Inline Execution** - Execute in current session

Which approach?