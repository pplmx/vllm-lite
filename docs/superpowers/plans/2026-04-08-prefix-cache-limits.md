# 前缀缓存增强实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为前缀缓存添加最大缓存限制和自动淘汰功能

**Architecture:** 在 PrefixCache 中添加 max_entries 和 max_blocks 配置，在 insert 时检查限制并自动触发淘汰

**Tech Stack:** Rust

---

## 问题分析

### 当前实现问题
1. 没有最大缓存大小限制 - 缓存可以无限增长
2. 淘汰需要手动调用 evict() - 没有自动触发
3. evict() 只淘汰一个条目 - 不够高效

### 目标
1. 添加 `max_entries` 配置 - 最大缓存条目数
2. 添加 `max_blocks` 配置 - 最大 KV block 数量
3. 在 insert 时自动检查并淘汰
4. 优化淘汰逻辑 - 一次淘汰多个条目直到满足限制

---

## 文件清单

| 文件 | 操作 |
|------|------|
| `crates/core/src/scheduler/cache/prefix_cache.rs` | 修改 |
| `crates/core/src/scheduler/cache/mod.rs` | 修改 (CacheManager) |
| `crates/core/src/scheduler/memory/mod.rs` | 修改 (MemoryManager 传递配置) |
| `crates/core/src/scheduler/engine.rs` | 修改 (初始化配置) |

---

## 任务分解

### Task 1: 修改 PrefixCache 结构体添加配置

**Files:**
- Modify: `crates/core/src/scheduler/cache/prefix_cache.rs`

- [ ] **Step 1: 添加配置字段**

修改 PrefixCache 结构体，添加配置选项：

```rust
pub struct PrefixCacheConfig {
    pub max_entries: Option<usize>,
    pub max_blocks: Option<usize>,
}

impl Default for PrefixCacheConfig {
    fn default() -> Self {
        Self {
            max_entries: Some(100),      // 默认最多 100 个缓存条目
            max_blocks: Some(1000),       // 默认最多 1000 个 KV blocks
        }
    }
}

pub struct PrefixCache {
    // ... existing fields ...
    config: PrefixCacheConfig,
    total_blocks: usize,  // 跟踪当前使用的 block 总数
}
```

- [ ] **Step 2: 修改 new() 方法接受配置**

```rust
pub fn new(config: PrefixCacheConfig) -> Self {
    Self {
        // ... existing init ...
        config,
        total_blocks: 0,
    }
}
```

- [ ] **Step 3: 修改 insert_arc 追踪 block 数量**

在 insert_arc 中追踪 total_blocks：

```rust
pub fn insert_arc(&mut self, key: CacheKey, blocks: Arc<Vec<BlockId>>, token_count: usize) {
    // ... existing code ...
    
    // 更新 total_blocks
    let new_block_count = blocks.len();
    self.total_blocks = self.total_blocks.saturating_add(new_block_count);
    
    // 检查限制并触发淘汰
    self.maybe_evict(allocator);
}
```

- [ ] **Step 4: 添加 maybe_evict 方法**

```rust
fn maybe_evict(&mut self, allocator: Option<&mut BlockAllocator>) {
    // 检查是否超过限制
    let needs_eviction = match (&self.config.max_entries, &self.config.max_blocks) {
        (Some(max_entries), _) if self.entries.len() > *max_entries => true,
        (_, Some(max_blocks)) if self.total_blocks > *max_blocks => true,
        _ => false,
    };
    
    if !needs_eviction {
        return;
    }
    
    // 批量淘汰直到满足限制
    while let Some(max) = self.config.max_entries {
        if self.entries.len() <= max {
            break;
        }
        if let Some(alloc) = allocator {
            self.evict_single(alloc);
        } else {
            self.evict_single_internal();
        }
    }
    
    while let Some(max) = self.config.max_blocks {
        if self.total_blocks <= max {
            break;
        }
        if let Some(alloc) = allocator {
            self.evict_single(alloc);
        } else {
            self.evict_single_internal();
        }
    }
}
```

- [ ] **Step 5: 添加 evict_single_internal 方法 (无需 allocator)**
  
在无法访问 allocator 时使用：

```rust
fn evict_single_internal(&mut self) {
    if let Some(oldest_key) = self.lru_order.pop_back() {
        if let Some(entry) = self.entries.remove(&oldest_key) {
            self.total_blocks = self.total_blocks.saturating_sub(entry.blocks.len());
            self.stats.evictions += 1;
            self.stats.entries = self.entries.len();
        }
    }
}
```

- [ ] **Step 6: 修改 evict 方法使用 maybe_evict**

将现有 evict 方法改为批量淘汰：

```rust
pub fn evict(&mut self, allocator: &mut BlockAllocator) {
    self.maybe_evict(Some(allocator));
}
```

- [ ] **Step 7: 验证编译**

Run: `cargo build -p vllm-core`

- [ ] **Step 8: 运行测试**

Run: `cargo test -p vllm-core -- prefix_cache`

- [ ] **Step 9: Commit**

```bash
git add crates/core/src/scheduler/cache/prefix_cache.rs
git commit -m "feat(core): add max cache limits to PrefixCache"
```

---

### Task 2: 更新 CacheManager 传递配置

**Files:**
- Modify: `crates/core/src/scheduler/cache/mod.rs`

- [ ] **Step 1: 添加配置到 CacheManager**

```rust
pub struct CacheManager {
    prefix_cache: PrefixCache,
}

impl CacheManager {
    pub fn new(config: PrefixCacheConfig) -> Self {
        Self {
            prefix_cache: PrefixCache::new(config),
        }
    }
    // ... 其他方法保持不变
}
```

- [ ] **Step 2: 验证编译**

Run: `cargo build -p vllm-core`

- [ ] **Step 3: Commit**

```bash
git add crates/core/src/scheduler/cache/mod.rs
git commit -m "feat(core): pass config to CacheManager"
```

---

### Task 3: 更新 SchedulerEngine 初始化

**Files:**
- Modify: `crates/core/src/scheduler/engine.rs`

- [ ] **Step 1: 更新 new() 初始化 CacheManager**

```rust
use crate::scheduler::cache::CacheManager;
use crate::scheduler::cache::PrefixCacheConfig;

// 在 new() 方法中:
cache: CacheManager::new(PrefixCacheConfig {
    max_entries: Some(100),
    max_blocks: Some(config.max_num_seqs * 16),  // 根据 max_seqs 动态计算
}),
```

- [ ] **Step 2: 验证编译**

Run: `cargo build -p vllm-core`

- [ ] **Step 3: 运行测试**

Run: `cargo test -p vllm-core`

- [ ] **Step 4: Commit**

```bash
git add crates/core/src/scheduler/engine.rs
git commit -m "feat(core): configure prefix cache limits"
```

---

### Task 4: 最终验证

**Files:**
- None

- [ ] **Step 1: 运行完整测试**

Run: `cargo test -p vllm-core`

- [ ] **Step 2: 运行 clippy**

Run: `cargo clippy -p vllm-core -- -D warnings`

- [ ] **Step 3: 运行 fmt**

Run: `cargo fmt --all --check`

- [ ] **Step 4: Commit final**

```bash
git add -A
git commit -m "feat(core): complete prefix cache limits implementation"
```
