# SchedulerEngine 重构实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 SchedulerEngine 拆分为独立的 MemoryManager 和 CacheManager，实现完全解耦

**Architecture:** 创建 scheduler/memory/ 和 scheduler/cache/ 模块，移动 BlockAllocator、EvictionPolicy、PrefixCache 到对应模块

**Tech Stack:** Rust

---

## 文件清单

| 文件 | 操作 |
|------|------|
| `crates/core/src/scheduler/memory/mod.rs` | 创建 |
| `crates/core/src/scheduler/memory/allocator.rs` | 创建 (从 kv_cache 移动) |
| `crates/core/src/scheduler/memory/eviction.rs` | 创建 (从 scheduler 移动) |
| `crates/core/src/scheduler/cache/mod.rs` | 创建 |
| `crates/core/src/scheduler/cache/prefix_cache.rs` | 创建 (从 kv_cache 移动) |
| `crates/core/src/scheduler/engine.rs` | 修改 |
| `crates/core/src/kv_cache/mod.rs` | 修改 |
| `crates/core/tests/prefix_cache.rs` | 修改 import |

---

## 任务分解

### Task 1: 创建 scheduler/memory/ 模块

**Files:**
- Create: `crates/core/src/scheduler/memory/mod.rs`
- Create: `crates/core/src/scheduler/memory/allocator.rs`
- Create: `crates/core/src/scheduler/memory/eviction.rs`

- [ ] **Step 1: 创建目录和 mod.rs**

创建 `crates/core/src/scheduler/memory/mod.rs`:

```rust
pub mod allocator;
pub mod eviction;

pub use allocator::BlockAllocator;
pub use eviction::EvictionPolicy;

use crate::types::{Sequence, SeqId};
use crate::scheduler::queue_manager::QueueManager;
use crate::scheduler::eviction::EvictionPolicy;
use crate::scheduler::preemption::PreemptionManager;
use crate::scheduler::SchedulerConfig;

pub struct MemoryManager {
    allocator: BlockAllocator,
    eviction_policy: EvictionPolicy,
    preemption_manager: PreemptionManager,
}

impl MemoryManager {
    pub fn new(num_blocks: usize, config: SchedulerConfig) -> Self {
        Self {
            allocator: BlockAllocator::new(num_blocks),
            eviction_policy: EvictionPolicy::new(),
            preemption_manager: PreemptionManager::new(config),
        }
    }

    pub fn allocate(&mut self, num_blocks: usize) -> Option<Vec<usize>> {
        self.allocator.allocate(num_blocks)
    }

    pub fn free(&mut self, blocks: &[usize]) {
        self.allocator.free(blocks);
    }

    pub fn available(&self) -> usize {
        self.allocator.available()
    }

    pub fn total(&self) -> usize {
        self.allocator.total()
    }

    pub fn select_victims(&self, running: &[Sequence], count: usize) -> Vec<usize> {
        self.eviction_policy.select_victims(running, count)
    }

    pub fn release_blocks(&mut self, blocks: &[usize]) {
        self.eviction_policy.release_blocks(blocks);
        self.allocator.free(blocks);
    }

    pub fn record_blocks(&self, blocks: &[usize]) {
        self.eviction_policy.record_blocks(blocks);
    }

    pub fn should_preempt(&self, running: &[Sequence], waiting: usize) -> bool {
        self.preemption_manager.should_preempt(running, waiting)
    }

    pub fn execute_preemption(
        &mut self,
        running: &mut Vec<Sequence>,
        queue: &mut QueueManager,
    ) {
        // 实现抢占逻辑
    }
}
```

- [ ] **Step 2: 移动 BlockAllocator 到 allocator.rs**

从 `crates/core/src/kv_cache/block_allocator.rs` 复制内容到新文件

- [ ] **Step 3: 移动 EvictionPolicy 到 eviction.rs**

从 `crates/core/src/scheduler/engine.rs` 中的 EvictionPolicy 相关代码移动

- [ ] **Step 4: 验证编译**

Run: `cargo build -p vllm-core 2>&1 | head -20`
Expected: 可能需要修复 import

- [ ] **Step 5: Commit**

```bash
git add crates/core/src/scheduler/memory/
git commit -m "refactor(core): create scheduler/memory module"
```

---

### Task 2: 创建 scheduler/cache/ 模块

**Files:**
- Create: `crates/core/src/scheduler/cache/mod.rs`
- Create: `crates/core/src/scheduler/cache/prefix_cache.rs`

- [ ] **Step 1: 创建目录和 mod.rs**

创建 `crates/core/src/scheduler/cache/mod.rs`:

```rust
pub mod prefix_cache;

pub use prefix_cache::{CachedEntry, CacheKey, PrefixCache, hash_tokens};

pub struct CacheManager {
    prefix_cache: PrefixCache,
}

impl CacheManager {
    pub fn new() -> Self {
        Self {
            prefix_cache: PrefixCache::new(),
        }
    }

    pub fn get(&mut self, key: CacheKey) -> Option<&CachedEntry> {
        self.prefix_cache.get(key)
    }

    pub fn insert(&mut self, key: CacheKey, blocks: Vec<usize>, token_count: usize) {
        self.prefix_cache.insert(key, blocks, token_count);
    }

    pub fn insert_arc(&mut self, key: CacheKey, blocks: std::sync::Arc<Vec<usize>>, token_count: usize) {
        self.prefix_cache.insert_arc(key, blocks, token_count);
    }

    pub fn find_prefix_match(&mut self, tokens: &[u32]) -> Option<&CachedEntry> {
        self.prefix_cache.find_prefix_match(tokens)
    }

    pub fn find_reverse_prefix_match(&self, tokens: &[u32]) -> Option<(std::sync::Arc<Vec<usize>>, usize)> {
        self.prefix_cache.find_reverse_prefix_match(tokens)
    }

    pub fn evict(&mut self, allocator: &mut super::memory::BlockAllocator) {
        self.prefix_cache.evict(allocator);
    }

    pub fn len(&self) -> usize {
        self.prefix_cache.len()
    }

    pub fn is_empty(&self) -> bool {
        self.prefix_cache.is_empty()
    }

    pub fn contains_key(&self, key: &CacheKey) -> bool {
        self.prefix_cache.contains_key(key)
    }

    pub fn stats(&self) -> prefix_cache::PrefixCacheStats {
        self.prefix_cache.stats()
    }

    pub fn hit_rate(&self) -> f64 {
        self.prefix_cache.hit_rate()
    }
}

impl Default for CacheManager {
    fn default() -> Self {
        Self::new()
    }
}
```

- [ ] **Step 2: 移动 PrefixCache 到 prefix_cache.rs**

从 `crates/core/src/kv_cache/prefix_cache.rs` 复制内容到新文件

- [ ] **Step 3: 验证编译**

Run: `cargo build -p vllm-core 2>&1 | head -20`
Expected: 可能需要修复 import

- [ ] **Step 4: Commit**

```bash
git add crates/core/src/scheduler/cache/
git commit -m "refactor(core): create scheduler/cache module"
```

---

### Task 3: 重构 SchedulerEngine 使用新模块

**Files:**
- Modify: `crates/core/src/scheduler/engine.rs`

- [ ] **Step 1: 更新 import**

将：
```rust
use crate::kv_cache::{BlockAllocator, PrefixCache, hash_tokens};
```

改为：
```rust
use crate::scheduler::memory::{MemoryManager, BlockAllocator};
use crate::scheduler::cache::{CacheManager, PrefixCache, hash_tokens};
```

- [ ] **Step 2: 修改结构体**

将：
```rust
pub struct SchedulerEngine {
    queue_manager: QueueManager,
    batch_planner: BatchPlanner,
    kv_allocator: BlockAllocator,
    prefix_cache: PrefixCache,
    eviction_policy: EvictionPolicy,
    preemption_manager: PreemptionManager,
    config: SchedulerConfig,
    stats: SchedulerStats,
    next_seq_id: SeqId,
    running: Vec<Sequence>,
    finished: Vec<Sequence>,
    observers: SchedulerObservers,
}
```

改为：
```rust
pub struct SchedulerEngine {
    queue_manager: QueueManager,
    batch_planner: BatchPlanner,
    memory: MemoryManager,
    cache: CacheManager,
    config: SchedulerConfig,
    stats: SchedulerStats,
    next_seq_id: SeqId,
    running: Vec<Sequence>,
    finished: Vec<Sequence>,
    observers: SchedulerObservers,
}
```

- [ ] **Step 3: 更新 new() 方法**

修改构造函数以使用新模块

- [ ] **Step 4: 更新所有方法调用**

将 `self.kv_allocator` 改为 `self.memory.allocator` 或通过 MemoryManager 方法
将 `self.prefix_cache` 改为 `self.cache`

- [ ] **Step 5: 验证编译**

Run: `cargo build -p vllm-core`
Expected: SUCCESS

- [ ] **Step 6: 运行测试**

Run: `cargo test -p vllm-core`
Expected: ALL PASS

- [ ] **Step 7: Commit**

```bash
git add crates/core/src/scheduler/engine.rs
git commit -m "refactor(core): integrate MemoryManager and CacheManager into SchedulerEngine"
```

---

### Task 4: 清理旧模块

**Files:**
- Modify: `crates/core/src/kv_cache/mod.rs`
- Delete: `crates/core/src/kv_cache/block_allocator.rs` (可选)
- Delete: `crates/core/src/kv_cache/prefix_cache.rs` (可选)

- [ ] **Step 1: 更新 kv_cache/mod.rs**

移除对 block_allocator 和 prefix_cache 的引用（如果已移动）

- [ ] **Step 2: 更新测试中的 import**

修改测试文件中的 import 路径

- [ ] **Step 3: 验证编译**

Run: `cargo build -p vllm-core`
Expected: SUCCESS

- [ ] **Step 4: 运行所有测试**

Run: `cargo test -p vllm-core`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add crates/core/src/kv_cache/
git commit -m "refactor(core): clean up kv_cache after migration"
```

---

### Task 5: 最终验证

**Files:**
- None

- [ ] **Step 1: 运行完整测试套件**

Run: `cargo test --workspace`
Expected: ALL PASS

- [ ] **Step 2: 运行 clippy**

Run: `cargo clippy --workspace -- -D warnings`
Expected: NO WARNINGS

- [ ] **Step 3: 运行 fmt**

Run: `cargo fmt --all --check`
Expected: PASS

- [ ] **Step 4: Commit final**

```bash
git add -A
git commit -m "refactor(core): complete scheduler decoupling"
```
