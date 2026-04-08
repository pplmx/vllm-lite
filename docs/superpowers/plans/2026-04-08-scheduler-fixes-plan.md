# Scheduler 架构问题修复实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 SchedulerEngine 中的 5 个架构问题：running_count 返回 0、add_request 重复代码、双重状态存储、EvictionPolicy 未集成、PreemptionManager 未集成

**Architecture:** 
- Task 1-2: 修复简单问题（running_count、状态分离）
- Task 3: 重构 add_request 消除重复代码
- Task 4-5: 集成 EvictionPolicy 和 PreemptionManager
- Task 6: 全面测试验证

**Tech Stack:** Rust, vllm-core crate

---

## 背景分析

### 问题清单

| # | 问题 | 文件 | 难度 |
|---|------|------|------|
| 1 | `running_count()` 永远返回 0 | engine.rs:601-603 | 简单 |
| 2 | `running` 和 `queue_manager` 双重存储 | engine.rs | 中等 |
| 3 | `add_request` 重复代码 (~250行) | engine.rs:100-356 | 中等 |
| 4 | EvictionPolicy 未集成 | engine.rs:507-516 | 复杂 |
| 5 | PreemptionManager 未使用 | engine.rs | 复杂 |

### 关键文件

- `crates/core/src/scheduler/engine.rs` - 主调度引擎
- `crates/core/src/scheduler/queue_manager.rs` - 队列管理
- `crates/core/src/scheduler/eviction.rs` - 淘汰策略
- `crates/core/src/scheduler/preemption.rs` - 抢占管理
- `crates/core/src/scheduler/batch_planner.rs` - 批次规划
- `crates/core/src/kv_cache/block_allocator.rs` - 块分配

---

## 任务 1: 修复 running_count() 返回 0

**Files:**
- Modify: `crates/core/src/scheduler/engine.rs:601-603`
- Test: `crates/core/src/scheduler/engine.rs` (现有测试)

- [ ] **Step 1: 查看当前 running_count 实现**

```rust
// 当前代码 (engine.rs:601-603)
pub fn running_count(&self) -> usize {
    0  // 硬编码返回 0！
}
```

- [ ] **Step 2: 修改为返回实际 running 长度**

```rust
pub fn running_count(&self) -> usize {
    self.running.len()
}
```

- [ ] **Step 3: 同时修复 SchedulerStateViewImpl 中的实现**

```rust
// engine.rs:635-637 修改为:
fn running_count(&self) -> usize {
    self.running.len()  // 需要把 running 传入
}
```

注意: `SchedulerStateViewImpl` 目前只有 `queue_manager` 和 `kv_allocator`，需要添加 `running` 字段。

- [ ] **Step 4: 更新 SchedulerStateViewImpl 结构体**

```rust
// engine.rs:625-628 修改为:
struct SchedulerStateViewImpl<'a> {
    queue_manager: &'a QueueManager,
    kv_allocator: &'a BlockAllocator,
    running: &'a Vec<Sequence>,  // 添加
}
```

- [ ] **Step 5: 更新 build_batch 中的 state_view 创建**

```rust
// engine.rs:394-397 修改为:
let state_view = SchedulerStateViewImpl {
    queue_manager: &self.queue_manager,
    kv_allocator: &self.kv_allocator,
    running: &self.running,  // 添加
};
```

- [ ] **Step 6: 运行测试验证**

Run: `cargo test -p vllm-core -- scheduler --nocapture`
Expected: 现有测试通过

---

## 任务 2: 分析并优化双重状态存储

**Files:**
- Modify: `crates/core/src/scheduler/engine.rs`
- Test: `crates/core/tests/integration.rs`

- [ ] **Step 1: 分析 running 和 queue_manager 的使用差异**

运行命令查看使用位置:
```bash
cd /home/mystvio/repos/vllm-lite
grep -n "self.running" crates/core/src/scheduler/engine.rs
grep -n "queue_manager" crates/core/src/scheduler/engine.rs
```

分析结果:
- `running`: 在 `build_batch` 中添加当前批次请求，在 `update` 中更新状态
- `queue_manager`: 持久化等待队列，按优先级管理

设计决策: 保持双源但明确职责，或者合并。选择保持分离但清理冗余代码。

- [ ] **Step 2: 简化 update 中的双重更新逻辑**

当前问题: `update` 函数在 `running` 和 `queue_manager` 中重复更新相同序列。

修复方案: 只在一个地方更新，因为数据应该同步。

查看当前 update 代码 (engine.rs:489-544)，分析哪些序列应该在 running，哪些在 queue_manager。

- [ ] **Step 3: 统一更新逻辑**

对于已完成的序列:
- 从 `running` 中移除
- 从 `queue_manager` 中移除并加入 prefix_cache

对于正在运行的序列:
- 只在 `running` 中更新（因为已不在 queue 中）

- [ ] **Step 4: 运行测试**

Run: `cargo test -p vllm-core -- scheduler --nocapture`
Expected: 所有测试通过

---

## 任务 3: 重构 add_request 消除重复代码

**Files:**
- Modify: `crates/core/src/scheduler/engine.rs:100-380`
- Test: `crates/core/src/scheduler/engine.rs` (添加测试)

- [ ] **Step 1: 提取序列创建逻辑为辅助函数**

在 `add_request` 函数之前添加:

```rust
fn create_sequence_from_cache(
    seq_id: SeqId,
    prompt: Vec<TokenId>,
    kv_blocks: Arc<Vec<BlockId>>,
    num_computed_tokens: usize,
    prompt_len: usize,
    max_tokens: usize,
    sampling_params: SamplingParams,
    priority: Priority,
    status: Status,
) -> Sequence {
    Sequence {
        id: seq_id,
        tokens: prompt,
        kv_blocks,
        num_computed_tokens,
        prompt_len,
        status,
        max_tokens,
        sampling_params,
        consecutive_decode_rounds: 0,
        priority,
    }
}
```

- [ ] **Step 2: 提取缓存处理逻辑**

添加枚举表示缓存命中类型:

```rust
enum CacheHit {
    Exact(Arc<Vec<BlockId>>, usize),           // blocks, token_count
    Prefix(Arc<Vec<BlockId>>, usize),          // blocks, tokens_to_skip
    ReversePrefix(Vec<BlockId>, usize),        // blocks, cached_tokens
    None,
}
```

- [ ] **Step 3: 添加缓存检查辅助方法**

在 SchedulerEngine impl 中添加:

```rust
fn check_prefix_cache(&mut self, prompt: &[TokenId]) -> CacheHit {
    // 1. 精确匹配
    let key = hash_tokens(prompt);
    if let Some(entry) = self.prefix_cache.get(key) {
        return CacheHit::Exact(entry.blocks.clone(), entry.token_count);
    }
    
    // 2. 前缀匹配
    if let Some(entry) = self.prefix_cache.find_prefix_match(prompt) {
        return CacheHit::Prefix(entry.blocks.clone(), entry.token_count);
    }
    
    // 3. 反向前缀匹配
    if let Some((blocks, cached_tokens)) = self.prefix_cache.find_reverse_prefix_match(prompt) {
        return CacheHit::ReversePrefix(blocks, cached_tokens);
    }
    
    CacheHit::None
}
```

- [ ] **Step 4: 重写 add_request 使用辅助方法**

将原来的 ~250 行重复代码替换为:

```rust
pub fn add_request(&mut self, mut req: Request) -> SeqId {
    if req.id == 0 {
        req.id = self.next_seq_id;
        self.next_seq_id += 1;
    }

    let seq_id = req.id;
    let priority = req.priority.clone();
    let prompt_len = req.prompt.len();

    // Check prefix cache
    match self.check_prefix_cache(&req.prompt) {
        CacheHit::Exact(blocks, token_count) => {
            let seq = create_sequence_from_cache(
                seq_id, req.prompt, blocks, token_count,
                prompt_len, req.max_tokens, req.sampling_params,
                priority.clone(), Status::Waiting,
            );
            self.queue_manager.enqueue(seq.clone(), priority);
            self.running.push(seq);
        }
        CacheHit::Prefix(blocks, tokens_to_skip) => {
            let seq = create_sequence_from_cache(
                seq_id, req.prompt.clone(), blocks, tokens_to_skip,
                prompt_len, req.max_tokens, req.sampling_params,
                priority.clone(), Status::Waiting,
            );
            self.queue_manager.enqueue(seq.clone(), priority);
            self.running.push(seq);
        }
        CacheHit::ReversePrefix(blocks, cached_tokens) => {
            let seq = create_sequence_from_cache(
                seq_id, req.prompt.clone(), blocks, cached_tokens,
                prompt_len, req.max_tokens, req.sampling_params,
                priority.clone(), Status::Waiting,
            );
            self.queue_manager.enqueue(seq.clone(), priority);
            self.running.push(seq);
        }
        CacheHit::None => {
            // Allocate new blocks
            let blocks_needed = prompt_len.div_ceil(BLOCK_SIZE);
            let blocks = self.kv_allocator
                .allocate(blocks_needed)
                .unwrap_or_default();

            let seq = create_sequence_from_cache(
                seq_id, req.prompt, Arc::new(blocks), 0,
                prompt_len, req.max_tokens, req.sampling_params,
                priority, Status::Waiting,
            );
            self.queue_manager.enqueue(seq, priority);
        }
    }

    seq_id
}
```

- [ ] **Step 5: 添加测试验证缓存逻辑**

```rust
#[test]
fn test_add_request_cache_exact_hit() {
    let mut engine = SchedulerEngine::default();
    
    // First request - cache miss
    engine.add_request(Request::new(0, vec![1, 2, 3], 10));
    // ... process and finish to populate cache ...
    
    // Second request with same prompt - should hit cache
    let id = engine.add_request(Request::new(0, vec![1, 2, 3], 10));
    assert!(id > 0);
}
```

- [ ] **Step 6: 运行测试**

Run: `cargo test -p vllm-core -- scheduler --nocapture`
Expected: 所有测试通过

---

## 任务 4: 集成 EvictionPolicy

**Files:**
- Modify: `crates/core/src/scheduler/engine.rs:500-520`
- Modify: `crates/core/src/scheduler/engine.rs:69-81` (添加字段)
- Test: `crates/core/src/scheduler/eviction.rs` (现有测试)

- [ ] **Step 1: 在 SchedulerEngine 中添加 EvictionPolicy 字段**

```rust
// engine.rs:69-81 修改为:
pub struct SchedulerEngine {
    // ... existing fields ...
    eviction_policy: EvictionPolicy,  // 添加
}
```

- [ ] **Step 2: 在 new() 中初始化**

```rust
// engine.rs:84-98 修改为:
pub fn new(config: SchedulerConfig, num_kv_blocks: usize) -> Self {
    Self {
        // ... existing fields ...
        eviction_policy: EvictionPolicy::new(),  // 添加
    }
}
```

- [ ] **Step 3: 修改 update 中的块分配逻辑**

当前代码 (engine.rs:507-516):
```rust
while seq.kv_blocks.len() < blocks_needed {
    if let Some(new_blocks) = self.kv_allocator.allocate(1) {
        // ...分配新块
    } else {
        self.stats.record_preemption();
        break;  // 问题: 没有真正执行淘汰
    }
}
```

修改为:
```rust
while seq.kv_blocks.len() < blocks_needed {
    if let Some(new_blocks) = self.kv_allocator.allocate(1) {
        let mut blocks = (*seq.kv_blocks).clone();
        blocks.extend(new_blocks);
        seq.kv_blocks = std::sync::Arc::new(blocks);
    } else {
        // 内存不足，尝试淘汰
        let running_seqs: Vec<_> = self.running.iter()
            .filter(|s| s.id != seq_id && s.status != Status::Finished)
            .cloned()
            .collect();
        
        let victims = self.eviction_policy.select_victims(
            &running_seqs,
            1,  // 需要 1 个块
        );
        
        if victims.is_empty() {
            self.stats.record_preemption();
            break;
        }
        
        // 释放被淘汰的块
        self.eviction_policy.release_blocks(&victims);
        self.kv_allocator.free(&victims);
        self.stats.record_eviction();
        
        // 重试分配
        if let Some(new_blocks) = self.kv_allocator.allocate(1) {
            let mut blocks = (*seq.kv_blocks).clone();
            blocks.extend(new_blocks);
            seq.kv_blocks = std::sync::Arc::new(blocks);
        } else {
            self.stats.record_preemption();
            break;
        }
    }
}
```

- [ ] **Step 4: 在 build_batch 中记录块引用**

在将序列加入 running 时:
```rust
// engine.rs:446 后添加
self.eviction_policy.record_blocks(&seq.kv_blocks);
```

- [ ] **Step 5: 运行测试**

Run: `cargo test -p vllm-core -- scheduler --nocapture`
Expected: 所有测试通过

---

## 任务 5: 集成 PreemptionManager

**Files:**
- Modify: `crates/core/src/scheduler/engine.rs`
- Modify: `crates/core/src/scheduler/preemption.rs` (如需要)
- Test: `crates/core/src/scheduler/preemption.rs`

- [ ] **Step 1: 在 SchedulerEngine 中添加 PreemptionManager 字段**

```rust
// engine.rs:69-81 修改为:
pub struct SchedulerEngine {
    // ... existing fields ...
    preemption_manager: PreemptionManager,  // 添加
}
```

- [ ] **Step 2: 在 new() 中初始化**

```rust
// engine.rs:84-98 修改为:
pub fn new(config: SchedulerConfig, num_kv_blocks: usize) -> Self {
    Self {
        // ... existing fields ...
        preemption_manager: PreemptionManager::new(config.clone()),  // 添加
    }
}
```

- [ ] **Step 3: 在 add_request 中检查是否需要抢占**

在分配块之前添加:
```rust
// 检查是否需要抢占
if blocks_needed > self.kv_allocator.available() {
    let should_preempt = self.preemption_manager.should_preempt(
        self.running.len(),
        self.queue_manager.len(),
        blocks_needed,
        self.kv_allocator.available(),
    );
    
    if should_preempt {
        // 执行抢占
        self.execute_preemption(blocks_needed);
    }
}
```

- [ ] **Step 4: 实现 execute_preemption 方法**

在 SchedulerEngine impl 中添加:
```rust
fn execute_preemption(&mut self, blocks_needed: usize) {
    let running_seqs: Vec<_> = self.running.iter()
        .filter(|s| s.status == Status::Decoding)
        .cloned()
        .collect();
    
    let victims = self.eviction_policy.select_victims(&running_seqs, blocks_needed);
    
    for victim in victims {
        // 找到对应的序列
        if let Some(seq) = self.running.iter_mut().find(|s| s.kv_blocks.contains(&victim)) {
            // 标记为等待
            seq.status = Status::Waiting;
            // 移入 preempted 队列
            self.queue_manager.enqueue_preempted(seq.clone());
            // 释放块
            self.eviction_policy.release_blocks(&seq.kv_blocks);
            self.kv_allocator.free(seq.kv_blocks.as_ref());
            // 从 running 中移除
            self.running.retain(|s| s.id != seq.id);
        }
    }
}
```

- [ ] **Step 5: 运行测试**

Run: `cargo test -p vllm-core -- preemption --nocapture`
Expected: 所有测试通过

---

## 任务 6: 全面测试和验证

**Files:**
- Test: `crates/core/tests/integration.rs`
- Test: `crates/core/src/scheduler/`

- [ ] **Step 1: 运行所有 core 测试**

Run: `cargo test -p vllm-core -- --nocapture`
Expected: 所有测试通过

- [ ] **Step 2: 运行 clippy 检查**

Run: `cargo clippy -p vllm-core -- -D warnings`
Expected: 无警告

- [ ] **Step 3: 运行格式检查**

Run: `cargo fmt --all --check`
Expected: 格式正确

- [ ] **Step 4: 运行完整 workspace 测试**

Run: `cargo test --workspace`
Expected: 所有测试通过

---

## 任务 7: 提交更改

- [ ] **Step 1: 查看更改**

Run: `git diff --stat`

- [ ] **Step 2: 提交**

```bash
git add -A
git commit -m "fix(scheduler): resolve architecture issues

- Fix running_count() returning 0
- Refactor add_request to eliminate duplicate code
- Integrate EvictionPolicy into block allocation
- Integrate PreemptionManager for memory pressure handling
- Simplify update logic to reduce state duplication"
```

---

## 执行顺序

1. 任务 1 (简单) - 先修复 running_count
2. 任务 2 (中等) - 分析和简化状态管理
3. 任务 3 (中等) - 重构 add_request
4. 任务 4 (复杂) - 集成 EvictionPolicy
5. 任务 5 (复杂) - 集成 PreemptionManager
6. 任务 6 - 全面测试
7. 任务 7 - 提交

每个任务独立可测试。建议按顺序执行。
