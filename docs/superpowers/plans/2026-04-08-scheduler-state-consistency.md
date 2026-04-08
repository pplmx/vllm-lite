# SchedulerEngine 状态一致性修复计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 SchedulerEngine 中 running 和 queue_manager 的数据冗余问题，确保序列状态一致性

**Architecture:** 在 build_batch 中从 queue_manager 移除序列到 running，而不是克隆；在 update 中只更新 running

**Tech Stack:** Rust

---

## 问题分析

### 当前实现问题
1. `build_batch` 从 `queue_manager.all_waiting()` 获取序列，克隆到 `running`
2. 原序列**保留在 queue_manager 中**
3. 结果：同一序列同时存在于 `queue_manager` 和 `running`
4. `update` 双重更新两个地方 (line 428-497 更新 running，line 500-520 更新 queue_manager)
5. 状态可能不一致：两个地方的状态可能不同步

### 修复方案（最优）
- **修改 build_batch**: 使用 `dequeue()` 从 queue_manager **移除**序列，而非克隆
- **修改 update**: 只更新 running，移除对 queue_manager 的冗余更新
- **修改 preemption**: 调整逻辑以适应新的数据流
- **核心原则**: running 是"正在处理的序列"的唯一状态来源

---

## 文件清单

| 文件 | 操作 |
|------|------|
| `crates/core/src/scheduler/engine.rs` | 修改 build_batch 和 update |
| `crates/core/tests/integration.rs` | 测试 - 验证功能正常 |

---

## 任务分解

### Task 1: 修改 build_batch - 从 queue_manager 移除序列

**Files:**
- Modify: `crates/core/src/scheduler/engine.rs:303-414`

- [ ] **Step 1: 修改获取序列的方式**

将现有的 "peek, don't remove" 逻辑改为 "dequeue"：

找到约 Line 333-358，原始代码：
```rust
// Get sequences from queue for batch (peek, don't remove)
let all_seqs: Vec<_> = {
    let all = self.queue_manager.all_waiting();
    all.into_iter().take(batch_size).cloned().collect()
};

// Apply token budget: collect sequences until we hit max_tokens limit
let mut current_tokens = 0;
let mut batch_seqs = Vec::new();

for seq in all_seqs {
    let is_prefilling = seq.status == Status::Waiting;
    let seq_tokens = if is_prefilling {
        seq.tokens.len().saturating_sub(seq.num_computed_tokens)
    } else {
        1 // decode is 1 token
    };

    // Check if adding this sequence would exceed token budget
    if current_tokens + seq_tokens > max_tokens {
        break;
    }

    current_tokens += seq_tokens;
    batch_seqs.push(seq);
}
```

修改为：
```rust
let mut batch_seqs = Vec::new();
let mut current_tokens = 0;

// 使用 dequeue 从队列中移除，而非克隆
while batch_seqs.len() < batch_size {
    if let Some(seq) = self.queue_manager.dequeue() {
        let is_prefilling = seq.status == Status::Waiting;
        let seq_tokens = if is_prefilling {
            seq.tokens.len().saturating_sub(seq.num_computed_tokens)
        } else {
            1
        };

        if current_tokens + seq_tokens > max_tokens {
            // 超出预算，放回队列并停止
            self.queue_manager.enqueue(seq, seq.priority.clone());
            break;
        }

        current_tokens += seq_tokens;
        batch_seqs.push(seq);
    } else {
        break;
    }
}
```

- [ ] **Step 2: 验证编译通过**

Run: `cargo build -p vllm-core 2>&1 | head -30`
Expected: SUCCESS or COMPILE ERROR (will fix)

- [ ] **Step 3: Commit**

```bash
git add crates/core/src/scheduler/engine.rs
git commit -m "fix(core): dequeue sequences from queue in build_batch"
```

---

### Task 2: 修改 update - 移除对 queue_manager 的冗余更新

**Files:**
- Modify: `crates/core/src/scheduler/engine.rs:413-550`

- [ ] **Step 1: 删除冗余的 queue_manager 更新代码**

找到约 Line 499-520，删除这段代码：

```rust
// 删除 - 不再需要
// Also check queue_manager for any sequences that didn't go through running
if let Some(seq) = self.queue_manager.get_mut(*seq_id) {
    if seq.status == Status::Waiting {
        seq.num_computed_tokens += input_count;
        if seq.num_computed_tokens >= seq.prompt_len {
            seq.status = Status::Decoding;
        } else {
            seq.status = Status::Prefilling;
        }
    }

    seq.tokens.push(token);
    seq.consecutive_decode_rounds += 1;

    if seq.tokens.len() >= seq.max_tokens {
        seq.status = Status::Finished;
    }

    if seq.status == Status::Finished {
        updated_seq = Some(seq.clone());
    }
}
```

- [ ] **Step 2: 验证编译通过**

Run: `cargo build -p vllm-core`
Expected: SUCCESS

- [ ] **Step 3: Commit**

```bash
git add crates/core/src/scheduler/engine.rs
git commit -m "fix(core): remove redundant queue_manager updates in update()"
```

---

### Task 3: 调整 preemption 逻辑

**Files:**
- Modify: `crates/core/src/scheduler/engine.rs:257-301`

- [ ] **Step 1: 分析 preemption 逻辑变化**

原有的 preemption 会将从 running 中取出的序列放回 queue_manager：

```rust
if let Some(running_seq) = self.running.iter().find(|s| s.id == seq.id) {
    self.queue_manager.enqueue_preempted(running_seq.clone());
    self.running.retain(|s| s.id != seq.id);
}
```

这个逻辑不需要改变，因为：
- 被 preemption 的序列来自 running
- 放回 queue_manager 是正确的行为

验证这一点：

- [ ] **Step 2: 运行测试验证 preemption 正常工作**

Run: `cargo test -p vllm-core --test integration`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add crates/core/src/scheduler/engine.rs
git commit -m "fix(core): adjust preemption logic for new data flow"
```

---

### Task 4: 最终验证

**Files:**
- None

- [ ] **Step 1: 运行所有测试**

Run: `cargo test -p vllm-core`
Expected: ALL PASS

- [ ] **Step 2: 运行 clippy**

Run: `cargo clippy -p vllm-core -- -D warnings`
Expected: NO WARNINGS

- [ ] **Step 3: 运行 fmt**

Run: `cargo fmt --all --check`
Expected: PASS

- [ ] **Step 4: Commit final**

```bash
git add -A
git commit -m "fix(core): resolve scheduler state consistency"
```
