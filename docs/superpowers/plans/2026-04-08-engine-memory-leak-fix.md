# Engine 内存泄漏修复实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 Engine 模块中两处内存泄漏：1) finished Vec 永远不清理导致累积；2) 取消请求时不清理 response_txs

**Architecture:** 在 SchedulerEngine 中添加 clear_finished 方法，在 Engine 的 step 和 step_speculative 中调用清理逻辑。同时在 cancel_request 时同步清理 response_txs

**Tech Stack:** Rust

---

## 问题分析

### 问题 1: finished Vec 永远不清理

- **位置**: `crates/core/src/scheduler/engine.rs:81` (字段定义), line 150 (push)
- **现象**: 每当 sequence 完成，被添加到 `self.finished` Vec，但从未被清空
- **影响**: 长时间运行后内存持续增长

### 问题 2: 取消请求时不清理 response_txs

- **位置**: `crates/core/src/scheduler/engine.rs:164-174` (cancel_request 方法)
- **现象**: `Engine.response_txs` 在请求完成后会清理 (batch.rs:35)，但取消请求时不会
- **影响**: 取消的请求对应的 channel 永远留在内存中

---

## 文件清单

| 文件 | 操作 |
|------|------|
| `crates/core/src/scheduler/engine.rs` | 修改 - 添加 clear_finished 方法 |
| `crates/core/src/engine.rs` | 修改 - 添加 cancel_request 方法供外部调用 |
| `crates/core/src/scheduler/batch.rs` | 修改 - 调用 clear_finished |
| `crates/core/src/engine/speculative.rs` | 修改 - 调用 clear_finished |
| `crates/core/tests/integration.rs` | 测试 - 验证内存泄漏修复 |

---

## 任务分解

### Task 1: 在 SchedulerEngine 中添加 clear_finished 方法

**Files:**
- Modify: `crates/core/src/scheduler/engine.rs:560-563`

- [ ] **Step 1: 添加 clear_finished 方法**

在 `finished_sequences` 方法后添加：

```rust
pub fn clear_finished(&mut self) {
    self.finished.clear();
}
```

- [ ] **Step 2: 验证编译通过**

Run: `cargo build -p vllm-core`
Expected: SUCCESS

- [ ] **Step 3: Commit**

```bash
git add crates/core/src/scheduler/engine.rs
git commit -m "fix(core): add clear_finished method to SchedulerEngine"
```

---

### Task 2: 在 Engine 中暴露 cancel_request 方法并同步清理 response_txs

**Files:**
- Modify: `crates/core/src/engine.rs:86-92`

- [ ] **Step 1: 添加 cancel_request 公共方法**

在 `Engine<M>` impl 块中添加：

```rust
pub fn cancel_request(&mut self, seq_id: SeqId) -> bool {
    let canceled = self.scheduler.cancel_request(seq_id);
    if canceled {
        self.response_txs.remove(&seq_id);
    }
    canceled
}
```

- [ ] **Step 2: 验证编译通过**

Run: `cargo build -p vllm-core`
Expected: SUCCESS

- [ ] **Step 3: Commit**

```bash
git add crates/core/src/engine.rs
git commit -m "fix(core): add cancel_request that cleans response_txs"
```

---

### Task 3: 在 batch.rs 的 step 方法中调用 clear_finished

**Files:**
- Modify: `crates/core/src/scheduler/batch.rs:34-36`

- [ ] **Step 1: 修改 step 方法**

将：
```rust
for seq in self.scheduler.finished_sequences() {
    self.response_txs.remove(&seq.id);
}
```

修改为：
```rust
let finished = self.scheduler.finished_sequences();
for seq in &finished {
    self.response_txs.remove(&seq.id);
}
self.scheduler.clear_finished();
```

- [ ] **Step 2: 验证编译通过**

Run: `cargo build -p vllm-core`
Expected: SUCCESS

- [ ] **Step 3: 运行相关测试**

Run: `cargo test -p vllm-core --test integration -- test_engine_streaming --nocapture`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add crates/core/src/scheduler/batch.rs
git commit -m "fix(core): clear finished sequences after cleanup"
```

---

### Task 4: 在 speculative.rs 的 step_speculative 方法中调用 clear_finished

**Files:**
- Modify: `crates/core/src/engine/speculative.rs:29-31`

- [ ] **Step 1: 修改 step_speculative 方法**

将：
```rust
for seq in self.scheduler.finished_sequences() {
    self.response_txs.remove(&seq.id);
}
```

修改为：
```rust
let finished = self.scheduler.finished_sequences();
for seq in &finished {
    self.response_txs.remove(&seq.id);
}
self.scheduler.clear_finished();
```

- [ ] **Step 2: 验证编译通过**

Run: `cargo build -p vllm-core`
Expected: SUCCESS

- [ ] **Step 3: Commit**

```bash
git add crates/core/src/engine/speculative.rs
git commit -m "fix(core): clear finished sequences in speculative mode"
```

---

### Task 5: 添加集成测试验证内存泄漏修复

**Files:**
- Modify: `crates/core/tests/integration.rs`

- [ ] **Step 1: 添加测试验证 finished Vec 清理**

在 `test_request_cancellation` 后添加：

```rust
#[test]
fn test_finished_sequences_cleared() {
    let config = SchedulerConfig {
        max_num_seqs: 10,
        max_num_batched_tokens: 100,
        max_consecutive_decode: 10,
        enable_pd_separation: false,
        prefill_chunk_size: 512,
        decode_preference_ratio: 0.7,
        enable_priority_scheduling: false,
        enable_dynamic_batching: false,
        min_batch_size: 1,
        max_batch_size: 256,
    };
    let mut engine = Engine::with_config(IncrementModel, IncrementModel, config, 4, 1024);

    let (tx, _rx) = mpsc::channel(64);
    engine.add_request(Request::new(1, vec![10, 20], 2), tx);

    // Complete the request
    for _ in 0..3 {
        engine.step().unwrap();
    }

    assert!(!engine.has_pending());
    
    // Verify finished is cleared
    assert!(engine.scheduler.finished_sequences().is_empty());
}
```

- [ ] **Step 4: 运行测试验证**

Run: `cargo test -p vllm-core --test integration -- test_finished_sequences_cleared --nocapture`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/core/tests/integration.rs
git commit -m "test(core): add test for finished sequences cleanup"
```

---

### Task 6: 最终验证

**Files:**
- None

- [ ] **Step 1: 运行所有相关测试**

Run: `cargo test -p vllm-core --test integration`
Expected: ALL PASS (including new tests: test_finished_sequences_cleared, test_cancel_request_cleans_response_txs)

- [ ] **Step 2: 运行 clippy**

Run: `cargo clippy -p vllm-core -- -D warnings`
Expected: NO WARNINGS

- [ ] **Step 3: 运行 fmt**

Run: `cargo fmt --all --check`
Expected: PASS

- [ ] **Step 4: Commit final**

```bash
git add -A
git commit -m "fix(core): resolve memory leaks in Engine module"
```
