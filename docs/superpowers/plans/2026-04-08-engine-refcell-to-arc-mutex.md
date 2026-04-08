# Engine 模型状态管理重构计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 Engine 中的 `RefCell<Box<M>>` 改为 `Arc<Mutex<dyn ModelBackend>>`，支持模型跨线程共享

**Architecture:** 使用 Arc 包装 Mutex 包裹的模型，支持多线程并发访问

**Tech Stack:** Rust (std::sync::Arc, std::sync::Mutex)

---

## 问题分析

### 当前实现
- `target_model: RefCell<Box<M>>`
- `draft_model: RefCell<Box<M>>`
- 限制：模型无法跨线程共享

### 目标
- `target_model: Arc<Mutex<dyn ModelBackend>>`
- `draft_model: Arc<Mutex<dyn ModelBackend>>`
- 好处：支持多线程共享，为未来并行化做准备

---

## 文件清单

| 文件 | 操作 |
|------|------|
| `crates/core/src/engine.rs` | 修改 - RefCell → Arc<Mutex> |
| `crates/core/src/scheduler/batch.rs` | 修改 - borrow_mut → lock |
| `crates/core/src/engine/speculative.rs` | 修改 - borrow_mut → lock |
| `crates/core/tests/integration.rs` | 测试 - 验证功能正常 |

---

## 任务分解

### Task 1: 修改 Engine 结构体定义和构造方法

**Files:**
- Modify: `crates/core/src/engine.rs`

- [ ] **Step 1: 添加 use 语句，移除不需要的 import**

将：
```rust
use std::cell::RefCell;
```

改为：
```rust
use std::sync::{Arc, Mutex};
```

- [ ] **Step 2: 修改结构体字段**

将：
```rust
pub target_model: RefCell<Box<M>>,
pub draft_model: RefCell<Box<M>>,
```

改为：
```rust
pub target_model: Arc<Mutex<dyn ModelBackend>>,
pub draft_model: Arc<Mutex<dyn ModelBackend>>,
```

- [ ] **Step 3: 添加 'static 约束到 impl 块**

将：
```rust
impl<M: ModelBackend> Engine<M> {
```

改为：
```rust
impl<M: ModelBackend + 'static> Engine<M> {
```

- [ ] **Step 4: 修改 with_config 方法**

将内部实现改为：
```rust
pub fn with_config(
    target_model: M,
    draft_model: M,
    config: SchedulerConfig,
    max_draft_tokens: usize,
    num_kv_blocks: usize,
) -> Self {
    let max_seqs = config.max_num_seqs;
    Self {
        scheduler: SchedulerEngine::new(config, num_kv_blocks),
        target_model: Arc::new(Mutex::new(target_model)),
        draft_model: Arc::new(Mutex::new(draft_model)),
        max_draft_tokens,
        speculative_mode: false,
        error_count: 0,
        last_error: None,
        metrics: MetricsCollector::new(),
        response_txs: HashMap::with_capacity(max_seqs),
        sleep_policy: SleepPolicy::default(),
    }
}
```

- [ ] **Step 5: 验证编译通过**

Run: `cargo build -p vllm-core 2>&1 | head -30`
Expected: COMPILE ERROR (expected, will fix in subsequent tasks)

- [ ] **Step 6: Commit**

```bash
git add crates/core/src/engine.rs
git commit -m "refactor(core): change model fields from RefCell to Arc<Mutex>"
```

---

### Task 3: 修改 batch.rs 中的模型调用

**Files:**
- Modify: `crates/core/src/scheduler/batch.rs:12`

- [ ] **Step 1: 修改 forward 调用**

将：
```rust
let output = self.target_model.borrow_mut().forward(
```

改为：
```rust
let output = self.target_model.lock().unwrap().forward(
```

- [ ] **Step 2: 添加 Mutex import**

在文件顶部添加：
```rust
use std::sync::Mutex;
```

- [ ] **Step 3: 验证编译**

Run: `cargo build -p vllm-core`
Expected: SUCCESS

- [ ] **Step 4: 运行测试**

Run: `cargo test -p vllm-core --test integration -- streaming`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/core/src/scheduler/batch.rs
git commit -m "refactor(core): update batch.rs to use Mutex"
```

---

### Task 4: 修改 speculative.rs 中的模型调用

**Files:**
- Modify: `crates/core/src/engine/speculative.rs`

- [ ] **Step 1: 修改 draft_model 调用 (line 64)**

将：
```rust
let output = self.draft_model.borrow_mut().forward(
```

改为：
```rust
let output = self.draft_model.lock().unwrap().forward(
```

- [ ] **Step 2: 修改 target_model 调用 (line 94, 118)**

将所有：
```rust
self.target_model.borrow_mut().forward(
```

改为：
```rust
self.target_model.lock().unwrap().forward(
```

- [ ] **Step 3: 添加 Mutex import**

在文件顶部添加：
```rust
use std::sync::Mutex;
```

- [ ] **Step 4: 验证编译**

Run: `cargo build -p vllm-core`
Expected: SUCCESS

- [ ] **Step 5: Commit**

```bash
git add crates/core/src/engine/speculative.rs
git commit -m "refactor(core): update speculative.rs to use Mutex"
```

---

### Task 5: 修改 engine.rs 中的模型调用

**Files:**
- Modify: `crates/core/src/engine.rs`

- [ ] **Step 1: 修改 forward_logits 调用 (line ~216)**

将：
```rust
let logits = self.target_model.borrow_mut().forward_logits(
```

改为：
```rust
let logits = self.target_model.lock().unwrap().forward_logits(
```

- [ ] **Step 2: 修改 embed 调用 (line ~130)**

找到 `EngineMessage::GetEmbeddings` 处理：
```rust
match self
    .target_model
    .borrow_mut()
    .embed(&input_tokens, &positions)
```

改为：
```rust
match self
    .target_model
    .lock()
    .unwrap()
    .embed(&input_tokens, &positions)
```

- [ ] **Step 3: 验证编译**

Run: `cargo build -p vllm-core`
Expected: SUCCESS

- [ ] **Step 4: Commit**

```bash
git add crates/core/src/engine.rs
git commit -m "refactor(core): update engine.rs model calls to use Mutex"
```

---

### Task 6: 最终验证

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

- [ ] **Step 4: 删除不再需要的 RefCell import**

检查并删除：
```rust
use std::cell::RefCell;
```

- [ ] **Step 5: Commit final**

```bash
git add -A
git commit -m "refactor(core): complete RefCell to Arc<Mutex> migration"
```
