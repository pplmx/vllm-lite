# BlockId 类型统一计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 统一 BlockId 类型定义，消除重复定义

**Tech Stack:** Rust

---

## 问题分析

### 当前问题
1. BlockId 在 `crates/core/src/types.rs` 定义为 `usize`
2. BlockId 在 `crates/model/src/qwen3/model.rs` 也定义了一次
3. 重复定义可能导致不一致

### 目标
- 在 `vllm_traits` 中统一定义 BlockId
- 所有其他 crate 引用 vllm_traits 中的定义
- 删除重复的定义

---

## 文件清单

| 文件 | 操作 |
|------|------|
| `crates/traits/src/types.rs` | 添加 BlockId 定义 |
| `crates/core/src/types.rs` | 删除本地 BlockId 定义 |
| `crates/model/src/qwen3/model.rs` | 删除本地 BlockId 定义 |
| `crates/model/src/llama/mod.rs` | 检查是否有重复 |
| `crates/model/src/mistral/model.rs` | 检查是否有重复 |
| 其他 model 文件 | 检查并更新引用 |

---

## 任务分解

### Task 1: 在 vllm_traits 中添加 BlockId

**Files:**
- Modify: `crates/traits/src/types.rs`

- [ ] **Step 1: 添加 BlockId 定义**

```rust
pub const BLOCK_SIZE: usize = 16;
pub type BlockId = usize;  // 添加这行
pub type TokenId = u32;
pub type SeqId = u64;
```

- [ ] **Step 2: 验证编译**

Run: `cargo build -p vllm-traits`

- [ ] **Step 3: Commit**

```bash
git add crates/traits/src/types.rs
git commit -m "feat(traits): add BlockId type definition"
```

---

### Task 2: 更新 core/types.rs

**Files:**
- Modify: `crates/core/src/types.rs`

- [ ] **Step 1: 移除本地 BlockId 定义**

将:
```rust
pub use vllm_traits::BLOCK_SIZE;
pub type BlockId = usize;
```

改为:
```rust
pub use vllm_traits::{BLOCK_SIZE, BlockId};
```

- [ ] **Step 2: 验证编译**

Run: `cargo build -p vllm-core`

- [ ] **Step 3: 运行测试**

Run: `cargo test -p vllm-core`

- [ ] **Step 4: Commit**

```bash
git add crates/core/src/types.rs
git commit -m "refactor(core): use BlockId from vllm_traits"
```

---

### Task 3: 更新 model/qwen3

**Files:**
- Modify: `crates/model/src/qwen3/model.rs`

- [ ] **Step 1: 检查并移除本地定义**

查找是否有 `pub type BlockId = usize;`，如果有，删除它

- [ ] **Step 2: 确保引用 vllm_traits**

确保文件顶部有:
```rust
use vllm_traits::BlockId;
```

- [ ] **Step 3: 验证编译**

Run: `cargo build -p vllm-model`

- [ ] **Step 4: Commit**

```bash
git add crates/model/src/qwen3/model.rs
git commit -m "refactor(model): use BlockId from vllm_traits"
```

---

### Task 4: 检查其他 model 文件

**Files:**
- 检查其他 model 文件是否有重复定义

- [ ] **Step 1: 搜索重复定义**

Run: `grep -r "type BlockId" crates/model/`

- [ ] **Step 2: 修复任何重复**

- [ ] **Step 3: 验证编译**

Run: `cargo build -p vllm-model`

- [ ] **Step 4: Commit**

```bash
git add crates/model/
git commit -m "refactor(model): unify BlockId type across all models"
```

---

### Task 5: 最终验证

**Files:**
- None

- [ ] **Step 1: 运行完整测试**

Run: `cargo test --workspace`

- [ ] **Step 2: 运行 clippy**

Run: `cargo clippy --workspace -- -D warnings`

- [ ] **Step 3: 运行 fmt**

Run: `cargo fmt --all --check`

- [ ] **Step 4: Commit final**

```bash
git add -A
git commit -m "refactor: complete BlockId type unification"
```
