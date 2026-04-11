# MambaBlock 权重加载实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 Qwen35Model 实现完整的 MambaBlock 权重加载，使其能从真实权重文件加载

**Architecture:** 为 MambaBlock (qwen3_5/ssm.rs) 添加 from_weights 方法，更新 Qwen35Model::from_weights 加载所有层权重

**Tech Stack:** Rust, Candle, vllm-model

---

## 文件结构

```
crates/model/src/qwen3_5/
├── ssm.rs          # 添加 MambaBlock::from_weights (已完成)
└── model.rs        # 更新 Qwen35Model::from_weights
```

---

### Task 1: 为 MambaBlock 添加 from_weights 方法 ✅ 已完成

- [x] 添加 from_weights 方法到 MambaBlock
- [x] 添加 HashMap import
- [x] 运行 clippy
- [x] 提交

---

### Task 2: 检查并添加必要的 imports

**Files:**
- Modify: `crates/model/src/qwen3_5/model.rs:1-10`

- [ ] **Step 1: 检查当前 imports**

查看当前文件头部，确认是否有：
```rust
use crate::qwen3_5::ssm::{MambaBlock, SSMConfig};
```

如果没有则添加。

- [ ] **Step 2: 运行 clippy 确认无新增警告**

Run: `cargo clippy -p vllm-model -- -D warnings`

- [ ] **Step 3: 提交**

```bash
git add crates/model/src/qwen3_5/model.rs
git commit -m "chore(qwen3_5): ensure MambaBlock imports"
```

---

### Task 3: 更新 embed_tokens 加载（支持 fallback）

**Files:**
- Modify: `crates/model/src/qwen3_5/model.rs:75-79`

- [ ] **Step 1: 修改 embed_tokens 加载逻辑**

当前代码：
```rust
// Load embed_tokens
if let Some(w) = weights.get("model.language_model.embed_tokens.weight") {
    model.embed_tokens = Embedding::new(w.clone(), w.dims()[1]);
    println!("Loaded embed_tokens");
}
```

修改为：
```rust
// Load embed_tokens (try multiple names)
if let Some(w) = weights.get("model.language_model.embed_tokens.weight") {
    model.embed_tokens = Embedding::new(w.clone(), w.dims()[1]);
    println!("Loaded embed_tokens");
} else if let Some(w) = weights.get("model.embed_tokens.weight") {
    model.embed_tokens = Embedding::new(w.clone(), w.dims()[1]);
    println!("Loaded embed_tokens (fallback)");
}
```

- [ ] **Step 2: 运行 clippy**

Run: `cargo clippy -p vllm-model -- -D warnings`

- [ ] **Step 3: 提交**

```bash
git add crates/model/src/qwen3_5/model.rs
git commit -m "feat(qwen3_5): improve embed_tokens loading with fallback"
```

---

### Task 4: 添加 MambaBlock 层加载循环

**Files:**
- Modify: `crates/model/src/qwen3_5/model.rs:81` (替换 TODO)

- [ ] **Step 1: 添加 MambaBlock 层加载代码**

在 `// TODO: Load layer weights...` 位置添加：

```rust
// Load MambaBlock layers
let num_layers = config.num_hidden_layers();
let hidden_size = config.hidden_size();
let d_state = 16; // Default from SSMConfig

for i in 0..num_layers {
    match MambaBlock::from_weights(hidden_size, d_state, i, &weights) {
        Ok(layer) => {
            model.layers[i] = layer;
            println!("Loaded MambaBlock layer {}", i);
        }
        Err(e) => {
            return Err(candle_core::Error::msg(format!(
                "Failed to load MambaBlock layer {}: {}",
                i, e
            )));
        }
    }
}
```

- [ ] **Step 2: 运行 clippy**

Run: `cargo clippy -p vllm-model -- -D warnings`

- [ ] **Step 3: 运行测试**

Run: `cargo test -p vllm-model -- qwen3_5`

- [ ] **Step 4: 提交**

```bash
git add crates/model/src/qwen3_5/model.rs
git commit -m "feat(qwen3_5): load MambaBlock layers in from_weights"
```

---

### Task 5: 添加 final norm 权重加载

**Files:**
- Modify: `crates/model/src/qwen3_5/model.rs`

- [ ] **Step 1: 在层加载循环后添加 final norm 加载**

在 `model.layers[i] = layer;` 之后添加：

```rust
// Load final norm
if let Some(w) = weights.get("model.norm.weight") {
    model.norm = candle_nn::LayerNorm::new(w.clone(), config.rms_norm_eps())?;
    println!("Loaded final norm");
} else {
    return Err(candle_core::Error::msg("Missing model.norm.weight"));
}
```

- [ ] **Step 2: 运行 clippy**

Run: `cargo clippy -p vllm-model -- -D warnings`

- [ ] **Step 3: 提交**

```bash
git add crates/model/src/qwen3_5/model.rs
git commit -m "feat(qwen3_5): load final norm weight"
```

---

### Task 6: 添加 lm_head 权重加载（支持 fallback 和 tie_word_embeddings）

**Files:**
- Modify: `crates/model/src/qwen3_5/model.rs`

- [ ] **Step 1: 在 final norm 加载后添加 lm_head 加载**

```rust
// Load lm_head (try multiple names)
let lm_head_w = weights
    .get("lm_head.weight")
    .or_else(|| weights.get("output.weight"))
    .cloned();

if let Some(w) = lm_head_w {
    model.lm_head = candle_nn::Linear::new(w, None)?;
    println!("Loaded lm_head");
} else if config.tie_word_embeddings() {
    // If tied, use embed_tokens weights
    println!("Using tied embeddings for lm_head");
} else {
    return Err(candle_core::Error::msg("Missing lm_head.weight or output.weight"));
}
```

- [ ] **Step 2: 运行 clippy**

Run: `cargo clippy -p vllm-model -- -D warnings`

- [ ] **Step 3: 运行测试**

Run: `cargo test -p vllm-model -- qwen3_5`

- [ ] **Step 4: 提交**

```bash
git add crates/model/src/qwen3_5/model.rs
git commit -m "feat(qwen3_5): load lm_head weight with fallback"
```

---

### Task 7: 移除 TODO 注释

**Files:**
- Modify: `crates/model/src/qwen3_5/model.rs`

- [ ] **Step 1: 删除 TODO 注释**

删除 `// TODO: Load layer weights when from_weights is implemented for MambaBlock`

- [ ] **Step 2: 运行 clippy**

Run: `cargo clippy -p vllm-model -- -D warnings`

- [ ] **Step 3: 提交**

```bash
git add crates/model/src/qwen3_5/model.rs
git commit -m "chore(qwen3_5): remove TODO after weight loading complete"
```

---

### Task 8: 验证编译

**Files:**
- Build: `crates/model/`

- [ ] **Step 1: 运行完整编译**

Run: `cargo build -p vllm-model`

- [ ] **Step 2: 运行 clippy**

Run: `cargo clippy -p vllm-model -- -D warnings`

- [ ] **Step 3: 提交**

```bash
git add -A && git commit -m "build: verify MambaBlock weight loading compiles"
```

---

### Task 9: 运行单元测试

**Files:**
- Test: `crates/model/src/qwen3_5/`

- [ ] **Step 1: 运行 qwen3_5 相关测试**

Run: `cargo test -p vllm-model -- qwen3_5`

- [ ] **Step 2: 运行所有 model 测试**

Run: `cargo test -p vllm-model`

- [ ] **Step 3: 提交**

```bash
git add -A && git commit -m "test(qwen3_5): run tests for weight loading"
```

---

### Task 10: 最终验证

**Files:**
- Full: `crates/model/`, `crates/core/`, `crates/server/`

- [ ] **Step 1: 运行 workspace 编译**

Run: `cargo build --workspace`

- [ ] **Step 2: 运行 workspace clippy**

Run: `cargo clippy --workspace -- -D warnings`

- [ ] **Step 3: 运行全部测试**

Run: `just test`

- [ ] **Step 4: 提交**

```bash
git add -A && git commit -m "chore: final verification for MambaBlock weight loading"
```

---

### Task 11: 更新 CHANGELOG

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: 添加 changelog 条目**

```markdown
## [Unreleased]

### Added
- MambaBlock weight loading for Qwen3.5 Mamba models
- Support for loading all Mamba layer weights (in_proj, x_proj, A_log, D, conv1d, out_proj, norm)
- Fallback support for embed_tokens and lm_head weights
```

- [ ] **Step 2: 提交**

```bash
git add CHANGELOG.md
git commit -m "chore: update changelog for MambaBlock weight loading"
```
