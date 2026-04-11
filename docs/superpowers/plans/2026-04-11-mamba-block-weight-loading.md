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

### Task 2: 更新 Qwen35Model::from_weights 加载所有权重

这是核心任务，一次性实现所有权重加载：

**Files:**
- Modify: `crates/model/src/qwen3_5/model.rs:67-84`

- [ ] **Step 1: 替换整个 from_weights 方法体**

将现有的 `from_weights` 方法替换为完整实现：

```rust
pub fn from_weights(
    config: Qwen3Config,
    device: Device,
    weights: HashMap<String, Tensor>,
    num_kv_blocks: usize,
) -> CandleResult<Self> {
    let mut model = Self::new(config.clone(), device.clone(), num_kv_blocks)?;

    // Load embed_tokens (try multiple names)
    if let Some(w) = weights.get("model.language_model.embed_tokens.weight") {
        model.embed_tokens = Embedding::new(w.clone(), w.dims()[1]);
        println!("Loaded embed_tokens");
    } else if let Some(w) = weights.get("model.embed_tokens.weight") {
        model.embed_tokens = Embedding::new(w.clone(), w.dims()[1]);
        println!("Loaded embed_tokens (fallback)");
    }

    // Load MambaBlock layers
    let num_layers = config.num_hidden_layers();
    let hidden_size = config.hidden_size();
    let d_state = 16;

    for i in 0..num_layers {
        let layer = MambaBlock::from_weights(hidden_size, d_state, i, &weights)
            .map_err(|e| candle_core::Error::msg(format!("Failed to load layer {}: {}", i, e)))?;
        model.layers[i] = layer;
        println!("Loaded MambaBlock layer {}", i);
    }

    // Load final norm
    if let Some(w) = weights.get("model.norm.weight") {
        model.norm = candle_nn::LayerNorm::new(w.clone(), config.rms_norm_eps())?;
        println!("Loaded final norm");
    } else {
        return Err(candle_core::Error::msg("Missing model.norm.weight"));
    }

    // Load lm_head (try multiple names, or use tied embeddings)
    let lm_head_w = weights
        .get("lm_head.weight")
        .or_else(|| weights.get("output.weight"))
        .cloned();

    if let Some(w) = lm_head_w {
        model.lm_head = candle_nn::Linear::new(w, None)?;
        println!("Loaded lm_head");
    } else if config.tie_word_embeddings() {
        println!("Using tied embeddings for lm_head");
    } else {
        return Err(candle_core::Error::msg("Missing lm_head.weight"));
    }

    // 删除 TODO 注释: // TODO: Load layer weights when from_weights is implemented for MambaBlock

    Ok(model)
}
```

- [ ] **Step 2: 运行 clippy 检查**

Run: `cargo clippy -p vllm-model -- -D warnings`

- [ ] **Step 3: 提交**

```bash
git add crates/model/src/qwen3_5/model.rs
git commit -m "feat(qwen3_5): load all MambaBlock weights in from_weights"
```

---

### Task 3: 测试验证

- [ ] **Step 1: 编译验证**

Run: `cargo build -p vllm-model`

- [ ] **Step 2: 运行测试**

Run: `cargo test -p vllm-model`

- [ ] **Step 3: 提交**

```bash
git add -A && git commit -m "test(qwen3_5): verify MambaBlock weight loading"
```

---

### Task 4: 最终验证与更新 CHANGELOG

- [ ] **Step 1: Workspace 编译**

Run: `cargo build --workspace`

- [ ] **Step 2: Workspace clippy**

Run: `cargo clippy --workspace -- -D warnings`

- [ ] **Step 3: 运行全部测试**

Run: `just test`

- [ ] **Step 4: 更新 CHANGELOG**

```markdown
## [Unreleased]

### Added
- MambaBlock weight loading for Qwen3.5 Mamba models
```

- [ ] **Step 5: 提交**

```bash
git add CHANGELOG.md
git commit -m "chore: final verification and update changelog"
```
