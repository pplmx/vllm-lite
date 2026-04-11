# MambaBlock 权重加载实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 Qwen35Model 实现完整的 MambaBlock 权重加载，使其能从真实权重文件加载

**Architecture:** 为 MambaBlock (qwen3_5/ssm.rs) 添加 from_weights 方法，更新 Qwen35Model::from_weights 加载所有层权重

**Tech Stack:** Rust, Candle, vllm-model

---

## 文件结构

```
crates/model/src/qwen3_5/
├── ssm.rs          # 添加 MambaBlock::from_weights
└── model.rs        # 更新 Qwen35Model::from_weights
```

---

### Task 1: 为 MambaBlock 添加 from_weights 方法

**Files:**
- Modify: `crates/model/src/qwen3_5/ssm.rs:103-196`

- [ ] **Step 1: 在 MambaBlock impl 末尾添加 from_weights 方法**

在 `impl MambaBlock {` 块中添加：

```rust
pub fn from_weights(
    d_model: usize,
    d_state: usize,
    layer_idx: usize,
    weights: &HashMap<String, Tensor>,
) -> Result<Self> {
    let config = SSMConfig {
        d_model,
        d_state,
        d_conv: 4,
        expand: 2,
    };
    let d_inner = config.d_inner();

    let in_proj_w = weights
        .get(&format!("model.layers.{}.mamba.in_proj.weight", layer_idx))
        .cloned()
        .ok_or_else(|| Error::msg(format!("Missing in_proj weight for layer {}", layer_idx)))?;
    let in_proj = candle_nn::Linear::new(in_proj_w, None);

    let x_proj_w = weights
        .get(&format!("model.layers.{}.mamba.x_proj.weight", layer_idx))
        .cloned()
        .ok_or_else(|| Error::msg(format!("Missing x_proj weight for layer {}", layer_idx)))?;
    let x_proj = candle_nn::Linear::new(x_proj_w, None);

    let a_log_w = weights
        .get(&format!("model.layers.{}.mamba.A_log.weight", layer_idx))
        .cloned()
        .ok_or_else(|| Error::msg(format!("Missing A_log weight for layer {}", layer_idx)))?;
    let a_log = candle_nn::Linear::new(a_log_w, None);

    let d_w = weights
        .get(&format!("model.layers.{}.mamba.D.weight", layer_idx))
        .cloned()
        .ok_or_else(|| Error::msg(format!("Missing D weight for layer {}", layer_idx)))?;
    let d = candle_nn::Linear::new(d_w, None);

    let conv_w = weights
        .get(&format!("model.layers.{}.mamba.conv1d.weight", layer_idx))
        .cloned()
        .ok_or_else(|| Error::msg(format!("Missing conv1d weight for layer {}", layer_idx)))?;
    let conv_cfg = candle_nn::Conv1dConfig {
        padding: config.d_conv - 1,
        ..Default::default()
    };
    let conv = candle_nn::conv1d(d_inner, d_inner, config.d_conv, conv_cfg, conv_w)?;

    let out_proj_w = weights
        .get(&format!("model.layers.{}.mamba.out_proj.weight", layer_idx))
        .cloned()
        .ok_or_else(|| Error::msg(format!("Missing out_proj weight for layer {}", layer_idx)))?;
    let out_proj = candle_nn::Linear::new(out_proj_w, None);

    let norm_w = weights
        .get(&format!("model.layers.{}.mamba.norm.weight", layer_idx))
        .cloned()
        .ok_or_else(|| Error::msg(format!("Missing norm weight for layer {}", layer_idx)))?;
    let norm = candle_nn::LayerNorm::new(norm_w, 1e-5);

    let ssm = SSMLayer {
        x_proj,
        a_log,
        d,
        conv,
        d_inner,
        d_state,
    };

    Ok(Self {
        input_proj: in_proj,
        ssm,
        output_proj: out_proj,
        norm,
    })
}
```

- [ ] **Step 2: 添加 HashMap import**

在文件顶部添加：
```rust
use std::collections::HashMap;
```

- [ ] **Step 3: 运行 clippy 检查**

Run: `cargo clippy -p vllm-model -- -D warnings`

- [ ] **Step 4: 提交**

```bash
git add crates/model/src/qwen3_5/ssm.rs
git commit -m "feat(qwen3_5): add MambaBlock::from_weights method"
```

---

### Task 2: 更新 Qwen35Model::from_weights 加载所有层

**Files:**
- Modify: `crates/model/src/qwen3_5/model.rs:67-84`

- [ ] **Step 1: 更新 from_weights 方法**

替换现有的 `from_weights` 方法体：

```rust
pub fn from_weights(
    config: Qwen3Config,
    device: Device,
    weights: HashMap<String, Tensor>,
    num_kv_blocks: usize,
) -> CandleResult<Self> {
    let mut model = Self::new(config.clone(), device.clone(), num_kv_blocks)?;

    // Load embed_tokens
    if let Some(w) = weights.get("model.language_model.embed_tokens.weight") {
        model.embed_tokens = Embedding::new(w.clone(), w.dims()[1]);
        println!("Loaded embed_tokens");
    } else if let Some(w) = weights.get("model.embed_tokens.weight") {
        model.embed_tokens = Embedding::new(w.clone(), w.dims()[1]);
        println!("Loaded embed_tokens");
    }

    // Load MambaBlock layers
    let num_layers = config.num_hidden_layers();
    let hidden_size = config.hidden_size();
    let d_state = 16; // Default from SSMConfig
    
    for i in 0..num_layers {
        let layer = MambaBlock::from_weights(hidden_size, d_state, i, &weights)?;
        model.layers[i] = layer;
        println!("Loaded MambaBlock layer {}", i);
    }

    // Load final norm
    if let Some(w) = weights.get("model.norm.weight") {
        model.norm = candle_nn::LayerNorm::new(w.clone(), config.rms_norm_eps())?;
        println!("Loaded final norm");
    }

    // Load lm_head (try multiple names)
    let lm_head_w = weights
        .get("lm_head.weight")
        .or_else(|| weights.get("output.weight"))
        .cloned();
    
    if let Some(w) = lm_head_w {
        model.lm_head = candle_nn::Linear::new(w, None)?;
        println!("Loaded lm_head");
    }

    Ok(model)
}
```

- [ ] **Step 2: 运行 clippy 检查**

Run: `cargo clippy -p vllm-model -- -D warnings`

- [ ] **Step 3: 运行测试**

Run: `cargo test -p vllm-model -- qwen3_5`

- [ ] **Step 4: 提交**

```bash
git add crates/model/src/qwen3_5/model.rs
git commit -m "feat(qwen3_5): load all MambaBlock weights in from_weights"
```

---

### Task 3: 测试验证（可选，需要真实权重）

**Files:**
- Test: `crates/model/src/qwen3_5/`

- [ ] **Step 1: 验证编译**

Run: `cargo build -p vllm-model`

- [ ] **Step 2: 运行现有测试**

Run: `cargo test -p vllm-model`

- [ ] **Step 3: 提交**

```bash
git add -A && git commit -m "test(qwen3_5): verify MambaBlock weight loading"
```
