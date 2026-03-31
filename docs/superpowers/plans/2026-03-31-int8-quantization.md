# INT8 Quantization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development

**Goal:** 实现 INT8 量化，减少显存 50%，保持精度 < 1% 损失

**Tech Stack:** Rust, Candle

---

## Task 1: 量化工具函数

**Files:**

- Create: `crates/model/src/quantize.rs`

- [ ] **Step 1: 添加 QuantizedTensor 结构**

```rust
pub struct QuantizedTensor {
    pub data: Vec<i8>,
    pub scale: f32,
    pub shape: Vec<usize>,
}
```

- [ ] **Step 2: 实现 quantize/dequantize**

```rust
pub fn quantize(tensor: &Tensor) -> Result<QuantizedTensor> { ... }
pub fn dequantize(quant: &QuantizedTensor) -> Result<Tensor> { ... }
```

- [ ] **Step 3: 提交**

```bash
git commit -m "feat(model): add quantization utilities"
```

---

## Task 2: 模型权重量化

**Files:**

- Modify: `crates/model/src/qwen3/model.rs`

- [ ] **Step 1: 添加量化字段**

```rust
pub struct Qwen3Model {
    // 原始权重
    embed_tokens: Embedding,
    // 量化权重 (可选)
    quantized_weights: Option<HashMap<String, QuantizedTensor>>,
}
```

- [ ] **Step 2: 实现量化加载**

```rust
pub fn from_weights_quantized(
    config: Qwen3Config,
    device: Device,
    weights: HashMap<String, Tensor>,
) -> Result<Self> {
    // 加载并量化权重
    // ...
}
```

- [ ] **Step 3: 提交**

```bash
git commit -m "feat(model): add quantized weight loading"
```

---

## Task 3: KV Cache 量化

**Files:**

- Modify: `crates/model/src/kv_cache.rs`

- [ ] **Step 1: 添加量化支持**

```rust
pub struct PagedKvCache {
    key_cache: Vec<Tensor>,     // FP16
    value_cache: Vec<Tensor>,   // FP16
    quantized: bool,
    scales: Vec<f32>,
}
```

- [ ] **Step 2: 实现量化读写**

```rust
pub fn write_kv_quantized(...) { ... }
pub fn read_kv_dequantized(...) { ... }
```

- [ ] **Step 3: 提交**

```bash
git commit -m "feat(model): add quantized KV cache support"
```

---

## Task 4: 校准脚本

**Files:**

- Create: `crates/model/examples/calibrate.rs`

- [ ] **Step 1: 实现校准**

```rust
fn calibrate(model: &mut Qwen3Model, prompts: &[&str]) {
    // 运行推理收集统计
    // 计算 scales
}
```

- [ ] **Step 2: 提交**

```bash
git commit -m "feat(model): add quantization calibration"
```

---

## Task 5: 测试

**Files:**

- Add: `crates/model/tests/quantize.rs`

- [ ] **Step 1: 添加测试**

```rust
#[test]
fn test_quantization_accuracy() { ... }

#[test]
fn test_memory_savings() { ... }
```

- [ ] **Step 2: 提交**

```bash
git commit -m "test(model): add quantization tests"
```

---

## Verification Checklist

- [ ] 量化精度 < 1% 损失
- [ ] 显存节省 > 40%
- [ ] 测试通过
