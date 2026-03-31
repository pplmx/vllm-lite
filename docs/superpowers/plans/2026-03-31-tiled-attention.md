# Tiled Attention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development

**Goal:** 实现 Tiled Attention，减小显存占用，优化 decode 和 prefill 性能

**Architecture:** 分块计算 attention，减少显存从 O(n²) 到 O(tile_size × n)

**Tech Stack:** Rust, Candle

---

## Task 1: 添加 AttentionConfig

**Files:**
- Modify: `crates/model/src/qwen3/attention.rs`

- [ ] **Step 1: 添加配置结构**

```rust
pub struct AttentionConfig {
    pub tile_size: Option<usize>,  // None = 使用标准 attention
    pub use_fused: bool,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            tile_size: None,
            use_fused: true,
        }
    }
}
```

- [ ] **Step 2: 添加到 GqaAttention**

```rust
pub struct GqaAttention {
    // ... existing fields
    config: AttentionConfig,
}
```

- [ ] **Step 3: 提交**

```bash
git commit -m "feat(model): add AttentionConfig for tiled attention"
```

---

## Task 2: 实现 tiled_attention

**Files:**
- Modify: `crates/model/src/qwen3/attention.rs`

- [ ] **Step 1: 实现核心算法**

```rust
fn tiled_attention(
    &self,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    seq_len: usize,
) -> Result<Tensor> {
    let tile_size = self.config.tile_size.unwrap_or(16);
    
    // 分块计算
    // ...
}
```

- [ ] **Step 2: 实现 causal_mask_tile**

```rust
fn causal_mask_tile(
    batch_size: usize,
    num_heads: usize,
    tile_len: usize,
    device: &Device,
) -> Result<Tensor> {
    // 只对 tile 内部做 causal mask
    // ...
}
```

- [ ] **Step 3: 提交**

```bash
git commit -m "feat(model): implement tiled attention kernel"
```

---

## Task 3: 自动选择逻辑

**Files:**
- Modify: `crates/model/src/qwen3/attention.rs`

- [ ] **Step 1: 添加 forward 方法**

```rust
pub fn forward(
    &self,
    x: &Tensor,
    kv_cache: &mut PagedKvCache,
    // ...
) -> Result<Tensor> {
    let tile_size = self.config.tile_size.unwrap_or(16);
    
    // 短序列用标准，长序列用 tiled
    if seq_len > tile_size {
        self.tiled_attention(...)
    } else {
        self.standard_attention(...)
    }
}
```

- [ ] **Step 2: 提交**

```bash
git commit -m "feat(model): add automatic attention strategy selection"
```

---

## Task 4: 测试验证

**Files:**
- Add: `crates/model/tests/tiled_attention.rs`

- [ ] **Step 1: 添加测试**

```rust
#[test]
fn test_tiled_attention_short_seq() {
    // 短序列用标准 attention
}

#[test]
fn test_tiled_attention_long_seq() {
    // 长序列用 tiled，与标准结果对比
}
```

- [ ] **Step 2: 运行测试**

```bash
cargo test -p vllm-model
```

- [ ] **Step 3: 提交**

```bash
git commit -m "test(model): add tiled attention tests"
```

---

## Verification Checklist

- [ ] AttentionConfig 添加正确
- [ ] tiled_attention 实现正确
- [ ] 自动选择逻辑工作
- [ ] 测试通过
- [ ] Clippy clean