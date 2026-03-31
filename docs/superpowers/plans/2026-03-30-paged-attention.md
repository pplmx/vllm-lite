# Paged Attention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现真正的 Paged Attention - 让 KV cache 写入 GPU paged memory，decode 阶段复用缓存的 KV

**Architecture:**

- 增强 PagedKvCache 添加 write_kv/read_kv 方法
- 修改 GqaAttention 支持 prefill（写 cache）和 decode（读 cache）
- 修改 Qwen3Model/TransformerBlock 封装 prefill/decode 路径
- 修改 Engine/Scheduler 区分 prefill/decode 调用

**Tech Stack:** Rust, Candle, vllm-core, vllm-model

---

## File Structure

```text
crates/model/src/kv_cache.rs     # 增强: write_kv, read_kv
crates/model/src/qwen3/attention.rs  # 新增: forward_prefill, forward_decode
crates/model/src/qwen3/block.rs  # 新增: forward_prefill, forward_decode
crates/model/src/qwen3/model.rs  # 修改: forward_with_cache
crates/core/src/engine.rs        # 修改: 区分 prefill/decode
crates/core/src/scheduler.rs     # 修改: batch 分类
crates/core/src/types.rs         # 可能需要添加 is_prefill 字段
```

---

## Task 1: 增强 PagedKvCache

**Files:**

- Modify: `crates/model/src/kv_cache.rs:1-94`

- [ ] **Step 1: Read current PagedKvCache implementation**

```bash
cat crates/model/src/kv_cache.rs
```

- [ ] **Step 2: 增强 PagedKvCache 添加 write_kv 方法**

在 impl PagedKvCache 中添加:

```rust
/// Write K, V to cache at specific block and token offset
pub fn write_kv(
    &mut self,
    layer_idx: usize,
    block_id: usize,
    token_offset: usize,
    k: &Tensor,
    v: &Tensor,
) -> Result<()> {
    // k, v shape: [1, num_kv_heads, head_dim]
    // Write to key_cache[layer_idx][block_id, :, token_offset, :]
    let num_kv_heads = self.num_kv_heads;
    let head_dim = self.head_dim;

    for h in 0..num_kv_heads {
        let k_head = k.squeeze(0)?.narrow(0, h, 1)?.squeeze(0)?; // [head_dim]
        let v_head = v.squeeze(0)?.narrow(0, h, 1)?.squeeze(0)?; // [head_dim]

        // Write to cache[block, head, offset, :]
        self.key_cache[layer_idx] = self.key_cache[layer_idx].index_set(
            &[
                block_id, h, token_offset, 
                candle_core::Range::full()
            ], 
            &k_head.unsqueeze(0)?.unsqueeze(0)?
        )?;
        self.value_cache[layer_idx] = self.value_cache[layer_idx].index_set(
            &[
                block_id, h, token_offset,
                candle_core::Range::full()
            ], 
            &v_head.unsqueeze(0)?.unsqueeze(0)?
        )?;
    }
    Ok(())
}
```

- [ ] **Step 3: 添加 read_kv 方法**

```rust
/// Read K, V from cache for given block_ids and sequence length
pub fn read_kv(
    &self,
    layer_idx: usize,
    block_ids: &[usize],
    seq_len: usize,
) -> Result<(Tensor, Tensor)> {
    let mut k_parts = Vec::new();
    let mut v_parts = Vec::new();

    let num_blocks = block_ids.len();

    for block_idx in 0..num_blocks {
        let block_id = block_ids[block_idx];
        let start_token = block_idx * self.block_size;
        let end_token = std::cmp::min(start_token + self.block_size, seq_len);
        let block_len = end_token - start_token;

        // Read K: [block_len, num_kv_heads, head_dim]
        let k_block = self.key_cache[layer_idx]
            .narrow(0, block_id, 1)?
            .narrow(1, 0, self.num_kv_heads)?
            .narrow(2, 0, block_len)?
            .squeeze(0)?;

        let v_block = self.value_cache[layer_idx]
            .narrow(0, block_id, 1)?
            .narrow(1, 0, self.num_kv_heads)?
            .narrow(2, 0, block_len)?
            .squeeze(0)?;

        k_parts.push(k_block);
        v_parts.push(v_block);
    }

    // Concatenate: [seq_len, num_kv_heads, head_dim]
    let k = Tensor::cat(&k_parts, 0)?;
    let v = Tensor::cat(&v_parts, 0)?;

    Ok((k, v))
}
```

- [ ] **Step 4: 运行测试验证**

```bash
cd /home/mystvio/repos/vllm-lite && cargo test -p vllm-model paged_kv_cache -- --nocapture
```

- [ ] **Step 5: Commit**

```bash
git add crates/model/src/kv_cache.rs
git commit -m "feat(model): add write_kv and read_kv to PagedKvCache"
```

---

## Task 2: 修改 GqaAttention 支持 Paged Attention

**Files:**

- Modify: `crates/model/src/qwen3/attention.rs:1-137`

- [ ] **Step 1: Read current attention implementation**

```bash
cat crates/model/src/qwen3/attention.rs
```

- [ ] **Step 2: 添加 forward_prefill 方法**

在 impl GqaAttention 中添加:

```rust
/// Prefill: compute attention and write KV to cache
pub fn forward_prefill(
    &self,
    x: &Tensor,
    kv_cache: &mut PagedKvCache,
    layer_idx: usize,
    block_ids: &[usize],
) -> Result<Tensor> {
    let batch_size = x.dims()[0];
    let seq_len = x.dims()[1];

    let q = self.q_proj.forward(x)?;
    let k = self.k_proj.forward(x)?;
    let v = self.v_proj.forward(x)?;

    // Reshape: [batch, seq, heads, dim] -> [batch, heads, seq, dim]
    let q = q.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?;
    let k = k.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;
    let v = v.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;

    // Write K, V to cache
    for token_idx in 0..seq_len {
        let block_id = token_idx / crate::kv_cache::BLOCK_SIZE;
        let offset = token_idx % crate::kv_cache::BLOCK_SIZE;

        // k[:, :, token_idx, :]
        let k_slice = k.narrow(2, token_idx, 1)?.transpose(0, 1)?; // [num_kv_heads, 1, head_dim]
        let v_slice = v.narrow(2, token_idx, 1)?.transpose(0, 1)?; // [num_kv_heads, 1, head_dim]

        // Reshape to [1, num_kv_heads, head_dim]
        let k_slice = k_slice.reshape((1, self.num_kv_heads, self.head_dim))?;
        let v_slice = v_slice.reshape((1, self.num_kv_heads, self.head_dim))?;

        kv_cache.write_kv(layer_idx, block_id, offset, &k_slice, &v_slice)?;
    }

    // Expand KV for GQA (num_heads vs num_kv_heads)
    let k_expanded = self.expand_kv(&k.transpose(1, 2)?, self.num_heads, self.num_kv_heads)?;
    let v_expanded = self.expand_kv(&v.transpose(1, 2)?, self.num_heads, self.num_kv_heads)?;
    let k_expanded = k_expanded.transpose(1, 2)?; // [batch, num_heads, seq, dim]
    let v_expanded = v_expanded.transpose(1, 2)?;

    // Attention computation
    self.paged_attention(&q, &k_expanded, &v_expanded, seq_len)
}
```

- [ ] **Step 3: 添加 forward_decode 方法**

```rust
/// Decode: read from cache, compute attention
pub fn forward_decode(
    &self,
    x: &Tensor,
    kv_cache: &PagedKvCache,
    layer_idx: usize,
    block_ids: &[usize],
    num_computed_tokens: usize,
) -> Result<Tensor> {
    let batch_size = x.dims()[0];

    let q = self.q_proj.forward(x)?;
    let q = q.reshape((batch_size, 1, self.num_heads, self.head_dim))?.transpose(1, 2)?;

    // Read K, V from paged cache
    let (k, v) = kv_cache.read_kv(layer_idx, block_ids, num_computed_tokens)?;
    // k, v shape: [seq_len, num_kv_heads, head_dim]

    // Reshape to [batch, num_kv_heads, seq, dim]
    let k = k.unsqueeze(0)?.transpose(1, 2)?; // [1, num_kv_heads, seq, dim]
    let v = v.unsqueeze(0)?.transpose(1, 2)?;

    // Expand KV for GQA
    let k_expanded = self.expand_kv(&k, self.num_heads, self.num_kv_heads)?;
    let v_expanded = self.expand_kv(&v, self.num_heads, self.num_kv_heads)?;
    let k_expanded = k_expanded.transpose(1, 2)?; // [batch, num_heads, seq, dim]
    let v_expanded = v_expanded.transpose(1, 2)?;

    // Paged attention with causal mask
    self.paged_attention(&q, &k_expanded, &v_expanded, num_computed_tokens + 1)
}
```

- [ ] **Step 4: 添加 paged_attention 辅助方法**

```rust
fn paged_attention(
    &self,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    seq_len: usize,
) -> Result<Tensor> {
    // q: [batch, num_heads, 1, head_dim]
    // k: [batch, num_heads, seq_len, head_dim]
    // v: [batch, num_heads, seq_len, head_dim]

    let qk = Tensor::matmul(q, &k.transpose(2, 3)?)?;

    // Causal mask: only attend to previous tokens
    let mask = self.causal_mask(seq_len, q.device())?;
    let qk = (&qk + &mask)?;

    let scale = 1.0 / (self.head_dim as f32).sqrt();
    let qk = qk.mul(&Tensor::new(&[scale], q.device())?)?;
    let attn_weights = candle_nn::ops::softmax(&qk, 3)?;

    let attn_output = Tensor::matmul(&attn_weights, v)?;

    // Reshape: [batch, num_heads, 1, head_dim] -> [batch, 1, num_heads * head_dim]
    let attn_output = attn_output.transpose(1, 2)?;
    let attn_output = attn_output.reshape((q.dims()[0], 1, self.num_heads * self.head_dim))?;

    let o = self.o_proj.forward(&attn_output)?;
    Ok(o)
}

fn causal_mask(&self, seq_len: usize, device: &Device) -> Result<Tensor> {
    let mask: Vec<f32> = (0..seq_len)
        .flat_map(|i| {
            (0..seq_len).map(move |j| {
                if i >= j { 0.0 } else { f32::NEG_INFINITY }
            })
        })
        .collect();
    Tensor::from_slice(&mask, (1, 1, seq_len, seq_len), device)
}
```

- [ ] **Step 5: 添加必要的 import**

确保文件顶部有:

```rust
use crate::kv_cache::PagedKvCache;
use candle_core::Device;
```

- [ ] **Step 6: 运行测试验证**

```bash
cd /home/mystvio/repos/vllm-lite && cargo check -p vllm-model
```

- [ ] **Step 7: Commit**

```bash
git add crates/model/src/qwen3/attention.rs
git commit -m "feat(model): add forward_prefill and forward_decode to GqaAttention"
```

---

## Task 3: 修改 TransformerBlock

**Files:**

- Modify: `crates/model/src/qwen3/block.rs`

- [ ] **Step 1: Read current block implementation**

```bash
cat crates/model/src/qwen3/block.rs
```

- [ ] **Step 2: 添加 forward_prefill 方法**

```rust
pub fn forward_prefill(
    &mut self,
    x: &Tensor,
    kv_cache: &mut PagedKvCache,
    layer_idx: usize,
    block_ids: &[usize],
) -> Result<Tensor> {
    let residual = x.clone();
    let x = self.input_layernorm.forward(x)?;
    let x = self.attention.forward_prefill(x, kv_cache, layer_idx, block_ids)?;
    let x = (&x + &residual)?;

    let residual = x.clone();
    let x = self.post_attention_layernorm.forward(x)?;
    let x = self.mlp.forward(&x)?;
    let x = (&x + &residual)?;

    Ok(x)
}
```

- [ ] **Step 3: 添加 forward_decode 方法**

```rust
pub fn forward_decode(
    &self,
    x: &Tensor,
    kv_cache: &PagedKvCache,
    layer_idx: usize,
    block_ids: &[usize],
    num_computed_tokens: usize,
) -> Result<Tensor> {
    let residual = x.clone();
    let x = self.input_layernorm.forward(x)?;
    let x = self.attention.forward_decode(x, kv_cache, layer_idx, block_ids, num_computed_tokens)?;
    let x = (&x + &residual)?;

    let residual = x.clone();
    let x = self.post_attention_layernorm.forward(x)?;
    let x = self.mlp.forward(&x)?;
    let x = (&x + &residual)?;

    Ok(x)
}
```

- [ ] **Step 4: 添加 import**

```rust
use crate::kv_cache::PagedKvCache;
```

- [ ] **Step 5: 运行测试**

```bash
cd /home/mystvio/repos/vllm-lite && cargo check -p vllm-model
```

- [ ] **Step 6: Commit**

```bash
git add crates/model/src/qwen3/block.rs
git commit -m "feat(model): add forward_prefill and forward_decode to TransformerBlock"
```

---

## Task 4: 修改 Qwen3Model

**Files:**

- Modify: `crates/model/src/qwen3/model.rs:215-284`

- [ ] **Step 1: 添加 forward_with_cache 方法**

在 impl ModelBackend for Qwen3Model 之后添加:

```rust
impl Qwen3Model {
    pub fn forward_with_cache(
        &mut self,
        tokens: &[TokenId],
        num_computed_tokens: usize,
        block_ids: &[BlockId],
        is_prefill: bool,
    ) -> EngineResult<(Tensor, Tensor)> {
        if tokens.is_empty() {
            return Err(EngineError::ModelError("Empty tokens".to_string()));
        }

        let token_tensor = Tensor::new(tokens, &self.device)
            .map_err(|e| EngineError::ModelError(e.to_string()))?;
        let hidden = self.embed_tokens.forward(&token_tensor)
            .map_err(|e| EngineError::ModelError(e.to_string()))?
            .unsqueeze(0)
            .map_err(|e| EngineError::ModelError(e.to_string()))?;

        let mut hidden = hidden;

        if is_prefill {
            for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
                hidden = layer.forward_prefill(&hidden, &mut self.kv_cache, layer_idx, block_ids)
                    .map_err(|e| EngineError::ModelError(e.to_string()))?;
            }
        } else {
            for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
                hidden = layer.forward_decode(&hidden, &self.kv_cache, layer_idx, block_ids, num_computed_tokens)
                    .map_err(|e| EngineError::ModelError(e.to_string()))?;
            }
        }

        hidden = self.norm.forward(&hidden)
            .map_err(|e| EngineError::ModelError(e.to_string()))?;
        let logits = self.lm_head.forward(&hidden)
            .map_err(|e| EngineError::ModelError(e.to_string()))?;

        Ok((logits, hidden))
    }
}
```

- [ ] **Step 2: 修改现有的 forward 方法**

将现有的 forward 方法改为使用 forward_with_cache:

```rust
impl ModelBackend for Qwen3Model {
    fn forward(
        &self,
        seq_ids: &[SeqId],
        input_tokens: &[Vec<TokenId>],
        positions: &[Vec<usize>],
    ) -> EngineResult<BatchOutput> {
        // For now, delegate to single sequence processing
        // Full batched implementation comes later
        let mut next_tokens = Vec::with_capacity(seq_ids.len());

        for (seq_idx, tokens) in input_tokens.iter().take(seq_ids.len()).enumerate() {
            if tokens.is_empty() {
                next_tokens.push(0);
                continue;
            }

            // Create single-element blocks
            let block_ids: Vec<BlockId> = (0..tokens.len().div_ceil(16)).collect();

            let (logits, _) = self.forward_with_cache(tokens, 0, &block_ids, true)?;

            let last_logits = logits.squeeze(0)
                .map_err(|e| EngineError::ModelError(e.to_string()))?
                .get(logits.dims()[1] - 1)
                .map_err(|e| EngineError::ModelError(e.to_string()))?;

            let max_idx = last_logits.argmax(0)
                .map_err(|e| EngineError::ModelError(e.to_string()))?
                .to_scalar::<u32>()
                .unwrap_or(0);

            next_tokens.push(max_idx as TokenId);
        }

        Ok(BatchOutput {
            seq_ids: seq_ids.to_vec(),
            next_tokens,
        })
    }
}
```

- [ ] **Step 3: 添加 BlockId import**

在文件顶部添加:

```rust
use crate::types::BlockId;
```

- [ ] **Step 4: 运行测试**

```bash
cd /home/mystvio/repos/vllm-lite && cargo check -p vllm-model
```

- [ ] **Step 5: Commit**

```bash
git add crates/model/src/qwen3/model.rs
git commit -m "feat(model): add forward_with_cache to Qwen3Model"
```

---

## Task 5: 集成测试

**Files:**

- Test: `crates/model/src/kv_cache.rs` (add tests)

- [ ] **Step 1: 添加 write_kv/read_kv 测试**

在 kv_cache.rs 的 tests 模块中添加:

```rust
#[test]
fn test_write_and_read_kv() -> Result<()> {
    let device = Device::Cpu;
    let mut cache = PagedKvCache::new(2, 4, 32, 10, device.clone())?;

    let k = Tensor::randn(0f32, 1f32, (1, 4, 32), &device)?;
    let v = Tensor::randn(0f32, 1f32, (1, 4, 32), &device)?;

    // Write to block 0, offset 0
    cache.write_kv(0, 0, 0, &k, &v)?;

    // Read back
    let (k_read, v_read) = cache.read_kv(0, &[0], 1)?;

    // Verify shapes
    assert_eq!(k_read.dims(), &[1, 4, 32]);
    assert_eq!(v_read.dims(), &[1, 4, 32]);

    Ok(())
}

#[test]
fn test_write_multiple_blocks() -> Result<()> {
    let device = Device::Cpu;
    let mut cache = PagedKvCache::new(1, 2, 16, 10, device.clone())?;

    // Write 20 tokens across 2 blocks
    for i in 0..20 {
        let k = Tensor::new(vec![i as f32; 32].as_slice(), (1, 2, 16), &device)?;
        let v = Tensor::new(vec![i as f32; 32].as_slice(), (1, 2, 16), &device)?;
        cache.write_kv(0, i / 16, i % 16, &k, &v)?;
    }

    // Read all
    let (k_read, v_read) = cache.read_kv(0, &[0, 1], 20)?;
    assert_eq!(k_read.dims(), &[20, 2, 16]);

    Ok(())
}
```

- [ ] **Step 2: 运行测试**

```bash
cd /home/mystvio/repos/vllm-lite && cargo test -p vllm-model -- --nocapture
```

- [ ] **Step 3: Commit**

```bash
git add crates/model/src/kv_cache.rs
git commit -m "test(model): add write_kv and read_kv tests"
```

---

## Task 6: 端到端集成验证

- [ ] **Step 1: 运行完整测试套件**

```bash
cd /home/mystvio/repos/vllm-lite && cargo test --workspace -- --nocapture
```

- [ ] **Step 2: 检查编译警告**

```bash
cd /home/mystvio/repos/vllm-lite && cargo clippy --workspace -- -D warnings
```

- [ ] **Step 3: Commit 完整实现**

```bash
git add -A && git commit -m "feat: implement paged attention with KV cache

- Add write_kv/read_kv to PagedKvCache
- Add forward_prefill/forward_decode to GqaAttention  
- Add forward_prefill/forward_decode to TransformerBlock
- Add forward_with_cache to Qwen3Model
- Add comprehensive tests"
```

---

## Verification Checklist

- [ ] PagedKvCache.write_kv 正确写入 GPU memory
- [ ] PagedKvCache.read_kv 正确读取 KV
- [ ] GqaAttention.forward_prefill 写入 cache 并计算 attention
- [ ] GqaAttention.forward_decode 从 cache 读取并计算 attention
- [ ] TransformerBlock 正确封装
- [ ] Qwen3Model 可调用 forward_with_cache
- [ ] 所有测试通过
- [ ] clippy 无警告

---

**Plan complete and saved to `docs/superpowers/plans/2026-03-30-paged-attention.md`.**

Two execution options:

1. **Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration
2. **Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
