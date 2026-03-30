# vLLM-lite Paged Attention Design

## 1. Overview

实现真正的 Paged Attention，让 KV cache 写入 GPU paged memory，decode 阶段复用缓存的 KV，避免重复计算。

**当前问题：**
- `PagedKvCache` 已分配 GPU 内存但从未使用
- 每次 forward 都重新计算完整 attention
- Prefix cache (block ID) 与 GPU paged cache 脱节

**目标：**
- Prefill: 计算 KV 并写入 paged cache
- Decode: 从 paged cache 读取 KV 进行 attention
- 简单 block 管理（固定大小）

## 2. 核心架构

### 2.1 数据流

```
Request arrives
    │
    ├─ Prefill phase
    │   ├─ Embed tokens
    │   ├─ For each layer:
    │   │   ├─ Compute Q, K, V
    │   │   ├─ Write K, V to paged cache (block_id → offset)
    │   │   ├─ Self-attention (causal mask)
    │   │   └─ Write output to residual
    │   └─ Output: next token
    │
    └─ Decode phase
        ├─ Embed new token
        ├─ For each layer:
        │   ├─ Compute Q (1 token)
        │   ├─ Read K, V from paged cache (block_ids)
        │   ├─ Paged attention (causal, block-wise)
        │   └─ Write new K, V to next block
        └─ Output: next token
```

### 2.2 Block 映射

```rust
// core/types.rs - 已有的 Sequence 结构
pub struct Sequence {
    pub id: SeqId,
    pub tokens: Vec<TokenId>,           // 已生成的 tokens
    pub kv_blocks: Vec<BlockId>,        // 分配的 block IDs
    pub num_computed_tokens: usize,     // 已计算的 token 数
    // ...
}

// model/kv_cache.rs - PagedKvCache 管理 GPU 内存
pub struct PagedKvCache {
    key_cache: Vec<Tensor>,   // [layer, block, head, block_size, head_dim]
    value_cache: Vec<Tensor>,
    // ...
}
```

### 2.3 Block 分配策略

- **Prefill**: 一次性分配 `prompt_len / BLOCK_SIZE` 个 blocks
- **Decode**: 每次 decode 1 个 token，写入下一个 block
- **固定 block size**: 16 tokens（已有常量）

```rust
const BLOCK_SIZE: usize = 16;

// 计算需要多少 blocks
fn num_blocks_needed(token_count: usize) -> usize {
    token_count.div_ceil(BLOCK_SIZE)
}
```

## 3. 核心实现

### 3.1 PagedKvCache 增强

```rust
// crates/model/src/kv_cache.rs

impl PagedKvCache {
    /// Write K, V to cache at specific block and offset
    pub fn write_kv(
        &mut self,
        layer_idx: usize,
        block_id: usize,
        offset: usize,  // 0-15
        k: &Tensor,    // [batch, 1, num_kv_heads, head_dim]
        v: &Tensor,    // [batch, 1, num_kv_heads, head_dim]
    ) -> Result<()> {
        let key_cache = &mut self.key_cache[layer_idx];
        let value_cache = &mut self.value_cache[layer_idx];

        // Write K: key_cache[block, head, offset, :]
        // Write V: value_cache[block, head, offset, :]
        // ...
    }

    /// Read K, V from cache for decode attention
    pub fn read_kv(
        &self,
        layer_idx: usize,
        block_ids: &[usize],
        seq_len: usize,
    ) -> Result<(Tensor, Tensor)> {
        // Collect K, V from all blocks
        // Return [batch, seq_len, num_kv_heads, head_dim]
    }

    /// Get block size constant
    pub fn block_size(&self) -> usize {
        self.block_size
    }
}
```

### 3.2 Attention with Paged Cache

```rust
// crates/model/src/qwen3/attention.rs

impl GqaAttention {
    /// Prefill: compute attention and write KV to cache
    pub fn forward_prefill(
        &self,
        x: &Tensor,           // [batch, seq_len, hidden]
        kv_cache: &mut PagedKvCache,
        layer_idx: usize,
        block_ids: &[BlockId],
    ) -> Result<Tensor> {
        let batch_size = x.dims()[0];
        let seq_len = x.dims()[1];

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Write each token's K, V to cache
        for token_idx in 0..seq_len {
            let block_id = token_idx / BLOCK_SIZE;
            let offset = token_idx % BLOCK_SIZE;

            let k_slice = k.get(1)?.get(token_idx)?;
            let v_slice = v.get(1)?.get(token_idx)?;

            kv_cache.write_kv(layer_idx, block_id, offset, &k_slice, &v_slice)?;
        }

        // Attention: use full K, V (from computation or cache)
        self.paged_attention(&q, &k, &v, seq_len)
    }

    /// Decode: read from cache, compute attention
    pub fn forward_decode(
        &self,
        x: &Tensor,           // [batch, 1, hidden]
        kv_cache: &PagedKvCache,
        layer_idx: usize,
        block_ids: &[BlockId],
        num_computed_tokens: usize,
    ) -> Result<Tensor> {
        let q = self.q_proj.forward(x)?;

        // Read K, V from paged cache
        let (k, v) = kv_cache.read_kv(layer_idx, block_ids, num_computed_tokens)?;

        // Paged attention
        self.paged_attention(&q, &k, &v, num_computed_tokens + 1)
    }

    fn paged_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        seq_len: usize,
    ) -> Result<Tensor> {
        // Expand KV for GQA
        let k = self.expand_kv(&k, self.num_heads, self.num_kv_heads)?;
        let v = self.expand_kv(&v, self.num_heads, self.num_kv_heads)?;

        // Reshape for block-wise computation
        let q = q.transpose(1, 2)?;  // [batch, num_heads, 1, head_dim]
        let k = k.transpose(1, 2)?;  // [batch, num_kv_heads, seq_len, head_dim]
        let v = v.transpose(1, 2)?;  // [batch, num_kv_heads, seq_len, head_dim]

        // Expand K, V to num_heads (if num_kv_heads < num_heads)
        let k = self.expand_kv(&k, self.num_heads, self.num_kv_heads)?;
        let v = self.expand_kv(&v, self.num_heads, self.num_kv_heads)?;

        // Compute attention
        let qk = Tensor::matmul(&q, &k.transpose(2, 3)?)?;

        // Causal mask: only attend to previous tokens
        let mask = self.causal_mask(seq_len, q.device())?;
        let qk = &qk + &mask;

        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let qk = qk.mul(&Tensor::new(&[scale], q.device())?)?;
        let attn_weights = candle_nn::ops::softmax(&qk, 3)?;

        let attn_output = Tensor::matmul(&attn_weights, &v)?;

        // Reshape and project
        let attn_output = attn_output.transpose(1, 2)?;
        let attn_output = attn_output.reshape((x.dims()[0], 1, self.num_heads * self.head_dim))?;

        let o = self.o_proj.forward(&attn_output)?;
        Ok(o)
    }

    fn causal_mask(&self, seq_len: usize, device: &Device) -> Result<Tensor> {
        // Create causal mask: mask[i, j] = 0 if i >= j, else -inf
        // [1, 1, seq_len, seq_len]
        let mask: Vec<f32> = (0..seq_len)
            .flat_map(|i| (0..seq_len).map(move |j| if i >= j { 0.0 } else { f32::NEG_INFINITY }))
            .collect();
        Tensor::from_slice(&mask, (1, 1, seq_len, seq_len), device)
    }
}
```

### 3.3 Model Forward 集成

```rust
// crates/model/src/qwen3/model.rs

impl ModelBackend for Qwen3Model {
    fn forward(
        &self,
        seq_ids: &[SeqId],
        input_tokens: &[Vec<TokenId>],
        positions: &[Vec<usize>],
    ) -> EngineResult<BatchOutput> {
        // Determine if prefill or decode based on num_computed_tokens
        // Use appropriate forward method
        // ...
    }

    fn forward_with_cache(
        &mut self,
        tokens: &[TokenId],
        num_computed_tokens: usize,
        block_ids: &[BlockId],
        is_prefill: bool,
    ) -> EngineResult<(Tensor, Tensor)> {
        // Embed
        let hidden = self.embed_tokens.forward(&Tensor::new(tokens, &self.device)?)?.unsqueeze(0)?;

        let mut hidden = hidden;

        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            if is_prefill {
                hidden = layer.forward_prefill(&hidden, &mut self.kv_cache, layer_idx, block_ids)?;
            } else {
                hidden = layer.forward_decode(&hidden, &self.kv_cache, layer_idx, block_ids, num_computed_tokens)?;
            }
        }

        // Final norm and lm_head
        hidden = self.norm.forward(&hidden)?;
        let logits = self.lm_head.forward(&hidden)?;

        Ok((logits, hidden))
    }
}
```

### 3.4 TransformerBlock 封装

```rust
// crates/model/src/qwen3/block.rs

impl TransformerBlock {
    pub fn forward_prefill(
        &mut self,
        x: &Tensor,
        kv_cache: &mut PagedKvCache,
        layer_idx: usize,
        block_ids: &[BlockId],
    ) -> Result<Tensor> {
        let residual = x.clone();
        let x = self.input_layernorm.forward(x)?;
        let x = self.attention.forward_prefill(x, kv_cache, layer_idx, block_ids)?;
        let x = (x + residual)?;

        let residual = x.clone();
        let x = self.post_attention_layernorm.forward(x)?;
        let x = self.mlp.forward(&x)?;
        let x = (x + residual)?;

        Ok(x)
    }

    pub fn forward_decode(
        &self,
        x: &Tensor,
        kv_cache: &PagedKvCache,
        layer_idx: usize,
        block_ids: &[BlockId],
        num_computed_tokens: usize,
    ) -> Result<Tensor> {
        let residual = x.clone();
        let x = self.input_layernorm.forward(x)?;
        let x = self.attention.forward_decode(x, kv_cache, layer_idx, block_ids, num_computed_tokens)?;
        let x = (x + residual)?;

        let residual = x.clone();
        let x = self.post_attention_layernorm.forward(x)?;
        let x = self.mlp.forward(&x)?;
        let x = (x + residual)?;

        Ok(x)
    }
}
```

## 4. Scheduler 集成

### 4.1 Batch 分类

```rust
// Scheduler 在 build_batch 时区分 prefill 和 decode

pub fn build_batch(&mut self) -> Option<Batch> {
    // waiting: prefill sequences (not computed yet)
    // running: decode sequences

    // Build prefill batch from waiting
    // Build decode batch from running
    
    // Process separately or combined (chunks)
}
```

### 4.2 Engine 调用

```rust
// crates/core/src/engine.rs

fn step(&mut self) -> EngineResult<()> {
    let batch = self.scheduler.build_batch()?;
    
    if let Some(batch) = batch {
        if batch.has_prefill() {
            // Prefill: compute KV and cache
            for seq in batch.prefill_sequences() {
                let (logits, _) = self.model.forward_with_cache(
                    &seq.tokens,
                    0,
                    &seq.kv_blocks,
                    true,  // is_prefill
                )?;
                // Sample next token
                // Update sequence
            }
        }
        
        if batch.has_decode() {
            // Decode: read from cache
            for seq in batch.decode_sequences() {
                let (logits, _) = self.model.forward_with_cache(
                    &seq.tokens[seq.num_computed_tokens..],
                    seq.num_computed_tokens,
                    &seq.kv_blocks,
                    false,  // is_decode
                )?;
                // Sample next token
                // Write new KV to cache
            }
        }
    }
    
    Ok(())
}
```

## 5. 测试场景

### 测试 1: Prefill 写入 cache

```
输入: "Hello world"
期望: KV 写入 paged cache, block 0 包含所有 KV
验证: read_kv 返回正确数据
```

### 测试 2: Decode 读取 cache

```
输入: prompt="Hello", 已 prefill
期望: decode 读取 prompt 的 KV, 计算 self-attention
验证: 输出与完整序列计算一致
```

### 测试 3: 多 block 序列

```
输入: 长度 32 的序列
期望: 使用 block 0 和 block 1
验证: 32 个 token 的 KV 都能正确读写
```

### 测试 4: 混合 prefill/decode batch

```
输入: waiting=[seq1], running=[seq2, seq3]
期望: 分别处理, 正确更新 kv_blocks
验证: 两者都能继续生成
```

## 6. 边界情况

1. **Block 不足**: OOM 时 evict prefix cache，释放 blocks
2. **单 token prefill**: 正常写入 block 0, offset 0
3. **Sequence 结束**: Cache 保留或释放（由 prefix cache 策略决定）
4. **CUDA vs CPU**: 先实现 CPU，CUDA 需要额外 kernel

## 7. 实现计划

- [ ] 增强 PagedKvCache: write_kv, read_kv 方法
- [ ] 修改 GqaAttention: forward_prefill, forward_decode
- [ ] 修改 TransformerBlock: prefill/decode 封装
- [ ] 修改 Qwen3Model: forward_with_cache
- [ ] 修改 Engine: 区分 prefill/decode 调用
- [ ] 修改 Scheduler: 正确设置 is_prefill 标志
- [ ] 集成测试

## 8. 预期收益

- **Decode 阶段**: O(1) KV 读取 vs 重新计算 O(n)
- **多 batch decode**: 共享 KV，减少计算
- **显存效率**: Block 粒度管理，避免碎片

## 9. 后续增强

- Flash attention kernel（GPU 优化）
- Block 动态扩展（按需分配）
- Prefix sharing（相同 prompt 共享 KV）
- 混合 attention（full + paged）