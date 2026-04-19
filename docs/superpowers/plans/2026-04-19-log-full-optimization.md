# 日志全面优化实现计划

**Goal:** 全面优化日志系统，修正消息格式、补充缺失日志、美化消息

---

## 需要修改的文件

| 文件 | 修改内容 |
|------|----------|
| crates/server/src/main.rs | 修正消息格式、添加耗时 |
| crates/server/src/openai/chat.rs | 添加失败日志、补充请求完成 |
| crates/model/src/gemma4/attention.rs | 添加 TRACE 日志 |
| crates/model/src/mixtral/sparse_moe.rs | 添加 TRACE 日志 |
| crates/model/src/components/block.rs | 添加层日志 |
| crates/core/src/scheduler/preemption.rs | 添加 DEBUG 日志 |
| crates/core/src/engine.rs | 统一字段命名 |

---

## 实现任务

### Task 1: 修复 main.rs 日志格式

**文件**: `crates/server/src/main.rs`

- [ ] **Step 1: 修正设备日志**

```rust
// 替换前:
tracing::info!(device = ?device, "Using device");

// 替换后:
tracing::info!(device = ?device, "Device initialized: {}", device);
```

- [ ] **Step 2: 添加模型加载耗时**

```rust
// 替换前:
let model = loader.load_model()...

// 替换后:
let model_load_start = std::time::Instant::now();
let model = loader.load_model()...
tracing::info!(
    model_path = %model_path,
    device = ?device,
    elapsed_ms = model_load_start.elapsed().as_millis() as u64,
    "Model loaded"
);
```

- [ ] **Step 3: 添加 server listening 详细信息**

```rust
// 替换前:
tracing::info!(address = %addr, "Server listening");

// 替换后:
tracing::info!(
    host = %addr.ip(),
    port = addr.port(),
    "Server listening"
);
```

- [ ] **Step 4: 提交**

```bash
git add crates/server/src/main.rs && git commit -m "refactor(server): improve startup log messages"
```

---

### Task 2: 补充 chat.rs 缺失日志

**文件**: `crates/server/src/openai/chat.rs`

- [ ] **Step 1: 添加请求失败日志**

在函数错误返回前添加:

```rust
tracing::warn!(
    request_id = %request_id,
    error = %error_message,
    "Request failed"
);
```

- [ ] **Step 2: 统一字段命名**

确保所有日志使用一致的字段名:
- `request_id` (不是 `req_id`)
- `prompt_tokens` (不是 `prompt_tokens_len`)
- `output_tokens` (不是 `token_count`)
- `duration_ms` (不是 `elapsed_ms`)

- [ ] **Step 3: 提交**

```bash
git add crates/server/src/openai/chat.rs && git commit -m "feat(server): add request failure logging"
```

---

### Task 3: 添加 Gemma4 Attention 日志

**文件**: `crates/model/src/gemma4/attention.rs`

- [ ] **Step 1: 添加层日志**

```rust
pub fn forward(&self, ...) -> Result<Tensor> {
    tracing::trace!(
        layer_idx = self.layer_idx,
        batch_size = x.dims()[0],
        seq_len = x.dims().get(1).copied().unwrap_or(1),
        num_heads = self.num_heads,
        "Gemma4 attention forward"
    );
    // ... existing logic
}
```

- [ ] **Step 2: 提交**

```bash
git add crates/model/src/gemma4/attention.rs && git commit -m "feat(model): add Gemma4 attention logs"
```

---

### Task 4: 添加 Mixtral MoE 日志

**文件**: `crates/model/src/mixtral/sparse_moe.rs`

- [ ] **Step 1: 添加 MoE 层日志**

```rust
pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
    tracing::trace!(
        batch_size = hidden_states.dims()[0],
        seq_len = hidden_states.dims().get(1).copied().unwrap_or(1),
        num_experts = self.num_experts,
        top_k = self.top_k,
        "MoE forward started"
    );
    
    // ... existing logic
    
    tracing::trace!(
        output_shape = ?output.dims(),
        "MoE forward completed"
    );
    Ok(output)
}
```

- [ ] **Step 2: 提交**

```bash
git add crates/model/src/mixtral/sparse_moe.rs && git commit -m "feat(model): add Mixtral MoE logs"
```

---

### Task 5: 添加 TransformerBlock 日志

**文件**: `crates/model/src/components/block.rs`

- [ ] **Step 1: 添加前向传播日志**

```rust
pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
    tracing::trace!(
        input_shape = ?x.dims(),
        "TransformerBlock forward"
    );
    
    let residual = x.clone();
    let x = self.input_layernorm.forward(x)?;
    let x = self.attention.forward(&x)?;
    let x = (&x + &residual)?;

    let residual = x.clone();
    let x = self.post_attention_layernorm.forward(&x)?;
    let x = self.mlp.forward(&x)?;
    
    tracing::trace!(
        output_shape = ?x.dims(),
        "TransformerBlock forward completed"
    );
    
    &x + &residual
}
```

- [ ] **Step 2: 提交**

```bash
git add crates/model/src/components/block.rs && git commit -m "feat(model): add TransformerBlock logs"
```

---

### Task 6: 添加 Preemption 日志

**文件**: `crates/core/src/scheduler/preemption.rs`

- [ ] **Step 1: 添加抢占决策日志**

```rust
pub fn should_preempt(&self, ...) -> bool {
    let decision = /* existing logic */;
    
    tracing::debug!(
        decision = decision,
        running_count = running_sequences.len(),
        waiting_count = waiting_sequences.len(),
        free_blocks = free_blocks,
        reason = %self.get_preemption_reason(...),
        "Preemption decision"
    );
    
    decision
}
```

- [ ] **Step 2: 添加驱逐日志**

```rust
pub fn select_victims(&self, ...) -> Vec<BlockId> {
    tracing::debug!(
        candidates = running_sequences.len(),
        target_blocks = num_blocks,
        "Selecting preemption victims"
    );
    
    // ... existing logic
    
    tracing::debug!(
        selected = victims.len(),
        victims = ?victims,
        "Preemption victims selected"
    );
    
    victims
}
```

- [ ] **Step 3: 提交**

```bash
git add crates/core/src/scheduler/preemption.rs && git commit -m "feat(core): add preemption decision logs"
```

---

### Task 7: 统一 engine.rs 字段命名

**文件**: `crates/core/src/engine.rs`

- [ ] **Step 1: 确保字段命名一致**

当前使用的字段名:
- `batch_seq_ids` → `seq_ids`
- `batch_input_tokens_count` → `total_input_tokens`
- `output_seq_ids` → `seq_ids`

统一为:
- `seq_ids`, `total_tokens`, `duration_ms`

- [ ] **Step 2: 提交**

```bash
git add crates/core/src/engine.rs && git commit -m "refactor(core): standardize log field names"
```

---

### Task 8: 完整验证

- [ ] **Step 1: 运行测试**

```bash
cargo test --workspace --lib 2>&1 | tail -20
```

- [ ] **Step 2: 运行 clippy**

```bash
cargo clippy --workspace -- -D warnings 2>&1 | tail -10
```

- [ ] **Step 3: 提交验证**

```bash
git commit -m "test: verify logging improvements"
```

---

## 验收检查清单

- [ ] 所有日志消息格式统一
- [ ] 字段命名一致 (request_id, prompt_tokens, output_tokens, duration_ms)
- [ ] Gemma4/Mixtral 有层日志
- [ ] TransformerBlock 有日志
- [ ] Preemption 有决策日志
- [ ] 所有测试通过
- [ ] Clippy 无警告
