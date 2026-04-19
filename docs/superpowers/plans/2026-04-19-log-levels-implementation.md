# 日志分层实现计划

**Goal:** 在关键代码阶段添加系统化日志，覆盖完整请求生命周期

**Architecture:** 按阶段分层，不同级别覆盖不同细节度

---

## 文件清单

| 文件 | 操作 | 职责 |
|------|------|------|
| crates/core/src/engine.rs | 修改 | Forward 调用日志 |
| crates/core/src/scheduler/engine.rs | 修改 | 调度决策日志 |
| crates/core/src/scheduler/batch.rs | 修改 | 批处理日志 |
| crates/core/src/scheduler/batch_composer.rs | 修改 | 批次构建日志 |
| crates/core/src/scheduler/memory/allocator.rs | 修改 | 内存分配日志 |
| crates/core/src/scheduler/memory/mod.rs | 修改 | 内存管理日志 |
| crates/core/src/scheduler/request_queue.rs | 修改 | 请求队列日志 |
| crates/model/src/kernels/cuda_graph/executor.rs | 修改 | CUDA Graph 日志 |
| crates/model/src/components/attention/gqa.rs | 修改 | Attention 日志 |

---

## 实现任务

### Task 1: Engine Forward 日志 (DEBUG)

**文件**: `crates/core/src/engine.rs`

- [ ] **Step 1: 在 execute_regular 添加 DEBUG 日志**

在模型 forward 调用前后添加:

```rust
fn execute_regular(&mut self, batch: &vllm_traits::Batch) -> Result<BatchOutput> {
    let total_tokens: usize = batch.input_tokens.iter().map(|t| t.len()).sum();
    tracing::debug!(
        batch_size = batch.seq_ids.len(),
        total_tokens = total_tokens,
        is_prefill = batch.phase.is_prefill(),
        "Model forward started"
    );

    let start = std::time::Instant::now();
    let result = {
        let mut model = self.target_model.lock().unwrap();
        model.forward(...)
    };
    let elapsed = start.elapsed().as_millis() as u64;

    match result {
        Ok(output) => {
            tracing::debug!(
                elapsed_ms = elapsed,
                output_tokens = output.next_tokens.len(),
                "Model forward completed"
            );
            Ok(output)
        }
        Err(e) => {
            tracing::error!(error = %e, "Model forward failed");
            Err(e)
        }
    }
}
```

- [ ] **Step 2: 在 process_output 添加 TRACE Token 日志**

在处理每个 token 时:

```rust
fn process_output(...) -> Result<Vec<(SeqId, TokenId)>> {
    let mut results = Vec::new();
    for (seq_id, token) in output.seq_ids.iter().zip(&output.next_tokens) {
        tracing::trace!(
            seq_id = %seq_id,
            token_id = %token,
            "Token generated"
        );
        // ... 发送 token
        results.push((*seq_id, *token));
    }
}
```

- [ ] **Step 3: 验证编译**

```bash
cargo check -p vllm-core 2>&1 | tail -10
```

- [ ] **Step 4: 提交**

```bash
git add crates/core/src/engine.rs
git commit -m "feat(core): add model forward and token generation logs"
```

---

### Task 2: 调度器日志 (DEBUG)

**文件**: `crates/core/src/scheduler/engine.rs`

- [ ] **Step 1: 在 schedule 方法添加调度决策日志**

```rust
fn schedule(&mut self) -> Option<Batch> {
    let waiting = self.waiting_sequences.len();
    let running = self.running_sequences.len();

    tracing::debug!(
        waiting = waiting,
        running = running,
        free_blocks = self.memory_manager.free_block_count(),
        "Scheduling decision"
    );

    // ... 调度逻辑
}
```

- [ ] **Step 2: 在 build_batch 添加批次构建日志**

```rust
fn build_batch(&mut self) -> Option<Batch> {
    // ...

    if let Some(batch) = batch {
        tracing::debug!(
            batch_size = batch.seq_ids.len(),
            prefill_count = batch.is_prefill.iter().filter(|&&x| x).count(),
            total_tokens = batch.total_tokens,
            phase = ?batch.phase,
            "Batch built"
        );
    }
    batch
}
```

- [ ] **Step 3: 验证编译并提交**

```bash
cargo check -p vllm-core && git add crates/core/src/scheduler/engine.rs && git commit -m "feat(core): add scheduler decision logs"
```

---

### Task 3: 批次处理日志 (DEBUG)

**文件**: `crates/core/src/scheduler/batch.rs`

- [ ] **Step 1: 添加批次处理日志**

```rust
impl BatchProcessor {
    pub fn process(&mut self, batch: &Batch) -> Result<BatchOutput> {
        tracing::debug!(
            seq_count = batch.seq_ids.len(),
            total_input_tokens = batch.total_tokens,
            "Processing batch"
        );

        // ...
    }
}
```

- [ ] **Step 2: 验证并提交**

```bash
cargo check -p vllm-core && git add crates/core/src/scheduler/batch.rs && git commit -m "feat(core): add batch processing logs"
```

---

### Task 4: 批次构建日志 (DEBUG)

**文件**: `crates/core/src/scheduler/batch_composer.rs`

- [ ] **Step 1: 检查现有 DEBUG 日志**

查看现有日志，确认覆盖:
- 批次大小限制
- Token 预算
- Prefill/Decode 分离

如需补充，添加:

```rust
tracing::debug!(
    candidate_count = candidates.len(),
    batch_size_limit = config.max_batch_size,
    "Batch candidates selected"
);
```

- [ ] **Step 2: 验证并提交**

```bash
cargo check -p vllm-core && git add crates/core/src/scheduler/batch_composer.rs && git commit -m "feat(core): add batch composition logs"
```

---

### Task 5: 内存管理日志 (DEBUG → TRACE)

**文件**: `crates/core/src/scheduler/memory/allocator.rs`

- [ ] **Step 1: 在 allocate 添加 DEBUG 日志**

```rust
pub fn allocate(&mut self, num_blocks: usize) -> Option<Vec<BlockId>> {
    tracing::debug!(
        requested = num_blocks,
        available = self.free_blocks.len(),
        "Block allocation requested"
    );

    // ...
}
```

- [ ] **Step 2: 在 free 添加 TRACE 日志**

```rust
pub fn free(&mut self, blocks: &[BlockId]) {
    tracing::trace!(
        blocks = ?blocks,
        freed_count = blocks.len(),
        remaining_free = self.free_blocks.len(),
        "Blocks freed"
    );

    // ...
}
```

- [ ] **Step 3: 验证并提交**

```bash
cargo check -p vllm-core && git add crates/core/src/scheduler/memory/allocator.rs && git commit -m "feat(core): add memory allocation logs"
```

---

### Task 6: 请求队列日志 (DEBUG)

**文件**: `crates/core/src/scheduler/request_queue.rs`

- [ ] **Step 1: 在 enqueue 添加 DEBUG 日志**

```rust
pub fn enqueue(&mut self, request: Request) {
    tracing::debug!(
        request_id = request.id,
        prompt_tokens = request.prompt_tokens.len(),
        max_tokens = request.max_tokens,
        queue_size = self.waiting.len(),
        "Request enqueued"
    );

    // ...
}
```

- [ ] **Step 2: 验证并提交**

```bash
cargo check -p vllm-core && git add crates/core/src/scheduler/request_queue.rs && git commit -m "feat(core): add request queue logs"
```

---

### Task 7: Attention 日志 (TRACE)

**文件**: `crates/model/src/components/attention/gqa.rs`

- [ ] **Step 1: 添加层级别 TRACE 日志**

```rust
pub fn forward(&self, x: &Tensor, ...) -> Result<Tensor> {
    tracing::trace!(
        layer_idx = self.layer_idx,
        batch_size = x.dims()[0],
        seq_len = x.dims()[1],
        head_dim = self.head_dim,
        "Attention forward started"
    );

    // ... 计算

    tracing::trace!(
        layer_idx = self.layer_idx,
        output_shape = ?output.dims(),
        "Attention forward completed"
    );

    Ok(output)
}
```

- [ ] **Step 2: 验证并提交**

```bash
cargo check -p vllm-model && git add crates/model/src/components/attention/gqa.rs && git commit -m "feat(model): add attention layer logs"
```

---

### Task 8: KV Cache 日志 (TRACE)

**文件**: `crates/model/src/paged_tensor/tensor_store.rs`

- [ ] **Step 1: 在 write_kv 添加 TRACE 日志**

```rust
pub fn write_kv(&mut self, ...) -> Result<()> {
    tracing::trace!(
        layer_idx = layer_idx,
        block_ids = ?block_ids,
        start_pos = start_pos,
        tokens = tokens.len(),
        "KV cache write"
    );

    // ...
}
```

- [ ] **Step 2: 在 read_kv 添加 TRACE 日志**

```rust
pub fn read_kv(&self, ...) -> Result<(Tensor, Tensor)> {
    if block_ids.is_empty() {
        return Ok((empty_k, empty_v));
    }

    tracing::trace!(
        layer_idx = layer_idx,
        block_ids = ?block_ids,
        seq_len = seq_len,
        "KV cache read"
    );

    // ...
}
```

- [ ] **Step 3: 验证并提交**

```bash
cargo check -p vllm-model && git add crates/model/src/paged_tensor/tensor_store.rs && git commit -m "feat(model): add KV cache operation logs"
```

---

### Task 9: 前缀缓存日志 (DEBUG → TRACE)

**文件**: `crates/core/src/kv_cache/prefix_cache.rs`

- [ ] **Step 1: 添加缓存命中/未命中日志**

```rust
pub fn find_prefix_match(&mut self, tokens: &[u32]) -> Option<PrefixMatchResult> {
    let result = self.radix_tree.find_longest_prefix(tokens);

    match &result {
        Some(matched) => {
            tracing::trace!(
                matched_tokens = matched.matched_tokens,
                new_tokens = tokens.len() - matched.matched_tokens,
                blocks = ?matched.block_ids,
                "Prefix cache hit"
            );
        }
        None => {
            tracing::trace!(
                tokens = tokens.len(),
                "Prefix cache miss"
            );
        }
    }

    result
}
```

- [ ] **Step 2: 验证并提交**

```bash
cargo check -p vllm-core && git add crates/core/src/kv_cache/prefix_cache.rs && git commit -m "feat(core): add prefix cache logs"
```

---

### Task 10: 完整验证

- [ ] **Step 1: 运行所有测试**

```bash
cargo test --workspace --lib 2>&1 | tail -20
```

- [ ] **Step 2: 运行 clippy**

```bash
cargo clippy --workspace -- -D warnings 2>&1 | tail -10
```

- [ ] **Step 3: 运行 CI**

```bash
just ci 2>&1 | tail -20
```

- [ ] **Step 4: 测试 trace 级别输出**

```bash
RUST_LOG=trace cargo run -p vllm-server -- --model-path /path/to/model 2>&1 | head -100
```

- [ ] **Step 5: 提交验证**

```bash
git commit -m "test: verify layered logging implementation"
```

---

## 验收检查清单

- [ ] Engine forward 有 DEBUG 日志
- [ ] Token 生成有 TRACE 日志
- [ ] 调度决策有 DEBUG 日志
- [ ] 批次构建有 DEBUG 日志
- [ ] 内存分配有 DEBUG/TRACE 日志
- [ ] 请求队列有 DEBUG 日志
- [ ] Attention 层有 TRACE 日志
- [ ] KV Cache 有 TRACE 日志
- [ ] 前缀缓存有 TRACE 日志
- [ ] 所有测试通过
- [ ] Clippy 无警告
