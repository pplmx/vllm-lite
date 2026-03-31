# vLLM-lite Test Coverage Design

## 1. Overview

添加完整的测试覆盖，确保生产就绪。

**目标：**

- 单元测试：边界情况、错误处理
- 集成测试：API 端到端
- 压力测试：高并发、大 batch

## 2. 测试分层

### 2.1 单元测试

针对每个模块的独立测试：

| 模块         | 测试覆盖                                  |
| ------------ | ----------------------------------------- |
| kv_cache.rs  | allocate, free, evict, prefix_cache, hash |
| scheduler.rs | add_request, build_batch, prefill/decode  |
| engine.rs    | step, add_request, streaming              |
| sampling.rs  | greedy, top_k, top_p, temperature         |
| attention.rs | forward, paged_attention, causal_mask     |
| model.rs     | forward, weight loading                   |

### 2.2 集成测试

```rust
// crates/core/tests/
integration.rs     // Engine 端到端测试
prefix_cache.rs    // Prefix cache 测试
speculative.rs     // Speculative decoding 测试

// crates/server/tests/
api.rs             // HTTP API 测试
streaming.rs       // SSE 流式测试
```

### 2.3 压力测试

```rust
// tests/stress/
high_concurrency.rs    // 100+ 并发请求
large_batch.rs         // batch=64+ 大 batch
long_sequence.rs       // seq_len=2048 长序列
memory_stress.rs       // 显存压力测试
```

## 3. 测试用例设计

### 3.1 边界情况

```rust
// 空输入
test_empty_prompt() { }

// 超长序列
test_max_tokens_limit() { }

// 特殊 token
test_special_tokens() { }

// 并发冲突
test_concurrent_requests() { }

// OOM 处理
test_kv_cache_eviction() { }
```

### 3.2 错误处理

```rust
test_invalid_request() { }
test_missing_prompt() { }
test_negative_max_tokens() { }
test_invalid_sampling_params() { }
test_model_load_failure() { }
```

### 3.3 回归测试

```rust
// 确保修复的问题不再出现
test_prefix_cache_eviction_bug() { }
test_batch_size_calculation() { }
test_streaming_complete() { }
```

## 4. 测试框架

### 4.1 测试工具

```rust
// 测试辅助函数
fn create_test_engine() -> Engine<StubModel> { ... }
fn create_test_request(prompt: Vec<TokenId>) -> Request { ... }
fn wait_for_completion(engine: &mut Engine<StubModel>) { ... }
fn extract_tokens(response: Response) -> Vec<TokenId> { ... }
```

### 4.2 Fixtures

```rust
#[fixture]
fn test_config() -> SchedulerConfig {
    SchedulerConfig {
        max_num_seqs: 256,
        max_num_batched_tokens: 4096,
        max_consecutive_decode: 10,
    }
}
```

## 5. 覆盖率目标

| 类型           | 目标           |
| -------------- | -------------- |
| 单元测试覆盖率 | > 80%          |
| 集成测试       | 核心流程全覆盖 |
| 压力测试       | 3 个场景       |

## 6. 持续集成

### 6.1 测试命令

```bash
# 所有测试
cargo test --workspace

# 带覆盖率
cargo test --workspace -- --include-crate-type=lib

# 压力测试 (单独运行)
cargo test --test stress
```

### 6.2 CI 配置

```yaml
# .github/workflows/test.yml
test:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v3
    - run: cargo test --workspace
    - run: cargo clippy --workspace
```

## 7. 实现计划

- [ ] 补充边界情况单元测试
- [ ] 添加集成测试
- [ ] 添加压力测试
- [ ] 配置 CI
- [ ] 覆盖率检查

## 8. 测试矩阵

| 场景    | Batch | Seq Len | 并发 | 预期 |
| ------- | ----- | ------- | ---- | ---- |
| Decode  | 1-8   | 1-32    | 10   | 正常 |
| Prefill | 1-4   | 128-512 | 5    | 正常 |
| 压力    | 64+   | 128     | 100+ | 稳定 |
