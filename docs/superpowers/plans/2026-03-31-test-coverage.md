# Test Coverage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development

**Goal:** 添加完整测试覆盖，单元 + 集成 + 压力测试

---

## Task 1: 单元测试补充

**Files:**
- Modify: 各模块的 `#[cfg(test)]` 模块

- [ ] **Step 1: kv_cache 测试**

```rust
#[test]
fn test_allocate_exact() { }
#[test]
fn test_eviction_order() { }
#[test]
fn test_hash_collision() { }
```

- [ ] **Step 2: scheduler 测试**

```rust
#[test]
fn test_empty_batch() { }
#[test]
fn test_prefill_decode_mix() { }
#[test]
fn test_max_batch_size() { }
```

- [ ] **Step 3: 提交**

```bash
git commit -m "test(core): add boundary case unit tests"
```

---

## Task 2: 集成测试

**Files:**
- Add: `crates/core/tests/streaming.rs`
- Add: `crates/server/tests/error_handling.rs`

- [ ] **Step 1: 流式测试**

```rust
#[tokio::test]
async fn test_streaming_completion() { }

#[tokio::test]
async fn test_streaming_interrupt() { }
```

- [ ] **Step 2: 错误处理测试**

```rust
#[test]
fn test_invalid_json() { }
#[test]
fn test_missing_field() { }
#[test]
fn test_out_of_range() { }
```

- [ ] **Step 3: 提交**

```bash
git commit -m "test: add integration tests for streaming and errors"
```

---

## Task 3: 压力测试

**Files:**
- Add: `tests/stress/mod.rs`
- Add: `tests/stress/high_concurrency.rs`

- [ ] **Step 1: 高并发测试**

```rust
#[test]
fn test_100_concurrent_requests() {
    // 启动 100 个并发请求
    // 验证无死锁、无崩溃
}
```

- [ ] **Step 2: 大 batch 测试**

```rust
#[test]
fn test_batch_64() {
    // batch=64 验证资源管理正确
}
```

- [ ] **Step 3: 提交**

```bash
git commit -m "test: add stress tests for high concurrency"
```

---

## Task 4: CI 配置

**Files:**
- Add: `.github/workflows/test.yml`

- [ ] **Step 1: 添加 CI**

```yaml
name: Test
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - run: cargo test --workspace
      - run: cargo clippy --workspace
```

- [ ] **Step 2: 提交**

```bash
git commit -m "ci: add test workflow"
```

---

## Verification Checklist

- [ ] 单元测试 > 80% 覆盖率
- [ ] 集成测试覆盖核心流程
- [ ] 压力测试 3 个场景
- [ ] CI 通过