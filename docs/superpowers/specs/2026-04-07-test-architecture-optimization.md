# vllm-lite 测试架构优化规范

## 目标

优化测试基础设施，消除重复代码，统一命名规范，提升测试可维护性。

## 当前问题分析

### 1. Mock 模型重复定义 (4处)

| 位置                                | 名称                           | 用途           |
| ----------------------------------- | ------------------------------ | -------------- |
| `crates/core/tests/common/mod.rs`   | `IncrementModel`, `ConstModel` | 核心调度器测试 |
| `crates/core/tests/prefix_cache.rs` | `StubModel`                    | 前缀缓存测试   |
| `crates/core/src/engine.rs:273`     | `StubModel`                    | 引擎测试       |
| `crates/model/src/fake.rs`          | `FakeModel`                    | 模型测试       |

### 2. 测试文件命名不一致

| 当前名称                       | 问题                            |
| ------------------------------ | ------------------------------- |
| `scheduler_functional.rs`      | 应为 `scheduler_integration.rs` |
| `tiled_attention.rs`           | 需明确是测试还是功能            |
| `attention_batch_benchmark.rs` | 需移至标准 bench 目录           |

### 3. 缺少共享测试基础设施

- 无 centralized test fixtures
- 无共享 test builders/helpers
- dev-dependencies 不完整

---

## 实施计划

### Phase 1: 创建统一测试工具 Crate (vllm-testing)

#### 1.1 新建 `crates/testing/Cargo.toml`

```toml
[package]
name = "vllm-testing"
version.workspace = true
edition.workspace = true

[dependencies]
vllm-traits = { path = "../traits" }
candle-core = "0.8"
tokio = { version = "1", features = ["sync"] }
rand = "0.10"

[dev-dependencies]
criterion = "0.5"
proptest = "1.5"
```

#### 1.2 统一 Mock 模型 (`crates/testing/src/mocks/mod.rs`)

```rust
// 合并所有 mock 实现到一个 crate

pub struct StubModel;
pub struct IncrementModel;
pub struct ConstModel;
pub struct FakeModel;
pub struct NeverProgressModel;  // 用于超时测试
```

#### 1.3 测试 Builder 工具 (`crates/testing/src/builders/`)

```rust
// RequestBuilder - 简化请求创建
// BatchBuilder - 简化批处理构建
// EngineBuilder - 简化引擎初始化
```

#### 1.4 更新依赖

- `vllm-core` → 添加 `vllm-testing` dev-dependency
- `vllm-model` → 添加 `vllm-testing` dev-dependency
- `vllm-server` → 添加 `vllm-testing` dev-dependency
- 删除 `crates/core/tests/common/` (迁移到新 crate)

---

### Phase 2: 规范化测试文件命名

#### 2.1 重命名文件

| 原名称                                            | 新名称                     |
| ------------------------------------------------- | -------------------------- |
| `crates/core/tests/scheduler_functional.rs`       | `scheduler_integration.rs` |
| `crates/model/tests/tiled_attention.rs`           | `tiled_attention_test.rs`  |
| `crates/model/tests/attention_batch_benchmark.rs` | 移动到 `benches/`          |

#### 2.2 创建标准 benchmark 目录

```text
crates/
  core/
    benches/
      scheduler.rs
      batch.rs
  model/
    benches/
      attention.rs
      paged_kv_cache.rs
```

#### 2.3 添加 Cargo.toml 配置

```toml
[[bench]]
name = "scheduler"
harness = false

[[bench]]
name = "attention"
harness = false
```

---

### Phase 3: 增强测试基础设施

#### 3.1 添加 Test Fixtures

```rust
// crates/testing/src/fixtures/mod.rs

pub struct TestFixtures;
impl TestFixtures {
    pub fn small_batch() -> Batch { ... }
    pub fn prefill_decode_mix() -> Batch { ... }
    pub fn oom_scenario() -> SchedulerConfig { ... }
}
```

#### 3.2 添加 Property-Based Testing

- 使用 `proptest` 生成随机测试用例
- 覆盖: 调度决策、批处理构建、采样逻辑

#### 3.3 添加测试工具函数

```rust
// crates/testing/src/utils.rs

pub fn assert_batch_consistency(batch: &Batch);
pub fn simulate_engine_step(engine: &mut Engine, steps: usize);
pub fn generate_random_tokens(len: usize) -> Vec<TokenId>;
```

---

## 实施顺序

1. **Week 1**: 创建 `vllm-testing` crate，迁移 Mock 模型
2. **Week 2**: 更新所有依赖的 crate，删除重复代码
3. **Week 3**: 规范化测试文件命名
4. **Week 4**: 完善 benchmark 基础设施
5. **Week 5**: 添加 fixtures 和高级测试工具

## 验收标准

- [x] 消除所有重复的 Mock 模型实现
- [x] 测试文件命名统一 (`*_test.rs` 或 `*_integration.rs`)
- [x] Benchmark 可通过 `cargo bench` 运行
- [x] 新测试使用 `vllm-testing` crate
- [x] CI 流程不变

## 风险与回滚

- 依赖变更可能导致编译失败 → 分步迁移，每步验证
- 测试文件重命名可能导致 git history 断裂 → 使用 `git mv`
