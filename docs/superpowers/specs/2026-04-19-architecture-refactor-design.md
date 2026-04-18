# vllm-lite 架构重构设计方案

**日期**: 2026-04-19
**状态**: 已批准
**版本**: v2

## 概述

对 vllm-lite 项目进行全面的架构优化，目标是：

1. **编译时间**: 减少 25-35%
2. **二进制体积**: 减少 15-20%
3. **代码重复**: 通过共享组件层显著降低
4. **架构健康**: 消除跨 crate 循环依赖风险

## 目标

### 编译优化

| 优化项 | 当前状态 | 目标状态 | 预期收益 |
|--------|----------|----------|----------|
| tokio features | `full` | 精确指定 | 编译快 15-20% |
| panic 策略 | default | `abort` | binary 小 5% |
| strip 符号 | no | yes | binary 小 10% |
| LTO | `thin` | `fat` | 性能+5% |
| candle CUDA | 强制 | 可选 feature | CPU 开发快 |

### 代码组织

| 问题 | 当前状态 | 目标状态 |
|------|----------|----------|
| Block 重复 | Llama/Mistral/Mixtral 各自实现 | 共享基类 + 架构组合 |
| Attention 分散 | 在各架构目录重复 | 统一组件层 |
| 组件共享 | 部分共享 | 全面共享 (Attention, MLP, Norm, RoPE) |

### 架构健康

| 问题 | 当前状态 | 目标状态 |
|------|----------|----------|
| core → model 依赖 | 存在 | 解耦 |
| 依赖方向 | 单向有环风险 | 层级清晰 |

## 分阶段实施计划

### Phase 1: Cargo.toml 基础优化

**目标**: 最快见效，风险最低

#### 1.1 tokio features 精确化

```toml
# 修改前
tokio = { version = "1", features = ["full"] }

# 修改后 (根据实际使用)
tokio = { version = "1", features = ["sync", "rt", "macros"] }

# server 使用更多特性
[dependencies]
tokio = { version = "1", features = ["sync", "rt", "macros", "time", "io-util"] }

# core 使用
tokio = { version = "1", features = ["sync", "rt", "macros"] }
```

#### 1.2 Release 编译优化

```toml
[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
panic = "abort"
strip = true
```

#### 1.3 工作区依赖统一管理

```toml
[workspace.dependencies]
tokio = { version = "1", features = ["sync", "rt", "macros"] }
```

**交付物**: PR #1
**风险**: 极低

---

### Phase 2: Cargo Feature 重构

**目标**: 可选依赖正确配置，支持轻量级构建

#### 2.1 Candle CUDA 可选

```toml
# crates/model/Cargo.toml
[features]
default = []
cuda = ["candle-core/cuda", "candle-nn/cuda"]
real_weights = ["tiktoken", "tokenizers", "gguf"]
full = ["cuda", "real_weights"]

[dependencies]
candle-core = "0.10.2"
candle-nn = "0.10.2"
```

#### 2.2 GGUF 可选

```toml
gguf = { version = "0.1", optional = true }

[features]
gguf-support = ["dep:gguf"]
```

**交付物**: PR #2
**风险**: 低

---

### Phase 3: 共享组件层重构

**目标**: 建立统一的组件抽象层

#### 3.1 组件 trait 定义

```rust
// crates/model/src/components/traits.rs

pub trait Attention: Send + Sync {
    fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        block_tables: &[i32],
        context_lens: &[usize],
    ) -> Result<Tensor>;
}

pub trait FeedForward: Send + Sync {
    fn forward(&self, x: &Tensor) -> Result<Tensor>;
}

pub trait Norm: Send + Sync {
    fn forward(&self, x: &Tensor) -> Result<Tensor>;
}

pub trait PositionalEncoding: Send + Sync {
    fn forward(&self, positions: &[i64], head_dim: usize) -> Result<(Tensor, Tensor)>;
}
```

#### 3.2 组件实现目录

```
crates/model/src/components/
├── traits.rs          # trait 定义
├── attention/
│   ├── mod.rs
│   ├── gqa.rs         # GQA 实现
│   ├── mha.rs         # MHA 实现 (备用)
│   └── flash.rs       # Flash Attention 封装
├── mlp/
│   ├── mod.rs
│   ├── swiglu.rs      # SwiGLU 实现
│   └── geglu.rs       # GeGLU 实现 (未来)
├── norm/
│   ├── mod.rs
│   ├── rms_norm.rs    # RMSNorm
│   └── layer_norm.rs  # LayerNorm (备用)
├── positional/
│   ├── mod.rs
│   ├── rope.rs        # 标准 RoPE
│   ├── yarn.rs        # YaRN 缩放
│   └── mrope.rs       # MRoPE (Qwen3.5)
└── mod.rs             # 统一导出
```

#### 3.3 迁移策略

1. 将 `components/attention.rs` 中的 `GqaAttention` 移动到 `components/attention/gqa.rs`
2. 将 `qwen3/attention.rs` 中的 `GqaAttention` 改为使用共享实现
3. 对其他架构重复此过程

**交付物**: PR #3
**风险**: 中 (需要确保向后兼容)

---

### Phase 4: Block 重构

**目标**: 通过组合模式减少重复代码

#### 4.1 基类设计

```rust
// crates/model/src/components/block.rs

pub struct TransformerBlockConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f64,
    pub rms_eps: f64,
    pub sliding_window: Option<usize>,
}

pub struct TransformerBlock<Attn, FF, Norm> {
    attention: Attn,
    feed_forward: FF,
    input_layernorm: Norm,
    post_attention_layernorm: Norm,
}
```

#### 4.2 架构特定组合

```rust
// 架构使用示例
pub type LlamaBlock = TransformerBlock<
    GqaAttention,      // from components
    SwiGLU,            // from components
    RmsNorm,           // from components
>;

pub type MistralBlock = TransformerBlock<
    SlidingWindowGqaAttention,  // 扩展 attention
    SwiGLU,
    RmsNorm,
>;
```

#### 4.3 配置差异处理

```rust
// 对于 Mistral 的 sliding window，在配置中传入
pub struct SlidingWindowConfig {
    pub base: TransformerBlockConfig,
    pub sliding_window: usize,
}
```

**交付物**: PR #4
**风险**: 中

---

### Phase 5: core→model 解耦 + 文档

**目标**: 架构健康，文档完善

#### 5.1 解耦策略

**问题**: `vllm-core` 依赖 `vllm-model` 来使用 CUDA graph kernels

**方案**: 将 kernel wrappers 移到 `vllm-traits`

```
vllm-traits
├── model.rs          # ModelBackend trait
├── types.rs          # Batch, SeqId 等类型
└── kernels.rs        # Kernel trait definitions (新)

vllm-core
├── scheduler/
├── kv_cache/
└── engine.rs         # 使用 kernels trait，不直接依赖 model

vllm-model
├── kernels/
│   ├── cuda_graph.rs  # 实现 kernels 中的 trait
│   └── flash_attention.rs
└── components/       # 使用 traits 定义接口
```

#### 5.2 文档完善

- 架构决策记录 (ADR)
- 模块级别文档
- 示例和教程

**交付物**: PR #5
**风险**: 低

---

## 依赖关系

```
Phase 1 ──┬──> Phase 2 ──> Phase 3 ──> Phase 4 ──> Phase 5
          │         │          │          │
          │         │          └──────────┘
          │         │              │
          └─────────┴──────────────┘
              (可以并行尝试)
```

## 测试策略

每个 Phase 需要:
1. 现有测试全部通过
2. 新增组件有单元测试
3. 集成测试覆盖关键路径

## 回滚计划

| Phase | 回滚策略 |
|-------|----------|
| Phase 1 | 直接 revert PR |
| Phase 2 | Feature flag 禁用 |
| Phase 3 | 保留旧实现作为 fallback |
| Phase 4 | 架构类型 alias 回退 |
| Phase 5 | 依赖关系逐步恢复 |

## 验收标准

| 指标 | 基线 | 目标 |
|------|------|------|
| 编译时间 (release) | T | T * 0.7 |
| Binary 大小 | B | B * 0.8 |
| 代码重复行 | D | D * 0.4 |
| 跨 crate 循环依赖 | 1 | 0 |
| 测试覆盖率 | C | C + 5% |

## 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| 抽象过度 | 中 | 中 | 保持简单，不强行抽象 |
| 回归 bug | 中 | 高 | 完整测试覆盖 |
| 编译失败 | 低 | 高 | 小步提交，逐阶段验证 |
| 性能下降 | 低 | 中 | benchmark 监控 |
