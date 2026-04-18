# vllm-lite 架构重构实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 全面重构 vllm-lite 架构，优化编译时间、减少代码重复、提升架构健康度

**Architecture:** 分 8 个 Phase 逐步实施，从低风险的基础优化到高风险的组件重构

**Tech Stack:** Rust (Edition 2024), Candle, Tokio

---

## 文件结构概览

### 当前结构 (待修改)
```
crates/
├── model/src/components/attention.rs    # 782行，包含GqaAttention和工具函数
├── model/src/components/mod.rs          # 组件导出
├── model/src/llama/block.rs             # LlamaBlock 实现
├── model/src/mistral/block.rs           # MistralBlock 实现
├── model/src/mixtral/block.rs           # MixtralBlock 实现
├── model/src/qwen3/block.rs             # Qwen3Block 实现
├── Cargo.toml                           # workspace 配置
├── server/Cargo.toml                    # tokio = ["full"]
├── core/Cargo.toml                      # core 依赖 model
```

### 目标结构
```
crates/model/src/components/
├── attention/
│   ├── mod.rs              # 导出
│   ├── gqa.rs              # GqaAttention
│   └── flash.rs            # Flash Attention 封装
├── mlp/
│   ├── mod.rs
│   └── swiglu.rs           # SwiGLU
├── norm/
│   ├── mod.rs
│   └── rms_norm.rs         # RMSNorm
├── positional/
│   ├── mod.rs
│   ├── rope.rs             # 标准 RoPE
│   └── mrope.rs            # MRoPE
├── block.rs                # TransformerBlock 基类
├── mod.rs                  # 统一导出
└── traits.rs               # 可选：统一 trait 定义
```

---

## Phase 1: Cargo.toml 基础优化

**复杂度**: 低 | **风险**: 极低 | **目标 PR 数**: 1

### Task 1.1: 精确化 tokio features

**Files:**
- Modify: `Cargo.toml:21` - workspace 依赖
- Modify: `crates/server/Cargo.toml:20` - server 依赖

- [ ] **Step 1: 修改 workspace Cargo.toml**

修改 `Cargo.toml` 第 21 行:
```toml
# 修改前
tokio = { version = "1", features = ["full"] }

# 修改后
tokio = { version = "1", features = ["sync", "rt", "macros"] }
```

- [ ] **Step 2: 验证 core 依赖**

检查 `crates/core/Cargo.toml:11` 确认已使用精确 features:
```toml
tokio = { version = "1", features = ["sync", "rt", "macros"] }
```

- [ ] **Step 3: 修改 server Cargo.toml**

```toml
# 修改前
tokio = { version = "1", features = ["full"] }

# 修改后
tokio = { version = "1", features = ["full"] }  # server 需要更多特性
```

**注意**: server 使用 `full` 因为有文件系统操作、网络等

- [ ] **Step 4: 运行测试验证**

```bash
cargo test -p vllm-core --no-fail-fast 2>&1 | tail -20
cargo build -p vllm-server 2>&1 | tail -10
```

- [ ] **Step 5: 提交**

```bash
git add Cargo.toml crates/server/Cargo.toml
git commit -m "perf: precise tokio features in workspace"
```

---

### Task 1.2: Release 编译优化

**Files:**
- Modify: `Cargo.toml:38-41` - release profile

- [ ] **Step 1: 修改 release profile**

```toml
# 修改前
[profile.release]
opt-level = 3
lto = "thin"
codegen-units = 1

# 修改后
[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
panic = "abort"
strip = true
```

- [ ] **Step 2: 记录基线编译时间**

```bash
time cargo build --release -p vllm-server 2>&1 | tail -5
```

记录输出中的 "Finished" 行时间

- [ ] **Step 3: 验证 release 构建**

```bash
cargo build --release -p vllm-server
ls -lh target/release/vllm-server 2>/dev/null || ls -lh target/release/deps/vllm-server*
```

- [ ] **Step 4: 提交**

```bash
git add Cargo.toml
git commit -m "perf: optimize release profile (fat lto, panic=abort, strip)"
```

---

## Phase 2: Cargo Feature 重构 (CUDA/GGUF 可选)

**复杂度**: 低-中 | **风险**: 低 | **目标 PR 数**: 1

### Task 2.1: Candle CUDA 可选

**Files:**
- Modify: `crates/model/Cargo.toml:11-12` - candle 依赖
- Modify: `crates/model/Cargo.toml:31-33` - features
- Modify: `crates/server/Cargo.toml:19` - candle 依赖
- Modify: `crates/dist/Cargo.toml` - candle 依赖

- [ ] **Step 1: 修改 model Cargo.toml**

```toml
# 修改前
candle-core = { version = "0.10.2", features = ["cuda"] }
candle-nn = { version = "0.10.2", features = ["cuda"] }

[features]
default = []
real_weights = ["tiktoken", "tokenizers"]

# 修改后
candle-core = "0.10.2"
candle-nn = "0.10.2"

[features]
default = []
cuda = ["candle-core/cuda", "candle-nn/cuda"]
real_weights = ["tiktoken", "tokenizers", "gguf"]
full = ["cuda", "real_weights"]
```

- [ ] **Step 2: 修改 server Cargo.toml**

```toml
# 修改前
candle-core = { version = "0.10.2", features = ["cuda"] }

# 修改后
candle-core = "0.10.2"
```

- [ ] **Step 3: 修改 dist Cargo.toml**

检查并更新:
```toml
candle-core = { version = "0.10.2", features = ["cuda"] }
```
改为:
```toml
candle-core = "0.10.2"
```

- [ ] **Step 4: 测试 CPU 构建**

```bash
cargo build --release -p vllm-server --no-default-features
```

- [ ] **Step 5: 测试 full 构建**

```bash
cargo build --release -p vllm-server --features full
```

- [ ] **Step 6: 提交**

```bash
git add crates/model/Cargo.toml crates/server/Cargo.toml crates/dist/Cargo.toml
git commit -m "feat: make candle cuda optional with feature flags"
```

---

### Task 2.2: GGUF 可选

**Files:**
- Modify: `crates/model/Cargo.toml:22` - gguf 依赖
- Modify: `crates/model/Cargo.toml` - features

- [ ] **Step 1: 修改 model Cargo.toml**

```toml
# 修改前
gguf = "0.1"

[features]
default = []
cuda = ["candle-core/cuda", "candle-nn/cuda"]
real_weights = ["tiktoken", "tokenizers", "gguf"]
full = ["cuda", "real_weights"]

# 修改后
gguf = { version = "0.1", optional = true }

[features]
default = []
cuda = ["candle-core/cuda", "candle-nn/cuda"]
gguf = ["dep:gguf"]
real_weights = ["tiktoken", "tokenizers"]
full = ["cuda", "gguf", "real_weights"]
```

- [ ] **Step 2: 更新 loader 代码以支持可选 gguf**

检查 `crates/model/src/loader/format/` 中使用 gguf 的代码

- [ ] **Step 3: 验证无 gguf 构建**

```bash
cargo build --release -p vllm-model --no-default-features
```

- [ ] **Step 4: 验证 gguf 支持**

```bash
cargo build --release -p vllm-model --features gguf
```

- [ ] **Step 5: 提交**

```bash
git add crates/model/Cargo.toml
git commit -m "feat: make gguf optional"
```

---

## Phase 3: 共享组件层 - Attention 提取

**复杂度**: 高 | **风险**: 中 | **目标 PR 数**: 1-2

### Task 3.1: 创建 attention 子模块目录

**Files:**
- Create: `crates/model/src/components/attention/mod.rs`
- Create: `crates/model/src/components/attention/gqa.rs`
- Modify: `crates/model/src/components/mod.rs`

- [ ] **Step 1: 创建目录结构**

```bash
mkdir -p crates/model/src/components/attention
```

- [ ] **Step 2: 创建 attention/gqa.rs**

从 `crates/model/src/qwen3/attention.rs` 提取 `GqaAttention` 结构体和实现:
```rust
use candle_core::{Result, Tensor, Device};
use crate::components::attention::{expand_kv, causal_mask, paged_attention};

#[derive(Debug, Clone)]
pub struct GqaAttention {
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
}

impl GqaAttention {
    pub fn new(num_q_heads: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        Self {
            num_q_heads,
            num_kv_heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
        }
    }

    pub fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
    ) -> Result<Tensor> {
        let q_heads = q.dims()[1];
        let k_heads = k.dims()[1];
        
        let k_expanded = if q_heads != k_heads {
            expand_kv(k, self.num_q_heads, self.num_kv_heads)?
        } else {
            k.clone()
        };
        
        paged_attention(q, &k_expanded, v, self.num_q_heads, self.head_dim)
    }
}
```

- [ ] **Step 3: 创建 attention/mod.rs**

```rust
pub mod gqa;

pub use gqa::GqaAttention;

pub use crate::components::attention::{
    expand_kv, causal_mask, paged_attention, tiled_attention,
};
```

- [ ] **Step 4: 更新 components/mod.rs**

```rust
pub mod attention;
pub mod mlp;
pub mod norm;
pub mod positional;

pub use attention::{GqaAttention, expand_kv, causal_mask, paged_attention, tiled_attention};
```

- [ ] **Step 5: 编译验证**

```bash
cargo build -p vllm-model 2>&1 | grep -E "error|warning"
```

- [ ] **Step 6: 运行测试**

```bash
cargo test -p vllm-model -- attention 2>&1 | tail -20
```

- [ ] **Step 7: 提交**

```bash
git add crates/model/src/components/attention/
git add crates/model/src/components/mod.rs
git commit -m "refactor: extract GqaAttention to shared components layer"
```

---

### Task 3.2: 迁移 qwen3 使用共享 Attention

**Files:**
- Modify: `crates/model/src/qwen3/attention.rs` - 使用共享实现
- Modify: `crates/model/src/qwen3/block.rs` - 更新 import

- [ ] **Step 1: 简化 qwen3/attention.rs**

将 `qwen3/attention.rs` 中的 GqaAttention 改为 re-export:
```rust
// 修改前 - 完整实现 (~700行)

// 修改后
pub use crate::components::attention::GqaAttention;

pub type Qwen3Attention = GqaAttention;
```

- [ ] **Step 2: 验证编译**

```bash
cargo build -p vllm-model 2>&1
```

- [ ] **Step 3: 运行 qwen3 相关测试**

```bash
cargo test -p vllm-model -- qwen3 2>&1 | tail -20
```

- [ ] **Step 4: 提交**

```bash
git add crates/model/src/qwen3/
git commit -m "refactor(qwen3): use shared GqaAttention from components"
```

---

### Task 3.3: 迁移 llama/mistral/mixtral

**Files:**
- Modify: `crates/model/src/llama/` - 更新 attention 使用
- Modify: `crates/model/src/mistral/` - 更新 attention 使用
- Modify: `crates/model/src/mixtral/` - 更新 attention 使用

- [ ] **Step 1: 检查 llama 是否有独立的 attention 实现**

如果 `llama/attention.rs` 存在，迁移方式同上

- [ ] **Step 2: 更新各架构的 block.rs**

确保使用共享的 `GqaAttention`:
```rust
use crate::components::GqaAttention;
```

- [ ] **Step 3: 编译验证**

```bash
cargo build -p vllm-model 2>&1 | grep -E "^error"
```

- [ ] **Step 4: 测试所有架构**

```bash
cargo test -p vllm-model 2>&1 | tail -30
```

- [ ] **Step 5: 提交**

```bash
git add crates/model/src/llama/ crates/model/src/mistral/ crates/model/src/mixtral/
git commit -m "refactor: migrate llama/mistral/mixtral to shared GqaAttention"
```

---

## Phase 4: 共享组件层 - MLP/Norm 提取

**复杂度**: 中 | **风险**: 低-中 | **目标 PR 数**: 1

### Task 4.1: 创建 mlp 子模块

**Files:**
- Create: `crates/model/src/components/mlp/mod.rs`
- Create: `crates/model/src/components/mlp/swiglu.rs`
- Modify: `crates/model/src/components/mod.rs`

- [ ] **Step 1: 创建目录结构**

```bash
mkdir -p crates/model/src/components/mlp
```

- [ ] **Step 2: 创建 mlp/swiglu.rs**

从各架构提取 SwiGLU 实现:
```rust
use candle_core::{Result, Tensor, Device};
use candle_nn::Linear;

#[derive(Debug, Clone)]
pub struct SwiGLU {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl SwiGLU {
    pub fn new(vocab_size: usize, hidden_size: usize, intermediate_size: usize, bias: bool, device: &Device) -> Result<Self> {
        let gate_proj = candle_nn::linear(hidden_size, intermediate_size, bias, device)?;
        let up_proj = candle_nn::linear(hidden_size, intermediate_size, bias, device)?;
        let down_proj = candle_nn::linear(intermediate_size, hidden_size, bias, device)?;
        Ok(Self { gate_proj, up_proj, down_proj })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let activated = candle_nn::activation::silu(&gate)?;
        let down = self.down_proj.forward(&(&activated * &up)?)?;
        Ok(down)
    }
}
```

- [ ] **Step 3: 创建 mlp/mod.rs**

```rust
pub mod swiglu;

pub use swiglu::SwiGLU;
```

- [ ] **Step 4: 更新 components/mod.rs**

```rust
pub mod mlp;
pub use mlp::SwiGLU;
```

- [ ] **Step 5: 编译验证**

```bash
cargo build -p vllm-model 2>&1 | grep -E "^error"
```

- [ ] **Step 6: 提交**

```bash
git add crates/model/src/components/mlp/
git commit -m "refactor: extract SwiGLU to shared components"
```

---

### Task 4.2: 创建 norm 子模块

**Files:**
- Create: `crates/model/src/components/norm/mod.rs`
- Create: `crates/model/src/components/norm/rms_norm.rs`
- Modify: `crates/model/src/components/mod.rs`

- [ ] **Step 1: 创建目录结构**

```bash
mkdir -p crates/model/src/components/norm
```

- [ ] **Step 2: 创建 norm/rms_norm.rs**

```rust
use candle_core::{Result, Tensor, Device};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    pub fn new(dim: usize, eps: f64, device: &Device) -> Result<Self> {
        let weight = Tensor::ones(dim, candle_core::DType::F32, device)?;
        Ok(Self { weight, eps })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let hidden_size = x.dims().last().copied().unwrap_or(1);
        let x = x.reshape((*, hidden_size))?;
        let norm = (x.sqr()? + self.eps)?.rsqrt()? * &x;
        (norm * &self.weight)?.reshape(x.dims())
    }
}
```

- [ ] **Step 3: 创建 norm/mod.rs**

```rust
pub mod rms_norm;

pub use rms_norm::RmsNorm;
```

- [ ] **Step 4: 更新 components/mod.rs**

```rust
pub mod norm;
pub use norm::RmsNorm;
```

- [ ] **Step 5: 编译验证**

```bash
cargo build -p vllm-model 2>&1 | grep -E "^error"
```

- [ ] **Step 6: 提交**

```bash
git add crates/model/src/components/norm/
git commit -m "refactor: extract RmsNorm to shared components"
```

---

## Phase 5: 共享组件层 - RoPE 提取

**复杂度**: 中 | **风险**: 低 | **目标 PR 数**: 1

### Task 5.1: 创建 positional 子模块

**Files:**
- Create: `crates/model/src/components/positional/mod.rs`
- Create: `crates/model/src/components/positional/rope.rs`
- Create: `crates/model/src/components/positional/mrope.rs`
- Modify: `crates/model/src/components/mod.rs`

- [ ] **Step 1: 创建目录结构**

```bash
mkdir -p crates/model/src/components/positional
```

- [ ] **Step 2: 创建 positional/rope.rs**

从各架构提取标准 RoPE 实现:
```rust
use candle_core::{Result, Tensor, Device};

#[derive(Debug, Clone)]
pub struct RoPE {
    cos: Tensor,
    sin: Tensor,
}

impl RoPE {
    pub fn new(dim: usize, max_position: usize, base: f64, device: &Device) -> Result<Self> {
        let inv_freq: Vec<f32> = (0..dim / 2)
            .map(|i| 1.0 / (base.powf(4.0 * i as f64 / dim as f64) as f32))
            .collect();
        
        let inv_freq = Tensor::new(&inv_freq, device)?;
        let positions: Vec<f32> = (0..max_position as usize)
            .map(|p| p as f32)
            .collect();
        let positions = Tensor::new(&positions, device)?;
        
        let angles = positions.unsqueeze(1)? * inv_freq.unsqueeze(0)?;
        let cos = angles.cos()?;
        let sin = angles.sin()?;
        
        Ok(Self { cos, sin })
    }

    pub fn forward(&self, q: &Tensor, k: &Tensor, start_pos: usize) -> Result<(Tensor, Tensor)> {
        let q_len = q.dims()[1];
        let cos = self.cos.narrow(0, start_pos, q_len)?;
        let sin = self.sin.narrow(0, start_pos, q_len)?;
        
        // 应用 RoPE
        let q1 = (&q * &cos.unsqueeze(1)? - &q.t()?.unsqueeze(2)? * &sin.unsqueeze(1)?)?.t()?;
        let k1 = (&k * &cos.unsqueeze(1)? - &k.t()?.unsqueeze(2)? * &sin.unsqueeze(1)?)?.t()?;
        
        Ok((q1, k1))
    }
}
```

- [ ] **Step 3: 创建 positional/mrope.rs**

MRoPE 实现用于 Qwen3.5 (多维 RoPE)

- [ ] **Step 4: 创建 positional/mod.rs**

```rust
pub mod rope;
pub mod mrope;

pub use rope::RoPE;
pub use mrope::MRoPE;
```

- [ ] **Step 5: 更新 components/mod.rs**

```rust
pub mod positional;
pub use positional::{RoPE, MRoPE};
```

- [ ] **Step 6: 编译验证**

```bash
cargo build -p vllm-model 2>&1 | grep -E "^error"
```

- [ ] **Step 7: 提交**

```bash
git add crates/model/src/components/positional/
git commit -m "refactor: extract RoPE variants to shared components"
```

---

## Phase 6: Block 基类设计

**复杂度**: 高 | **风险**: 中 | **目标 PR 数**: 1-2

### Task 6.1: 设计 TransformerBlock 基类

**Files:**
- Create: `crates/model/src/components/block.rs`
- Modify: `crates/model/src/components/mod.rs`

- [ ] **Step 1: 创建 components/block.rs**

```rust
use candle_core::{Result, Tensor};
use crate::components::{GqaAttention, SwiGLU, RmsNorm};

#[derive(Debug, Clone)]
pub struct TransformerBlockConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub head_dim: usize,
}

pub struct TransformerBlock {
    pub attention: GqaAttention,
    pub feed_forward: SwiGLU,
    pub input_layernorm: RmsNorm,
    pub post_attention_layernorm: RmsNorm,
}

impl TransformerBlock {
    pub fn new(config: &TransformerBlockConfig, device: &Device) -> Result<Self> {
        let attention = GqaAttention::new(
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
        );
        
        let feed_forward = SwiGLU::new(
            config.hidden_size,
            config.hidden_size,
            config.intermediate_size,
            false,
            device,
        )?;
        
        let input_layernorm = RmsNorm::new(config.head_dim * config.num_attention_heads, 1e-5, device)?;
        let post_attention_layernorm = RmsNorm::new(config.head_dim * config.num_attention_heads, 1e-5, device)?;
        
        Ok(Self {
            attention,
            feed_forward,
            input_layernorm,
            post_attention_layernorm,
        })
    }
    
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x.clone();
        let x = self.input_layernorm.forward(x)?;
        let x = self.attention.forward(&x, &x, &x)?;
        let x = (&x + &residual)?;
        
        let residual = &x;
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = self.feed_forward.forward(&x)?;
        (&x + &residual)
    }
}
```

- [ ] **Step 2: 更新 components/mod.rs**

```rust
pub mod block;
pub use block::{TransformerBlock, TransformerBlockConfig};
```

- [ ] **Step 3: 编译验证**

```bash
cargo build -p vllm-model 2>&1 | grep -E "^error"
```

- [ ] **Step 4: 提交**

```bash
git add crates/model/src/components/block.rs
git commit -m "feat: add TransformerBlock base class"
```

---

## Phase 7: 架构迁移

**复杂度**: 高 | **风险**: 中 | **目标 PR 数**: 4+ (每个架构一个)

### Task 7.1: 迁移 Llama

**Files:**
- Modify: `crates/model/src/llama/block.rs` - 使用基类
- Modify: `crates/model/src/llama/model.rs` - 更新 Block 类型

- [ ] **Step 1: 分析当前 LlamaBlock 实现**

读取 `crates/model/src/llama/block.rs` 了解当前结构

- [ ] **Step 2: 迁移到基类**

```rust
use crate::components::{TransformerBlock, TransformerBlockConfig};

pub type LlamaBlock = TransformerBlock;

impl LlamaBlock {
    pub fn from_config(config: &LlamaConfig, device: &Device) -> Result<Self> {
        let block_config = TransformerBlockConfig {
            hidden_size: config.hidden_size,
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads,
            intermediate_size: config.intermediate_size,
            head_dim: config.head_dim,
        };
        TransformerBlock::new(&block_config, device)
    }
}
```

- [ ] **Step 3: 验证编译**

```bash
cargo build -p vllm-model 2>&1 | grep -E "^error"
```

- [ ] **Step 4: 运行 llama 测试**

```bash
cargo test -p vllm-model -- llama 2>&1 | tail -20
```

- [ ] **Step 5: 提交**

```bash
git add crates/model/src/llama/
git commit -m "refactor(llama): migrate to TransformerBlock base class"
```

---

### Task 7.2-7.5: 迁移 Mistral, Qwen3, Mixtral, Gemma4

对每个架构重复 Task 7.1 的步骤:
1. 分析当前实现
2. 迁移到基类 (或创建架构特定扩展)
3. 验证编译
4. 运行测试
5. 提交

**注意**: Mistral 需要处理 sliding window，Qwen3 需要处理 QK-Norm，Mixtral 需要处理 MoE

---

## Phase 8: core→model 解耦 + 文档

**复杂度**: 中 | **风险**: 低 | **目标 PR 数**: 1-2

### Task 8.1: 解耦 core 和 model

**Files:**
- Create: `crates/traits/src/kernels.rs` - Kernel trait 定义
- Modify: `crates/core/Cargo.toml` - 移除 model 依赖
- Modify: `crates/core/src/engine.rs` - 使用 trait 而非具体实现

- [ ] **Step 1: 分析当前 core 对 model 的依赖**

```bash
grep -r "vllm_model" crates/core/src/ | head -20
```

- [ ] **Step 2: 创建 traits/kernels.rs**

```rust
use candle_core::{Result, Tensor};

pub trait CudaGraph {
    fn capture(&mut self) -> Result<()>;
    fn replay(&self) -> Result<()>;
}

pub trait FlashAttention {
    fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        attn_mask: Option<&Tensor>,
    ) -> Result<Tensor>;
}
```

- [ ] **Step 3: 修改 core Cargo.toml**

```toml
# 修改前
vllm-model = { path = "../model" }

# 修改后
# 移除 vllm-model 依赖，改为仅依赖 vllm-traits
```

- [ ] **Step 4: 更新 engine.rs 使用 trait**

使用 `kernels::CudaGraph` trait 而非具体 `vllm_model::CudaGraph`

- [ ] **Step 5: 在 model 中实现 trait**

在 `vllm-model` 中为具体类型实现 `kernels::CudaGraph`

- [ ] **Step 6: 编译验证**

```bash
cargo build --workspace 2>&1 | grep -E "^error"
```

- [ ] **Step 7: 提交**

```bash
git add crates/traits/src/kernels.rs
git add crates/core/Cargo.toml crates/core/src/
git add crates/model/src/kernels/
git commit -m "refactor: decouple core from model dependency"
```

---

### Task 8.2: 文档完善

**Files:**
- Create: `docs/adr/` - 架构决策记录
- Modify: `docs/superpowers/specs/` - 更新设计文档

- [ ] **Step 1: 创建 ADR 目录**

```bash
mkdir -p docs/adr
```

- [ ] **Step 2: 创建 ADR-001: 组件共享策略**

记录为什么选择当前组件提取策略

- [ ] **Step 3: 创建 ADR-002: Feature Flag 设计**

记录 Cargo feature 的设计决策

- [ ] **Step 4: 更新 SPEC 文档**

在设计文档中添加已完成标记

- [ ] **Step 5: 提交**

```bash
git add docs/
git commit -m "docs: add architecture decision records"
```

---

## 验证清单

每个 Phase 完成后验证:

- [ ] `cargo build --workspace` 成功
- [ ] `cargo test --workspace` 成功
- [ ] `cargo clippy --workspace -- -D warnings` 无警告
- [ ] `cargo fmt --all --check` 通过
- [ ] 对比编译时间改善
- [ ] 对比二进制大小改善

---

## 执行选项

**Plan complete and saved.**

**Two execution options:**

**1. Subagent-Driven (recommended)** - Dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
