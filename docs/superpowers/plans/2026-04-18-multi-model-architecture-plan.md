# 多模型架构优化实现计划

> **状态**: ✅ 已完成
> 
> **完成日期**: 2026-04-18
> 
> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 重构 vllm-lite 多模型架构，消除代码重复，支持动态架构注册，提高组件可组合性

**Architecture:** 基于 trait 的组件化设计，分层抽象 (NormLayer → Attention → MlpLayer → TransformerBlock → Architecture)，使用注册机制替代 enum + match 分发

**Tech Stack:** Rust, candle-core, vllm-traits

**结果**: 18 个提交, 852 测试全部通过

---

## 文件结构

```
crates/model/src/
├── components/
│   ├── mod.rs           # 修改: 导出新 trait
│   ├── norm.rs          # 新增: NormLayer trait + 实现
│   ├── mlp.rs           # 新增: MlpLayer trait + 实现
│   ├── block.rs         # 新增: TransformerBlock trait
│   └── block_impl.rs    # 新增: GenericTransformerBlock
├── arch/
│   ├── mod.rs           # 新增: Architecture trait
│   └── registry.rs      # 新增: ArchitectureRegistry
├── llama/
│   ├── arch.rs          # 新增: LlamaArchitecture impl
│   └── register.rs      # 新增: llama 注册函数
├── mistral/
│   ├── arch.rs          # 新增: MistralArchitecture impl
│   └── register.rs      # 新增: mistral 注册函数
├── qwen3/
│   ├── arch.rs          # 新增: Qwen3Architecture impl
│   └── register.rs      # 新增: qwen3 注册函数
├── qwen3_5/
│   ├── arch.rs          # 新增: Qwen35Architecture impl
│   └── register.rs      # 新增: qwen35 注册函数
├── gemma4/
│   ├── arch.rs          # 新增: Gemma4Architecture impl
│   └── register.rs      # 新增: gemma4 注册函数
├── mixtral/
│   ├── arch.rs          # 新增: MixtralArchitecture impl
│   └── register.rs      # 新增: mixtral 注册函数
└── loader/
    ├── builder.rs       # 修改: 使用注册机制
    └── mod.rs           # 修改: 导出注册表
```

---

## Phase 1: Trait 基础

### Task 1: 定义 NormLayer trait

**Files:**
- Create: `crates/model/src/components/norm.rs`
- Modify: `crates/model/src/components/mod.rs`

- [ ] **Step 1: 创建 norm.rs 定义 NormLayer trait**

```rust
// crates/model/src/components/norm.rs

use candle_core::{Result, Tensor};

pub trait NormLayer: Send + Sync {
    fn forward(&self, x: &Tensor) -> Result<Tensor>;
    fn hidden_size(&self) -> usize;
}

#[cfg(feature = "candle")]
pub struct RmsNorm {
    weight: Tensor,
    eps: f32,
}

#[cfg(feature = "candle")]
impl RmsNorm {
    pub fn new(weight: Tensor, eps: f32) -> Self {
        Self { weight, eps }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let hidden_size = *x.dims().last().unwrap();
        let dims = x.dims();
        let batch_size = dims[0];
        let seq_len = dims[1];
        let total_len = batch_size * seq_len;

        let x_flat = x.reshape((total_len, hidden_size))?;
        let variance = x_flat.sqr()?.mean(1)?;
        let x_normed = x_flat.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        let x = x_normed.broadcast_mul(&self.weight)?;

        x.reshape((batch_size, seq_len, hidden_size))
    }
}

#[cfg(feature = "candle")]
impl NormLayer for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward(x)
    }

    fn hidden_size(&self) -> usize {
        self.weight.dims().last().copied().unwrap_or(0)
    }
}

#[cfg(feature = "candle")]
pub struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f32,
}

#[cfg(feature = "candle")]
impl LayerNorm {
    pub fn new(weight: Tensor, bias: Tensor, eps: f32) -> Self {
        Self { weight, bias, eps }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        candle_nn::layer_norm_custom(&self.weight, &self.bias, self.eps, x)
    }
}

#[cfg(feature = "candle")]
impl NormLayer for LayerNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward(x)
    }

    fn hidden_size(&self) -> usize {
        self.weight.dims().last().copied().unwrap_or(0)
    }
}
```

- [ ] **Step 2: 添加测试**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    #[test]
    fn test_rms_norm_shape_preserved() {
        let weight = Tensor::ones((32,), DType::F32, &Device::Cpu).unwrap();
        let rms = RmsNorm::new(weight, 1e-6);
        
        let input = Tensor::ones((2, 10, 32), DType::F32, &Device::Cpu).unwrap();
        let output = rms.forward(&input).unwrap();
        
        assert_eq!(output.dims(), &[2, 10, 32]);
    }

    #[test]
    fn test_layer_norm_shape_preserved() {
        let weight = Tensor::ones((32,), DType::F32, &Device::Cpu).unwrap();
        let bias = Tensor::zeros((32,), DType::F32, &Device::Cpu).unwrap();
        let ln = LayerNorm::new(weight, bias, 1e-6);
        
        let input = Tensor::ones((2, 10, 32), DType::F32, &Device::Cpu).unwrap();
        let output = ln.forward(&input).unwrap();
        
        assert_eq!(output.dims(), &[2, 10, 32]);
    }
}
```

- [ ] **Step 3: 更新 mod.rs 导出**

```rust
pub mod norm;

pub use norm::{LayerNorm, NormLayer, RmsNorm};
```

- [ ] **Step 4: 运行测试**

Run: `cargo test -p vllm-model -- norm --nocapture`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add crates/model/src/components/norm.rs crates/model/src/components/mod.rs
git commit -m "feat(model): add NormLayer trait with RmsNorm and LayerNorm implementations"
```

---

### Task 2: 定义 MlpLayer trait

**Files:**
- Create: `crates/model/src/components/mlp.rs`
- Modify: `crates/model/src/components/mod.rs`

- [ ] **Step 1: 创建 mlp.rs 定义 MlpLayer trait**

```rust
// crates/model/src/components/mlp.rs

use candle_core::{Result, Tensor};

pub trait MlpLayer: Send + Sync {
    fn forward(&self, x: &Tensor) -> Result<Tensor>;
}
```

- [ ] **Step 2: 更新 mod.rs 导出**

```rust
pub mod norm;
pub mod mlp;

pub use mlp::MlpLayer;
pub use norm::{LayerNorm, NormLayer, RmsNorm};
```

- [ ] **Step 3: 提交**

```bash
git add crates/model/src/components/mlp.rs crates/model/src/components/mod.rs
git commit -m "feat(model): add MlpLayer trait stub for component abstraction"
```

---

### Task 3: 定义 TransformerBlock trait

**Files:**
- Create: `crates/model/src/components/block.rs`
- Modify: `crates/model/src/components/mod.rs`

- [ ] **Step 1: 创建 block.rs 定义 TransformerBlock trait**

```rust
// crates/model/src/components/block.rs

#[cfg(feature = "candle")]
use candle_core::{Result, Tensor};

#[cfg(feature = "candle")]
pub trait TransformerBlock: Send + Sync {
    fn forward(
        &mut self,
        hidden_states: &Tensor,
        positions: &[usize],
        kv_block_ids: &[usize],
        num_computed: usize,
        is_prefill: bool,
    ) -> Result<Tensor>;

    fn inner_dim(&self) -> usize;
    fn num_kv_heads(&self) -> usize;
}
```

- [ ] **Step 2: 更新 mod.rs 导出**

```rust
pub mod block;
pub mod mlp;
pub mod norm;

pub use block::TransformerBlock;
pub use mlp::MlpLayer;
pub use norm::{LayerNorm, NormLayer, RmsNorm};
```

- [ ] **Step 3: 提交**

```bash
git add crates/model/src/components/block.rs crates/model/src/components/mod.rs
git commit -m "feat(model): add TransformerBlock trait for layer abstraction"
```

---

## Phase 2: 架构抽象

### Task 4: 创建 Architecture trait 和 Registry

**Files:**
- Create: `crates/model/src/arch/mod.rs`
- Create: `crates/model/src/arch/registry.rs`
- Modify: `crates/model/src/lib.rs`

- [ ] **Step 1: 创建 arch/mod.rs 定义 Architecture trait**

```rust
// crates/model/src/arch/mod.rs

use candle_core::{Device, Result, Tensor};
use std::collections::HashMap;
use vllm_traits::ModelBackend;

use crate::config::ModelConfig;
use crate::components::TransformerBlock;

pub trait Architecture: Send + Sync + 'static {
    const NAME: &'static str;

    fn detect(config_json: &serde_json::Value) -> bool;

    fn create_block(
        config: &ModelConfig,
        layer_idx: usize,
        weights: &HashMap<String, Tensor>,
        device: &Device,
    ) -> Result<Box<dyn TransformerBlock>>;

    fn create_model(
        config: ModelConfig,
        device: Device,
        weights: HashMap<String, Tensor>,
        num_kv_blocks: usize,
    ) -> Result<Box<dyn ModelBackend>>;

    fn remap_weights(weights: HashMap<String, Tensor>) -> HashMap<String, Tensor> {
        weights
    }
}
```

- [ ] **Step 2: 创建 arch/registry.rs 定义注册表**

```rust
// crates/model/src/arch/registry.rs

use std::collections::HashMap;
use std::sync::RwLock;
use once_cell::sync::Lazy;
use serde_json::Value;

use super::Architecture;

pub struct ArchitectureRegistry {
    architectures: RwLock<HashMap<String, Box<dyn Architecture>>>,
}

impl ArchitectureRegistry {
    pub fn new() -> Self {
        Self {
            architectures: RwLock::new(HashMap::new()),
        }
    }

    pub fn register<A: Architecture + 'static>(&self) {
        let arch = A::NAME.to_string();
        let builder: Box<dyn Architecture> = Box::new(A);
        self.architectures
            .write()
            .unwrap()
            .insert(arch, builder);
    }

    pub fn get(&self, name: &str) -> Option<Box<dyn Architecture>> {
        self.architectures.read().unwrap().get(name).cloned()
    }

    pub fn detect(&self, config_json: &Value) -> Option<String> {
        let regs = self.architectures.read().unwrap();
        for (name, arch) in regs.iter() {
            if arch.detect(config_json) {
                return Some(name.clone());
            }
        }
        None
    }
}

pub static ARCHITECTURE_REGISTRY: Lazy<ArchitectureRegistry> = Lazy::new(|| {
    let registry = ArchitectureRegistry::new();
    // 注册将在 loader 模块初始化时进行
    registry
});

pub fn register_all_archs(registry: &ArchitectureRegistry) {
    // 延迟导入避免循环依赖
    crate::llama::register::register(registry);
    crate::mistral::register::register(registry);
    crate::qwen3::register::register(registry);
    crate::qwen3_5::register::register(registry);
    crate::gemma4::register::register(registry);
    crate::mixtral::register::register(registry);
}
```

- [ ] **Step 3: 更新 lib.rs 导出**

```rust
pub mod arch;
pub mod components;
// ... existing mods
pub use arch::{Architecture, ArchitectureRegistry, ARCHITECTURE_REGISTRY, register_all_archs};
```

- [ ] **Step 4: 添加测试**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_registry_detect_empty() {
        let registry = ArchitectureRegistry::new();
        let config = json!({"model_type": "llama"});
        assert!(registry.detect(&config).is_none());
    }

    #[test]
    fn test_registry_register_and_get() {
        let registry = ArchitectureRegistry::new();
        // 注册后可以获取
        // 注意: 实际测试需要实现 Architecture trait
    }
}
```

- [ ] **Step 5: 提交**

```bash
git add crates/model/src/arch/ crates/model/src/lib.rs
git commit -m "feat(model): add Architecture trait and Registry for dynamic registration"
```

---

## Phase 3: 架构迁移

### Task 5: 为 Llama 添加 Architecture 实现

**Files:**
- Create: `crates/model/src/llama/arch.rs`
- Create: `crates/model/src/llama/register.rs`
- Modify: `crates/model/src/llama/mod.rs`

- [ ] **Step 1: 创建 llama/arch.rs 实现 LlamaArchitecture**

```rust
// crates/model/src/llama/arch.rs

use crate::arch::Architecture;
use crate::components::TransformerBlock;
use crate::config::ModelConfig;
use candle_core::{Device, Result, Tensor};
use std::collections::HashMap;
use vllm_traits::ModelBackend;

use super::block::LlamaBlock;
use super::model::LlamaModel;

pub struct LlamaArchitecture;

impl Architecture for LlamaArchitecture {
    const NAME: &'static str = "llama";

    fn detect(config_json: &serde_json::Value) -> bool {
        let model_type = config_json
            .get("model_type")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        matches!(
            model_type.to_lowercase().as_str(),
            "llama" | "llama2" | "llama3"
        )
    }

    fn create_block(
        config: &ModelConfig,
        layer_idx: usize,
        weights: &HashMap<String, Tensor>,
        _device: &Device,
    ) -> Result<Box<dyn TransformerBlock>> {
        Ok(Box::new(LlamaBlock::from_weights(config, layer_idx, weights)?))
    }

    fn create_model(
        config: ModelConfig,
        device: Device,
        weights: HashMap<String, Tensor>,
        num_kv_blocks: usize,
    ) -> Result<Box<dyn ModelBackend>> {
        let model = LlamaModel::from_weights(config, device, weights, num_kv_blocks)?;
        Ok(Box::new(model))
    }
}
```

- [ ] **Step 2: 创建 llama/register.rs 注册函数**

```rust
// crates/model/src/llama/register.rs

use crate::arch::{Architecture, ArchitectureRegistry};
use super::arch::LlamaArchitecture;

pub fn register(registry: &ArchitectureRegistry) {
    registry.register::<LlamaArchitecture>();
}
```

- [ ] **Step 3: 更新 llama/mod.rs 导出注册模块**

```rust
pub mod arch;
pub mod block;
pub mod model;
pub mod register;

pub use model::LlamaModel;
```

- [ ] **Step 4: 添加测试**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_llama_architecture_detect() {
        for model_type in ["llama", "llama2", "llama3"] {
            let config = json!({"model_type": model_type});
            assert!(LlamaArchitecture::detect(&config), "Failed for {}", model_type);
        }
    }

    #[test]
    fn test_llama_architecture_name() {
        assert_eq!(LlamaArchitecture::NAME, "llama");
    }
}
```

- [ ] **Step 5: 运行测试**

Run: `cargo test -p vllm-model -- llama::arch --nocapture`
Expected: PASS

- [ ] **Step 6: 提交**

```bash
git add crates/model/src/llama/arch.rs crates/model/src/llama/register.rs crates/model/src/llama/mod.rs
git commit -m "feat(model): add LlamaArchitecture impl for registry"
```

---

### Task 6: 为 Mistral 添加 Architecture 实现

**Files:**
- Create: `crates/model/src/mistral/arch.rs`
- Create: `crates/model/src/mistral/register.rs`
- Modify: `crates/model/src/mistral/mod.rs`

- [ ] **Step 1: 创建 mistral/arch.rs 实现 MistralArchitecture**

```rust
// crates/model/src/mistral/arch.rs

use crate::arch::Architecture;
use crate::components::TransformerBlock;
use crate::config::ModelConfig;
use candle_core::{Device, Result, Tensor};
use std::collections::HashMap;
use vllm_traits::ModelBackend;

use super::block::MistralBlock;
use super::model::MistralModel;

pub struct MistralArchitecture;

impl Architecture for MistralArchitecture {
    const NAME: &'static str = "mistral";

    fn detect(config_json: &serde_json::Value) -> bool {
        config_json
            .get("model_type")
            .and_then(|v| v.as_str())
            .map(|s| s.to_lowercase() == "mistral")
            .unwrap_or(false)
    }

    fn create_block(
        config: &ModelConfig,
        layer_idx: usize,
        weights: &HashMap<String, Tensor>,
        _device: &Device,
    ) -> Result<Box<dyn TransformerBlock>> {
        Ok(Box::new(MistralBlock::from_weights(config, layer_idx, weights)?))
    }

    fn create_model(
        config: ModelConfig,
        device: Device,
        weights: HashMap<String, Tensor>,
        num_kv_blocks: usize,
    ) -> Result<Box<dyn ModelBackend>> {
        let model = MistralModel::from_weights(config, device, weights, num_kv_blocks)?;
        Ok(Box::new(model))
    }
}
```

- [ ] **Step 2: 创建 mistral/register.rs 注册函数**

```rust
// crates/model/src/mistral/register.rs

use crate::arch::{Architecture, ArchitectureRegistry};
use super::arch::MistralArchitecture;

pub fn register(registry: &ArchitectureRegistry) {
    registry.register::<MistralArchitecture>();
}
```

- [ ] **Step 3: 更新 mistral/mod.rs 导出**

```rust
pub mod arch;
pub mod block;
pub mod model;
pub mod register;

pub use model::MistralModel;
```

- [ ] **Step 4: 提交**

```bash
git add crates/model/src/mistral/arch.rs crates/model/src/mistral/register.rs crates/model/src/mistral/mod.rs
git commit -m "feat(model): add MistralArchitecture impl for registry"
```

---

### Task 7-12: 为其他架构添加实现

按照 Task 5/6 的模式，为以下架构添加 Architecture 实现:

- Task 7: Qwen3 (`crates/model/src/qwen3/`)
- Task 8: Qwen35 (`crates/model/src/qwen3_5/`)
- Task 9: Gemma4 (`crates/model/src/gemma4/`)
- Task 10: Mixtral (`crates/model/src/mixtral/`)

每个任务包含:
1. 创建 `arch.rs` 实现 Architecture trait
2. 创建 `register.rs` 注册函数
3. 更新 `mod.rs` 导出
4. 添加测试
5. 提交

---

## Phase 4: 加载器集成

### Task 13: 更新 ModelLoader 使用注册表

**Files:**
- Modify: `crates/model/src/loader/builder.rs`
- Modify: `crates/model/src/loader/mod.rs`

- [ ] **Step 1: 更新 loader/builder.rs 使用注册表**

找到 `load()` 方法，替换 match 语句:

```rust
// 替换前的代码 (需要删除):
// match self.inner.architecture {
//     Architecture::Llama => { ... }
//     Architecture::Mistral => { ... }
//     ...
// }

// 替换后的代码:
use crate::arch::{ArchitectureRegistry, ARCHITECTURE_REGISTRY, register_all_archs};

pub fn load(&self) -> Result<Box<dyn vllm_traits::ModelBackend>> {
    // 确保所有架构已注册
    register_all_archs(&ARCHITECTURE_REGISTRY);

    // 检测架构
    let arch_name = ARCHITECTURE_REGISTRY
        .detect(&self.inner.config_json)
        .ok_or_else(|| candle_core::Error::msg("Unsupported architecture"))?;

    let arch = ARCHITECTURE_REGISTRY
        .get(&arch_name)
        .ok_or_else(|| candle_core::Error::msg("Architecture not found"))?;

    let config = ModelConfig::from_config_json(&self.inner.config_json)?;
    let mut weights = self.load_weights()?;
    let weights = arch.remap_weights(weights);

    arch.create_model(config, self.inner.device.clone(), weights, self.inner.num_kv_blocks)
}
```

- [ ] **Step 2: 运行测试验证加载器工作**

Run: `cargo test -p vllm-model -- loader --nocapture`
Expected: PASS

- [ ] **Step 3: 提交**

```bash
git add crates/model/src/loader/builder.rs
git commit -m "refactor(model): use ArchitectureRegistry instead of enum match in ModelLoader"
```

---

## Phase 5: 清理

### Task 14: 删除旧的 detect_architecture 函数

**Files:**
- Modify: `crates/model/src/loader/mod.rs`

- [ ] **Step 1: 移除旧代码**

删除 `detect_architecture()` 函数和相关的 Architecture enum 用法

- [ ] **Step 2: 运行完整测试**

Run: `cargo test -p vllm-model -- --nocapture`
Expected: PASS

- [ ] **Step 3: 提交**

```bash
git add crates/model/src/loader/mod.rs
git commit -m "refactor(model): remove legacy detect_architecture after registry migration"
```

---

### Task 15: 最终验证

- [ ] **Step 1: 运行完整 CI**

Run: `just ci`
Expected: 所有检查通过

- [ ] **Step 2: 确认代码行数减少**

```bash
git diff --stat HEAD~15 HEAD -- crates/model/src/
```

- [ ] **Step 3: 提交最终变更**

```bash
git commit -m "chore(model): complete multi-model architecture refactoring"
```

---

## 预期结果

| 指标 | 预期值 |
|------|--------|
| 新增文件 | ~18 个 |
| 修改文件 | ~10 个 |
| 减少代码行数 | ~500 行 (33%) |
| 测试覆盖 | 保持 100% |

---

## 验证检查清单

- [ ] 所有 Architecture trait 实现通过测试
- [ ] ModelLoader 使用注册表成功加载所有架构
- [ ] `cargo clippy --workspace -- -D warnings` 无警告
- [ ] `cargo fmt --all --check` 通过
- `just nextest` 通过
- 新架构添加只需 3 步: 创建 `arch.rs`, `register.rs`, 更新 `mod.rs`
