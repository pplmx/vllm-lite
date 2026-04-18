# 多模型架构优化 - Trait 方案设计

## 概述

为 vllm-lite 设计一套完美的 trait-based 架构，支持多模型、高可组合性、零运行时开销。

## 核心设计原则

1. **最小接口**: trait 只暴露必要方法
2. **可组合性**: Norm、Attention、MLP 可独立替换
3. **零运行时开销**: 泛型 + trait bound 优化
4. **易于测试**: 每个组件可独立 mock
5. **编译时安全**: 穷举检查不丢失

---

## 1. 组件 Trait 设计

### 1.1 NormLayer Trait

```rust
// crates/model/src/components/norm.rs

pub trait NormLayer: Send + Sync {
    fn forward(&self, x: &Tensor) -> Result<Tensor>;

    fn hidden_size(&self) -> usize;
}

pub struct RmsNorm {
    weight: Tensor,
    eps: f32,
}

pub struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f32,
}
```

### 1.2 Attention Trait

```rust
// crates/model/src/components/attention.rs

#[cfg(feature = "candle")]
pub trait Attention: Send + Sync {
    fn forward(
        &self,
        x: &Tensor,
        positions: &[usize],
        kv_cache: Option<(&Tensor, &Tensor)>,
        is_prefill: bool,
    ) -> Result<Tensor>;

    fn num_heads(&self) -> usize;
    fn num_kv_heads(&self) -> usize;
    fn head_dim(&self) -> usize;
}

#[cfg(feature = "candle")]
pub struct AttentionInput {
    pub query: Tensor,
    pub key: Tensor,
    pub value: Tensor,
}

#[cfg(feature = "candle")]
pub trait AttentionBuilder: Send + Sync {
    fn build(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope_theta: f32,
        config: &AttentionConfig,
    ) -> Result<Box<dyn Attention>>;

    fn build_with_weights(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope_theta: f32,
        q_weight: Tensor,
        k_weight: Tensor,
        v_weight: Tensor,
        o_weight: Tensor,
        config: &AttentionConfig,
    ) -> Result<Box<dyn Attention>>;
}
```

### 1.3 MLP Trait

```rust
// crates/model/src/components/mlp.rs

pub trait MlpLayer: Send + Sync {
    fn forward(&self, x: &Tensor) -> Result<Tensor>;
}

pub trait MlpBuilder: Send + Sync {
    fn build(hidden_size: usize, intermediate_size: usize) -> Result<Box<dyn MlpLayer>>;

    fn build_with_weights(
        hidden_size: usize,
        intermediate_size: usize,
        gate_weight: Tensor,
        up_weight: Tensor,
        down_weight: Tensor,
    ) -> Result<Box<dyn MlpLayer>>;
}
```

---

## 2. TransformerBlock Trait

### 2.1 核心接口

```rust
// crates/model/src/components/block.rs

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

### 2.2 通用实现

```rust
// crates/model/src/components/block_impl.rs

#[cfg(feature = "candle")]
pub struct GenericTransformerBlock {
    input_norm: Box<dyn NormLayer>,
    output_norm: Box<dyn NormLayer>,
    attention: Box<dyn Attention>,
    mlp: Box<dyn MlpLayer>,
    kv_cache: Option<(Tensor, Tensor)>,
}

#[cfg(feature = "candle")]
impl TransformerBlock for GenericTransformerBlock {
    fn forward(
        &mut self,
        hidden_states: &Tensor,
        positions: &[usize],
        kv_block_ids: &[usize],
        num_computed: usize,
        is_prefill: bool,
    ) -> Result<Tensor> {
        // Pre-norm 架构
        let residual = hidden_states.clone();
        let x = self.input_norm.forward(hidden_states)?;

        // Attention with KV cache
        let x = self.attention.forward(&x, positions, self.kv_cache.as_ref(), is_prefill)?;

        // Residual connection
        let x = (x + residual)?;

        // MLP
        let residual = x.clone();
        let x = self.output_norm.forward(&x)?;
        let x = self.mlp.forward(&x)?;

        x.add(&residual)
    }

    fn inner_dim(&self) -> usize {
        self.attention.num_heads() * self.attention.head_dim()
    }

    fn num_kv_heads(&self) -> usize {
        self.attention.num_kv_heads()
    }
}
```

---

## 3. 架构特定 Traits

### 3.1 Architecture Trait

```rust
// crates/model/src/arch/mod.rs

use candle_core::{Device, Result, Tensor};
use std::collections::HashMap;

use crate::config::ModelConfig;

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

### 3.2 注册机制

```rust
// crates/model/src/arch/registry.rs

use std::collections::HashMap;
use std::sync::RwLock;

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

    pub fn detect(config_json: &serde_json::Value) -> Option<String> {
        let regs = self.architectures.read().unwrap();
        for (name, arch) in regs.iter() {
            if arch.detect(config_json) {
                return Some(name.clone());
            }
        }
        None
    }
}

// 初始化所有架构
pub fn register_all_archs(registry: &ArchitectureRegistry) {
    registry.register::<LlamaArchitecture>();
    registry.register::<MistralArchitecture>();
    registry.register::<Qwen3Architecture>();
    registry.register::<Qwen35Architecture>();
    registry.register::<Gemma4Architecture>();
    registry.register::<MixtralArchitecture>();
}
```

---

## 4. 架构实现示例

### 4.1 Llama 实现

```rust
// crates/model/src/llama/arch.rs

use super::block::LlamaBlock;
use crate::arch::{Architecture, TransformerBlock};
use crate::config::ModelConfig;
use crate::kv_cache::PagedKvCache;
use candle_core::{Device, Result, Tensor};
use std::collections::HashMap;
use vllm_traits::ModelBackend;

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
        let model = crate::llama::model::LlamaModel::from_weights(
            config, device, weights, num_kv_blocks,
        )?;
        Ok(Box::new(model))
    }
}
```

### 4.2 Mistral 实现

```rust
// crates/model/src/mistral/arch.rs

use super::block::MistralBlock;
use crate::arch::{Architecture, TransformerBlock};
use crate::config::ModelConfig;
use crate::kv_cache::PagedKvCache;
use candle_core::{Device, Result, Tensor};
use std::collections::HashMap;
use vllm_traits::ModelBackend;

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
        let model = crate::mistral::model::MistralModel::from_weights(
            config, device, weights, num_kv_blocks,
        )?;
        Ok(Box::new(model))
    }
}
```

---

## 5. 加载器集成

```rust
// crates/model/src/loader/builder.rs

use crate::arch::{register_all_archs, ArchitectureRegistry, ARCHITECTURE_REGISTRY};

pub struct ModelLoader {
    // ... existing fields
}

impl ModelLoader {
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
        let weights = self.load_weights()?;
        let weights = arch.remap_weights(weights);

        arch.create_model(config, self.inner.device.clone(), weights, self.inner.num_kv_blocks)
    }
}
```

---

## 6. 目录结构

```
crates/model/src/
├── lib.rs
├── components/
│   ├── mod.rs
│   ├── norm.rs              # NormLayer trait
│   ├── attention.rs         # Attention trait
│   ├── mlp.rs               # MlpLayer trait
│   ├── block.rs             # TransformerBlock trait
│   └── block_impl.rs        # GenericTransformerBlock
├── arch/
│   ├── mod.rs               # Architecture trait
│   ├── registry.rs          # ArchitectureRegistry
│   └── traits.rs            # ArchitectureExt for impls
├── config/
│   └── model_config.rs
├── llama/
│   ├── mod.rs
│   ├── block.rs
│   ├── arch.rs              # LlamaArchitecture impl
│   └── register.rs          # 显式注册函数
├── mistral/
│   ├── mod.rs
│   ├── block.rs
│   ├── arch.rs
│   └── register.rs
├── qwen3/
│   ├── mod.rs
│   ├── arch.rs
│   └── register.rs
├── qwen3_5/
│   ├── arch.rs
│   └── register.rs
├── gemma4/
│   ├── arch.rs
│   └── register.rs
├── mixtral/
│   ├── arch.rs
│   └── register.rs
└── loader/
    ├── builder.rs
    └── mod.rs
```

---

## 7. 实现步骤

### Phase 1: Trait 基础 (1-2天)
1. 定义 `NormLayer` trait
2. 定义 `Attention` trait
3. 定义 `MlpLayer` trait
4. 定义 `TransformerBlock` trait

### Phase 2: 架构抽象 (1-2天)
1. 定义 `Architecture` trait
2. 实现 `ArchitectureRegistry`
3. 创建 `register_all_archs()` 函数

### Phase 3: 模型迁移 (3-4天)
1. 迁移 Llama → `LlamaArchitecture`
2. 迁移 Mistral → `MistralArchitecture`
3. 迁移其他架构

### Phase 4: 清理 (1天)
1. 删除重复代码
2. 统一配置
3. 添加测试

---

## 8. 收益分析

| 指标 | 当前 | 重构后 | 改善 |
|------|------|--------|------|
| 代码行数 (估计) | ~1500 | ~1000 | -33% |
| 新架构添加 | 修改3+文件 | 只需添加模块 | +80% |
| 组件可测试性 | 低 | 高 | +++ |
| 编译时检查 | 部分 | 完整 | + |

---

## 9. 风险缓解

| 风险 | 缓解 |
|------|------|
| 破坏现有功能 | 增量修改，保持 API 兼容 |
| 性能下降 | 使用 trait object 但保证关键路径内联 |
| 过度设计 | YAGNI，只抽取当前需要的抽象 |

### 性能优化

对于性能敏感路径，可提供泛型版本:

```rust
pub fn forward_generic<B: TransformerBlock>(
    block: &mut B,
    hidden_states: &Tensor,
    // ...
) -> Result<Tensor> {
    block.forward(hidden_states, ...)
}
```

编译器会对泛型函数进行内联优化。

---

## 10. 测试策略

### 单元测试
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llama_architecture_detect() {
        let config = serde_json::json!({
            "model_type": "llama"
        });
        assert!(LlamaArchitecture::detect(&config));
    }

    #[test]
    fn test_mistral_architecture_detect() {
        let config = serde_json::json!({
            "model_type": "mistral"
        });
        assert!(MistralArchitecture::detect(&config));
    }
}
```

### 集成测试
```rust
#[test]
#[ignore = "需要真实模型文件"]
fn test_load_llama_model() {
    let loader = ModelLoader::builder(Device::Cpu)
        .with_model_dir("models/llama".to_string())
        .build()
        .unwrap();

    let model = loader.load().unwrap();
    // 验证模型可正常工作
}
```

### 回归测试
```rust
#[test]
fn test_output_matches_original() {
    // 加载优化前后的模型
    // 比对输出确保一致
}
```
