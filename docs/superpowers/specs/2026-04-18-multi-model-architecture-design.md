# 多模型架构优化设计

## 概述

优化 vllm-lite 的多模型支持架构，解决代码重复、提高可扩展性、增强组件复用性。

## 当前状态

### 问题

1. **代码重复**: `LlamaModel` 和 `MistralModel` 结构几乎完全相同 (166行重复)
2. **架构扩展困难**: 使用 `Architecture` enum + match 分发，新架构需改核心代码
3. **配置不一致**: 混用 `ModelConfig` 和 `Qwen3Config`
4. **组件复用不足**: `components/` 目录存在但未充分利用

### 当前架构

```
loader/mod.rs (detect_architecture)
        ↓
    Architecture enum
        ↓
loader/builder.rs (match dispatch)
        ↓
   各个 Model 模块 (大量重复)
```

## 目标

1. 消除模型间的代码重复
2. 支持动态注册新架构 (无需修改核心代码)
3. 统一配置管理
4. 提高组件可组合性

## 设计方案

### 1. 核心接口

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

### 2. 基类结构

```rust
// crates/model/src/model.rs

pub struct TransformerModel {
    config: Arc<ModelConfig>,
    embed_tokens: Embedding,
    layers: Vec<Box<dyn TransformerBlock>>,
    norm: NormLayer,
    lm_head: Linear,
    kv_cache: PagedKvCache,
    device: Device,
}

impl TransformerModel {
    pub fn from_weights(
        config: ModelConfig,
        device: Device,
        weights: HashMap<String, Tensor>,
        num_kv_blocks: usize,
    ) -> Result<Self>;

    pub fn num_layers(&self) -> usize;
    pub fn hidden_size(&self) -> usize;
}
```

### 3. 架构注册机制

```rust
// crates/model/src/registry.rs

pub struct ArchitectureInfo {
    pub name: &'static str,
    pub detect_fn: fn(&serde_json::Value) -> Option<Architecture>,
    pub block_factory: fn(&ModelConfig, usize) -> Result<Box<dyn TransformerBlock>>,
    pub weight_remap: Option<fn(HashMap<String, Tensor>) -> HashMap<String, Tensor>>,
}

pub struct Registry {
    architectures: HashMap<Architecture, ArchitectureInfo>,
    detectors: Vec<fn(&serde_json::Value) -> Option<Architecture>>,
}

impl Registry {
    pub fn register(&mut self, info: ArchitectureInfo);

    pub fn detect(&self, config: &serde_json::Value) -> Option<Architecture>;

    pub fn create_block(
        &self,
        arch: Architecture,
        config: &ModelConfig,
        layer_idx: usize,
    ) -> Result<Box<dyn TransformerBlock>>;
}

// 全局注册表
lazy_static::lazy_static! {
    pub static ref ARCHITECTURE_REGISTRY: Registry = {
        let mut r = Registry::new();
        llama::register(&mut r);
        mistral::register(&mut r);
        qwen3::register(&mut r);
        // 新架构只需调用 register!
        r
    };
}
```

### 4. 架构模块改造

```rust
// crates/model/src/llama/mod.rs

use crate::registry::{ArchitectureInfo, Registry};

pub struct LlamaBlock { /* ... */ }

impl TransformerBlock for LlamaBlock { /* ... */ }

pub fn register(registry: &mut Registry) {
    registry.register(ArchitectureInfo {
        name: "llama",
        detect_fn: |config| {
            let model_type = config.get("model_type")?.as_str()?;
            if ["llama", "llama2", "llama3"].contains(&model_type) {
                Some(Architecture::Llama)
            } else {
                None
            }
        },
        block_factory: |config, idx| {
            Ok(Box::new(LlamaBlock::from_weights(config, idx)?))
        },
        weight_remap: None,
    });
}
```

### 5. 配置统一

```rust
// crates/model/src/config/model_config.rs

pub struct ModelConfig {
    // 通用配置
    pub hidden_size: usize,
    pub num_layers: usize,
    pub vocab_size: usize,
    pub head_dim: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,

    // 架构特定配置
    pub rope_config: Option<RoPEConfig>,
    pub mlp_type: MlpType,
    pub attention_type: AttentionType,
    pub sliding_window: Option<usize>,
    pub tie_word_embeddings: bool,

    // 其他
    pub max_position_embeddings: usize,
}

impl ModelConfig {
    pub fn from_config_json(json: &serde_json::Value) -> Result<Self>;
}
```

### 6. 加载流程

```rust
// crates/model/src/loader/builder.rs

impl ModelLoader {
    pub fn load(&self) -> Result<Box<dyn ModelBackend>> {
        let arch = ARCHITECTURE_REGISTRY.detect(&self.inner.config_json)
            .ok_or_else(|| Error::msg("Unsupported architecture"))?;

        let config = ModelConfig::from_config_json(&self.inner.config_json)?;
        let weights = self.load_weights()?;

        // 使用注册机制而非 match
        let model = TransformerModel::from_architecture(
            arch,
            config,
            self.inner.device.clone(),
            weights,
            self.inner.num_kv_blocks,
        )?;

        Ok(Box::new(model))
    }
}
```

## 目录结构

```
crates/model/src/
├── lib.rs
├── components/           # 已存在
│   ├── attention.rs
│   ├── mlp.rs
│   ├── norm.rs
│   └── positional.rs
├── model.rs              # 新增: TransformerModel 基类
├── registry.rs           # 新增: 架构注册表
├── config/
│   └── model_config.rs   # 改造: 统一配置
├── llama/
│   ├── mod.rs
│   ├── block.rs
│   └── register.rs       # 新增: 注册函数
├── mistral/
│   ├── mod.rs
│   ├── block.rs
│   └── register.rs
├── qwen3/
│   ├── mod.rs
│   ├── block.rs
│   └── register.rs
└── ... (其他架构类似)
```

## 实现步骤

### Phase 1: 接口定义
1. 定义 `TransformerBlock` trait
2. 创建 `TransformerModel` 基类
3. 迁移共享逻辑

### Phase 2: 注册机制
1. 实现 `Registry` 结构
2. 为现有架构添加注册函数
3. 更新 loader 使用注册表

### Phase 3: 代码迁移
1. 将 `LlamaBlock` 实现为 `TransformerBlock`
2. 将 `MistralBlock` 实现为 `TransformerBlock`
3. 删除重复代码

### Phase 4: 清理
1. 统一配置
2. 简化 `Architecture` enum (或转为注册表驱动)
3. 文档和测试

## 测试策略

1. **接口测试**: 验证 `TransformerBlock` trait 实现
2. **集成测试**: 确保各架构加载正常
3. **回归测试**: 比对优化前后输出

## 风险和缓解

| 风险 | 缓解 |
|------|------|
| 破坏现有功能 | 增量修改，保持 API 兼容 |
| 性能下降 | 使用 trait object 但保证关键路径内联 |
| 过度设计 | YAGNI，只抽取当前需要的抽象 |

## 收益

1. **代码量减少**: 预计减少 500+ 行重复代码
2. **扩展性**: 新架构只需实现 trait + 注册
3. **可维护性**: 单一职责，组件可独立测试
