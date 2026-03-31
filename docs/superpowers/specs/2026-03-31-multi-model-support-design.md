# Multi-Model Support Design

> Date: 2026-03-31
> Status: Draft

## Goal

让 vLLM-lite 支持部署以下四个模型，并能正常使用 GPU 进行推理：

| Model | Path | Architecture | Vocab | Layers | Hidden | Size |
|-------|------|--------------|-------|--------|--------|------|
| Qwen3-0.6B | /models/Qwen3-0.6B | Qwen3 (GQA) | 151936 | 28 | 1024 | ~1.5GB |
| Qwen2.5-0.5B-Instruct | /models/Qwen2.5-0.5B-Instruct | Qwen2 (GQA) | 151936 | 24 | 896 | ~1GB |
| DeepSeek-R1-0528-Qwen3-8B | /models/DeepSeek-R1-0528-Qwen3-8B | Qwen3 (GQA+YARN) | 151936 | 36 | 4096 | ~16GB |
| Qwen3.5-0.8B | /models/Qwen3.5-0.8B | Qwen3.5 (Mamba/SSM) | 248320 | 24 | 1024 | ~1.7GB |

## Current State

### Already Working
- `ModelLoader::load_weights()` - 可加载 safetensors 权重 ✅
- Qwen2.5-0.5B 权重加载验证通过 (290 weights, 29s) ✅
- Qwen3.5-0.8B 权重加载验证通过 (335 weights, 45s) ✅
- 视觉权重自动跳过 (已实现) ✅

### 实际权重结构分析

| 模型 | 权重 Key 格式 | 需要改动 |
|------|--------------|----------|
| Qwen3-0.6B | `model.layers.{i}.self_attn.*` + `q_norm`, `k_norm` | 需加 QKNorm |
| Qwen2.5-0.5B | `model.layers.{i}.self_attn.{q,k,v,o}_proj.*` | 基本兼容 |
| DeepSeek-R1 | `model.layers.{i}.self_attn.*` + `q_norm`, `k_norm` + YARN | 需加 QKNorm + RoPE |
| Qwen3.5-0.8B | `model.language_model.layers.{i}.linear_attn.*` (Mamba) | 需新模型 |

### Issues to Fix
1. `main.rs` 未使用 loader，只创建零张量
2. 需要支持 `tie_word_embeddings` (部分模型 weight sharing)
3. Qwen3 需要支持 q_norm/k_norm (Qwen3-0.6B, DeepSeek-R1)
4. DeepSeek-R1 需要 RoPE YARN Scaling
5. Qwen3.5 需要全新 Mamba 模型 (linear_attn 结构完全不同)

## Architecture

### File Changes

```
crates/model/src/
├── config.rs          # 添加 model_type, rope_scaling, vision_config
├── loader.rs          # 添加 key 前缀适配
├── model_registry.rs  # 新建：模型工厂
├── lib.rs             # 导出新模块
├── qwen3/
│   ├── model.rs       # 适配权重 key 前缀
│   └── rope.rs        # 实现 YARN scaling
└── qwen3_5/           # 新建：Qwen3.5 支持
    ├── model.rs
    └── mamba.rs       # Mamba/SSM 层
```

### Key Changes

#### 1. Config (config.rs)

```rust
#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3Config {
    // 现有字段...
    
    #[serde(default)]
    pub model_type: Option<String>,
    
    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,
    
    #[serde(default)]
    pub vision_config: Option<VisionConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeScaling {
    pub rope_type: String,  // "yarn", "linear"
    pub factor: f32,
    pub original_max_position_embeddings: Option<usize>,
    pub attn_factor: Option<f32>,
}

impl Qwen3Config {
    pub fn model_type(&self) -> ModelType {
        match self.model_type.as_deref() {
            Some("qwen2") => ModelType::Qwen2,
            Some("qwen3") => ModelType::Qwen3,
            Some("qwen3_5") => ModelType::Qwen3_5,
            _ => ModelType::Qwen3, // default
        }
    }
}

pub enum ModelType {
    Qwen2,
    Qwen3,
    Qwen3_5,
}
```

#### 2. Model Registry (model_registry.rs)

```rust
use crate::config::{ModelType, Qwen3Config};
use crate::qwen3::model::Qwen3Model;
use crate::qwen3_5::model::Qwen35Model;
use candle_core::Device;

pub struct ModelRegistry;

impl ModelRegistry {
    pub fn create_model(
        config: Qwen3Config,
        device: Device,
    ) -> Result<Box<dyn ModelBackend>, Box<dyn std::error::Error>> {
        match config.model_type() {
            ModelType::Qwen2 | ModelType::Qwen3 => {
                Ok(Box::new(Qwen3Model::new(config, device)?))
            }
            ModelType::Qwen3_5 => {
                Ok(Box::new(Qwen35Model::new(config, device)?))
            }
        }
    }
    
    pub fn load_model(
        model_dir: &str,
        device: Device,
    ) -> Result<Box<dyn ModelBackend>, Box<dyn std::error::Error>> {
        let loader = ModelLoader::new(device);
        let config = loader.load_config(model_dir)?;
        
        match config.model_type() {
            ModelType::Qwen2 | ModelType::Qwen3 => {
                let weights = loader.load_weights(model_dir)?;
                let model = Qwen3Model::from_weights(config, device, weights)?;
                Ok(Box::new(model))
            }
            ModelType::Qwen3_5 => {
                let weights = loader.load_weights(model_dir)?;
                let model = Qwen35Model::from_weights(config, device, weights)?;
                Ok(Box::new(model))
            }
        }
    }
}
```

#### 3. Qwen3 QK Norm Support (qwen3/attention.rs)

Qwen3 (DeepSeek-R1) has additional `q_norm` and `k_norm` weights in attention:

```rust
pub struct GqaAttention {
    // existing fields...
    q_norm: Option<LayerNorm>,  // NEW
    k_norm: Option<LayerNorm>,  // NEW
}

impl GqaAttention {
    pub fn forward(&self, x: &Tensor, ...) -> Result<Tensor> {
        // Apply q_norm and k_norm after projection if present
        // ...
    }
}
```

**注意:** Qwen2 没有 q_norm/k_norm，需要用 `Option<LayerNorm>` 处理。

#### 5. tie_word_embeddings Support

Different models have different `tie_word_embeddings` settings:

| Model | tie_word_embeddings |
|-------|---------------------|
| Qwen3-0.6B | true |
| Qwen2.5-0.5B | true |
| DeepSeek-R1 | false |
| Qwen3.5 | true |

```rust
pub struct Qwen3Model {
    // ... existing fields
    embed_tokens: Embedding,
    lm_head: Linear,
    tie_word_embeddings: bool,
}

impl Qwen3Model {
    pub fn from_weights(...) -> Result<Self> {
        // ...
        let lm_head = if config.tie_word_embeddings {
            // Share weights with embed_tokens
            Linear::new(embed_tokens.weight().clone(), None)
        } else {
            // Separate lm_head weights
            // ...
        };
    }
}
```

#### 4. Qwen3.5 Model (qwen3_5/model.rs)

Qwen3.5 使用完全不同的 Mamba/SSM 架构，权重 key 完全不兼容，需要新建模型:

```rust
// qwen3_5/model.rs
pub struct Qwen35Model {
    embed_tokens: Embedding,
    layers: Vec<MambaBlock>,  // Different from TransformerBlock
    norm: LayerNorm,
    lm_head: Linear,
    kv_cache: PagedKvCache,
}

pub struct MambaBlock {
    // Mamba/SSM parameters (from weights)
    in_proj_a: Linear,    // A state projection
    in_proj_b: Linear,    // B state projection  
    in_proj_qkv: Linear,  // Query/Key/Value
    in_proj_z: Linear,    // Gating
    conv1d: Conv1d,       # Depthwise convolution
    out_proj: Linear,
    A_log: Tensor,        # SSM A matrix (log)
    dt_bias: Tensor,      # Delta bias
    norm: LayerNorm,
}
```

权重加载 key 对照:
- `model.language_model.layers.{i}.linear_attn.in_proj_qkv` → `in_proj_qkv`
- `model.language_model.layers.{i}.linear_attn.A_log` → `A_log`
- `model.language_model.layers.{i}.linear_attn.dt_bias` → `dt_bias`
- etc.

#### 4. RoPE Scaling (qwen3/rope.rs)

```rust
pub struct RoPE {
    theta: f32,
    scaling_factor: f32,
    original_max_pos: usize,
    attn_factor: Option<f32>,
}

impl RoPE {
    pub fn new(config: &Qwen3Config) -> Self {
        let rope_scaling = config.rope_scaling.as_ref();
        
        Self {
            theta: config.rope_theta(),
            scaling_factor: rope_scaling.map(|r| r.factor).unwrap_or(1.0),
            original_max_pos: rope_scaling
                .and_then(|r| r.original_max_position_embeddings)
                .unwrap_or(config.max_position_embeddings()),
            attn_factor: rope_scaling.and_then(|r| r.attn_factor),
        }
    }
    
    pub fn apply(&self, query: &Tensor, position_ids: &Tensor) -> Result<Tensor> {
        // YARN scaling implementation
        // 1. Extend position range by scaling_factor
        // 2. Apply attention factor adjustment
        // 3. Standard RoPE rotation
        todo!()
    }
}
```

#### 5. Main Integration (main.rs)

```rust
#[derive(Clone)]
struct AppState {
    pub engine_tx: mpsc::UnboundedSender<EngineMessage>,
    pub tokenizer: Arc<Tokenizer>,
    pub model_name: String,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_path = args
        .iter()
        .position(|a| a == "--model")
        .and_then(|i| args.get(i + 1))
        .unwrap_or("/models/Qwen2.5-0.5B-Instruct");
    
    let enable_speculative = args.iter().any(|a| a == "--speculative");
    
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    tracing::info!(device = ?device, "Using device");
    
    // Load model with real weights
    let loader = ModelLoader::new(device.clone());
    let model = loader.load_model(model_path)
        .expect("Failed to load model");
    
    // For speculative decoding, create a smaller draft model
    // or reuse the same model if --speculative not specified
    let draft_model = if enable_speculative {
        // TODO: Create a lighter draft model (smaller model or fewer layers)
        loader.load_model(model_path).ok()
    } else {
        None
    };
    
    // ... rest of initialization
}
```

## Implementation Phases

### Phase 1: Basic Integration + Qwen2 (Priority: High)
- [ ] Modify main.rs to accept `--model` parameter
- [ ] Integrate ModelLoader to load real weights
- [ ] Add tie_word_embeddings support to Qwen3Model
- [ ] Add q_norm/k_norm Option support to GqaAttention
- [ ] Test Qwen2.5-0.5B - 验证权重 key 匹配 (无 q_norm)
- [ ] Test Qwen3-0.6B - 验证 q_norm 处理
- [ ] 确保 GPU 正常使用

### Phase 2: DeepSeek-R1 YARN Support (Priority: High)
- [ ] Add RoPE YARN scaling to rope.rs
- [ ] Test DeepSeek-R1-0528-Qwen3-8B
- [ ] Verify extended context (131K tokens)

### Phase 3: Qwen3.5 Support (Priority: Medium)
- [ ] Create qwen3_5/ module
- [ ] Implement MambaBlock (not TransformerBlock)
- [ ] Handle linear_attn weights (完全不同的结构)
- [ ] Test text generation (skip vision for now)

## Testing

```bash
# Test each model (in order of complexity)
cargo run --package vllm-server -- --model /models/Qwen2.5-0.5B-Instruct
cargo run --package vllm-server -- --model /models/Qwen3-0.6B
cargo run --package vllm-server -- --model /models/DeepSeek-R1-0528-Qwen3-8B
cargo run --package vllm-server -- --model /models/Qwen3.5-0.8B

# Verify GPU usage
nvidia-smi

# Test inference
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, how are you?", "max_tokens": 32}'
```

## Risks

1. **Qwen3.5 Mamba Layer**: 需要实现全新的 Mamba/SSM 架构，权重结构完全不同，工作量大
2. **Memory**: DeepSeek-R1-8B 需要 ~16GB VRAM，测试需要 GPU
3. **RoPE Scaling**: YARN 算法复杂，可能需要参考 HuggingFace 实现
4. **Qwen3 QKNorm**: 需要在 attention 层添加 q_norm/k_norm 支持
5. **加载时间**: DeepSeek-R1 权重 16GB，加载时间长

## Success Criteria

- [ ] Qwen3-0.6B 可以正常启动并生成文本
- [ ] Qwen2.5-0.5B 可以正常启动并生成文本
- [ ] DeepSeek-R1 可以正常启动并生成文本
- [ ] Qwen3.5 可以正常启动并生成文本（文本模式）
- [ ] 所有模型在有 GPU 时使用 GPU
- [ ] 可以通过命令行指定模型路径

## Notes

- Qwen3.5 视觉功能 (ViT encoder) 暂时跳过，后续单独处理
- 权重加载已验证可以正常工作 (测试通过)
- 只需要解决架构差异和 main 集成问题
- Qwen3-0.6B 配置显示 `rope_scaling: null`，不需要 YARN，但需要 q_norm/k_norm
- Qwen3-0.6B 和 Qwen2.5 都使用 `tie_word_embeddings: true`，需要 weight sharing