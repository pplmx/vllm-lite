# Unified Model Architecture Design

**Date**: 2026-04-03
**Status**: Approved
**Goal**: Unified architecture supporting Qwen3, Llama, Mistral with easy extensibility

## Problem Statement

Current architecture:
- Each model (Qwen3, Qwen3.5-Mamba) has its own monolithic implementation
- Hard to add new architectures (Llama, Mistral)
- Configuration and weight loading not unified

## Target Architecture

```rust
// 1. Architecture Enum
pub enum Architecture {
    Qwen3,
    Llama,
    Mistral,
    // Easy to extend
}

// 2. Unified ModelConfig
pub struct ModelConfig {
    pub architecture: Architecture,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub intermediate_size: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f64,
    pub sliding_window: Option<usize>,  // Mistral specific
    pub tie_word_embeddings: bool,
    pub max_position_embeddings: usize,
}

// 3. Architecture Spec Trait
pub trait ArchitectureSpec: Send + Sync {
    fn attention_type() -> AttentionType;
    fn norm_type() -> NormType;
    fn mlp_type() -> MlpType;
    fn use_rope() -> bool;
    fn share_embeddings() -> bool;
    fn sliding_window() -> Option<usize>;
}

pub enum AttentionType {
    Mha,           // Multi-Head Attention (Llama 7B)
    Gqa,           // Grouped-Query Attention (Llama 70B, Qwen3)
    SlidingWindow, // Mistral sliding window
}

pub enum NormType {
    RmsNorm,   // Llama, Mistral
    LayerNorm, // Qwen3
}

pub enum MlpType {
    SwiGLU,    // Qwen3, Llama, Mistral
    GatedMLP,  // Alternative
}

// 4. Unified TransformerBlock using components
pub struct TransformerBlock {
    norm1: LayerNorm,
    norm2: LayerNorm,
    attention: GqaAttention,
    mlp: SwiGLU,
    // Architecture-specific config
    attention_type: AttentionType,
    sliding_window: Option<usize>,
}

impl TransformerBlock {
    pub fn new(config: &ModelConfig, layer_idx: usize) -> Result<Self>;
    pub fn forward(&self, x: &Tensor) -> Result<Tensor>;
    pub fn forward_prefill(&self, x: &Tensor, kv_cache: &mut PagedKvCache, ...) -> Result<Tensor>;
    pub fn forward_decode(&self, x: &Tensor, kv_cache: &PagedKvCache, ...) -> Result<Tensor>;
}

// 5. Unified Model with Architecture Trait
pub trait ModelTrait: Send + Sync {
    fn config(&self) -> &ModelConfig;
    fn architecture(&self) -> Architecture;
    fn forward(&self, input_ids: &[TokenId], ...) -> Result<BatchOutput>;
}

// 6. ModelRegistry with auto-detection
pub struct ModelRegistry;

impl ModelRegistry {
    pub fn create_model(config: &ModelConfig, device: Device, num_kv_blocks: usize) -> Result<Box<dyn ModelBackend>> {
        match config.architecture {
            Architecture::Llama => Ok(Box::new(LlamaModel::new(config, device, num_kv_blocks)?)),
            Architecture::Mistral => Ok(Box::new(MistralModel::new(config, device, num_kv_blocks)?)),
            Architecture::Qwen3 => Ok(Box::new(Qwen3Model::new(...)?)),
        }
    }

    pub fn from_huggingface(path: &str) -> Result<ModelConfig>;
}
```

## Model Implementations

### LlamaModel
```rust
pub struct LlamaModel {
    config: ModelConfig,
    embed_tokens: Embedding,
    layers: Vec<TransformerBlock>,
    norm: RmsNorm,
    lm_head: Linear,
    kv_cache: PagedKvCache,
    device: Device,
}

impl LlamaModel {
    pub fn new(config: &ModelConfig, device: Device, num_kv_blocks: usize) -> Result<Self>;
    pub fn forward_prefill(&self, input_ids: &[TokenId], ...) -> Result<BatchOutput>;
    pub fn forward_decode(&self, input_ids: &[TokenId], ...) -> Result<BatchOutput>;
}
```

### MistralModel
```rust
pub struct MistralModel {
    config: ModelConfig,
    embed_tokens: Embedding,
    layers: Vec<TransformerBlock>,
    norm: RmsNorm,
    lm_head: Linear,
    kv_cache: PagedKvCache,
    device: Device,
}

impl MistralModel {
    pub fn new(config: &ModelConfig, device: Device, num_kv_blocks: usize) -> Result<Self>;
    // Uses sliding window attention
    pub fn forward_prefill(&self, input_ids: &[TokenId], ...) -> Result<BatchOutput>;
    pub fn forward_decode(&self, input_ids: &[TokenId], ...) -> Result<BatchOutput>;
}
```

## Weight Loading

```rust
pub struct ModelLoader;

impl ModelLoader {
    pub fn load_llama(path: &str, device: &Device) -> Result<LlamaModel>;
    pub fn load_mistral(path: &str, device: &Device) -> Result<MistralModel>;
    pub fn load_qwen3(path: &str, device: &Device) -> Result<Qwen3Model>;
    
    // Generic loader
    pub fn load(path: &str, device: &Device) -> Result<Box<dyn ModelTrait>>;
}
```

## Module Structure

```
model/src/
├── config/
│   ├── mod.rs
│   ├── model_config.rs    # Unified ModelConfig
│   └── architecture.rs    # Architecture enum + Spec trait
│
├── llama/
│   ├── mod.rs
│   ├── block.rs
│   └── model.rs
│
├── mistral/
│   ├── mod.rs
│   ├── block.rs           # With sliding window attention
│   └── model.rs
│
├── loader.rs              # Unified weight loading
├── registry.rs            # Model registry
└── components/            # Already extracted
```

## Differences Between Models

| Feature | Llama | Mistral | Qwen3 |
|---------|-------|---------|-------|
| Norm | RMSNorm | RMSNorm | LayerNorm |
| Attention | GQA | Sliding + GQA | GQA + q/k norm |
| Position | RoPE | RoPE | RoPE |
| MLP | SwiGLU | SwiGLU | SwiGLU |
| Sliding Window | None | 4096 | None |

## Implementation Order

1. **Phase 1: Infrastructure**
   - Create `config/model_config.rs`
   - Create `config/architecture.rs`
   - Update `loader.rs` for unified loading

2. **Phase 2: Llama**
   - Create `llama/block.rs`
   - Create `llama/model.rs`
   - Implement ModelBackend

3. **Phase 3: Mistral**
   - Create `mistral/block.rs` (with sliding window)
   - Create `mistral/model.rs`
   - Implement ModelBackend

4. **Phase 4: Integration**
   - Update registry
   - Add to server model loading

## Acceptance Criteria

- [ ] Can load Llama models from HuggingFace format
- [ ] Can load Mistral models from HuggingFace format
- [ ] Both implement ModelBackend trait
- [ ] Components (attention, mlp, rope) are reused
- [ ] Easy to add new architectures (just add enum variant)
