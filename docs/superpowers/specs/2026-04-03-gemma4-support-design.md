# Gemma 4 Support Design

**Date**: 2026-04-03
**Status**: In Progress
**Goal:** Add support for Gemma 4 (text-only, dense models: E2B, E4B)

---

## 1. Overview

Gemma 4 is a family of multimodal models from Google DeepMind. This spec covers text-only inference support for dense models (E2B, E4B). Full multimodal support (vision, audio) is out of scope for now.

### Scope
- ✅ Text-only inference
- ✅ Gemma 4 E2B (2.3B effective)
- ✅ Gemma 4 E4B (4.5B effective)
- ❌ Vision encoder (future work)
- ❌ Audio encoder (future work)
- ❌ MoE variant (26B A4B) (future work)

---

## 2. Architecture Comparison

| Feature | Llama | Mistral | Gemma 4 |
|---------|--------|----------|-----------|
| Norm | RMSNorm | RMSNorm | RMSNorm |
| Position | RoPE | RoPE | p-RoPE |
| Attention | GQA | Sliding + GQA | Hybrid Sliding + MQA |
| MLP | SwiGLU | SwiGLU | GeGLU |
| Sliding Window | None | 4096 | 512 |
| Global Layers | All | All | Every ~5th layer |
| KV Heads | num_kv_heads | num_kv_heads | 1 (MQA) |

---

## 3. Configuration

### Model Config Structure

```rust
// crates/model/src/config/model_config.rs

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
    pub sliding_window: Option<usize>,
    pub tie_word_embeddings: bool,
    pub max_position_embeddings: usize,
    // Gemma 4 specific
    pub layer_types: Vec<LayerType>,  // sliding_attention or full_attention
    pub rope_configs: Vec<RoPEConfig>, // per-layer RoPE settings
    pub use_double_wide_mlp: bool,
}
```

### Layer Types

```rust
// crates/model/src/config/architecture.rs

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    SlidingAttention,
    FullAttention,
}

#[derive(Debug, Clone)]
pub struct RoPEConfig {
    pub rope_theta: f32,
    pub partial_rotary_factor: f32,  // For full attention layers
}
```

### Architecture Spec

```rust
pub struct Gemma4Spec;
impl ArchitectureSpec for Gemma4Spec {
    fn architecture() -> Architecture { Architecture::Gemma4 }
    fn attention_type() -> AttentionType { AttentionType::HybridMqa }
    fn norm_type() -> NormType { NormType::RmsNorm }
    fn mlp_type() -> MlpType { MlpType::GeGLU }
    fn use_rope() -> bool { true }
    fn share_embeddings() -> bool { true }
    fn sliding_window() -> Option<usize> { Some(512) }
}
```

---

## 4. Module Structure

```
crates/model/src/
├── config/
│   ├── model_config.rs    # Add Gemma4 fields
│   └── architecture.rs    # Add LayerType, RoPEConfig
│
├── gemma4/               # NEW
│   ├── mod.rs
│   ├── block.rs          # Gemma4Block with hybrid attention
│   ├── attention.rs       # Sliding + Full attention
│   ├── mlp.rs            # GeGLU MLP
│   ├── rope.rs            # p-RoPE implementation
│   └── model.rs           # Gemma4Model
│
└── components/
    └── ...                # Reuse existing components
```

---

## 5. Implementation Details

### 5.1 Hybrid Attention

Gemma 4 alternates between sliding window attention and full attention:

```rust
// 35 layers pattern from config:
// ["sliding", "sliding", "sliding", "sliding", "full", 
//  "sliding", "sliding", "sliding", "sliding", "full", ...]
// Every 5th layer (index 4, 9, 14, 19, 24, 29, 34) is full attention

pub struct Gemma4Block {
    attention: GqaAttention,
    mlp: GeGLU,
    input_layernorm: Linear,
    post_attention_layernorm: Linear,
    layer_type: LayerType,
    rope_config: RoPEConfig,
}

impl Gemma4Block {
    pub fn forward(&self, x: &Tensor, positions: &[usize]) -> Result<Tensor> {
        let residual = x.clone();
        let x = self.rms_norm(x, &self.input_layernorm)?;
        
        // Attention based on layer type
        x = match self.layer_type {
            LayerType::FullAttention => self.attention.forward(x)?,
            LayerType::SlidingAttention => self.attention.forward_sliding(x, positions)?,
        };
        
        x = (x + residual)?;
        
        // MLP (GeGLU)
        let residual = x.clone();
        let x = self.rms_norm(&x, &self.post_attention_layernorm)?;
        x = self.mlp.forward(&x)?;
        x.add(&residual)
    }
}
```

### 5.2 GeGLU MLP

Gemma 4 uses GeGLU (Gaussian Error Linear Unit):

```rust
// crates/model/src/gemma4/mlp.rs

pub struct GeGLU {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl GeGLU {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        
        // GeGLU activation: x * gelu(gate)
        let activated = gate.gelu(&up)?;  // Different from SwiGLU's silu
        self.down_proj.forward(&activated)
    }
}
```

Note: Gemma uses `gelu_pytorch_tanh` which is slightly different from standard GELU.

### 5.3 p-RoPE (Proportional RoPE)

Different layers have different RoPE configurations:

```rust
// crates/model/src/gemma4/rope.rs

pub struct Gemma4RoPE {
    rope_theta: f32,
    partial_rotary_factor: f32,  // 0.25 for full attention layers
    head_dim: usize,
}

impl Gemma4RoPE {
    pub fn new(rope_config: &RoPEConfig, head_dim: usize) -> Self {
        Self {
            rope_theta: rope_config.rope_theta,
            partial_rotary_factor: rope_config.partial_rotary_factor,
            head_dim,
        }
    }
    
    pub fn apply(&self, q: &Tensor, k: &Tensor, positions: &[i64]) -> Result<(Tensor, Tensor)> {
        // Only apply to first partial_rotary_factor of head_dim
        // For full attention layers: partial_rotary_factor = 0.25
        // For sliding attention: partial_rotary_factor = 1.0 (full)
        let rot_dim = (self.head_dim as f32 * self.partial_rotary_factor) as usize;
        // ... apply RoPE to first rot_dim dimensions
    }
}
```

### 5.4 Weight Loading

Weight key patterns for Gemma 4:
```
model.embed_tokens.weight
model.layers.{i}.self_attn.q_proj.weight
model.layers.{i}.self_attn.k_proj.weight  
model.layers.{i}.self_attn.v_proj.weight
model.layers.{i}.self_attn.o_proj.weight
model.layers.{i}.mlp.gate_proj.weight
model.layers.{i}.mlp.up_proj.weight
model.layers.{i}.mlp.down_proj.weight
model.layers.{i}.input_layernorm.weight
model.layers.{i}.post_attention_layernorm.weight
model.final_lm_head.weight (tied with embed_tokens if tie_word_embeddings=true)
```

---

## 6. Model Config Parsing

```rust
impl ModelConfig {
    pub fn from_gemma4_config(value: &serde_json::Value) -> Result<Self> {
        let text_config = value.get("text_config")
            .or_else(|| value.get("config"))
            .ok_or_else(|| Error::msg("Missing text_config"))?;

        let layer_types: Vec<LayerType> = text_config.get("layer_types")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str())
                    .map(|s| match s {
                        "full_attention" => LayerType::FullAttention,
                        _ => LayerType::SlidingAttention,
                    })
                    .collect()
            })
            .unwrap_or_else(|| vec![LayerType::SlidingAttention; 35]);

        // Parse RoPE configs
        let rope_params = text_config.get("rope_parameters")
            .and_then(|v| v.as_object());
        
        let full_rope = rope_params.and_then(|r| r.get("full_attention"));
        let sliding_rope = rope_params.and_then(|r| r.get("sliding_attention"));

        let rope_configs: Vec<RoPEConfig> = layer_types.iter().map(|lt| {
            match lt {
                LayerType::FullAttention => RoPEConfig {
                    rope_theta: full_rope.and_then(|v| v.get("rope_theta"))
                        .and_then(|v| v.as_f64())
                        .unwrap_or(1000000.0) as f32,
                    partial_rotary_factor: full_rope.and_then(|v| v.get("partial_rotary_factor"))
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.25) as f32,
                },
                LayerType::SlidingAttention => RoPEConfig {
                    rope_theta: sliding_rope.and_then(|v| v.get("rope_theta"))
                        .and_then(|v| v.as_f64())
                        .unwrap_or(10000.0) as f32,
                    partial_rotary_factor: 1.0,
                },
            }
        }).collect();

        Ok(Self {
            architecture: Architecture::Gemma4,
            hidden_size: text_config.get("hidden_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(1536) as usize,
            num_layers: text_config.get("num_hidden_layers")
                .and_then(|v| v.as_u64())
                .unwrap_or(35) as usize,
            num_heads: text_config.get("num_attention_heads")
                .and_then(|v| v.as_u64())
                .unwrap_or(8) as usize,
            num_kv_heads: text_config.get("num_key_value_heads")
                .and_then(|v| v.as_u64())
                .unwrap_or(1) as usize,
            head_dim: text_config.get("head_dim")
                .and_then(|v| v.as_u64())
                .unwrap_or(256) as usize,
            vocab_size: text_config.get("vocab_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(262144) as usize,
            intermediate_size: text_config.get("intermediate_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(6144) as usize,
            rope_theta: 10000.0, // Default, actual per-layer
            rms_norm_eps: text_config.get("rms_norm_eps")
                .and_then(|v| v.as_f64())
                .unwrap_or(1e-6),
            sliding_window: text_config.get("sliding_window")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize),
            tie_word_embeddings: text_config.get("tie_word_embeddings")
                .and_then(|v| v.as_bool())
                .unwrap_or(true),
            max_position_embeddings: text_config.get("max_position_embeddings")
                .and_then(|v| v.as_u64())
                .unwrap_or(131072) as usize,
            // Gemma 4 specific
            layer_types,
            rope_configs,
            use_double_wide_mlp: text_config.get("use_double_wide_mlp")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
        })
    }
}
```

---

## 7. Acceptance Criteria

- [ ] Can detect "gemma4" model type from config.json
- [ ] Can parse Gemma 4 config with layer_types and rope_parameters
- [ ] Gemma4Block correctly applies sliding or full attention based on layer
- [ ] GeGLU MLP works correctly
- [ ] p-RoPE applies different theta per layer
- [ ] Can load Gemma 4 weights from safetensors
- [ ] Model can run inference with correct output
- [ ] Tests pass for both E2B and E4B configurations

---

## 8. Implementation Order

1. **Phase 1: Infrastructure**
   - Add Architecture::Gemma4 to enum
   - Add LayerType, RoPEConfig types
   - Update ModelConfig parsing

2. **Phase 2: Core Components**
   - Create gemma4/ module structure
   - Implement GeGLU MLP
   - Implement p-RoPE
   - Implement hybrid attention

3. **Phase 3: Model Integration**
   - Implement Gemma4Block
   - Implement Gemma4Model
   - Add weight loading

4. **Phase 4: Integration**
   - Update ModelLoader
   - Update registry
   - Add tests

---

## 9. Notes

- Gemma 4 uses per-layer embeddings (PLE) for smaller models - we can ignore this for inference
- Vision/Audio encoders are separate components - skip for now
- tie_word_embeddings is typically true for Gemma 4
- vocab_size is very large (262K) due to tokenizers
