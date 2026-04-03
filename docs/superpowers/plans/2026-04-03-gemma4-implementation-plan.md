# Gemma 4 Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan.

**Goal:** Add Gemma 4 text-only inference support (E2B, E4B dense models)

**Tech Stack:** Rust, Candle

---

## Phase 1: Infrastructure

### Task 1: Add Gemma4 to Architecture enum

**Files:**
- Modify: `crates/model/src/config/architecture.rs`
- Modify: `crates/model/src/config/model_config.rs`

- [ ] **Step 1: Add LayerType and RoPEConfig to architecture.rs**

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    SlidingAttention,
    FullAttention,
}

#[derive(Debug, Clone)]
pub struct RoPEConfig {
    pub rope_theta: f32,
    pub partial_rotary_factor: f32,
}
```

- [ ] **Step 2: Update Architecture enum**

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Architecture {
    Qwen3,
    Llama,
    Mistral,
    Gemma4,  // Add this
}
```

- [ ] **Step 3: Add Gemma4 fields to ModelConfig**

```rust
pub struct ModelConfig {
    // ... existing fields ...
    pub layer_types: Vec<LayerType>,
    pub rope_configs: Vec<RoPEConfig>,
    pub use_double_wide_mlp: bool,
}
```

- [ ] **Step 4: Run build and commit**

---

### Task 2: Update ModelConfig parsing for Gemma4

**Files:**
- Modify: `crates/model/src/config/model_config.rs`
- Modify: `crates/model/src/loader.rs`

- [ ] **Step 1: Add from_gemma4_config method**

- [ ] **Step 2: Add detect_architecture case for gemma4**

```rust
pub fn detect_architecture(config: &serde_json::Value) -> Architecture {
    let model_type = config.get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_lowercase();

    match model_type.as_str() {
        "llama" | "llama2" | "llama3" => Architecture::Llama,
        "mistral" | "mixtral" => Architecture::Mistral,
        "qwen2" | "qwen2.5" => Architecture::Qwen3,
        "gemma4" => Architecture::Gemma4,  // Add this
        _ => Architecture::Llama,
    }
}
```

- [ ] **Step 3: Run build and commit**

---

## Phase 2: Core Components

### Task 3: Create gemma4 module structure

**Files:**
- Create: `crates/model/src/gemma4/mod.rs`
- Create: `crates/model/src/gemma4/block.rs`
- Create: `crates/model/src/gemma4/attention.rs`
- Create: `crates/model/src/gemma4/mlp.rs`
- Create: `crates/model/src/gemma4/rope.rs`
- Create: `crates/model/src/gemma4/model.rs`
- Modify: `crates/model/src/lib.rs`

- [ ] **Step 1: Create gemma4/mod.rs**

```rust
pub mod block;
pub mod attention;
pub mod mlp;
pub mod rope;
pub mod model;

pub use block::Gemma4Block;
pub use model::Gemma4Model;
```

- [ ] **Step 2: Update lib.rs**

```rust
pub mod gemma4;
```

- [ ] **Step 3: Run build and commit**

---

### Task 4: Implement GeGLU MLP

**Files:**
- Modify: `crates/model/src/gemma4/mlp.rs`

- [ ] **Step 1: Implement GeGLU**

```rust
use candle_core::{Module, Result, Tensor};
use candle_nn::Linear;

pub struct GeGLU {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl GeGLU {
    pub fn new(hidden_size: usize, intermediate_size: usize, vb: VarBuilder) -> Result<Self> {
        let gate_proj = Linear::new(hidden_size, intermediate_size, false, vb)?;
        let up_proj = Linear::new(hidden_size, intermediate_size, false, vb)?;
        let down_proj = Linear::new(intermediate_size, hidden_size, false, vb)?;
        
        Ok(Self { gate_proj, up_proj, down_proj })
    }
    
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        
        // GeGLU: x * gelu(gate)
        let activated = gate.gelu(&up)?;  // Different from SwiGLU's silu
        self.down_proj.forward(&activated)
    }
}
```

- [ ] **Step 2: Add tests and commit**

---

### Task 5: Implement p-RoPE

**Files:**
- Modify: `crates/model/src/gemma4/rope.rs`

- [ ] **Step 1: Implement Gemma4RoPE**

```rust
use crate::config::RoPEConfig;

pub struct Gemma4RoPE {
    rope_theta: f32,
    partial_rotary_factor: f32,
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
        // Apply RoPE only to first partial_rotary_factor dimensions
        let rot_dim = (self.head_dim as f32 * self.partial_rotary_factor) as usize;
        // ... (similar to standard RoPE but only first rot_dim dimensions)
    }
}
```

- [ ] **Step 2: Add tests and commit**

---

### Task 6: Implement Hybrid Attention

**Files:**
- Modify: `crates/model/src/gemma4/attention.rs`

- [ ] **Step 1: Implement Gemma4Attention**

```rust
use crate::config::LayerType;

pub struct Gemma4Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    sliding_window: usize,
    // RoPE for this layer
    rope: Gemma4RoPE,
}

impl Gemma4Attention {
    pub fn forward(&self, x: &Tensor, positions: &[usize]) -> Result<Tensor> {
        // Standard GQA forward with RoPE
    }
    
    pub fn forward_sliding(&self, x: &Tensor, positions: &[usize]) -> Result<Tensor> {
        // Apply sliding window mask
        // Only attend to tokens within sliding_window
    }
}
```

- [ ] **Step 2: Add tests and commit**

---

### Task 7: Implement Gemma4Block

**Files:**
- Modify: `crates/model/src/gemma4/block.rs`

- [ ] **Step 1: Implement Gemma4Block**

```rust
use crate::config::{LayerType, ModelConfig};

pub struct Gemma4Block {
    attention: Gemma4Attention,
    mlp: GeGLU,
    input_layernorm: Linear,
    post_attention_layernorm: Linear,
    layer_type: LayerType,
}

impl Gemma4Block {
    pub fn new(config: &ModelConfig, layer_idx: usize) -> Result<Self> {
        let layer_type = config.layer_types.get(layer_idx)
            .copied()
            .unwrap_or(LayerType::SlidingAttention);
        
        // Create attention and MLP
        // ...
    }
    
    pub fn forward(&self, x: &Tensor, positions: &[usize]) -> Result<Tensor> {
        let residual = x.clone();
        let x = self.rms_norm(x, &self.input_layernorm)?;
        
        // Attention based on layer type
        x = self.attention.forward(x, positions)?;
        
        x = (x + residual)?;
        
        // MLP
        let residual = x.clone();
        let x = self.rms_norm(&x, &self.post_attention_layernorm)?;
        x = self.mlp.forward(&x)?;
        x.add(&residual)
    }
    
    fn rms_norm(&self, x: &Tensor, weight: &Linear) -> Result<Tensor> {
        // Standard RMS norm
    }
}
```

- [ ] **Step 2: Add tests and commit**

---

## Phase 3: Model Integration

### Task 8: Implement Gemma4Model

**Files:**
- Modify: `crates/model/src/gemma4/model.rs`

- [ ] **Step 1: Implement Gemma4Model**

```rust
pub struct Gemma4Model {
    config: ModelConfig,
    embed_tokens: Embedding,
    layers: Vec<Gemma4Block>,
    norm: Linear,
    lm_head: Linear,
    kv_cache: PagedKvCache,
    device: Device,
}

impl Gemma4Model {
    pub fn new(config: ModelConfig, device: Device, num_kv_blocks: usize) -> Result<Self> {
        // Similar to LlamaModel
    }
    
    pub fn from_weights(...) -> Result<Self> {
        // Load weights
    }
}

impl ModelBackend for Gemma4Model {
    fn forward(...) -> Result<BatchOutput> {
        // Implement inference
    }
}
```

- [ ] **Step 2: Commit**

---

## Phase 4: Integration

### Task 9: Update registry and loader

**Files:**
- Modify: `crates/model/src/registry.rs`
- Modify: `crates/model/src/loader.rs`

- [ ] **Step 1: Add Gemma4Model::from_config_json support**

- [ ] **Step 2: Update ModelLoader::load**

```rust
match config.architecture {
    Architecture::Gemma4 => {
        let model = Gemma4Model::from_weights(config, self.device.clone(), weights, num_kv_blocks)?;
        Ok(Box::new(model))
    }
    // ... existing
}
```

- [ ] **Step 2: Commit**

---

### Task 10: Final verification

- [ ] **Step 1: Run cargo build**

- [ ] **Step 2: Run tests**

- [ ] **Step 3: Run clippy**

- [ ] **Step 4: Commit**

---

## Summary

### Phase 1: Infrastructure (2 tasks)
- Task 1: Add Gemma4 to Architecture enum
- Task 2: Update ModelConfig parsing

### Phase 2: Core Components (4 tasks)
- Task 3: Create gemma4 module
- Task 4: Implement GeGLU
- Task 5: Implement p-RoPE
- Task 6: Implement Hybrid Attention
- Task 7: Implement Gemma4Block

### Phase 3: Model Integration (1 task)
- Task 8: Implement Gemma4Model

### Phase 4: Integration (2 tasks)
- Task 9: Update registry/loader
- Task 10: Final verification
