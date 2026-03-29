# vLLM-lite Real Weight Loading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development

**Goal:** Load real Qwen3 model weights from SafeTensors files instead of using random initialization.

**Architecture:** Implement ModelLoader to load weights from HuggingFace format SafeTensors files, integrate with Qwen3Model.

**Tech Stack:** Rust, Candle, SafeTensors

---

## Current State

- Qwen3Model uses random weights from `VarBuilder::zeros()`
- Forward pass produces random logits
- Need to load real weights for actual inference

## Target

- Load model from local directory (e.g., `./models/qwen3-7b/`)
- Parse config.json for model architecture
- Load weights from .safetensors files
- Run inference with real weights

---

### Task W1: Model Loader Implementation

**Files:**
- Create: `crates/model/src/loader.rs`

- [ ] **Step 1: Create ModelLoader**

`crates/model/src/loader.rs`:

```rust
use candle_core::{Device, Result, Tensor};
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::path::Path;

use crate::config::Qwen3Config;
use crate::qwen3::model::Qwen3Model;

pub struct ModelLoader {
    device: Device,
}

impl ModelLoader {
    pub fn new(device: Device) -> Self {
        Self { device }
    }

    pub fn load_model(&self, model_dir: &str) -> Result<Qwen3Model> {
        let config = self.load_config(model_dir)?;
        let weights = self.load_weights(model_dir)?;
        
        // Build model with loaded weights
        // ... 
    }

    fn load_config(&self, model_dir: &str) -> Result<Qwen3Config> {
        let config_path = Path::new(model_dir).join("config.json");
        let content = std::fs::read_to_string(config_path)?;
        let config: Qwen3Config = serde_json::from_str(&content)
            .map_err(|e| candle_core::Error::msg(e.to_string()))?;
        Ok(config)
    }

    fn load_weights(&self, model_dir: &str) -> Result<HashMap<String, Tensor>> {
        let model_path = Path::new(model_dir).join("model.safetensors");
        let file = SafeTensors::read(model_path.to_str().unwrap())
            .map_err(|e| candle_core::Error::msg(e.to_string()))?;
        
        let mut weights = HashMap::new();
        for (name, tensor) in file.tensors() {
            weights.insert(name, tensor.to_device(&self.device)?);
        }
        Ok(weights)
    }
}
```

- [ ] **Step 2: Update Qwen3Model to accept weights**

Modify `crates/model/src/qwen3/model.rs` to add constructor that takes weights.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "feat(model): add ModelLoader for SafeTensors"
```

---

### Task W2: Server Integration

**Files:**
- Modify: `crates/server/src/main.rs`

- [ ] **Step 1: Add model loading to server**

```rust
let model = ModelLoader::new(device)
    .load_model(&model_path)?;
```

- [ ] **Step 2: Add CLI argument for model path**

- [ ] **Step 3: Commit**

```bash
git add -A  
git commit -m "feat(server): load real model weights"
```

---

### Task W3: Test with Real Model

- [ ] **Step 1: Download test model** (or use mock)

- [ ] **Step 2: Run inference**

```bash
cargo run -p vllm-server

curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 10}'
```

- [ ] **Step 3: Verify output is coherent** (not random)

---

## Verification

```bash
# Build
cargo build --workspace

# Run server with model
MODEL_PATH=/path/to/qwen3-7b cargo run -p vllm-server
```

## Expected Outcome

- Model loads weights from SafeTensors
- Forward pass produces meaningful logits
- Generated text is coherent (not random)