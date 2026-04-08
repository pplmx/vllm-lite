# Unified Weight Loading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) to implement this plan.

**Goal:** Extend ModelLoader to support Llama and Mistral weight loading

---

## Task 1: Add architecture detection

**Files:**

- Modify: `crates/model/src/loader.rs`

- [ ] **Step 1: Add detect_architecture function**

```rust
use crate::config::Architecture;

pub fn detect_architecture(config: &serde_json::Value) -> Architecture {
    let model_type = config
        .get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_lowercase();

    match model_type.as_str() {
        "llama" | "llama2" | "llama3" => Architecture::Llama,
        "mistral" | "mixtral" => Architecture::Mistral,
        "qwen2" | "qwen2.5" => Architecture::Qwen3,
        _ => Architecture::Llama,
    }
}
```

- [ ] **Step 2: Run build and commit**

---

## Task 2: Add ModelConfig from JSON

**Files:**

- Modify: `crates/model/src/config/model_config.rs`

- [ ] **Step 1: Add from_config_json method**

```rust
impl ModelConfig {
    pub fn from_config_json(value: &serde_json::Value) -> Result<Self> {
        let architecture = detect_architecture(value);

        let hidden_size = value.get("hidden_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(4096) as usize;
        let num_layers = value.get("num_hidden_layers")
            .and_then(|v| v.as_u64())
            .unwrap_or(32) as usize;
        let num_heads = value.get("num_attention_heads")
            .and_then(|v| v.as_u64())
            .unwrap_or(32) as usize;
        let num_kv_heads = value.get("num_key_value_heads")
            .or_else(|| value.get("num_local_heads"))
            .and_then(|v| v.as_u64())
            .unwrap_or(num_heads as u64) as usize;
        let head_dim = hidden_size / num_heads;
        let vocab_size = value.get("vocab_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(32000) as usize;
        let intermediate_size = value.get("intermediate_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(11008) as usize;
        let rope_theta = value.get("rope_theta")
            .or_else(|| value.get("rotary_base"))
            .and_then(|v| v.as_f64())
            .unwrap_or(10000.0) as f32;
        let rms_norm_eps = value.get("rms_norm_eps")
            .or_else(|| value.get("layer_norm_eps"))
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-5) as f64;
        let sliding_window = value.get("sliding_window")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);
        let tie_word_embeddings = value.get("tie_word_embeddings")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let max_position_embeddings = value.get("max_position_embeddings")
            .and_then(|v| v.as_u64())
            .unwrap_or(2048) as usize;

        Ok(Self {
            architecture,
            hidden_size,
            num_layers,
            num_heads,
            num_kv_heads,
            head_dim,
            vocab_size,
            intermediate_size,
            rope_theta,
            rms_norm_eps,
            sliding_window,
            tie_word_embeddings,
            max_position_embeddings,
        })
    }
}
```

- [ ] **Step 2: Run build and commit**

---

## Task 3: Add LlamaModel::from_weights

**Files:**

- Modify: `crates/model/src/llama/model.rs`

- [ ] **Step 1: Add from_weights method to LlamaModel**

```rust
impl LlamaModel {
    pub fn from_weights(
        config: ModelConfig,
        device: Device,
        weights: HashMap<String, Tensor>,
        num_kv_blocks: usize,
    ) -> Result<Self> {
        // Extract weights by key mapping
        let embed_key = "model.embed_tokens.weight";
        let embed_tokens = weights.get(embed_key)
            .ok_or_else(|| candle_core::Error::msg(format!("Missing {}", embed_key)))?;
        let embed_tokens = Embedding::new(embed_tokens.clone(), config.hidden_size);

        let mut layers = Vec::new();
        for layer_idx in 0..config.num_layers {
            layers.push(LlamaBlock::from_weights(&config, layer_idx, &weights)?);
        }

        let norm_weight = weights.get("model.norm.weight")
            .ok_or_else(|| candle_core::Error::msg("Missing model.norm.weight"))?;
        let norm = candle_nn::linear(config.hidden_size, config.hidden_size, false)?;
        // Copy weight

        let lm_head_weight = weights.get("lm_head.weight")
            .or_else(|| weights.get("model.embed_tokens.weight"))
            .cloned()
            .ok_or_else(|| candle_core::Error::msg("Missing lm_head.weight"))?;
        let lm_head = candle_nn::linear(config.hidden_size, config.vocab_size, false)?;

        let kv_cache = PagedKvCache::new(
            config.num_layers,
            config.num_kv_heads,
            config.head_dim,
            num_kv_blocks,
            device.clone(),
            false,
        )?;

        Ok(Self { config, embed_tokens, layers, norm, lm_head, kv_cache, device })
    }
}

impl LlamaBlock {
    pub fn from_weights(
        config: &ModelConfig,
        layer_idx: usize,
        weights: &HashMap<String, Tensor>,
    ) -> Result<Self> {
        let prefix = format!("model.layers.{}", layer_idx);

        // Extract weights for this layer
        // ... (similar to Qwen3 implementation)

        // For now, create with random weights
        Self::new(config, layer_idx)
    }
}
```

- [ ] **Step 2: Run build and commit**

---

## Task 4: Add MistralModel::from_weights

**Files:**

- Modify: `crates/model/src/mistral/model.rs`

- [ ] **Step 1: Similar to Task 3 for Mistral**

- [ ] **Step 2: Commit**

---

## Task 5: Update ModelLoader::load

**Files:**

- Modify: `crates/model/src/loader.rs`

- [ ] **Step 1: Update load method to auto-detect**

```rust
pub fn load(
    &self,
    model_dir: &str,
    num_kv_blocks: usize,
) -> Result<Box<dyn ModelBackend>> {
    let config_path = Path::new(model_dir).join("config.json");
    let content = std::fs::read_to_string(config_path)?;
    let value: serde_json::Value = serde_json::from_str(&content)?;

    let config = ModelConfig::from_config_json(&value)?;
    let weights = self.load_weights(model_dir)?;

    match config.architecture {
        Architecture::Llama => {
            let model = LlamaModel::from_weights(config, self.device.clone(), weights, num_kv_blocks)?;
            Ok(Box::new(model))
        }
        Architecture::Mistral => {
            let model = MistralModel::from_weights(config, self.device.clone(), weights, num_kv_blocks)?;
            Ok(Box::new(model))
        }
        Architecture::Qwen3 => {
            // Keep existing behavior
            let config = self.load_config(model_dir)?;
            let model = Qwen3Model::from_weights(config, self.device.clone(), weights, num_kv_blocks)?;
            Ok(Box::new(model))
        }
    }
}
```

- [ ] **Step 2: Run tests and commit**

---

## Summary

- Task 1: Add architecture detection
- Task 2: Add ModelConfig from JSON
- Task 3: Add LlamaModel::from_weights
- Task 4: Add MistralModel::from_weights
- Task 5: Update ModelLoader::load
