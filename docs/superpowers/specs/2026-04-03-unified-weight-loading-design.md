# Unified Weight Loading Design

**Date**: 2026-04-03
**Status**: Approved
**Goal**: Extend ModelLoader to support Llama and Mistral weight loading

## Current State

- `ModelLoader` only supports Qwen3
- Uses `Qwen3Config` and `Qwen3Model::from_weights()`
- Architecture is implicitly Qwen3

## Target Architecture

```rust
// 1. Auto-detect architecture from config.json
pub fn detect_architecture(config: &serde_json::Value) -> Architecture {
    let model_type = config.get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_lowercase();

    match model_type.as_str() {
        "llama" | "llama2" | "llama3" => Architecture::Llama,
        "mistral" | "mixtral" => Architecture::Mistral,
        "qwen2" | "qwen2.5" => Architecture::Qwen3,
        _ => Architecture::Llama, // Default
    }
}

// 2. Architecture-specific weight prefixes
pub struct WeightMapping {
    pub embed_tokens: &'static str,
    pub layer_prefix: &'static str,
    pub norm: &'static str,
    pub lm_head: &'static str,
}

impl WeightMapping {
    pub fn for_architecture(arch: Architecture) -> &'static Self {
        match arch {
            Architecture::Llama => &LLAMA_MAPPING,
            Architecture::Mistral => &MISTRAL_MAPPING,
            Architecture::Qwen3 => &QWEN3_MAPPING,
        }
    }
}

static LLAMA_MAPPING: WeightMapping = WeightMapping {
    embed_tokens: "model.embed_tokens",
    layer_prefix: "model.layers",
    norm: "model.norm",
    lm_head: "lm_head",
};

static MISTRAL_MAPPING: WeightMapping = WeightMapping {
    embed_tokens: "model.embed_tokens",
    layer_prefix: "model.layers",
    norm: "model.norm",
    lm_head: "lm_head",
};

static QWEN3_MAPPING: WeightMapping = WeightMapping {
    embed_tokens: "model.embed_tokens",
    layer_prefix: "model.layers",
    norm: "model.norm",
    lm_head: "lm_head",
};

// 3. Unified ModelLoader
impl ModelLoader {
    pub fn load_config(&self, model_dir: &str) -> Result<ModelConfig> {
        let config_path = Path::new(model_dir).join("config.json");
        let content = std::fs::read_to_string(config_path)?;
        let value: serde_json::Value = serde_json::from_str(&content)?;

        let architecture = detect_architecture(&value);

        // Parse config based on architecture
        match architecture {
            Architecture::Llama => self.parse_llama_config(&value),
            Architecture::Mistral => self.parse_mistral_config(&value),
            Architecture::Qwen3 => self.parse_qwen3_config(&value),
        }
    }

    pub fn load_model<T: ModelTrait>(&self, ...) -> Result<T>;

    pub fn load(
        &self, 
        model_dir: &str, 
        num_kv_blocks: usize,
    ) -> Result<Box<dyn ModelBackend>> {
        let config = self.load_config(model_dir)?;
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
                let model = Qwen3Model::from_weights(config, self.device.clone(), weights, num_kv_blocks)?;
                Ok(Box::new(model))
            }
        }
    }
}
```

## Implementation Steps

1. Add `detect_architecture()` function
2. Add `WeightMapping` for each architecture
3. Implement `ModelConfig::from_config_json()`
4. Add `LlamaModel::from_weights()` and `MistralModel::from_weights()`
5. Update `ModelLoader::load()` to auto-detect and load

## Weight Key Patterns

**Llama:**

- `model.embed_tokens.weight`
- `model.layers.{i}.self_attn.q_proj.weight`
- `model.layers.{i}.self_attn.k_proj.weight`
- `model.layers.{i}.self_attn.v_proj.weight`
- `model.layers.{i}.self_attn.o_proj.weight`
- `model.layers.{i}.mlp.gate_proj.weight`
- `model.layers.{i}.mlp.up_proj.weight`
- `model.layers.{i}.mlp.down_proj.weight`
- `model.layers.{i}.input_layernorm.weight`
- `model.layers.{i}.post_attention_layernorm.weight`
- `model.norm.weight`
- `lm_head.weight`

**Mistral:**

- Same as Llama

**Qwen3:**

- `model.embed_tokens.weight`
- `model.layers.{i}.self_attn.q_proj.weight`
- etc.
