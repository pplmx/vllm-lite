# Real Weight Loading Design

**Date**: 2026-04-03
**Status**: Approved
**Goal:** Implement real weight loading for Llama and Mistral from HuggingFace safetensors

## Current State

- `ModelLoader::load()` now auto-detects architecture
- `LlamaModel::from_weights()` and `MistralModel::from_weights()` exist but don't actually load weights
- Qwen3 has complete weight loading implementation to reference

## Target Implementation

### Weight Key Patterns

**Llama / Mistral:**

```text
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
model.norm.weight
lm_head.weight (or tie with embed_tokens)
```

### Implementation Approach

Follow Qwen3's pattern:

1. Look up weights by key pattern (with fallback keys)
2. Extract weights for each layer
3. Pass weights to model/block constructors

### LlamaBlock Changes

Update `LlamaBlock` to support `new_with_weights`:

```rust
impl LlamaBlock {
    pub fn new_with_weights(
        config: &ModelConfig,
        layer_idx: usize,
        weights: &HashMap<String, Tensor>,
    ) -> Result<Self> {
        // Extract weights for this layer
        let q_key = format!("model.layers.{}.self_attn.q_proj.weight", layer_idx);
        let k_key = format!("model.layers.{}.self_attn.k_proj.weight", layer_idx);
        let v_key = format!("model.layers.{}.self_attn.v_proj.weight", layer_idx);
        let o_key = format!("model.layers.{}.self_attn.o_proj.weight", layer_idx);

        // Create attention with weights
        let attention = GqaAttention::new_with_weights(...)?;

        // Similar for MLP and norms
    }
}
```

### Updated LlamaModel::from_weights

```rust
impl LlamaModel {
    pub fn from_weights(
        config: ModelConfig,
        device: Device,
        weights: HashMap<String, Tensor>,
        num_kv_blocks: usize,
    ) -> Result<Self> {
        // Embed tokens
        let embed_key = "model.embed_tokens.weight";
        let embed_weight = weights.get(embed_key).cloned()
            .ok_or_else(|| Error::msg(format!("Missing {}", embed_key)))?;
        let embed_tokens = Embedding::new(embed_weight, config.hidden_size);

        // Layers
        let mut layers = Vec::new();
        for i in 0..config.num_layers {
            layers.push(LlamaBlock::new_with_weights(&config, i, &weights)?);
        }

        // Norm
        let norm_weight = weights.get("model.norm.weight").cloned()
            .ok_or_else(|| Error::msg("Missing model.norm.weight"))?;
        let norm = Linear::new(norm_weight, None)?;

        // LM Head
        let lm_head = if config.tie_word_embeddings {
            Linear::new(embed_weight, None)?
        } else {
            weights.get("lm_head.weight")
                .cloned()
                .map(|w| Linear::new(w, None)?)
                .ok_or_else(|| Error::msg("Missing lm_head.weight"))?
        };

        // KV Cache
        let kv_cache = PagedKvCache::new(...)?;

        Ok(Self { config, embed_tokens, layers, norm, lm_head, kv_cache, device })
    }
}
```

### Same Pattern for Mistral

Mistral uses the same weight key patterns as Llama, so the implementation will be nearly identical.

## Implementation Tasks

1. Update `LlamaBlock::new_with_weights()`
2. Update `LlamaModel::from_weights()` to actually load weights
3. Update `MistralBlock::new_with_weights()`
4. Update `MistralModel::from_weights()` to actually load weights
5. Test with real Llama/Mistral model weights

## Acceptance Criteria

- [ ] Can load Llama 7B weights from HuggingFace format
- [ ] Can load Mistral 7B weights from HuggingFace format
- [ ] Model produces meaningful output (not random/fake)
- [ ] Tests pass
