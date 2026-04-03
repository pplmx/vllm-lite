# Real Weight Loading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan.

**Goal:** Implement real weight loading for Llama and Mistral

---

## Task 1: Update LlamaBlock to support weights

**Files:**
- Modify: `crates/model/src/llama/block.rs`

- [ ] **Step 1: Add new_with_weights method to LlamaBlock**

```rust
impl LlamaBlock {
    pub fn new_with_weights(
        config: &ModelConfig,
        layer_idx: usize,
        weights: &HashMap<String, Tensor>,
    ) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_heads;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;
        let intermediate_size = config.intermediate_size;
        let theta = config.rope_theta;
        let rms_norm_eps = config.rms_norm_eps;

        // Get layer prefix
        let layer_prefix = format!("model.layers.{}", layer_idx);

        // Attention weights
        let q_key = format!("{}.self_attn.q_proj.weight", layer_prefix);
        let k_key = format!("{}.self_attn.k_proj.weight", layer_prefix);
        let v_key = format!("{}.self_attn.v_proj.weight", layer_prefix);
        let o_key = format!("{}.self_attn.o_proj.weight", layer_prefix);

        let q_weight = weights.get(&q_key).cloned()
            .ok_or_else(|| Error::msg(format!("Missing {}", q_key)))?;
        let k_weight = weights.get(&k_key).cloned()
            .ok_or_else(|| Error::msg(format!("Missing {}", k_key)))?;
        let v_weight = weights.get(&v_key).cloned()
            .ok_or_else(|| Error::msg(format!("Missing {}", v_key)))?;
        let o_weight = weights.get(&o_key).cloned()
            .ok_or_else(|| Error::msg(format!("Missing {}", o_key)))?;

        // MLP weights
        let gate_key = format!("{}.mlp.gate_proj.weight", layer_prefix);
        let up_key = format!("{}.mlp.up_proj.weight", layer_prefix);
        let down_key = format!("{}.mlp.down_proj.weight", layer_prefix);

        let gate_weight = weights.get(&gate_key).cloned()
            .ok_or_else(|| Error::msg(format!("Missing {}", gate_key)))?;
        let up_weight = weights.get(&up_key).cloned()
            .ok_or_else(|| Error::msg(format!("Missing {}", up_key)))?;
        let down_weight = weights.get(&down_key).cloned()
            .ok_or_else(|| Error::msg(format!("Missing {}", down_key)))?;

        // Norm weights
        let input_ln_key = format!("{}.input_layernorm.weight", layer_prefix);
        let post_attn_ln_key = format!("{}.post_attention_layernorm.weight", layer_prefix);

        let input_ln_weight = weights.get(&input_ln_key).cloned()
            .ok_or_else(|| Error::msg(format!("Missing {}", input_ln_key)))?;
        let post_attn_ln_weight = weights.get(&post_attn_ln_key).cloned()
            .ok_or_else(|| Error::msg(format!("Missing {}", post_attn_ln_key)))?;

        // Create attention with weights
        let attention = GqaAttention::new_with_weights(
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            theta,
            q_weight,
            k_weight,
            v_weight,
            o_weight,
            AttentionConfig::default(),
            false, // No qk_norm for Llama
            None,  // q_norm
            None,  // k_norm
        )?;

        // Create MLP with weights
        let mlp = SwiGLU::new_with_weights(
            hidden_size,
            intermediate_size,
            gate_weight,
            up_weight,
            down_weight,
        )?;

        // Create norms
        let input_layernorm = Linear::new(input_ln_weight, None)?;
        let post_attention_layernorm = Linear::new(post_attn_ln_weight, None)?;

        Ok(Self {
            input_layernorm,
            post_attention_layernorm,
            attention,
            mlp,
        })
    }
}
```

- [ ] **Step 2: Run build and commit**

---

## Task 2: Update LlamaModel::from_weights

**Files:**
- Modify: `crates/model/src/llama/model.rs`

- [ ] **Step 1: Update from_weights to actually load weights**

```rust
impl LlamaModel {
    pub fn from_weights(
        config: ModelConfig,
        device: Device,
        weights: HashMap<String, Tensor>,
        num_kv_blocks: usize,
    ) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_layers = config.num_layers;

        // Embed tokens
        let embed_key = "model.embed_tokens.weight";
        let embed_weight = weights.get(embed_key).cloned()
            .ok_or_else(|| Error::msg(format!("Missing {}", embed_key)))?;
        let embed_tokens = Embedding::new(embed_weight.clone(), hidden_size);

        // Layers
        let mut layers = Vec::new();
        for i in 0..num_layers {
            layers.push(LlamaBlock::new_with_weights(&config, i, &weights)?);
        }

        // Final norm
        let norm_key = "model.norm.weight";
        let norm_weight = weights.get(norm_key).cloned()
            .ok_or_else(|| Error::msg(format!("Missing {}", norm_key)))?;
        let norm = Linear::new(norm_weight, None)?;

        // LM head
        let lm_head = if config.tie_word_embeddings {
            Linear::new(embed_weight, None)?
        } else {
            let lm_key = "lm_head.weight";
            let lm_weight = weights.get(lm_key).cloned()
                .or_else(|| weights.get("model.embed_tokens.weight").cloned())
                .ok_or_else(|| Error::msg("Missing lm_head.weight"))?;
            Linear::new(lm_weight, None)?
        };

        // KV Cache
        let kv_cache = PagedKvCache::new(
            num_layers,
            config.num_kv_heads,
            config.head_dim,
            num_kv_blocks,
            device.clone(),
            false,
        )?;

        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
            lm_head,
            kv_cache,
            device,
        })
    }
}
```

- [ ] **Step 2: Run build and commit**

---

## Task 3: Update MistralBlock

**Files:**
- Modify: `crates/model/src/mistral/block.rs`

- [ ] **Step 1: Add new_with_weights method (similar to LlamaBlock)**

- [ ] **Step 2: Run build and commit**

---

## Task 4: Update MistralModel::from_weights

**Files:**
- Modify: `crates/model/src/mistral/model.rs`

- [ ] **Step 1: Update from_weights (similar to LlamaModel)**

- [ ] **Step 2: Run build and commit**

---

## Task 5: Final verification

- [ ] **Step 1: Run cargo build**

- [ ] **Step 2: Run tests**

- [ ] **Step 3: Run clippy**

- [ ] **Step 4: Commit**

---

## Summary

- Task 1: LlamaBlock with weights
- Task 2: LlamaModel from_weights
- Task 3: MistralBlock with weights
- Task 4: MistralModel from_weights
- Task 5: Final verification
