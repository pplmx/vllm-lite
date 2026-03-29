# vLLM-lite Phase 5: Qwen3 Real Model Integration

## 1. Overview

Implement Qwen3 model inference with Candle, supporting:
- GQA (Grouped Query Attention)
- Sliding Window Attention
- RoPE (Rotary Position Embedding)
- SwiGLU MLP
- SafeTensors weight loading
- CUDA GPU inference

---

## 2. Technical Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Model | Qwen3 | Popular, well-documented |
| Backend | Candle | Pure Rust, CUDA support |
| Weights | SafeTensors | HuggingFace standard |
| Device | CUDA | Fast inference |
| Tokenizer | tokenizers crate | HuggingFace compatible |

---

## 3. Architecture

### 3.1 Model Crate Structure

```
crates/model/src/
├── lib.rs
├── loader.rs           # SafeTensors loading
├── config.rs           # Qwen3Config
└── qwen3/
    ├── mod.rs
    ├── attention.rs    # GQA + Sliding Window
    ├── mlp.rs          # SwiGLU
    ├── rope.rs         # RoPE
    ├── block.rs        # Transformer Block
    └── model.rs        # Qwen3Model
```

### 3.2 Qwen3Config

```rust
pub struct Qwen3Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub sliding_window: Option<usize>,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
}
```

---

## 4. Core Components

### 4.1 Weight Loading

```rust
pub struct ModelWeights {
    pub embed_tokens: Tensor,
    pub layers: Vec<LayerWeights>,
    pub norm: Tensor,
    pub lm_head: Tensor,
}

pub struct LayerWeights {
    pub attn_q_proj: Tensor,
    pub attn_k_proj: Tensor,
    pub attn_v_proj: Tensor,
    pub attn_o_proj: Tensor,
    pub mlp_gate_proj: Tensor,
    pub mlp_up_proj: Tensor,
    pub mlp_down_proj: Tensor,
    pub input_layernorm: Tensor,
    pub post_attention_layernorm: Tensor,
}
```

### 4.2 GQA + Sliding Window Attention

- **GQA**: num_key_value_heads < num_attention_heads
  - K, V 在 head 维度投影 num_key_value_heads
  - Q 投影 num_attention_heads
  - 推理时 expand K, V 到 Q 的 head 数

- **Sliding Window**: 每个 token 只看前后 `sliding_window` 个 token
  - 超过 window 的位置 mask 掉
  - Qwen3 典型值: 32768

### 4.3 RoPE

- 2D 旋转矩阵: `cos(mθ), sin(mθ)`
- 频率: `θ_i = base^(-2i/d)`
- 位置 m 从 0 开始

### 4.4 SwiGLU MLP

```
FFN(x) = SwiGLU(x) = SiLU(x @ W_gate) * (x @ W_up) @ W_down
intermediate_size = 4 * hidden_size (not 2*)
```

---

## 5. Forward Pass

```
Input: batch.input_ids [batch_size, seq_len]
Output: logits [batch_size, vocab_size]

1. embed = embedding(input_ids)
2. hidden = embed

3. For each layer:
   a. normalized = rms_norm(hidden, layernorm.weight)
   b. q = normalized @ q_proj.weight
   c. k = normalized @ k_proj.weight
   d. v = normalized @ v_proj.weight
   e. expand k, v to num_attention_heads (GQA)
   f. apply RoPE to q, k
   g. apply sliding window mask
   h. attn_output = attention(q, k, v)
   i. attn_output = attn_output @ o_proj.weight
   j. hidden = hidden + attn_output (residual)
   
   k. normalized = rms_norm(hidden, post_layernorm.weight)
   l. mlp_output = swiglu(normalized)
   m. hidden = hidden + mlp_output (residual)

4. hidden = rms_norm(hidden, final_norm.weight)
5. logits = hidden @ lm_head.weight.T
```

---

## 6. Integration

### 6.1 ModelBackend Implementation

```rust
impl ModelBackend for Qwen3Model {
    fn forward(
        &self,
        seq_ids: &[SeqId],
        input_tokens: &[Vec<TokenId>],
        positions: &[Vec<usize>],
    ) -> Result<BatchOutput> {
        // Convert tokens to embeddings
        // Run forward through all layers
        // Return logits
    }
}
```

### 6.2 Server Integration

```rust
// main.rs
let model = Qwen3Model::from_pretrained(
    "Qwen/Qwen3-7B",
    &Device::Cuda(0),
)?;
let engine = Engine::with_config(model, config, num_kv_blocks);
```

---

## 7. Dependencies

```toml
# crates/model/Cargo.toml
[dependencies]
candle-core = "0.8"
candle-nn = "0.8"
safetensors = "0.44"
tokenizers = "0.20"
serde = { version = "1", features = ["derive"] }
thiserror = "2"
```

---

## 8. Implementation Plan

### Phase 5.1: Infrastructure
- [ ] Add dependencies to Cargo.toml
- [ ] Implement Qwen3Config
- [ ] Implement SafeTensors loader

### Phase 5.2: Core Components
- [ ] Implement RoPE
- [ ] Implement RMSNorm
- [ ] Implement SwiGLU MLP

### Phase 5.3: Attention
- [ ] Implement GQA
- [ ] Implement Sliding Window Attention
- [ ] Implement full Transformer Block

### Phase 5.4: Model Integration
- [ ] Implement Qwen3Model
- [ ] Implement ModelBackend trait
- [ ] Connect to Engine

### Phase 5.5: Server Integration
- [ ] Update server to load real model
- [ ] Add tokenizer
- [ ] End-to-end test

---

## 9. Success Criteria

- [ ] Model loads from SafeTensors
- [ ] Forward pass produces valid logits
- [ ] Greedy sampling returns coherent text
- [ ] Streaming works end-to-end
- [ ] CUDA GPU used for inference