# MLA (Multi-head Latent Attention) Implementation Design

**Date:** 2026-04-19
**Status:** Draft
**Reference:** DeepSeek-V3 MLA architecture

## Overview

Implement MLA (Multi-head Latent Attention) as a new attention mechanism in vllm-lite, following the DeepSeek-V3 architecture pattern. MLA reduces KV cache size through low-rank compression while maintaining attention quality.

## Reference Configuration

From DeepSeek-V3-Base:

```json
{
  "q_lora_rank": 1536,
  "kv_lora_rank": 512,
  "qk_nope_head_dim": 128,
  "qk_rope_head_dim": 64,
  "v_head_dim": 128,
  "num_attention_heads": 128,
  "num_key_value_heads": 128,
  "hidden_size": 7168
}
```

## Core Parameters

| Parameter | Source | Description |
|-----------|--------|-------------|
| `q_lora_rank` | config.q_len | Q compression dimension |
| `kv_lora_rank` | config.kv_len | KV compression dimension |
| `qk_nope_head_dim` | config.qk_nope_dim | Q_no_rope per-head dimension |
| `qk_rope_head_dim` | config.qk_rope_dim | Q_rope per-head dimension |
| `v_head_dim` | config.v_head_dim | V per-head dimension |
| `num_heads` | config.num_attention_heads | Number of Q heads |
| `num_kv_heads` | config.num_key_value_heads | Number of K/V heads |
| `hidden_size` | config.hidden_size | Hidden dimension |

## Data Flow

### Q Projection

```
hidden_states: [batch, seq, hidden_size]
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ Q Projection (W_q)                                          │
│ W_q: [hidden_size, q_lora_rank]                            │
│ q_compressed: [batch, seq, q_lora_rank]                    │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼ split (qk_nope + qk_rope)
        ┌──────────────────────────────────────┐
        │ q_nope: [batch, seq, qk_nope_head_dim * num_heads]   │
        │ q_rope: [batch, seq, qk_rope_head_dim * num_heads]   │
        └──────────────────────────────────────┘
                    │
                    ▼ apply RoPE to q_rope
        ┌──────────────────────────────────────┐
        │ q_rope_rotated: [batch, seq, qk_rope_head_dim * num_heads] │
        └──────────────────────────────────────┘
                    │
                    ▼ concat
        ┌──────────────────────────────────────┐
        │ Q: [batch, seq, (qk_nope + qk_rope) * num_heads] │
        │   = [batch, num_heads, seq, head_dim]               │
        └──────────────────────────────────────┘
```

### KV Compression

```
hidden_states: [batch, seq, hidden_size]
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ KV Compression (W_kv)                                       │
│ W_kv: [hidden_size, kv_lora_rank]                          │
│ kv_compressed: [batch, seq, kv_lora_rank]                   │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼ cache (compressed format)
        ┌──────────────────────────────────────┐
        │ kv_cache: [batch, seq, kv_lora_rank]              │
        └──────────────────────────────────────┘
```

### KV Decompression (during attention)

```
kv_cache: [batch, seq, kv_lora_rank]
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ K Decompression (W_K)                                       │
│ W_K: [kv_lora_rank, num_kv_heads * v_head_dim]             │
│ k_decompressed: [batch, seq, num_kv_heads * v_head_dim]     │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼ reshape
        ┌──────────────────────────────────────┐
        │ K: [batch, num_kv_heads, seq, v_head_dim]           │
        └──────────────────────────────────────┘

kv_cache: [batch, seq, kv_lora_rank]
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ V Decompression (W_V)                                       │
│ W_V: [kv_lora_rank, num_kv_heads * v_head_dim]             │
│ v_decompressed: [batch, seq, num_kv_heads * v_head_dim]     │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼ reshape
        ┌──────────────────────────────────────┐
        │ V: [batch, num_kv_heads, seq, v_head_dim]           │
        └──────────────────────────────────────┘
```

## Architecture Components

### 1. Attention Trait (统一接口)

```rust
pub trait Attention: Send + Sync {
    fn forward(
        &self,
        hidden_states: &Tensor,
        positions: &[i64],
        is_prefill: bool,
    ) -> Result<Tensor>;
    
    fn num_heads(&self) -> usize;
    fn num_kv_heads(&self) -> usize;
    fn head_dim(&self) -> usize;
    fn num_layers(&self) -> usize;
}
```

### 2. MlaAttention Struct

```rust
pub struct MlaAttention {
    // Q projections
    q_proj: Linear,           // W_q: [hidden_size, q_lora_rank]
    q_nope_proj: Linear,      // For q_nope part
    q_rope_proj: Linear,      // For q_rope part
    
    // KV compression
    kv_proj: Linear,          // W_kv: [hidden_size, kv_lora_rank]
    
    // KV decompression
    k_decompress: Linear,     // W_K: [kv_lora_rank, num_kv_heads * v_head_dim]
    v_decompress: Linear,     // W_V: [kv_lora_rank, num_kv_heads * v_head_dim]
    
    // O projection
    o_proj: Linear,           // W_o: [num_heads * head_dim, hidden_size]
    
    // RoPE for q_rope
    rope: RoPE,
    
    // Configuration
    q_lora_rank: usize,
    kv_lora_rank: usize,
    qk_nope_dim: usize,
    qk_rope_dim: usize,
    v_head_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    
    // Compressed KV cache (internal)
    kv_cache: Tensor,         // [num_layers, batch, seq, kv_lora_rank]
}
```

### 3. TransformerBlock with Generic Attention

```rust
pub struct TransformerBlock<Attn: Attention> {
    attention: Attn,
    input_layernorm: LnLayerNorm,
    post_attention_layernorm: LnLayerNorm,
    mlp: SwiGLU,
}

impl<Attn: Attention> TransformerBlock<Attn> {
    pub fn new(
        attention: Attn,
        hidden_size: usize,
        intermediate_size: usize,
        eps: f64,
        device: &Device,
    ) -> Result<Self> {
        // ...
    }
    
    pub fn forward(&self, hidden_states: &Tensor, positions: &[i64], is_prefill: bool) -> Result<Tensor> {
        let residual = hidden_states.clone();
        let x = self.input_layernorm.forward(hidden_states)?;
        let x = self.attention.forward(&x, positions, is_prefill)?;
        let x = (&x + &residual)?;
        
        let residual = x.clone();
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        &x + &residual
    }
}

// Type aliases
pub type GqaTransformerBlock = TransformerBlock<GqaAttention>;
pub type MlaTransformerBlock = TransformerBlock<MlaAttention>;
```

### 4. Config Extensions

Update `Qwen3Config` to properly parse MLA parameters:

```rust
pub struct Qwen3Config {
    // ... existing fields ...
    pub q_lora_rank: Option<usize>,
    pub kv_lora_rank: Option<usize>,
    pub qk_nope_dim: Option<usize>,
    pub qk_rope_dim: Option<usize>,
    pub v_head_dim: Option<usize>,
}

impl Qwen3Config {
    pub fn q_lora_rank(&self) -> usize {
        self.q_lora_rank.or(self.q_len).unwrap_or(512)
    }
    
    pub fn kv_lora_rank(&self) -> usize {
        self.kv_lora_rank.or(self.kv_len).unwrap_or(512)
    }
    
    pub fn qk_nope_dim(&self) -> usize {
        self.qk_nope_dim.unwrap_or(self.q_lora_rank() - self.qk_rope_dim().unwrap_or(128))
    }
    
    pub fn qk_rope_dim(&self) -> usize {
        self.qk_rope_dim.unwrap_or(64)
    }
    
    pub fn v_head_dim(&self) -> usize {
        self.v_head_dim.unwrap_or(128)
    }
}
```

## Implementation Plan

### Phase 1: Core MLA Attention

1. Create `crates/model/src/components/attention/mla.rs`
2. Implement `MlaAttention::new()` with all projections
3. Implement `forward()` with Q projection, KV compression, attention, O projection
4. Add RoPE integration for q_rope

### Phase 2: Attention Trait

1. Define `Attention` trait in `crates/model/src/components/attention/mod.rs`
2. Update `GqaAttention` to implement `Attention`
3. Update `MlaAttention` to implement `Attention`

### Phase 3: Generic TransformerBlock

1. Update `TransformerBlock` to be generic over `Attention`
2. Create `MlaBlock` type alias
3. Update model creation to select block type based on `AttentionType`

### Phase 4: Model Integration

1. Update `Qwen3Model` to support MLA
2. Add cache management for compressed KV
3. Handle prefill vs decode paths

### Phase 5: Tests

| Test | Description |
|------|-------------|
| `test_mla_q_projection_shape` | Verify Q projection output shape |
| `test_mla_kv_compression_shape` | Verify KV compression output shape |
| `test_mla_forward_prefill` | Full forward pass in prefill mode |
| `test_mla_forward_decode` | Full forward pass in decode mode |
| `test_mla_kv_cache_expansion` | Verify KV cache expansion works |
| `test_mla_rope_application` | Verify RoPE is applied to q_rope |
| `test_mla_deterministic` | Output is deterministic |

## Memory Efficiency

| Attention Type | KV Cache Size (per token) |
|----------------|---------------------------|
| GQA (num_kv_heads=8) | 8 * head_dim * 2 |
| MLA (kv_lora_rank=512) | 512 * 2 |

For DeepSeek-V3 (MHA):
- GQA equivalent: 128 * 128 * 2 = 32KB per token
- MLA: 512 * 2 = 1KB per token
- **Reduction: 32x**

## Compatibility

MLA requires:
- `q_len` or `q_lora_rank` in config
- `kv_len` or `kv_lora_rank` in config
- `qk_nope_dim` and `qk_rope_dim` for Q decomposition
- RoPE configuration for q_rope

If these are missing, fall back to standard GQA.

## 6. Logging

### 6.1 Log Levels

Follow the existing logging conventions in `gqa.rs` and `block.rs`:

| Level | Usage | Style |
|-------|-------|-------|
| **trace** | Forward start/complete, shape info | `trace!(field1, field2, "MlaAttention forward started")` |
| **debug** | KV compression, RoPE application | `debug!(compressed_dim, "KV compressed")` |
| **warn** | Fallback to GQA | `warn!("MLA config missing, falling back to GQA")` |

### 6.2 Log Messages

```rust
// Initialization
trace!(
    layers,
    kv_lora_rank,
    num_heads,
    "MlaAttention initialized"
);

// Forward pass
trace!(
    batch_size,
    seq_len,
    kv_lora_rank,
    "MlaAttention forward started"
);

trace!(
    output_shape = ?o.dims(),
    "MlaAttention forward completed"
);

// KV compression
trace!(
    kv_lora_rank,
    original_dim,
    "KV compressed to latent space"
);

trace!(
    num_kv_heads,
    v_head_dim,
    "KV decompressed from latent space"
);

// RoPE
trace!(
    qk_rope_dim,
    "RoPE applied to q_rope"
);

// Fallback
tracing::warn!(
    missing_field = ?field,
    "MLA config incomplete, falling back to GQA"
);
```

### 6.3 Field Format

- Simple values: `field_name` (no prefix)
- Debug formatting: `field_name = ?expr`
- Messages: Short English, PascalCase class name + verb

Example from existing code:
```rust
// gqa.rs:122-127
trace!(
    batch_size,
    seq_len,
    head_dim = self.head_dim,
    "GqaAttention forward started"
);
```
