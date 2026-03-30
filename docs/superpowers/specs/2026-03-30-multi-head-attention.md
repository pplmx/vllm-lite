# Multi-Head Attention Design Spec

## Overview

Support MHA, GQA, MQA, and MLA attention mechanisms in Qwen3 model inference.

## Problem

Current implementation fails with:
```
shape mismatch in reshape, lhs: [1, 4, 896], rhs: [1, 14, 64]
```
- Qwen2.5-0.5B has 14 query heads, 2 KV heads (GQA)
- Current code doesn't expand KV heads to match Q heads

## Attention Types

| Type | num_q_heads | num_kv_heads | KV Expansion |
|------|-------------|--------------|--------------|
| MHA | N | N | None (1:1) |
| MQA | N | 1 | Repeat N times |
| GQA | N | M (M < N) | Repeat N/M times |
| MLA | Compressed | Compressed | Decompress from latent |

## Architecture

```
GqaAttention
├── q_proj: Linear(hidden_size, num_q_heads * head_dim)
├── k_proj: Linear(hidden_size, num_kv_heads * head_dim)  
├── v_proj: Linear(hidden_size, num_kv_heads * head_dim)
├── o_proj: Linear(num_q_heads * head_dim, hidden_size)
└── kv_cache: PagedKvCache

Forward flow:
1. Project x to Q, K, V
2. Reshape: Q -> [batch, num_q_heads, head_dim]
           K -> [batch, num_kv_heads, head_dim]
           V -> [batch, num_kv_heads, head_dim]
3. Expand K, V to match Q heads (GQA/MQA)
4. Compute attention scores
5. Output projection
```

## Key Changes

### 1. Fix Sequence Length in Reshape

Current bug: Reshape drops seq_len dimension.

```rust
// Current (WRONG):
let q = q.reshape((q.dims()[0], self.num_heads, self.head_dim))?;

// Correct - preserve seq_len:
let batch_size = q.dims()[0];
let seq_len = q.dims()[1] / self.num_heads;
let q = q.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?;
// Or more commonly: flatten seq_len into batch for computation
let q = q.reshape((batch_size * seq_len, self.num_heads, self.head_dim))?;
```

### 2. KV Expansion for GQA/MQA

```rust
fn expand_kv(k: &Tensor, num_q_heads: usize, num_kv_heads: usize) -> Result<Tensor> {
    if num_q_heads == num_kv_heads {
        Ok(k.clone())  // MHA
    } else {
        let repeat_factor = num_q_heads / num_kv_heads;
        // Reshape then tile: [batch, seq_len, num_kv, head] -> [batch, seq_len, num_q, head]
        let k = k.reshape((k.dims()[0], k.dims()[1], num_kv_heads, self.head_dim))?;
        let k_expanded = tile_heads(k, repeat_factor)?;  // Expand KV heads
        Ok(k_expanded)
    }
}

// Tile heads along head dimension
fn tile_heads(tensor: &Tensor, repeat: usize) -> Result<Tensor> {
    let (batch, seq, heads, dim) = tensor.dims4()?;
    let tensor = tensor.unsqueeze(3)?;  // [batch, seq, heads, 1, dim]
    let tensor = tensor.repeat(3, repeat)?;  // [batch, seq, heads*repeat, 1, dim]
    let tensor = tensor.reshape((batch, seq, heads * repeat, dim))?;  // [batch, seq, heads*repeat, dim]
    Ok(tensor)
}
```

### 3. MLA Support

MLA (Multi-Latent Attention) from Qwen2.5 uses compressed KV:

```rust
// MLA weight structure:
// q_a_proj: [hidden, q_len] - projects to latent Q
// q_b_proj: [q_len, num_heads * (head_dim + qk_rope_dim)] - expands Q + rope
// kv_a_proj_m: [hidden, kv_len] - projects to latent KV  
// kv_b_proj: [kv_len, num_heads * (head_dim + qk_nope_dim)] - expands KV

// Simplified flow:
// 1. q_hidden = q_a_proj(x) -> [batch, seq, q_len]
// 2. q = q_b_proj(q_hidden) -> [batch, seq, num_heads, head_dim + qk_dim]
// 3. Split q into q_nope (no position) and q_rope (with position)
// 4. Similar for KV: decompress from latent
```

### 4. Config Updates

Add fields for MLA detection:

```rust
#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3Config {
    // ... existing fields ...
    #[serde(default)]
    pub q_len: Option<usize>,        // MLA Q projection dim
    #[serde(default)]
    pub qk_nope_dim: Option<usize>,  // MLA non-position-aware dim
    #[serde(default)]
    pub qk_rope_dim: Option<usize>,  // MLA rope dim
    #[serde(default)]
    pub kv_len: Option<usize>,       // MLA KV projection dim
}

impl Qwen3Config {
    pub fn attention_type(&self) -> AttentionType {
        if self.q_len.is_some() || self.kv_len.is_some() {
            AttentionType::MLA
        } else if self.num_key_value_heads == 1 {
            AttentionType::MQA
        } else if self.num_key_value_heads == self.num_attention_heads {
            AttentionType::MHA
        } else {
            AttentionType::GQA
        }
    }
}

enum AttentionType {
    MHA,
    MQA, 
    GQA,
    MLA,
}
```

## File Changes

```
crates/model/src/qwen3/
├── attention.rs       # MODIFY: Fix reshape, add expand_kv, tile_heads, MLA forward
├── block.rs           # MODIFY: Pass correct num_kv_heads
└── config.rs          # MODIFY: Add MLA config fields (q_len, qk_nope_dim, etc.)
```

## Implementation Details

### Candle Tensor Operations

Need to verify available operations:
- `repeat` or `tile` for head expansion
- `reshape` with proper dimensions
- 4D tensor handling for attention

### Error Handling

- Handle num_q_heads % num_kv_heads != 0 (should not happen in valid models)
- Validate head_dim consistency

## Implementation Priority

1. **Phase 1**: Fix GQA expansion (current bug)
2. **Phase 2**: Add MQA support  
3. **Phase 3**: Add MLA support

## Testing

Test with different models:
- Qwen2.5-0.5B: GQA (14 Q heads, 2 KV heads)
- Qwen2-7B: GQA (32 Q heads, 2 KV heads)
- Qwen2-0.5B: MHA (14 Q heads, 14 KV heads)

## Out of Scope

- Sliding window attention
- Flash attention optimization
- Prefix caching integration