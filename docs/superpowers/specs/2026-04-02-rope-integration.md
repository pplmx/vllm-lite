# RoPE Integration Design

## Overview

Implement complete Rotary Position Embedding (RoPE) in vLLM-lite to make the `positions` parameter functional across the model backend. This enables position-aware attention where different sequence positions produce different outputs.

## Motivation

The `ModelBackend` trait includes a `positions` parameter that is currently unused:

```rust
fn forward(
    &self,
    seq_ids: &[SeqId],
    input_tokens: &[Vec<TokenId>],
    positions: &[Vec<usize>],  // Currently unused
) -> Result<BatchOutput>;
```

This causes:
- Warnings in tests about unused variables
- Incomplete model implementation (position information lost)
- Inability to test position-dependent behavior

## Architecture

### Data Flow

```
Engine::forward() 
  → ModelBackend::forward(seq_ids, input_tokens, positions)
      → Qwen3Model::forward(positions)
          → embed_tokens
          → TransformerBlock::forward(x, positions)
              → GqaAttention::forward(x, positions)
                  → q_proj, k_proj
                  → apply_rope(query, key, position_ids)  ← NEW
                  → attention computation
```

### Key Changes

1. **rope.rs**: Implement actual RoPE computation
2. **GqaAttention**: Add positions parameter, apply RoPE to Q/K
3. **TransformerBlock**: Pass positions through layers
4. **Qwen3Model**: Wire positions from trait to attention
5. **Other models**: Update to use positions (FakeModel, Qwen5)

## Implementation Details

### 1. RoPE Algorithm (rope.rs)

Standard RoPE applies rotary transformation to query and key vectors:

For each position `m` and dimension `i`:
- Angle = m * θ_i where θ_i = base^(-2i/d)
- Rotate by angle using complex number representation

```rust
pub fn apply_rope(query: &Tensor, position_ids: &Tensor, theta: f32) -> Result<Tensor>
```

Input shapes:
- query: [batch, seq_len, num_heads, head_dim]
- position_ids: [seq_len]

Output: Same shape as query, with rotation applied

### 2. GqaAttention Changes

Add `positions` parameter to all forward methods:

```rust
pub fn forward(&self, x: &Tensor, positions: &[usize]) -> Result<Tensor>
pub fn forward_prefill(&self, x: &Tensor, positions: &[usize], ...) -> Result<Tensor>
pub fn forward_decode(&self, x: &Tensor, positions: &[usize], ...) -> Result<Tensor>
```

Apply RoPE after Q/K projection but before attention:

```rust
let q = self.q_proj.forward(x)?;
let k = self.k_proj.forward(x)?;

// Reshape to [batch, seq, heads, head_dim]
let q = q.reshape((batch, seq, self.num_heads, self.head_dim))?;
let k = k.reshape((batch, seq, self.num_kv_heads, self.head_dim))?;

let position_ids = Tensor::new(positions, x.device())?;
let q = apply_rope(&q, &position_ids, self.theta)?;
let k = apply_rope(&k, &position_ids, self.theta)?;
```

### 3. TransformerBlock Changes

Pass positions through:

```rust
pub fn forward(&self, x: &Tensor, positions: &[usize]) -> Result<Tensor>
```

### 4. Qwen3Model Changes

Store RoPE config and thread positions:

```rust
pub struct Qwen3Model {
    // ... existing fields
    rope: RoPE,
}

impl ModelBackend for Qwen3Model {
    fn forward(&self, seq_ids: &[SeqId], input_tokens: &[Vec<TokenId>], positions: &[Vec<usize>]) -> ... {
        for (seq_idx, (tokens, pos)) in input_tokens.iter().zip(positions).enumerate() {
            // Pass positions to transformer layers
        }
    }
}
```

## Testing Strategy

1. **Unit tests for apply_rope**:
   - Verify output shape matches input
   - Verify different positions produce different outputs
   - Verify same position produces same output (deterministic)

2. **Integration test**:
   - Same input tokens at different positions should produce different next tokens

3. **Mock model update**:
   - Make MockModel actually use positions parameter

## Scope

- Implement for all ModelBackend implementations: Qwen3Model, Qwen5Model, FakeModel
- Standard RoPE (not Linear/Nerf)
- Backward compatible: positions can be simple indices [0, 1, 2, ...]

## Trade-offs

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| RoPE variant | Standard | Simpler to implement, works with current attention |
| Where to apply | After Q/K projection | Standard practice, matches HuggingFace |
| Positions format | Simple indices | Easy to generate, sufficient for testing |

## Risks

- Performance: RoPE adds computation per forward pass
- Compatibility: Must ensure existing tests still pass
- CUDA: Need to verify works on GPU (may need kernel optimization later)

## Success Criteria

1. `cargo nextest run` passes with no warnings
2. Different positions produce measurably different attention outputs
3. All existing ModelBackend implementations updated
4. At least one test verifies position-dependent behavior