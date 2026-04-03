# Component Extraction Design

**Date**: 2026-04-03
**Status**: Approved
**Goal**: Extract common components (Attention, MLP, RoPE) to support multiple model architectures (Qwen3, Llama, Mistral)

## Problem Statement

**Current Issues:**
1. Qwen3 components (attention, mlp, rope) are tightly coupled to qwen3 module
2. RoPE is in qwen3/rope.rs but is a general component used by Llama/Mistral
3. Hard to add new architectures (Llama, Mistral) without code duplication

## Target Architecture

```
model/src/
├── components/
│   ├── mod.rs
│   ├── attention.rs    # gqa_forward, paged_attention_*
│   ├── mlp.rs          # swiglu_forward, gated_mlp_forward
│   ├── norm.rs         # rms_norm, layer_norm
│   └── positional.rs   # RoPE, apply_rope, apply_alibi
│
├── qwen3/
│   ├── block.rs        # Uses components
│   ├── attention.rs    # Simplified, delegates to components
│   └── model.rs
│
├── llama/              # NEW: Llama architecture support
│   ├── mod.rs
│   ├── block.rs
│   └── model.rs
│
└── registry.rs         # Updated for multi-architecture
```

## Component APIs

### attention.rs

```rust
#[allow(clippy::too_many_arguments)]
pub fn gqa_forward(
    x: &Tensor,
    q_proj: &Linear,
    k_proj: &Linear,
    v_proj: &Linear,
    o_proj: &Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<Tensor>;

#[allow(clippy::too_many_arguments)]
pub fn gqa_forward_with_rope(
    x: &Tensor,
    q_proj: &Linear,
    k_proj: &Linear,
    v_proj: &Linear,
    o_proj: &Linear,
    positions: &[usize],
    rope: &RoPE,
    q_norm: Option<&LayerNorm>,
    k_norm: Option<&LayerNorm>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<Tensor>;

#[allow(clippy::too_many_arguments)]
pub fn paged_attention_prefill(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    kv_cache: &mut PagedKvCache,
    layer_idx: usize,
    block_ids: &[usize],
    positions: &[usize],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    tile_size: Option<usize>,
) -> Result<Tensor>;

#[allow(clippy::too_many_arguments)]
pub fn paged_attention_decode(
    q: &Tensor,
    kv_cache: &PagedKvCache,
    layer_idx: usize,
    block_ids: &[usize],
    num_computed_tokens: usize,
    num_heads: usize,
    head_dim: usize,
    tile_size: Option<usize>,
) -> Result<Tensor>;
```

### mlp.rs

```rust
pub fn swiglu_forward(
    x: &Tensor,
    gate_proj: &Linear,
    up_proj: &Linear,
    down_proj: &Linear,
) -> Result<Tensor>;

pub fn gated_mlp_forward(
    x: &Tensor,
    gate_proj: &Linear,
    up_proj: &Linear,
    down_proj: &Linear,
) -> Result<Tensor>;
```

### positional.rs

```rust
pub struct RoPE {
    pub theta: f32,
    pub head_dim: usize,
    pub scaling_factor: f32,
}

impl RoPE {
    pub fn new(theta: f32, head_dim: usize) -> Self;
    pub fn apply(&self, x: &Tensor, positions: &[usize]) -> Result<Tensor>;
    pub fn apply_qk(&self, q: &Tensor, k: &Tensor, positions: &[usize]) -> Result<(Tensor, Tensor)>;
}
```

### norm.rs

```rust
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor>;
pub fn layer_norm(x: &Tensor, weight: &Tensor, bias: &Tensor, eps: f64) -> Result<Tensor>;
```

## Migration Plan

### Phase 1: Extract Independent Functions
1. Move `RoPE` to `components/positional.rs`
2. Move `apply_rope` to `components/positional.rs`
3. Move `swiglu_forward` to `components/mlp.rs`
4. Move normalization functions to `components/norm.rs`

### Phase 2: Migrate Attention
1. Create `components/attention.rs` with:
   - `gqa_forward_with_rope` (from qwen3/attention.rs)
   - `paged_attention_prefill` (from qwen3/attention.rs)
   - `paged_attention_decode` (from qwen3/attention.rs)
2. Simplify `qwen3/attention.rs` to delegate to components

### Phase 3: Add Llama Skeleton
1. Create `model/src/llama/`
2. Implement `LlamaBlock` using components
3. Add to model registry

## Testing Strategy

```rust
// components tests
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_swiglu_forward_shape() { ... }
    #[test]
    fn test_rope_apply_deterministic() { ... }
    #[test]
    fn test_gqa_forward_output_shape() { ... }
    #[test]
    fn test_paged_attention_prefill() { ... }
    #[test]
    fn test_paged_attention_decode() { ... }
}

// Existing Qwen3 tests remain unchanged
// Only internal implementation changes
```

## Acceptance Criteria

- [ ] `cargo test --workspace` passes
- [ ] `cargo clippy --workspace` has no warnings
- [ ] Qwen3 functionality unchanged
- [ ] New components have comprehensive tests
- [ ] Llama skeleton can be built

## Benefits

1. **Reusability**: Attention, MLP, RoPE can be reused across architectures
2. **Testability**: Pure functions are easy to unit test
3. **Maintainability**: Changes to core algorithms only need to be made once
4. **Extensibility**: New architectures (Llama, Mistral) can be added with minimal code
