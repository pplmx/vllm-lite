# Mixtral Support Design

**Date**: 2026-04-04
**Status**: In Progress
**Goal:** Add Mixtral MoE (8x7B) support

## Overview

Mixtral is a Sparse Mixture of Experts (SMoE) model from Mistral AI. Key features:

- 8 experts per layer, top-2 routing
- 4096 sliding window attention
- Similar to Mistral but with MoE

## Architecture Comparison

| Feature   | Mistral | Mixtral             |
| --------- | ------- | ------------------- |
| Norm      | RMSNorm | RMSNorm             |
| Attention | Sliding | Sliding             |
| MLP       | SwiGLU  | **MoE (8 experts)** |
| Experts   | 1       | 8 (top-2)           |

## Implementation

### 1. MixtralConfig

```rust
pub struct MixtralConfig {
    // Standard fields
    pub hidden_size: usize,      // 4096
    pub num_layers: usize,        // 32
    pub num_heads: usize,         // 32
    pub num_kv_heads: usize,     // 8
    pub vocab_size: usize,        // 32000

    // MoE specific
    pub num_experts: usize,      // 8
    pub top_k_experts: usize,     // 2
    pub expert_intermediate_size: usize,  // 14336
    pub sliding_window: usize,     // 4096
}
```

### 2. MixtralSparseMoe

```rust
pub struct MixtralSparseMoe {
    experts: Vec<SwiGLU>,  // 8 experts
    gate: Linear,
    top_k: usize,  // 2
}

impl MixtralSparseMoe {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // 1. Compute gate scores
        let gate_logits = self.gate.forward(x)?;

        // 2. Get top-k experts
        let (top_indices, top_weights) = gate_logits.topk(self.top_k, 1)?;

        // 3. Forward through selected experts
        // 4. Weighted sum of expert outputs
    }
}
```

### 3. MixtralBlock

- Similar to MistralBlock but uses MixtralSparseMoe instead of SwiGLU

## Files to Create/Modify

```text
crates/model/src/
├── mistral/
│   ├── mod.rs
│   └── sparse_moe.rs  # NEW: MixtralSparseMoe
├── mixtral/            # NEW
│   ├── mod.rs
│   ├── block.rs
│   └── model.rs
└── config/
    └── model_config.rs  # Add Mixtral
```

## Implementation Order

1. Add Mixtral to Architecture enum
2. Create sparse_moe.rs in mistral
3. Create mixtral/ module
4. Implement MixtralModel
5. Update loader/registry
6. Tests

## Acceptance Criteria

- [ ] Can detect "mixtral" model type
- [ ] MixtralSparseMoe implements top-2 routing
- [ ] Can load Mixtral weights
- [ ] Tests pass
