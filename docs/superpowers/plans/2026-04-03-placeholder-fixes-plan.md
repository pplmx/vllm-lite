# Placeholder Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix fake/placeholder implementations found during code review

**Architecture:** Two-phase approach: Simple fix first, then complex Mamba implementation

**Tech Stack:** Rust, Candle ML framework

---

## Task 1: Fix Embeddings Error Handling (Simple)

**Files:**

- Modify: `crates/server/src/openai/embeddings.rs`

- [ ] **Step 1: Review current implementation**

Read `crates/server/src/openai/embeddings.rs` lines 43-49 to see the fallback code.

- [ ] **Step 2: Replace fallback with error return**

Replace:

```rust
let embeddings = match rx.recv().await {
    Some(emb) => emb,
    None => {
        let embedding_dim = 1024;
        req.input.iter().map(|_| vec![0.0; embedding_dim]).collect()
    }
};
```

With:

```rust
let embeddings = rx.recv().await
    .map_err(|_| {
        (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse::new(
                "Failed to get embeddings from engine",
                "internal_error"
            ))
        )
    })?;
```

- [ ] **Step 3: Run server tests**

```bash
cargo test -p vllm-server embeddings
```

- [ ] **Step 4: Commit**

```bash
git add crates/server/src/openai/embeddings.rs
git commit -m "fix(server): return error instead of zero vectors on embeddings failure"
```

---

## Task 2: Complete Mamba Implementation (Complex)

**Files:**

- Modify: `crates/model/src/qwen3_5/model.rs`
- Create: `crates/model/src/qwen3_5/ssm.rs` (new file for SSM layer)
- Test: Add tests for Mamba model

**Background:**
Mamba is a State Space Model (SSM) architecture, different from transformer attention. The current implementation is just a simple linear layer stack (placeholder).

**Mamba Architecture Requirements:**

1. **SSM Core**: Selective State Space Model (S6) with hidden state
2. **Gated MLP**: SiLU gating mechanism
3. **Conv1D**: For handling sequential data
4. **Layer Norm**: RMSNorm (already present)

- [ ] **Step 1: Research Mamba architecture**

Review the current `MambaBlock` implementation in `crates/model/src/qwen3_5/model.rs`:

- Current: Simple linear layer
- Needed: Full SSM block with:
    - Input projection
    - Conv1D for local context
    - SSM (state space model) core
    - Gated MLP output

- [ ] **Step 2: Create SSM layer module**

Create `crates/model/src/qwen3_5/ssm.rs`:

```rust
use candle_core::{Tensor, Result};
use candle_nn::{Linear, Conv1d};

pub struct SSMState {
    pub x: Tensor,  // [batch, dim]
}

pub struct SSMLayer {
    pub x_proj: Linear,   // Projects input to ssm dimensions
    pub dt_proj: Linear,  // Delta time projection  
    pub A: Tensor,        // State matrix [dim]
    pub B: Tensor,        // Input matrix [dim]
    pub C: Tensor,        // Output matrix [dim]
    pub D: Tensor,        // Skip connection
    pub conv: Conv1d,     # Local convolution
}

impl SSMLayer {
    pub fn new(dim: usize, state_dim: usize, vb: VarBuilder) -> Result<Self> {
        // Implementation of SSM layer
    }

    pub fn forward(&self, x: &Tensor, state: &mut SSMState) -> Result<Tensor> {
        // Selective scan implementation
    }
}
```

- [ ] **Step 3: Rewrite MambaBlock to use SSM**

Replace the simple `MambaBlock` in `model.rs` with:

```rust
pub struct MambaBlock {
    pub ssm: SSMLayer,
    pub gating: Linear,
    pub norm: candle_nn::LayerNorm,
}

impl MambaBlock {
    pub fn new(hidden_size: usize, ssm_state_size: usize, vb: VarBuilder) -> CandleResult<Self> {
        // Initialize SSM layer, gating, and normalization
    }

    pub fn forward(&mut self, x: &Tensor, state: &mut SSMState) -> CandleResult<Tensor> {
        // 1. Conv1D for local context
        // 2. SSM forward pass
        // 3. Gating with SiLU
        // 4. Residual connection
    }
}
```

- [ ] **Step 4: Update Qwen35Model to handle state**

The model needs to maintain SSM state across timesteps:

```rust
pub struct Qwen35Model {
    // ... existing fields
    ssm_states: Vec<SSMState>,  // Per-sequence SSM state
}
```

- [ ] **Step 5: Remove placeholder warning**

Remove line 85:

```rust
eprintln!("Warning: Qwen3.5 Mamba implementation is simplified (placeholder)");
```

- [ ] **Step 6: Add tests**

```rust
#[test]
fn test_mamba_block_forward() {
    // Test SSM layer produces valid output
}

#[test]
fn test_mamba_state_persistence() {
    // Test SSM state is maintained across tokens
}
```

- [ ] **Step 7: Run tests**

```bash
cargo test -p vllm-model
```

- [ ] **Step 8: Commit**

```bash
git add crates/model/src/qwen3_5/
git commit -m "feat(model): implement Mamba SSM architecture"
```

---

## Success Criteria

- [ ] Embeddings endpoint returns proper error on failure
- [ ] No placeholder warnings in production
- [ ] Mamba implementation uses proper SSM architecture
- [ ] All tests pass
