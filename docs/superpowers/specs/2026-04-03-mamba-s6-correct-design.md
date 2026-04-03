# Correct Mamba S6 Implementation Design

**Date**: 2026-04-03  
**Status**: Draft  
**Issue**: Current Mamba implementation is incorrect - missing S6 selective scan

---

## 1. Executive Summary

This document outlines the correct implementation of Mamba S6 (Selective State Space Model) for Qwen3.5.

**Current Problem**: The current implementation only has Conv1D + SiLU, missing the core S6 selective scan algorithm.

---

## 2. Mamba S6 Architecture

### Core Components

```
Input x [batch, seq, d_model]
    │
    ▼
┌─────────────────────────────────────────┐
│           Input Projection              │
│    x → (z [gating], x_inner)            │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│              Conv1D                     │
│  x_inner → x_conv (local context)       │
│  + SiLU activation                      │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│            x_proj                       │
│  x_conv → (delta, B, C)                 │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│        Discretization                   │
│  delta → A_bar (using exp(delta*A))     │
│  delta → B_bar (using exp(delta*B))     │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│      Selective Scan (S6)                │
│  h_new = A_bar * h + B_bar * x          │
│  y = C * h                              │
│  (parallel scan algorithm)              │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│            Gating                        │
│  output = z * silu(y + D * x_conv)      │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│           Output Projection             │
│  output → [batch, seq, d_model]         │
│  + residual connection                  │
└─────────────────────────────────────────┘
```

### Learnable Parameters

| Parameter | Shape | Description |
|-----------|-------|-------------|
| A | [d_state, d_model] | State matrix (typically initialized to -1) |
| x_proj | [d_inner, d_inner * 3] | Projects to delta, B, C |
| conv | [d_inner, d_inner, d_conv] | 1D convolution |
| input_proj | [d_model, d_inner * 2] | Gated projection |
| output_proj | [d_inner, d_model] | Output projection |
| D | [d_inner] | Skip connection (learnable) |

### Key Hyperparameters

| Parameter | Typical Value | Description |
|-----------|---------------|-------------|
| d_model | 2048-4096 | Model hidden dimension |
| d_state | 16-64 | SSM state expansion |
| d_conv | 4 | Convolution width |
| expand | 2 | Inner dimension expansion |

---

## 3. Implementation Details

### 3.1 Discretization

The SSM operates in continuous time, discretized using bilinear transform:

```
A_bar = exp(delta * A)
B_bar = delta * B
```

Where delta = softplus(x_proj_delta)

### 3.2 Selective Scan

The key innovation of Mamba is **selectiveness** - delta, B, C are computed per-token, not shared:

```rust
// Pseudo-code for selective scan
for t in 0..seq_len:
    delta = softplus(delta_proj[x_t])
    A_bar = exp(delta * A)  // [d_state, d_state]
    B_bar = delta * B[t]    // [d_state]
    
    h = A_bar * h + B_bar * x_t  // State update
    y_t = C * h                  // Output
```

### 3.3 Parallel Scan

For efficiency, use parallel scan algorithm (associative scan) instead of sequential loop:

```
y = C @ scan(A_bar, B_bar ⊙ x)
```

---

## 4. Files to Modify

### 4.1 New/Modified Files

| File | Action | Description |
|------|--------|-------------|
| `crates/model/src/qwen3_5/ssm.rs` | Modify | Replace with correct S6 implementation |
| `crates/model/src/qwen3_5/model.rs` | Modify | Update to work with new SSM |
| `crates/model/src/qwen3_5/mod.rs` | No change | Already exports ssm |

### 4.2 Key Changes to ssm.rs

1. Add A, B, C, D as learnable parameters
2. Add proper discretization
3. Implement parallel scan
4. Fix gating: `output = z * silu(ssm_out + D * x_conv)`

---

## 5. Success Criteria

- [ ] S6 layer has learnable A, B, C, D parameters
- [ ] Discretization uses exp(delta * A) + delta * B
- [ ] Gating uses z * silu(ssm_out) (not just silu)
- [ ] Parallel scan for efficiency
- [ ] All existing tests pass