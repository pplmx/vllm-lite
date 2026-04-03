# Deep Dive: Mamba S6 Selective Scan Implementation

**Date**: 2026-04-03  
**Status**: Detailed Design

---

## 1. Mathematical Foundation

### 1.1 State Space Model (Continuous)

The continuous-time SSM:

```text
h'(t) = A * h(t) + B * x(t)
y(t) = C * h(t) + D * x(t)
```

Where:

- h(t): hidden state [d_state]
- x(t): input [d_inner]
- y(t): output [d_inner]
- A: state matrix [d_state, d_state]
- B: input matrix [d_state]
- C: output matrix [d_state]
- D: skip connection [d_inner]

### 1.2 Discretization (Zero-Order Hold)

Discretize using delta (step size):

```text
delta = softplus(Linear(x))
A_bar = exp(delta * A)
B_bar = delta * B
```

Then the discrete SSM:

```text
h_t = A_bar * h_{t-1} + B_bar * x_t
y_t = C * h_t + D * x_t
```

### 1.3 Selective Scan (The Mamba Innovation)

Key difference from standard SSM: **delta, B, C are per-token, not shared**

```text
For each token t:
    delta_t = softplus(delta_proj(x_t))
    B_t = B_proj(x_t)      # [d_state]
    C_t = C_proj(x_t)      # [d_state]

    A_bar_t = exp(delta_t * A)
    B_bar_t = delta_t * B_t

    h_t = A_bar_t * h_{t-1} + B_bar_t * x_t
    y_t = C_t * h_t
```

### 1.4 Parallel Scan

For sequence parallelization, use associative scan:

```text
# Associative scan for: y = C @ scan(A_bar, B_bar ⊙ x)
# 
# Define operator: (h1, y1) ⊕ (h2, y2) = (A_bar2 * h1, y2 + C2 * h1)
# 
# Result: y = y_N where H_N = A_bar_N * ... * A_bar_1 * h_0
```

---

## 2. Complete Parameter List

### SSMLayer Parameters

```rust
pub struct SSMLayer {
    // Projections from d_inner to (delta, B, C)
    x_proj: Linear,  // [d_inner, d_inner * 3]

    // State space matrices
    A: Tensor,       // [d_state, d_model] - initialized to -1
    D: Tensor,       // [d_inner] - skip connection

    // Convolution for local context
    conv: Conv1d,    // [d_inner, d_inner, d_conv]
}
```

### MambaBlock Parameters

```rust
pub struct MambaBlock {
    input_proj: Linear,   // [d_model, d_inner * 2] - gated
    ssm: SSMLayer,
    output_proj: Linear,  // [d_inner, d_model]
    norm: LayerNorm,
}
```

---

## 3. Forward Pass Detailed

### Step 1: Input Projection (Gated)

```rust
let x_proj = self.input_proj.forward(x)?;
// Split: [batch, seq, d_inner * 2] → [batch, seq, d_inner] * 2
let (z, x_inner) = x_proj.split_at(2, 1)?;  // split at dim 1
```

### Step 2: Conv1D + SiLU

```rust
let x_conv = x_inner.transpose(1, 2)?;  // [batch, d_inner, seq]
let x_conv = self.conv.forward(&x_conv)?;
let x_conv = x_conv.transpose(1, 2)?;  // [batch, seq, d_inner]
let x_conv = candle_nn::ops::silu(&x_conv)?;
```

### Step 3: Compute delta, B, C per token

```rust
let x_ssm = self.ssm.x_proj.forward(&x_conv)?;
// Split: [batch, seq, d_inner * 3] → (delta, B, C)
// Each is [batch, seq, d_inner] or [batch, seq, d_inner]
let (delta_flat, rest) = x_ssm.split_at(3, 2)?;
let (B_flat, C_flat) = rest.split_at(1, 2)?;

let delta = candle_nn::ops::softplus(&delta_flat)?;  // [batch, seq, d_inner]
```

### Step 4: Discretization

```rust
// A_bar = exp(delta * A)
// A is [d_state, d_model], delta is [batch, seq, d_inner]
// Result: A_bar is [batch, seq, d_state, d_state]

let A = self.ssm.A.broadcast_mul(&delta)?;  // Element-wise: delta * A
let A_bar = candle_core::ops::exp(&A)?;

// B_bar = delta * B
// B is [d_inner, d_state], delta is [batch, seq, d_inner]
let B_bar = delta_flat.broadcast_mul(&self.ssm.B)?;  // [batch, seq, d_state]
```

### Step 5: Selective Scan (Parallel)

```rust
// y = C @ scan(A_bar, B_bar ⊙ x)
// This is the expensive part - use parallel scan

// Simplified sequential version for correctness:
let mut h = Tensor::zeros(&[batch, d_state], DType::F32, device)?;
// Actually need h per batch element

for t in 0..seq_len {
    let x_t = x_conv.narrow(1, t, 1)?;        // [batch, 1, d_inner]
    let B_t = B_bar.narrow(1, t, 1)?;         // [batch, 1, d_state]
    let C_t = C_flat.narrow(1, t, 1)?;        // [batch, 1, d_state]
    let A_t = A_bar.narrow(1, t, 1)?;         // [batch, 1, d_state, d_state]

    // h = A_t * h + B_t * x_t
    h = A_t.matmul(&h.unsqueeze(-1))?.squeeze(-1)?;
    h = h + B_t.broadcast_mul(&x_t.squeeze(1))?;

    // y = C * h
    let y_t = C_t.matmul(&h.unsqueeze(-1))?.squeeze(-1)?;
    outputs.push(y_t);
}
```

### Step 6: Gating + Residual

```rust
// Gating: z * silu(y + D * x_conv)
let ssm_out = Tensor::stack(&outputs, 1)?;  // [batch, seq, d_inner]
let D_mul_x = self.ssm.D.broadcast_mul(&x_conv)?;  // [batch, seq, d_inner]
let ssm_gated = ssm_out + D_mul_x;
let gated = candle_nn::ops::silu(&ssm_gated)?;

// Apply gating from input projection
let gated = z * gated;

// Output projection + residual
let output = self.output_proj.forward(&gated)?;
let output = output + x;  // residual
let output = self.norm.forward(&output)?;

Ok(output)
```

---

## 4. Implementation Priority

Since full parallel scan is complex, here's a tiered approach:

### Tier 1: Correct Architecture (Simplified Scan)

- ✅ Proper parameter shapes (A, B, C, D)
- ✅ Correct discretization
- ⚠️ Sequential scan (slower but correct)
- ❌ No parallel optimization

### Tier 2: Optimized Scan

- Use associative scan for parallelization
- Memory-efficient chunking

### Tier 3: Full Mamba-2

- State space fusion (SSF)
- FlashAttention-like optimization

---

## 5. Key Code Patterns to Avoid

### ❌ Wrong: No per-token selectivity

```rust
// BAD: Shared delta, B, C for all tokens
let delta = self.delta_proj(x.mean(1)?);  // Pooled!
```

### ❌ Wrong: No discretization

```rust
// BAD: Direct SSM without discretization
let h = self.A * h + self.B * x;
```

### ❌ Wrong: Wrong gating

```rust
// BAD: No gating from input projection
let output = self.output_proj.forward(&ssm_out)?;
```

### ✅ Correct: Full selectivity

```rust
// GOOD: Per-token delta, B, C
let (delta, B, C) = self.x_proj(x).split_at(3, -1)?;
let delta = candle_nn::ops::softplus(delta);
let A_bar = (delta.unsqueeze(-1) * self.A.unsqueeze(0).unsqueeze(0))?.exp();
```

---

## 6. Testing Strategy

### Unit Tests

1. **test_ssm_discretization**: Verify A_bar = exp(delta * A)
2. **test_ssm_forward**: Verify output shape and non-zero values
3. **test_ssm_selectivity**: Verify different inputs produce different outputs
4. **test_mamba_block**: Verify residual connection works

### Integration Tests

1. **test_qwen35_mamba_forward**: Full model forward pass
2. **test_embedding_extraction**: Verify embeddings are not all zeros
