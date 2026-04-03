# Component Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract common components (Attention, MLP, RoPE, Norm) to support multiple model architectures

**Architecture:** Function-based composition - no trait objects, zero runtime overhead. Components are pure functions that can be reused across Qwen3, Llama, Mistral.

**Tech Stack:** Rust, Candle

---

## File Structure

```
crates/model/src/
├── components/
│   ├── mod.rs          # NEW: Module entry
│   ├── attention.rs    # NEW: GQA attention functions
│   ├── mlp.rs          # NEW: MLP forward functions
│   ├── norm.rs         # NEW: Normalization functions
│   └── positional.rs   # NEW: RoPE implementation
│
├── qwen3/
│   ├── attention.rs    # MODIFY: Delegate to components
│   ├── mlp.rs          # MODIFY: Keep thin wrapper or delete
│   ├── rope.rs         # MODIFY: Delete, use components
│   └── block.rs        # MODIFY: Use components
│
└── llama/              # NEW: Llama architecture
    ├── mod.rs
    ├── block.rs
    └── model.rs
```

---

## Task 1: Create components module structure

**Files:**
- Create: `crates/model/src/components/mod.rs`

- [ ] **Step 1: Create components/mod.rs**

```rust
pub mod attention;
pub mod mlp;
pub mod norm;
pub mod positional;

pub use attention::*;
pub use mlp::*;
pub use norm::*;
pub use positional::*;
```

- [ ] **Step 2: Run build to verify module compiles**

Run: `cargo build -p vllm-model`
Expected: Compiles (empty modules)

- [ ] **Step 3: Commit**

```bash
git add crates/model/src/components/ crates/model/src/lib.rs
git commit -m "feat(model): create components module structure"
```

---

## Task 2: Extract positional.rs (RoPE)

**Files:**
- Create: `crates/model/src/components/positional.rs`
- Modify: `crates/model/src/qwen3/rope.rs`
- Modify: `crates/model/src/qwen3/attention.rs:1-10` (remove import)

- [ ] **Step 1: Create components/positional.rs with RoPE**

```rust
use candle_core::{Result, Tensor};

pub struct RoPE {
    pub theta: f32,
    pub head_dim: usize,
    pub scaling_factor: f32,
}

impl RoPE {
    pub fn new(theta: f32, head_dim: usize) -> Self {
        Self {
            theta,
            head_dim,
            scaling_factor: 1.0,
        }
    }

    pub fn new_with_config(config: &super::super::config::Qwen3Config) -> Self {
        let rope_scaling = config.rope_scaling();
        Self {
            theta: config.rope_theta(),
            head_dim: config.hidden_size() / config.num_attention_heads(),
            scaling_factor: rope_scaling.and_then(|r| r.factor).unwrap_or(1.0),
        }
    }

    pub fn apply(&self, x: &Tensor, positions: &[usize]) -> Result<Tensor> {
        apply_rope(x, positions, self.theta)
    }
}

pub fn apply_rope(query: &Tensor, positions: &[usize], theta: f32) -> Result<Tensor> {
    let (batch, seq_len, num_heads, head_dim) = query.dims4()?;
    let query = query.transpose(1, 2)?;
    let positions = positions.iter().map(|&p| p as i64).collect::<Vec<_>>();
    let half_dim = head_dim / 2;

    let inv_freq: Vec<f32> = (0..half_dim)
        .map(|i| theta.powf(-2.0 * (i as f32) / (head_dim as f32)))
        .collect();

    let mut cos_matrix = Vec::with_capacity(seq_len * half_dim);
    let mut sin_matrix = Vec::with_capacity(seq_len * half_dim);

    for &pos in &positions {
        let pos_f = pos as f32;
        for &freq in &inv_freq {
            let angle = pos_f * freq;
            cos_matrix.push(angle.cos());
            sin_matrix.push(angle.sin());
        }
    }

    let cos = Tensor::new(cos_matrix.as_slice(), query.device())?
        .reshape((1, seq_len, half_dim))?
        .broadcast_as((batch, num_heads, seq_len, half_dim))?;
    let sin = Tensor::new(sin_matrix.as_slice(), query.device())?
        .reshape((1, seq_len, half_dim))?
        .broadcast_as((batch, num_heads, seq_len, half_dim))?;

    let first_half = query.narrow(3, 0, half_dim)?;
    let second_half = query.narrow(3, half_dim, half_dim)?;

    let rotated_first = first_half
        .broadcast_mul(&cos)?
        .broadcast_add(&second_half.broadcast_mul(&sin)?)?;
    let rotated_second = second_half
        .broadcast_mul(&cos)?
        .broadcast_sub(&first_half.broadcast_mul(&sin)?)?;

    let result = Tensor::cat(&[&rotated_first, &rotated_second], 3)?;
    result.transpose(1, 2)
}
```

- [ ] **Step 2: Update qwen3/rope.rs to re-export from components**

```rust
pub use crate::components::{apply_rope, precompute_rope_cache, RoPE};

pub fn precompute_rope_cache(seq_len: usize, head_dim: usize, theta: f32) -> Vec<(f32, f32)> {
    let mut cache = Vec::with_capacity(seq_len * head_dim / 2);
    for pos in 0..seq_len {
        for i in 0..head_dim / 2 {
            let freq = (pos as f32).powf(-2.0 * (i as f32) / (head_dim as f32)) * theta;
            cache.push((freq.cos(), freq.sin()));
        }
    }
    cache
}
```

- [ ] **Step 3: Run tests to verify RoPE still works**

Run: `cargo test -p vllm-model -- rope`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add crates/model/src/components/positional.rs crates/model/src/qwen3/rope.rs
git commit -m "feat(model): extract RoPE to components/positional.rs"
```

---

## Task 3: Extract mlp.rs (SwiGLU)

**Files:**
- Create: `crates/model/src/components/mlp.rs`
- Modify: `crates/model/src/qwen3/mlp.rs`

- [ ] **Step 1: Create components/mlp.rs**

```rust
use candle_core::{Module, Result, Tensor};
use candle_nn::Linear;

pub fn swiglu_forward(
    x: &Tensor,
    gate_proj: &Linear,
    up_proj: &Linear,
    down_proj: &Linear,
) -> Result<Tensor> {
    let gate = gate_proj.forward(x)?;
    let up = up_proj.forward(x)?;
    let silu = gate.silu()?;
    let activated = silu.broadcast_mul(&up)?;
    down_proj.forward(&activated)
}

pub fn gated_mlp_forward(
    x: &Tensor,
    gate_proj: &Linear,
    up_proj: &Linear,
    down_proj: &Linear,
) -> Result<Tensor> {
    let gate = gate_proj.forward(x)?;
    let up = up_proj.forward(x)?;
    let activated = gate.broadcast_mul(&up)?;
    down_proj.forward(&activated)
}
```

- [ ] **Step 2: Update qwen3/mlp.rs to use components**

```rust
pub use crate::components::swiglu_forward;

impl SwiGLU {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        swiglu_forward(x, &self.gate_proj, &self.up_proj, &self.down_proj)
    }
}
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p vllm-model -- swiglu`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add crates/model/src/components/mlp.rs crates/model/src/qwen3/mlp.rs
git commit -m "feat(model): extract SwiGLU to components/mlp.rs"
```

---

## Task 4: Extract norm.rs

**Files:**
- Create: `crates/model/src/components/norm.rs`

- [ ] **Step 1: Create components/norm.rs**

```rust
use candle_core::{Module, Result, Tensor};
use candle_nn::LayerNorm;

pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let (batch, seq, hidden) = x.dims3()?;
    let x = x.reshape((batch * seq, hidden))?;
    let norm = LayerNorm::new(weight.clone(), Tensor::zeros(hidden, x.dtype(), x.device())?, eps);
    let x = norm.forward(&x)?;
    x.reshape((batch, seq, hidden))
}

pub fn layer_norm(
    x: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    eps: f64,
) -> Result<Tensor> {
    let (batch, seq, hidden) = x.dims3()?;
    let x = x.reshape((batch * seq, hidden))?;
    let norm = LayerNorm::new(weight.clone(), bias.clone(), eps);
    let x = norm.forward(&x)?;
    x.reshape((batch, seq, hidden))
}
```

- [ ] **Step 2: Run build**

Run: `cargo build -p vllm-model`
Expected: Compiles

- [ ] **Step 3: Commit**

```bash
git add crates/model/src/components/norm.rs
git commit -m "feat(model): add components/norm.rs"
```

---

## Task 5: Extract attention.rs

**Files:**
- Create: `crates/model/src/components/attention.rs`
- Modify: `crates/model/src/qwen3/attention.rs`

- [ ] **Step 1: Create components/attention.rs**

```rust
#![allow(clippy::too_many_arguments)]

use crate::kv_cache::PagedKvCache;
use candle_core::{Result, Tensor};
use candle_nn::Linear;

pub struct AttentionConfig {
    pub tile_size: Option<usize>,
    pub use_fused: bool,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            tile_size: None,
            use_fused: true,
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn gqa_forward_with_rope(
    x: &Tensor,
    q_proj: &Linear,
    k_proj: &Linear,
    v_proj: &Linear,
    o_proj: &Linear,
    positions: &[usize],
    rope: &super::positional::RoPE,
    q_norm: Option<&candle_nn::LayerNorm>,
    k_norm: Option<&candle_nn::LayerNorm>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    config: &AttentionConfig,
) -> Result<Tensor> {
    let batch_size = x.dims()[0];
    let seq_len = x.dims()[1];

    let q = q_proj.forward(x)?;
    let k = k_proj.forward(x)?;
    let v = v_proj.forward(x)?;

    let q = q.reshape((batch_size, seq_len, num_heads, head_dim))?;
    let k = k.reshape((batch_size, seq_len, num_kv_heads, head_dim))?;
    let v = v.reshape((batch_size, seq_len, num_kv_heads, head_dim))?;

    let q = apply_q_norm(q, q_norm, batch_size, num_heads, seq_len, head_dim)?;
    let k = apply_k_norm(k, k_norm, batch_size, num_kv_heads, seq_len, head_dim)?;

    let k = expand_kv(&k, num_heads, num_kv_heads)?;
    let v = expand_kv(&v, num_heads, num_kv_heads)?;

    let k = k.transpose(2, 3)?;
    let qk = Tensor::matmul(&q, &k)?;

    let scale = 1.0 / (head_dim as f32).sqrt();
    let qk = qk.mul(&Tensor::new(&[scale], q.device())?.broadcast_as(qk.dims())?)?;
    let attn_weights = candle_nn::ops::softmax(&qk, 3)?;

    let attn_output = Tensor::matmul(&attn_weights, &v)?;

    let attn_output = attn_output.reshape((batch_size, seq_len, num_heads * head_dim))?;
    let o = o_proj.forward(&attn_output)?;
    Ok(o)
}

fn apply_q_norm(
    q: Tensor,
    q_norm: Option<&candle_nn::LayerNorm>,
    batch_size: usize,
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
) -> Result<Tensor> {
    if let Some(norm) = q_norm {
        let q = q.transpose(1, 2)?;
        let q = q.reshape((batch_size * num_heads * seq_len, head_dim))?;
        let q = norm.forward(&q)?;
        let q = q.reshape((batch_size, num_heads, seq_len, head_dim))?;
        q.transpose(1, 2)
    } else {
        Ok(q)
    }
}

fn apply_k_norm(
    k: Tensor,
    k_norm: Option<&candle_nn::LayerNorm>,
    batch_size: usize,
    num_kv_heads: usize,
    seq_len: usize,
    head_dim: usize,
) -> Result<Tensor> {
    if let Some(norm) = k_norm {
        let k = k.transpose(1, 2)?;
        let k = k.reshape((batch_size * num_kv_heads * seq_len, head_dim))?;
        let k = norm.forward(&k)?;
        let k = k.reshape((batch_size, num_kv_heads, seq_len, head_dim))?;
        k.transpose(1, 2)
    } else {
        Ok(k)
    }
}

pub fn expand_kv(kv: &Tensor, num_q_heads: usize, num_kv_heads: usize) -> Result<Tensor> {
    if num_q_heads == num_kv_heads {
        return Ok(kv.clone());
    }

    let repeat_factor = num_q_heads / num_kv_heads;
    let (batch, seq, heads, dim) = kv.dims4()?;

    let kv = kv.reshape((batch, seq, heads, 1, dim))?;
    let expanded = kv.broadcast_as((batch, seq, heads, repeat_factor, dim))?;
    let expanded = expanded.reshape((batch, seq, heads * repeat_factor, dim))?;

    Ok(expanded)
}
```

- [ ] **Step 2: Update qwen3/attention.rs to delegate to components**

Keep the struct definition but simplify forward methods:

```rust
impl GqaAttention {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        super::super::components::gqa_forward_with_rope(
            x,
            &self.q_proj,
            &self.k_proj,
            &self.v_proj,
            &self.o_proj,
            &[],  // positions - not needed for basic forward
            &super::super::components::RoPE::new(self.theta, self.head_dim),
            self.q_norm.as_ref(),
            self.k_norm.as_ref(),
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            &self.config,
        )
    }
    
    // Keep prefill/decode methods as they handle KV cache
}
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p vllm-model -- attention`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add crates/model/src/components/attention.rs crates/model/src/qwen3/attention.rs
git commit -m "feat(model): extract attention to components/attention.rs"
```

---

## Task 6: Update qwen3/block.rs to use components

**Files:**
- Modify: `crates/model/src/qwen3/block.rs`

- [ ] **Step 1: Update TransformerBlock to use components**

```rust
use crate::components::{swiglu_forward, GqaAttention, AttentionConfig};

impl TransformerBlock {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x.clone();
        let x = self.input_layernorm.forward(x)?;
        let x = self.attention.forward(&x)?;
        let x = (x + residual)?;

        let residual = x.clone();
        let x = self.post_attention_layernorm.forward(&x)?;
        x = swiglu_forward(&x, &self.mlp.gate_proj, &self.mlp.up_proj, &self.mlp.down_proj)?;
        x.add(&residual)
    }
}
```

- [ ] **Step 2: Run tests**

Run: `cargo test -p vllm-model -- block`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add crates/model/src/qwen3/block.rs
git commit -m "refactor(model): update qwen3/block.rs to use components"
```

---

## Task 7: Create Llama skeleton

**Files:**
- Create: `crates/model/src/llama/mod.rs`
- Create: `crates/model/src/llama/block.rs`
- Create: `crates/model/src/llama/model.rs`
- Modify: `crates/model/src/lib.rs`

- [ ] **Step 1: Create llama/mod.rs**

```rust
pub mod block;
pub mod model;

pub use block::LlamaBlock;
pub use model::LlamaModel;
```

- [ ] **Step 2: Create llama/block.rs**

```rust
#![allow(dead_code)]

use crate::components::{gqa_forward_with_rope, swiglu_forward, AttentionConfig};
use crate::kv_cache::PagedKvCache;
use candle_core::{Module, Result, Tensor};
use candle_nn::{Embedding, LayerNorm, Linear};

pub struct LlamaBlock {
    input_layernorm: LayerNorm,
    post_attention_layernorm: LayerNorm,
    attention: crate::qwen3::attention::GqaAttention,
    mlp: crate::qwen3::mlp::SwiGLU,
}

impl LlamaBlock {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
        theta: f32,
        rms_norm_eps: f64,
    ) -> Result<Self> {
        todo!("Implement LlamaBlock")
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x.clone();
        let x = self.input_layernorm.forward(x)?;
        let x = self.attention.forward(&x)?;
        let x = (x + residual)?;

        let residual = x.clone();
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = swiglu_forward(&x, &self.mlp.gate_proj, &self.mlp.up_proj, &self.mlp.down_proj)?;
        x.add(&residual)
    }
}
```

- [ ] **Step 3: Create llama/model.rs**

```rust
#![allow(dead_code)]

pub struct LlamaModel;

impl LlamaModel {
    pub fn new() -> Self {
        todo!("Implement LlamaModel")
    }
}
```

- [ ] **Step 4: Update lib.rs**

```rust
pub mod llama;
```

- [ ] **Step 5: Run build**

Run: `cargo build -p vllm-model`
Expected: Compiles (with todo!())

- [ ] **Step 6: Commit**

```bash
git add crates/model/src/llama/ crates/model/src/lib.rs
git commit -m "feat(model): add llama module skeleton"
```

---

## Task 8: Final verification

**Files:**
- All modified files

- [ ] **Step 1: Run full test suite**

Run: `cargo test --workspace`
Expected: All tests pass

- [ ] **Step 2: Run clippy**

Run: `cargo clippy --workspace -- -D warnings`
Expected: No warnings

- [ ] **Step 3: Run format check**

Run: `cargo fmt --all --check`
Expected: No formatting issues

- [ ] **Step 4: Final commit**

```bash
git commit -m "refactor(model): complete component extraction for multi-arch support"
```

---

## Summary

- Task 1: Create components module structure
- Task 2: Extract positional.rs (RoPE)
- Task 3: Extract mlp.rs (SwiGLU)
- Task 4: Extract norm.rs
- Task 5: Extract attention.rs
- Task 6: Update qwen3/block.rs
- Task 7: Create Llama skeleton
- Task 8: Final verification
