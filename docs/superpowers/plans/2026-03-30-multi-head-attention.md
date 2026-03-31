# Multi-Head Attention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development

**Goal:** Fix GQA/KV expansion bug and add support for MHA, GQA, MQA, and MLA attention mechanisms.

**Architecture:** Fix reshape to preserve seq_len, add tile_heads for KV expansion, add MLA config fields.

**Tech Stack:** Rust, Candle

**Spec:** `docs/superpowers/specs/2026-03-30-multi-head-attention.md`

---

## Current State

- `crates/model/src/qwen3/attention.rs` has broken reshape (drops seq_len)
- No KV head expansion for GQA/MQA
- Config missing MLA fields

---

### Task MHA-1: Fix Reshape to Preserve Sequence Length

**Files:**

- Modify: `crates/model/src/qwen3/attention.rs:68-88`

- [ ] **Step 1: Fix forward method reshape**

Replace current forward method to properly handle seq_len:

```rust
pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
    // x shape: [batch, seq_len, hidden_size]
    let batch_size = x.dims()[0];
    let seq_len = x.dims()[1];

    let q = self.q_proj.forward(x)?;
    let k = self.k_proj.forward(x)?;
    let v = self.v_proj.forward(x)?;

    // Reshape to: [batch, seq_len, num_heads, head_dim]
    let q = q.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?;
    let k = k.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?;
    let v = v.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?;

    // Expand KV heads to match Q heads for GQA/MQA
    let k = self.expand_kv(&k, self.num_heads, self.num_kv_heads)?;
    let v = self.expand_kv(&v, self.num_heads, self.num_kv_heads)?;

    // Compute attention: Q @ K^T
    // Transpose K to: [batch, seq_len, head_dim, num_heads]
    let k = k.transpose(2, 3)?;
    let qk = Tensor::matmul(&q, &k)?;  // [batch, seq_len, num_heads, num_heads]

    let scale = 1.0 / (self.head_dim as f32).sqrt();
    let qk = qk.mul(&Tensor::new(&[scale], q.device())?)?;
    let attn_weights = candle_nn::ops::softmax(&qk, 3)?;  // softmax on head dim

    // Attention output: attn_weights @ V
    let attn_output = Tensor::matmul(&attn_weights, &v)?;  // [batch, seq_len, num_heads, head_dim]

    // Reshape back: [batch, seq_len, num_heads * head_dim]
    let attn_output = attn_output.reshape((batch_size, seq_len, self.num_heads * self.head_dim))?;

    let o = self.o_proj.forward(&attn_output)?;
    Ok(o)
}

fn expand_kv(&self, kv: &Tensor, num_q_heads: usize, num_kv_heads: usize) -> Result<Tensor> {
    if num_q_heads == num_kv_heads {
        return Ok(kv.clone());  // MHA - no expansion needed
    }

    // Expand KV heads to match Q heads
    let repeat_factor = num_q_heads / num_kv_heads;
    let (batch, seq, heads, dim) = kv.dims4()?;

    // Reshape: [batch, seq, num_kv_heads, head_dim] -> [batch, seq, num_kv_heads, 1, head_dim]
    let kv = kv.unsqueeze(3)?;
    // Repeat: [batch, seq, num_kv_heads, repeat, head_dim]
    let kv = kv.repeat(3, repeat_factor)?;
    // Reshape: [batch, seq, num_q_heads, head_dim]
    let kv = kv.reshape((batch, seq, num_q_heads, dim))?;

    Ok(kv)
}
```

- [ ] **Step 2: Test compilation**

```bash
cargo build -p vllm-model
```

- [ ] **Step 3: Test with Qwen2.5-0.5B**

```bash
pkill -f vllm-server 2>/dev/null
MODEL_PATH=/models/Qwen2.5-0.5B-Instruct cargo run -p vllm-server 2>&1 | head -30
```

- [ ] **Step 4: Commit**

```bash
git add crates/model/src/qwen3/attention.rs
git commit -m "fix(model): fix seq_len in attention reshape, add KV expansion for GQA/MQA"
```

---

### Task MHA-2: Add MLA Config Fields

**Files:**

- Modify: `crates/model/src/config.rs`

- [ ] **Step 1: Add MLA fields to Qwen3Config**

Add after existing fields in `crates/model/src/config.rs`:

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
```

- [ ] **Step 2: Add attention_type method**

Add after existing methods:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionType {
    MHA,
    MQA,
    GQA,
    MLA,
}

impl Qwen3Config {
    // ... existing methods ...

    pub fn attention_type(&self) -> AttentionType {
        if self.q_len.is_some() || self.kv_len.is_some() {
            AttentionType::MLA
        } else if self.num_key_value_heads().unwrap_or(32) == 1 {
            AttentionType::MQA
        } else if self.num_attention_heads().unwrap_or(32) == self.num_key_value_heads().unwrap_or(32) {
            AttentionType::MHA
        } else {
            AttentionType::GQA
        }
    }
}
```

Note: Need to update num_attention_heads and num_key_value_heads to return Option<usize> for comparison.

- [ ] **Step 3: Test compilation**

```bash
cargo build -p vllm-model
```

- [ ] **Step 4: Commit**

```bash
git add crates/model/src/config.rs
git commit -m "feat(model): add MLA config fields and attention_type detection"
```

---

### Task MHA-3: Add MLA Attention Support (Phase 3)

**Files:**

- Modify: `crates/model/src/qwen3/attention.rs`

This is more complex and can be done after Phase 1-2 are working.

- [ ] **Step 1: Detect attention type and route**

- [ ] **Step 2: Implement MLA forward path**

---

## Verification

```bash
# Build
cargo build --workspace

# Test with Qwen2.5-0.5B (GQA: 14 Q heads, 2 KV heads)
MODEL_PATH=/models/Qwen2.5-0.5B-Instruct cargo run -p vllm-server

# Test inference
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 10}'
```

Expected: No shape mismatch error, generates coherent text.

## Spec Coverage

| Spec Section             | Covered By |
| ------------------------ | ---------- |
| Fix seq_len in reshape   | Task MHA-1 |
| KV expansion for GQA/MQA | Task MHA-1 |
| MLA config fields        | Task MHA-2 |
| Attention type detection | Task MHA-2 |
| MLA forward (future)     | Task MHA-3 |
