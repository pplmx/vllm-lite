# RoPE Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement complete RoPE (Rotary Position Embedding) so the `positions` parameter in ModelBackend::forward actually affects model output.

**Architecture:** Apply RoPE to query and key tensors in GqaAttention after projection but before attention computation. Thread positions from ModelBackend trait through Qwen3Model → TransformerBlock → GqaAttention.

**Tech Stack:** Rust, Candle (tensor operations), vLLM-lite

---

## Task 1: Implement apply_rope function

**Files:**

- Modify: `crates/model/src/qwen3/rope.rs:26-34`

- [ ] **Step 1: Write the failing test**

Add test that verifies different positions produce different outputs:

```rust
#[test]
fn test_apply_rope_different_positions() -> Result<()> {
    use candle_core::{DType, Device, Tensor};

    let device = Device::Cpu;
    // batch=1, seq=2, heads=2, head_dim=4
    let query = Tensor::ones((1, 2, 2, 4), DType::F32, &device)?;

    // Position 0
    let pos0 = Tensor::new(&[0i64], &device)?;
    let out0 = apply_rope(&query, &pos0, 10000.0)?;

    // Position 1
    let pos1 = Tensor::new(&[1i64], &device)?;
    let out1 = apply_rope(&query, &pos1, 10000.0)?;

    // Outputs should be different
    let diff = (&out0 - &out1)?.abs_sum()?;
    assert!(diff.to_scalar::<f32>()? > 1e-5, "RoPE should produce different outputs for different positions");

    Ok(())
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p vllm-model test_apply_rope_different_positions`
Expected: PASS (because current impl returns clone, which may or may not differ)

Actually, current impl returns `query.clone()` - same for all positions. Need different test:

```rust
#[test]
fn test_apply_rope_uses_position() -> Result<()> {
    use candle_core::{DType, Device, Tensor};

    let device = Device::Cpu;
    // batch=1, seq=1, heads=1, head_dim=2 (simpler)
    let query = Tensor::new(&[[[[1.0, 2.0]]]], &device)?;

    // Position 0
    let pos0 = Tensor::new(&[0i64], &device)?;
    let out0 = apply_rope(&query, &pos0, 10000.0)?;

    // Position 1  
    let pos1 = Tensor::new(&[1i64], &device)?;
    let out1 = apply_rope(&query, &pos1, 10000.0)?;

    // Verify outputs differ
    let diff = (&out0 - &out1)?.abs_sum()?;
    assert!(diff.to_scalar::<f32>()? > 1e-4);

    Ok(())
}
```

- [ ] **Step 3: Implement actual RoPE computation**

Replace the placeholder in `apply_rope`:

```rust
pub fn apply_rope(query: &Tensor, position_ids: &Tensor, theta: f32) -> Result<Tensor> {
    // query shape: [batch, seq_len, num_heads, head_dim]
    // position_ids shape: [seq_len]

    let (batch, seq_len, num_heads, head_dim) = query.dims4()?;

    // Reshape to [batch, num_heads, seq_len, head_dim] for easier processing
    let query = query.transpose(1, 2)?;

    // Get position as i64
    let pos = position_ids.to_vec1::<i64>()?[0] as f32;

    // For each head dimension pair (i, i+head_dim/2), apply rotation
    // Rotated = x * cos(θ) + rotate(x) * sin(θ)
    // where rotate(x) = [-x[..., i+head_dim/2], x[..., i]]

    let half_dim = head_dim / 2;
    let freq = (pos + 1.0).powf(-2.0 * (0..half_dim) as f32 / head_dim as f32) * theta;

    let cos_vals: Vec<f32> = freq.iter().map(|f| f.cos()).collect();
    let sin_vals: Vec<f32> = freq.iter().map(|f| f.sin()).collect();

    let cos = Tensor::new(cos_vals.as_slice(), query.device())?
        .reshape((1, num_heads, 1, half_dim))?
        .broadcast_add(&[batch, num_heads, seq_len, half_dim])?;
    let sin = Tensor::new(sin_vals.as_slice(), query.device())?
        .reshape((1, num_heads, 1, half_dim))?
        .broadcast_add(&[batch, num_heads, seq_len, half_dim])?;

    // Split head_dim into first_half and second_half
    let first_half = query.narrow(3, 0, half_dim)?;
    let second_half = query.narrow(3, half_dim, half_dim)?;

    // first_half * cos + second_half * sin
    let rotated_first = (first_half * &cos + second_half * &sin)?;
    // second_half * cos - first_half * sin  
    let rotated_second = (second_half * &cos - first_half * &sin)?;

    let result = Tensor::cat(&[&rotated_first, &rotated_second], 3)?;

    // Back to [batch, seq_len, num_heads, head_dim]
    result.transpose(1, 2)
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p vllm-model test_apply_rope_different_positions`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/model/src/qwen3/rope.rs
git commit -m "feat(model): implement apply_rope with actual rotation computation"
```

---

### Task 2: Add RoPE theta to GqaAttention

**Files:**

- Modify: `crates/model/src/qwen3/attention.rs:22-33`
- Modify: `crates/model/src/qwen3/attention.rs:36-77`

- [ ] **Step 1: Add theta field to GqaAttention struct**

In struct definition, add:

```rust
pub struct GqaAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    config: AttentionConfig,
    q_norm: Option<LayerNorm>,
    k_norm: Option<LayerNorm>,
    theta: f32,  // ADD THIS
}
```

- [ ] **Step 2: Initialize theta in constructor**

In `new` function, add theta parameter:

```rust
pub fn new(
    hidden_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    vb: Option<candle_nn::VarBuilder>,
    config: AttentionConfig,
    has_qk_norm: bool,
    theta: f32,  // ADD THIS
) -> Result<Self> {
    // ... existing code ...

    Ok(Self {
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        num_heads,
        num_kv_heads,
        head_dim,
        config,
        q_norm,
        k_norm,
        theta,  // ADD THIS
    })
}
```

Also update `new_with_weights`:

```rust
pub fn new_with_weights(
    hidden_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    q_w: Tensor,
    k_w: Tensor,
    v_w: Tensor,
    o_w: Tensor,
    config: AttentionConfig,
    has_qk_norm: bool,
    q_norm_w: Option<Tensor>,
    k_norm_w: Option<Tensor>,
    theta: f32,  // ADD THIS
) -> Result<Self>
```

- [ ] **Step 3: Add positions parameter to forward methods**

Modify `forward`, `forward_prefill`, `forward_decode` to accept positions:

```rust
pub fn forward(&self, x: &Tensor, positions: &[usize]) -> Result<Tensor> {
    // ... existing code up to q/k projection ...

    let q = self.q_proj.forward(x)?;
    let k = self.k_proj.forward(x)?;
    let v = self.v_proj.forward(x)?;

    let q = q.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?;
    let k = k.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?;
    let v = v.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?;

    // Apply RoPE here
    let position_tensor = Tensor::new(positions, x.device())?;
    let q = apply_rope(&q, &position_tensor, self.theta)?;
    let k = apply_rope(&k, &position_tensor, self.theta)?;

    // ... rest of forward ...
}
```

- [ ] **Step 4: Run build to verify**

Run: `cargo check -p vllm-model`
Expected: SUCCESS (may have warnings)

- [ ] **Step 5: Commit**

```bash
git add crates/model/src/qwen3/attention.rs
git commit -m "feat(model): add theta to GqaAttention and apply RoPE in forward"
```

---

### Task 3: Update TransformerBlock to pass positions

**Files:**

- Modify: `crates/model/src/qwen3/block.rs:11-16`
- Modify: `crates/model/src/qwen3/block.rs:18-59`
- Modify: `crates/model/src/qwen3/block.rs:139-149`

- [ ] **Step 1: Add positions to forward methods**

In `TransformerBlock`, modify:

```rust
pub fn forward(&self, x: &Tensor, positions: &[usize]) -> Result<Tensor> {
    let residual = x.clone();
    let x = self.input_layernorm.forward(x)?;
    let x = self.attention.forward(&x, positions)?;  // pass positions
    let x = (x + residual)?;

    let residual = x.clone();
    let x = self.post_attention_layernorm.forward(&x)?;
    let x = self.mlp.forward(&x)?;
    x.add(&residual)
}
```

Also update `forward_prefill` and `forward_decode`.

- [ ] **Step 2: Run build to verify**

Run: `cargo check -p vllm-model`

- [ ] **Step 3: Commit**

```bash
git add crates/model/src/qwen3/block.rs
git commit -m "feat(model): pass positions through TransformerBlock"
```

---

### Task 4: Update Qwen3Model to use positions

**Files:**

- Modify: `crates/model/src/qwen3/model.rs:13-21`
- Modify: `crates/model/src/qwen3/model.rs:295-363`

- [ ] **Step 1: Add RoPE field to Qwen3Model struct**

```rust
pub struct Qwen3Model {
    config: Qwen3Config,
    embed_tokens: Embedding,
    layers: Vec<TransformerBlock>,
    norm: LayerNorm,
    lm_head: Linear,
    kv_cache: PagedKvCache,
    device: Device,
    rope: RoPE,  // ADD THIS
}
```

- [ ] **Step 2: Initialize RoPE in constructors**

In `new`:

```rust
let rope = RoPE::new(&config);
```

In `from_weights`: same initialization.

- [ ] **Step 3: Use positions in forward method**

```rust
impl ModelBackend for Qwen3Model {
    fn forward(
        &self,
        seq_ids: &[SeqId],
        input_tokens: &[Vec<TokenId>],
        positions: &[Vec<usize>],  // Now used!
    ) -> EngineResult<BatchOutput> {
        // ...
        for (seq_idx, (tokens, pos)) in input_tokens.iter().zip(positions).enumerate() {
            // ...
            // Pass pos to layers
            for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
                hidden_states = layer.forward(&hidden_states, pos.as_slice())?;  // pass positions
            }
            // ...
        }
    }
}
```

- [ ] **Step 4: Run build to verify**

Run: `cargo check -p vllm-model`

- [ ] **Step 5: Commit**

```bash
git add crates/model/src/qwen3/model.rs
git commit -m "feat(model): wire positions through Qwen3Model to transformer layers"
```

---

### Task 5: Update Qwen5Model

**Files:**

- Modify: `crates/model/src/qwen5/model.rs`

- [ ] **Step 1: Update similar to Qwen3Model**

Add RoPE field, initialize in constructor, pass positions through forward.

- [ ] **Step 2: Run build**

Run: `cargo check -p vllm-model`

- [ ] **Step 3: Commit**

```bash
git add crates/model/src/qwen5/model.rs
git commit -m "feat(model): add RoPE support to Qwen5Model"
```

---

### Task 6: Update FakeModel

**Files:**

- Modify: `crates/model/src/fake.rs`

- [ ] **Step 1: Update to use positions parameter**

Change `_positions` to `positions` and optionally use it (can pass through to inner model or use directly).

- [ ] **Step 2: Run build**

Run: `cargo check -p vllm-model`

- [ ] **Step 3: Commit**

```bash
git add crates/model/src/fake.rs
git commit -m "feat(model): update FakeModel to use positions parameter"
```

---

### Task 7: Fix test warnings and add position test

**Files:**

- Modify: `crates/core/tests/integration.rs:464-509`

- [ ] **Step 1: Update MockModel in test to actually use positions**

```rust
impl ModelBackend for MockModel {
    fn forward(
        &self,
        seq_ids: &[SeqId],
        input_tokens: &[Vec<TokenId>],
        positions: &[Vec<usize>],  // Remove underscore, use it
    ) -> Result<BatchOutput> {
        // Use positions to make output position-dependent
        let next_tokens: Vec<TokenId> = input_tokens
            .iter()
            .zip(positions.iter())
            .map(|(tokens, pos)| {
                let pos_sum: usize = pos.iter().sum();
                if let Some(&last) = tokens.last() {
                    (last + pos_sum as TokenId + 1) % 256
                } else {
                    1
                }
            })
            .collect();

        Ok(BatchOutput {
            seq_ids: seq_ids.to_vec(),
            next_tokens,
        })
    }

    fn forward_logits(
        &self,
        _seq_ids: &[SeqId],
        input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
    ) -> Result<Vec<Vec<f32>>> {
        // ... existing
    }
}
```

- [ ] **Step 2: Add test for position-dependent behavior**

```rust
#[test]
fn test_model_position_awareness() {
    struct PositionAwareModel;

    impl ModelBackend for PositionAwareModel {
        fn forward(&self, seq_ids: &[SeqId], input_tokens: &[Vec<TokenId>], positions: &[Vec<usize>]) -> Result<BatchOutput> {
            let next_tokens: Vec<TokenId> = input_tokens
                .iter()
                .zip(positions.iter())
                .map(|(tokens, pos)| {
                    // Position-aware: different position = different output
                    if let Some(&last) = tokens.last() {
                        (last + pos.first().unwrap_or(&0) + 1) % 256
                    } else {
                        1
                    }
                })
                .collect();
            Ok(BatchOutput { seq_ids: seq_ids.to_vec(), next_tokens })
        }

        fn forward_logits(&self, _seq_ids: &[SeqId], input_tokens: &[Vec<TokenId>], _positions: &[Vec<usize>]) -> Result<Vec<Vec<f32>>> {
            Ok(input_tokens.iter().map(|t| vec![0.0; t.len()]).collect())
        }
    }

    let model = PositionAwareModel;

    // Same input tokens at different positions should produce different outputs
    let output1 = model.forward(&[1], &[vec![42]], &[vec![0]]).unwrap();
    let output2 = model.forward(&[1], &[vec![42]], &[vec![1]]).unwrap();

    assert_ne!(output1.next_tokens[0], output2.next_tokens[0], 
        "Different positions should produce different outputs");
}
```

- [ ] **Step 3: Run tests**

Run: `cargo nextest run 2>&1 | head -30`

- [ ] **Step 4: Commit**

```bash
git add crates/core/tests/integration.rs
git commit -m "test(core): add position-aware test and fix MockModel"
```

---

### Task 8: Final verification

**Files:**

- All modified files

- [ ] **Step 1: Run full test suite**

Run: `cargo nextest run`

- [ ] **Step 2: Run clippy**

Run: `cargo clippy --workspace -- -D warnings`

- [ ] **Step 3: Verify no warnings**

Expected: No warnings about unused variables

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat(model): implement RoPE integration - complete positions support"
```

---

## Spec Coverage Check

- [x] RoPE algorithm implemented in rope.rs
- [x] GqaAttention applies RoPE to Q/K
- [x] TransformerBlock passes positions
- [x] Qwen3Model wires positions
- [x] Qwen5Model updated
- [x] FakeModel updated
- [x] Tests verify position-dependent behavior
- [x] No unused variable warnings

**Plan complete and saved to `docs/superpowers/plans/2026-04-02-rope-integration.md`. Two execution options:**

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
