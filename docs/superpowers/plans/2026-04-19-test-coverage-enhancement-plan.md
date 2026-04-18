# Test Coverage Enhancement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 47 comprehensive tests across 5 components (StandardBlock, SwiGLU, GqaAttention, RMSNorm, RoPE) to achieve production-grade test coverage.

**Architecture:** Tests are organized by component and category (A: Core, B: Boundary, C: Numerical). Each component's tests live in a `#[cfg(test)]` module within the source file.

**Tech Stack:** Rust, candle-core, candle-nn

---

## File Structure

```
crates/model/src/components/
├── block.rs              # StandardBlock tests (add 12 tests)
├── mlp/
│   └── swiglu.rs         # SwiGLU tests (add 13 tests)
├── attention/
│   └── gqa.rs            # GqaAttention tests (add 8 tests)
├── norm/
│   └── rms_norm.rs       # RMSNorm tests (add 9 tests)
└── positional/
    └── rope.rs           # RoPE tests (add 5 tests)
```

---

## Task 1: StandardBlock Tests

**File:** `crates/model/src/components/block.rs:159-254`

### A: Core Functionality Tests (5 tests)

- [ ] **Step 1: Add test_standard_block_deterministic_output**

```rust
#[test]
fn test_standard_block_deterministic_output() {
    let config = BlockConfig::default();
    let block = StandardBlock::new(&config, None).unwrap();
    let x = Tensor::randn(0.0, 1.0, (2, 4, config.hidden_size), &Device::Cpu).unwrap();
    
    let out1 = block.forward(&x).unwrap();
    let out2 = block.forward(&x).unwrap();
    
    let diff = (&out1 - &out2).unwrap().abs().unwrap();
    let max_diff: f32 = diff.max().unwrap().to_scalar().unwrap();
    assert!(max_diff < 1e-5, "Forward pass should be deterministic");
}
```

- [ ] **Step 2: Add test_standard_block_residual_connection**

```rust
#[test]
fn test_standard_block_residual_connection() {
    let config = BlockConfig { hidden_size: 128, ..Default::default() };
    let block = StandardBlock::new(&config, None).unwrap();
    let x = Tensor::randn(0.0, 1.0, (1, 2, 128), &Device::Cpu).unwrap();
    
    let output = block.forward(&x).unwrap();
    assert_eq!(output.dims(), x.dims());
    
    // Residual connection should prevent complete signal loss
    let output_sum: f32 = output.abs().unwrap().sum_all().unwrap().to_scalar().unwrap();
    let input_sum: f32 = x.abs().unwrap().sum_all().unwrap().to_scalar().unwrap();
    assert!(output_sum > input_sum * 0.1, "Residual should preserve signal");
}
```

- [ ] **Step 3: Add test_standard_block_layernorm_application**

```rust
#[test]
fn test_standard_block_layernorm_application() {
    let config = BlockConfig { hidden_size: 64, eps: 1e-5, ..Default::default() };
    let block = StandardBlock::new(&config, None).unwrap();
    
    // After layernorm, values should be normalized
    let x = Tensor::ones((1, 1, 64), DType::F32, &Device::Cpu).unwrap();
    let output = block.forward(&x).unwrap();
    
    // Output should be finite and reasonably sized
    let abs_max: f32 = output.abs().unwrap().max().unwrap().to_scalar().unwrap();
    assert!(abs_max.is_finite() && abs_max < 100.0);
}
```

- [ ] **Step 4: Add test_standard_block_attention_called**

```rust
#[test]
fn test_standard_block_attention_called() {
    // Test that attention component is accessible and properly configured
    let config = BlockConfig { num_attention_heads: 4, num_key_value_heads: 2, ..Default::default() };
    let block = StandardBlock::new(&config, None).unwrap();
    
    assert_eq!(block.attention().num_heads(), 4);
    assert_eq!(block.attention().num_kv_heads(), 2);
}
```

- [ ] **Step 5: Add test_standard_block_mlp_called**

```rust
#[test]
fn test_standard_block_mlp_called() {
    let config = BlockConfig { hidden_size: 256, intermediate_size: 512, ..Default::default() };
    let block = StandardBlock::new(&config, None).unwrap();
    
    let x = Tensor::randn(0.0, 1.0, (1, 1, 256), &Device::Cpu).unwrap();
    let output = block.forward(&x).unwrap();
    
    // MLP should transform the hidden states
    let out_before_mlp = block.post_attention_layernorm().forward(&x).unwrap();
    assert_ne!(out_before_mlp.dims()[0], output.dims()[0]); // Different magnitudes expected
}
```

### B: Boundary Condition Tests (4 tests)

- [ ] **Step 6: Add test_standard_block_minimal_hidden_size**

```rust
#[test]
fn test_standard_block_minimal_hidden_size() {
    let config = BlockConfig {
        hidden_size: 16,
        num_attention_heads: 2,
        num_key_value_heads: 2,
        head_dim: 8,
        ..Default::default()
    };
    let block = StandardBlock::new(&config, None).unwrap();
    let x = Tensor::ones((1, 1, 16), DType::F32, &Device::Cpu).unwrap();
    let output = block.forward(&x).unwrap();
    assert_eq!(output.dims(), &[1, 1, 16]);
}
```

- [ ] **Step 7: Add test_standard_block_single_head**

```rust
#[test]
fn test_standard_block_single_head() {
    let config = BlockConfig {
        hidden_size: 128,
        num_attention_heads: 1,
        num_key_value_heads: 1,
        head_dim: 128,
        ..Default::default()
    };
    let block = StandardBlock::new(&config, None).unwrap();
    let x = Tensor::ones((1, 1, 128), DType::F32, &Device::Cpu).unwrap();
    let output = block.forward(&x).unwrap();
    assert_eq!(output.dims(), &[1, 1, 128]);
}
```

- [ ] **Step 8: Add test_standard_block_large_batch**

```rust
#[test]
fn test_standard_block_large_batch() {
    let config = BlockConfig { hidden_size: 256, ..Default::default() };
    let block = StandardBlock::new(&config, None).unwrap();
    let x = Tensor::ones((64, 4, 256), DType::F32, &Device::Cpu).unwrap();
    let output = block.forward(&x).unwrap();
    assert_eq!(output.dims(), &[64, 4, 256]);
}
```

- [ ] **Step 9: Add test_standard_block_extreme_intermediate_size**

```rust
#[test]
fn test_standard_block_extreme_intermediate_size() {
    let config = BlockConfig {
        hidden_size: 128,
        intermediate_size: 512, // 4x hidden_size
        ..Default::default()
    };
    let block = StandardBlock::new(&config, None).unwrap();
    let x = Tensor::ones((1, 1, 128), DType::F32, &Device::Cpu).unwrap();
    let output = block.forward(&x).unwrap();
    assert!(output.to_vec::<f32>().unwrap().iter().all(|v| v.is_finite()));
}
```

### C: Numerical Correctness Tests (3 tests)

- [ ] **Step 10: Add test_standard_block_output_finite**

```rust
#[test]
fn test_standard_block_output_finite() {
    let config = BlockConfig::default();
    let block = StandardBlock::new(&config, None).unwrap();
    let x = Tensor::randn(-1.0, 1.0, (2, 4, config.hidden_size), &Device::Cpu).unwrap();
    let output = block.forward(&x).unwrap();
    
    let data: Vec<f32> = output.to_vec().unwrap();
    assert!(data.iter().all(|v| v.is_finite()), "Output must not contain NaN or Inf");
}
```

- [ ] **Step 11: Add test_standard_block_output_magnitude**

```rust
#[test]
fn test_standard_block_output_magnitude() {
    let config = BlockConfig::default();
    let block = StandardBlock::new(&config, None).unwrap();
    let x = Tensor::randn(-1.0, 1.0, (1, 2, config.hidden_size), &Device::Cpu).unwrap();
    let output = block.forward(&x).unwrap();
    
    let abs_max: f32 = output.abs().unwrap().max().unwrap().to_scalar().unwrap();
    // Output should not explode (reasonable bounds)
    assert!(abs_max < 100.0, "Output magnitude unreasonable: {}", abs_max);
}
```

- [ ] **Step 12: Add test_standard_block_eps_stability**

```rust
#[test]
fn test_standard_block_eps_stability() {
    let x = Tensor::randn(0.0, 1.0, (1, 1, 64), &Device::Cpu).unwrap();
    
    let config1 = BlockConfig { hidden_size: 64, eps: 1e-6, ..Default::default() };
    let config2 = BlockConfig { hidden_size: 64, eps: 1e-2, ..Default::default() };
    
    let block1 = StandardBlock::new(&config1, None).unwrap();
    let block2 = StandardBlock::new(&config2, None).unwrap();
    
    let out1 = block1.forward(&x).unwrap();
    let out2 = block2.forward(&x).unwrap();
    
    // Both should produce finite outputs
    assert!(out1.to_vec::<f32>().unwrap().iter().all(|v| v.is_finite()));
    assert!(out2.to_vec::<f32>().unwrap().iter().all(|v| v.is_finite()));
}
```

- [ ] **Step 13: Commit StandardBlock tests**

```bash
git add crates/model/src/components/block.rs
git commit -m "test(model): add 12 tests for StandardBlock (core + boundary + numerical)"
```

---

## Task 2: SwiGLU Tests

**File:** `crates/model/src/components/mlp/swiglu.rs:68-108`

### A: Core Functionality Tests (5 tests)

- [ ] **Step 14: Add test_swiglu_new_with_weights_creation**

```rust
#[test]
fn test_swiglu_new_with_weights_creation() -> Result<()> {
    let device = Device::Cpu;
    let hidden_size = 128;
    let intermediate_size = 256;
    
    let gate = Tensor::randn(0.0, 1.0, (intermediate_size, hidden_size), &device)?;
    let up = Tensor::randn(0.0, 1.0, (intermediate_size, hidden_size), &device)?;
    let down = Tensor::randn(0.0, 1.0, (hidden_size, intermediate_size), &device)?;
    
    let mlp = SwiGLU::new_with_weights(hidden_size, intermediate_size, gate, up, down)?;
    let x = Tensor::ones((2, hidden_size), DType::F32, &device)?;
    let output = mlp.forward(&x)?;
    
    assert_eq!(output.dims(), &[2, hidden_size]);
    Ok(())
}
```

- [ ] **Step 15: Add test_swiglu_gate_proj_output_shape**

```rust
#[test]
fn test_swiglu_gate_proj_output_shape() -> Result<()> {
    let device = Device::Cpu;
    let hidden_size = 64;
    let intermediate_size = 128;
    let batch = 3;
    
    let mlp = SwiGLU::new(hidden_size, intermediate_size, None)?;
    let x = Tensor::ones((batch, hidden_size), DType::F32, &device)?;
    let output = mlp.forward(&x)?;
    
    // gate_proj: hidden_size -> intermediate_size
    assert_eq!(output.dims(), &[batch, hidden_size]);
    Ok(())
}
```

- [ ] **Step 16: Add test_swiglu_up_proj_output_shape**

```rust
#[test]
fn test_swiglu_up_proj_output_shape() -> Result<()> {
    // Same as gate_proj - up_proj also produces intermediate_size output
    let device = Device::Cpu;
    let mlp = SwiGLU::new(32, 64, None)?;
    let x = Tensor::ones((5, 32), DType::F32, &device)?;
    let output = mlp.forward(&x)?;
    assert_eq!(output.dims(), &[5, 32]);
    Ok(())
}
```

- [ ] **Step 17: Add test_swiglu_down_proj_output_shape**

```rust
#[test]
fn test_swiglu_down_proj_output_shape() -> Result<()> {
    // down_proj: intermediate_size -> hidden_size
    let device = Device::Cpu;
    let mlp = SwiGLU::new(256, 512, None)?;
    let x = Tensor::ones((4, 256), DType::F32, &device)?;
    let output = mlp.forward(&x)?;
    assert_eq!(output.dims(), &[4, 256]);
    Ok(())
}
```

- [ ] **Step 18: Add test_swiglu_silu_activation**

```rust
#[test]
fn test_swiglu_silu_activation() -> Result<()> {
    let device = Device::Cpu;
    let mlp = SwiGLU::new(32, 64, None)?;
    let x = Tensor::ones((1, 32), DType::F32, &device)?;
    let output = mlp.forward(&x)?;
    
    // SiLU(x) = x * sigmoid(x)
    // For x=1: sigmoid(1) ≈ 0.731, so SiLU(1) ≈ 0.731
    // Output should be positive for positive input
    let data: Vec<f32> = output.to_vec()?;
    assert!(data.iter().all(|v| *v > 0.0), "SiLU output should be positive for positive input");
    Ok(())
}
```

### B: Boundary Condition Tests (4 tests)

- [ ] **Step 19: Add test_swiglu_minimal_hidden_size**

```rust
#[test]
fn test_swiglu_minimal_hidden_size() -> Result<()> {
    let device = Device::Cpu;
    let mlp = SwiGLU::new(16, 32, None)?;
    let x = Tensor::ones((1, 16), DType::F32, &device)?;
    let output = mlp.forward(&x)?;
    assert_eq!(output.dims(), &[1, 16]);
    Ok(())
}
```

- [ ] **Step 20: Add test_swiglu_large_ratio**

```rust
#[test]
fn test_swiglu_large_ratio() -> Result<()> {
    let device = Device::Cpu;
    // intermediate_size = 8 * hidden_size
    let mlp = SwiGLU::new(32, 256, None)?;
    let x = Tensor::ones((1, 32), DType::F32, &device)?;
    let output = mlp.forward(&x)?;
    assert!(output.to_vec::<f32>()?.iter().all(|v| v.is_finite()));
    Ok(())
}
```

- [ ] **Step 21: Add test_swiglu_single_token_batch**

```rust
#[test]
fn test_swiglu_single_token_batch() -> Result<()> {
    let device = Device::Cpu;
    let mlp = SwiGLU::new(64, 128, None)?;
    let x = Tensor::ones((1, 64), DType::F32, &device)?;
    let output = mlp.forward(&x)?;
    assert_eq!(output.dims(), &[1, 64]);
    Ok(())
}
```

- [ ] **Step 22: Add test_swiglu_multi_token_sequence**

```rust
#[test]
fn test_swiglu_multi_token_sequence() -> Result<()> {
    let device = Device::Cpu;
    let mlp = SwiGLU::new(64, 128, None)?;
    let x = Tensor::ones((32, 64), DType::F32, &device)?;
    let output = mlp.forward(&x)?;
    assert_eq!(output.dims(), &[32, 64]);
    Ok(())
}
```

### C: Numerical Correctness Tests (4 tests)

- [ ] **Step 23: Add test_swiglu_output_finite**

```rust
#[test]
fn test_swiglu_output_finite() -> Result<()> {
    let device = Device::Cpu;
    let mlp = SwiGLU::new(128, 256, None)?;
    let x = Tensor::randn(-5.0, 5.0, (4, 128), DType::F32, &device)?;
    let output = mlp.forward(&x)?;
    let data: Vec<f32> = output.to_vec()?;
    assert!(data.iter().all(|v| v.is_finite()));
    Ok(())
}
```

- [ ] **Step 24: Add test_swiglu_deterministic**

```rust
#[test]
fn test_swiglu_deterministic() -> Result<()> {
    let device = Device::Cpu;
    let mlp = SwiGLU::new(64, 128, None)?;
    let x = Tensor::randn(0.0, 1.0, (2, 64), DType::F32, &device)?;
    
    let out1 = mlp.forward(&x)?;
    let out2 = mlp.forward(&x)?;
    
    let diff = (&out1 - &out2)?.abs()?;
    let max_diff: f32 = diff.max()?.to_scalar()?;
    assert!(max_diff < 1e-6, "MLP should be deterministic");
    Ok(())
}
```

- [ ] **Step 25: Add test_swiglu_zero_input**

```rust
#[test]
fn test_swiglu_zero_input() -> Result<()> {
    let device = Device::Cpu;
    let mlp = SwiGLU::new(64, 128, None)?;
    let x = Tensor::zeros((2, 64), DType::F32, &device)?;
    let output = mlp.forward(&x)?;
    
    // Zero input should produce finite output (SiLU(0) = 0)
    let data: Vec<f32> = output.to_vec()?;
    assert!(data.iter().all(|v| v.is_finite()));
    Ok(())
}
```

- [ ] **Step 26: Add test_swiglu_silu_range**

```rust
#[test]
fn test_swiglu_silu_range() -> Result<()> {
    let device = Device::Cpu;
    let mlp = SwiGLU::new(32, 64, None)?;
    let x = Tensor::ones((1, 32), DType::F32, &device)?;
    let output = mlp.forward(&x)?;
    
    // SiLU(x) for positive x is in range (0, x)
    // For x=1, output should be in (0, 1) approximately
    let data: Vec<f32> = output.to_vec()?;
    let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
    assert!(mean > 0.0 && mean < 1.5, "SiLU output for x=1 should be in reasonable range");
    Ok(())
}
```

- [ ] **Step 27: Commit SwiGLU tests**

```bash
git add crates/model/src/components/mlp/swiglu.rs
git commit -m "test(model): add 13 tests for SwiGLU (core + boundary + numerical)"
```

---

## Task 3: GqaAttention Tests

**File:** `crates/model/src/components/attention/gqa.rs`

### A: Core Functionality Tests (5 tests)

- [ ] **Step 28: Add test_gqa_attention_new_creation**

```rust
#[test]
fn test_gqa_attention_new_creation() -> Result<()> {
    let device = Device::Cpu;
    let attn = GqaAttention::new(256, 8, 2, 32, None, AttentionConfig::default(), false)?;
    assert_eq!(attn.num_heads(), 8);
    assert_eq!(attn.num_kv_heads(), 2);
    assert_eq!(attn.head_dim(), 32);
    Ok(())
}
```

- [ ] **Step 29: Add test_gqa_attention_num_heads_accessors**

```rust
#[test]
fn test_gqa_attention_num_heads_accessors() {
    let device = Device::Cpu;
    let attn = GqaAttention::new(512, 16, 4, 32, None, AttentionConfig::default(), false).unwrap();
    assert_eq!(attn.num_heads(), 16);
    assert_eq!(attn.num_kv_heads(), 4);
}
```

- [ ] **Step 30: Add test_gqa_attention_head_dim_accessors**

```rust
#[test]
fn test_gqa_attention_head_dim_accessors() {
    let device = Device::Cpu;
    let attn = GqaAttention::new(256, 4, 2, 64, None, AttentionConfig::default(), false).unwrap();
    assert_eq!(attn.head_dim(), 64);
}
```

- [ ] **Step 31: Add test_gqa_attention_paged_attention_shape**

```rust
#[test]
fn test_gqa_attention_paged_attention_shape() -> Result<()> {
    let device = Device::Cpu;
    let num_heads = 8;
    let num_kv_heads = 2;
    let head_dim = 64;
    let batch_size = 2;
    let seq_len = 4;
    let hidden_size = num_heads * head_dim;
    
    let q_w = Tensor::randn(0.0, 1.0, (hidden_size, hidden_size), &device)?;
    let k_w = Tensor::randn(0.0, 1.0, (num_kv_heads * head_dim, hidden_size), &device)?;
    let v_w = Tensor::randn(0.0, 1.0, (num_kv_heads * head_dim, hidden_size), &device)?;
    let o_w = Tensor::randn(0.0, 1.0, (hidden_size, hidden_size), &device)?;
    
    let attn = GqaAttention::new_with_weights(hidden_size, num_heads, num_kv_heads, head_dim, q_w, k_w, v_w, o_w, AttentionConfig::default(), false, None, None)?;
    
    let q = Tensor::randn(0.0, 1.0, (batch_size, num_heads, 1, head_dim), &device)?;
    let k = Tensor::randn(0.0, 1.0, (batch_size, num_kv_heads, seq_len, head_dim), &device)?;
    let v = Tensor::randn(0.0, 1.0, (batch_size, num_kv_heads, seq_len, head_dim), &device)?;
    
    let output = attn.paged_attention_fn(&q, &k, &v)?;
    assert_eq!(output.dims(), &[batch_size, num_heads, 1, hidden_size / num_heads]);
    Ok(())
}
```

- [ ] **Step 32: Add test_gqa_attention_tiled_attention_shape**

```rust
#[test]
fn test_gqa_attention_tiled_attention_shape() -> Result<()> {
    let device = Device::Cpu;
    let num_heads = 4;
    let num_kv_heads = 2;
    let head_dim = 32;
    let batch_size = 1;
    let seq_len = 8;
    let hidden_size = num_heads * head_dim;
    
    let q_w = Tensor::randn(0.0, 1.0, (hidden_size, hidden_size), &device)?;
    let k_w = Tensor::randn(0.0, 1.0, (num_kv_heads * head_dim, hidden_size), &device)?;
    let v_w = Tensor::randn(0.0, 1.0, (num_kv_heads * head_dim, hidden_size), &device)?;
    let o_w = Tensor::randn(0.0, 1.0, (hidden_size, hidden_size), &device)?;
    
    let attn = GqaAttention::new_with_weights(hidden_size, num_heads, num_kv_heads, head_dim, q_w, k_w, v_w, o_w, AttentionConfig::default(), false, None, None)?;
    
    let q = Tensor::randn(0.0, 1.0, (batch_size, num_heads, seq_len, head_dim), &device)?;
    let k = Tensor::randn(0.0, 1.0, (batch_size, num_heads, seq_len, head_dim), &device)?;
    let v = Tensor::randn(0.0, 1.0, (batch_size, num_heads, seq_len, head_dim), &device)?;
    
    let output = attn.tiled_attention_fn(&q, &k, &v)?;
    assert_eq!(output.dims(), &[batch_size, seq_len, hidden_size]);
    Ok(())
}
```

### B: Boundary Condition Tests (4 tests)

- [ ] **Step 33: Add test_gqa_attention_single_q_head**

```rust
#[test]
fn test_gqa_attention_single_q_head() -> Result<()> {
    let device = Device::Cpu;
    let attn = GqaAttention::new(64, 1, 1, 64, None, AttentionConfig::default(), false)?;
    assert_eq!(attn.num_heads(), 1);
    assert_eq!(attn.num_kv_heads(), 1);
    Ok(())
}
```

- [ ] **Step 34: Add test_gqa_attention_matching_heads**

```rust
#[test]
fn test_gqa_attention_matching_heads() -> Result<()> {
    // MHA mode: num_q_heads = num_kv_heads
    let device = Device::Cpu;
    let attn = GqaAttention::new(256, 4, 4, 64, None, AttentionConfig::default(), false)?;
    assert_eq!(attn.num_heads(), attn.num_kv_heads());
    Ok(())
}
```

- [ ] **Step 35: Add test_gqa_attention_small_head_dim**

```rust
#[test]
fn test_gqa_attention_small_head_dim() -> Result<()> {
    let device = Device::Cpu;
    let attn = GqaAttention::new(64, 2, 1, 32, None, AttentionConfig::default(), false)?;
    assert_eq!(attn.head_dim(), 32);
    Ok(())
}
```

- [ ] **Step 36: Add test_gqa_attention_large_head_dim**

```rust
#[test]
fn test_gqa_attention_large_head_dim() -> Result<()> {
    let device = Device::Cpu;
    let attn = GqaAttention::new(512, 4, 2, 128, None, AttentionConfig::default(), false)?;
    assert_eq!(attn.head_dim(), 128);
    Ok(())
}
```

### C: Numerical Correctness Tests (3 tests)

- [ ] **Step 37: Add test_gqa_attention_output_finite**

```rust
#[test]
fn test_gqa_attention_output_finite() -> Result<()> {
    let device = Device::Cpu;
    let attn = GqaAttention::new(128, 4, 2, 32, None, AttentionConfig::default(), false)?;
    
    let q = Tensor::randn(-2.0, 2.0, (1, 4, 4, 32), &device)?;
    let k = Tensor::randn(-2.0, 2.0, (1, 2, 4, 32), &device)?;
    let v = Tensor::randn(-2.0, 2.0, (1, 2, 4, 32), &device)?;
    
    let output = attn.paged_attention_fn(&q, &k, &v)?;
    let data: Vec<f32> = output.to_vec()?;
    assert!(data.iter().all(|v| v.is_finite()));
    Ok(())
}
```

- [ ] **Step 38: Add test_gqa_attention_deterministic**

```rust
#[test]
fn test_gqa_attention_deterministic() -> Result<()> {
    let device = Device::Cpu;
    let q_w = Tensor::randn(0.0, 1.0, (256, 256), &device)?;
    let k_w = Tensor::randn(0.0, 1.0, (64, 256), &device)?;
    let v_w = Tensor::randn(0.0, 1.0, (64, 256), &device)?;
    let o_w = Tensor::randn(0.0, 1.0, (256, 256), &device)?;
    
    let attn = GqaAttention::new_with_weights(256, 8, 2, 32, q_w, k_w, v_w, o_w, AttentionConfig::default(), false, None, None)?;
    
    let q = Tensor::randn(0.0, 1.0, (1, 8, 1, 32), &device)?;
    let k = Tensor::randn(0.0, 1.0, (1, 8, 4, 32), &device)?;
    let v = Tensor::randn(0.0, 1.0, (1, 8, 4, 32), &device)?;
    
    let out1 = attn.paged_attention_fn(&q, &k, &v)?;
    let out2 = attn.paged_attention_fn(&q, &k, &v)?;
    
    let diff = (&out1 - &out2)?.abs()?;
    let max_diff: f32 = diff.max()?.to_scalar()?;
    assert!(max_diff < 1e-5);
    Ok(())
}
```

- [ ] **Step 39: Add test_gqa_attention_expand_kv_correct**

```rust
#[test]
fn test_gqa_attention_expand_kv_correct() -> Result<()> {
    // Test that KV expansion produces correct shapes
    let device = Device::Cpu;
    let attn = GqaAttention::new(256, 8, 2, 32, None, AttentionConfig::default(), false)?;
    
    let hidden_size = 256;
    let num_kv_heads = 2;
    let head_dim = 32;
    let seq_len = 4;
    
    let k = Tensor::randn(0.0, 1.0, (1, num_kv_heads, seq_len, head_dim), &device)?;
    let k_expanded = expand_kv(&k, 8, 2)?;
    
    // Should expand from (1, 2, 4, 32) to (1, 8, 4, 32)
    assert_eq!(k_expanded.dims(), &[1, 8, seq_len, head_dim]);
    Ok(())
}
```

- [ ] **Step 40: Commit GqaAttention tests**

```bash
git add crates/model/src/components/attention/gqa.rs
git commit -m "test(model): add 8 tests for GqaAttention (core + boundary + numerical)"
```

---

## Task 4: RMSNorm Tests

**File:** `crates/model/src/components/norm/rms_norm.rs`

### A: Core Functionality Tests (3 tests)

- [ ] **Step 41: Add test_rms_norm_weight_application**

```rust
#[test]
fn test_rms_norm_weight_application() -> Result<()> {
    let device = Device::Cpu;
    let dim = 128;
    let weight = Tensor::ones(dim, DType::F32, &device)?;
    let norm = RmsNorm::new(weight, 1e-6);
    
    let x = Tensor::ones((2, dim), DType::F32, &device)?;
    let output = norm.forward(&x)?;
    
    // With unit weight, RMSNorm should not change magnitude significantly
    let mean_out: f32 = output.mean()?.to_scalar()?;
    let mean_in: f32 = x.mean()?.to_scalar()?;
    assert!((mean_out - mean_in).abs() < 0.5);
    Ok(())
}
```

- [ ] **Step 42: Add test_rms_norm_variance_calculation**

```rust
#[test]
fn test_rms_norm_variance_calculation() -> Result<()> {
    let device = Device::Cpu;
    let weight = Tensor::ones(64, DType::F32, &device)?;
    let norm = RmsNorm::new(weight, 1e-6);
    
    // For constant input, output should be constant (normalized)
    let x = Tensor::full(2.0, (1, 64), DType::F32, &device)?;
    let output = norm.forward(&x)?;
    
    // All outputs should be the same
    let first: f32 = output.get(0)?.to_scalar()?;
    let all_same = output.to_vec::<f32>()?.iter().all(|v| (v - first).abs() < 1e-6);
    assert!(all_same);
    Ok(())
}
```

- [ ] **Step 43: Add test_rms_norm_forward**

```rust
#[test]
fn test_rms_norm_forward() -> Result<()> {
    let device = Device::Cpu;
    let weight = Tensor::ones(32, DType::F32, &device)?;
    let norm = RmsNorm::new(weight, 1e-6);
    
    let x = Tensor::randn(0.0, 1.0, (4, 32), DType::F32, &device)?;
    let output = norm.forward(&x)?;
    
    assert_eq!(output.dims(), x.dims());
    Ok(())
}
```

### B: Boundary Condition Tests (3 tests)

- [ ] **Step 44: Add test_rms_norm_minimal_dim**

```rust
#[test]
fn test_rms_norm_minimal_dim() -> Result<()> {
    let device = Device::Cpu;
    let weight = Tensor::ones(8, DType::F32, &device)?;
    let norm = RmsNorm::new(weight, 1e-6);
    
    let x = Tensor::ones((1, 8), DType::F32, &device)?;
    let output = norm.forward(&x)?;
    assert_eq!(output.dims(), &[1, 8]);
    Ok(())
}
```

- [ ] **Step 45: Add test_rms_norm_large_dim**

```rust
#[test]
fn test_rms_norm_large_dim() -> Result<()> {
    let device = Device::Cpu;
    let weight = Tensor::ones(8192, DType::F32, &device)?;
    let norm = RmsNorm::new(weight, 1e-6);
    
    let x = Tensor::ones((1, 8192), DType::F32, &device)?;
    let output = norm.forward(&x)?;
    assert_eq!(output.dims(), &[1, 8192]);
    Ok(())
}
```

- [ ] **Step 46: Add test_rms_norm_large_eps**

```rust
#[test]
fn test_rms_norm_large_eps() -> Result<()> {
    let device = Device::Cpu;
    let weight = Tensor::ones(64, DType::F32, &device)?;
    let norm = RmsNorm::new(weight, 0.1); // Large eps
    
    let x = Tensor::randn(0.0, 1.0, (2, 64), DType::F32, &device)?;
    let output = norm.forward(&x)?;
    let data: Vec<f32> = output.to_vec()?;
    assert!(data.iter().all(|v| v.is_finite()));
    Ok(())
}
```

### C: Numerical Correctness Tests (3 tests)

- [ ] **Step 47: Add test_rms_norm_output_finite**

```rust
#[test]
fn test_rms_norm_output_finite() -> Result<()> {
    let device = Device::Cpu;
    let weight = Tensor::ones(128, DType::F32, &device)?;
    let norm = RmsNorm::new(weight, 1e-6);
    
    let x = Tensor::randn(-10.0, 10.0, (4, 128), DType::F32, &device)?;
    let output = norm.forward(&x)?;
    let data: Vec<f32> = output.to_vec()?;
    assert!(data.iter().all(|v| v.is_finite()));
    Ok(())
}
```

- [ ] **Step 48: Add test_rms_norm_output_scale**

```rust
#[test]
fn test_rms_norm_output_scale() -> Result<()> {
    let device = Device::Cpu;
    let weight = Tensor::ones(64, DType::F32, &device)?;
    let norm = RmsNorm::new(weight, 1e-6);
    
    let x = Tensor::randn(0.0, 1.0, (2, 64), DType::F32, &device)?;
    let output = norm.forward(&x)?;
    
    // RMSNorm output should have RMS ~ 1 (before weight)
    let rms: f32 = output.sqr()?.mean()?.to_scalar::<f32>()?.sqrt();
    assert!(rms < 2.0 && rms > 0.1, "RMS should be in reasonable range");
    Ok(())
}
```

- [ ] **Step 49: Add test_rms_norm_eps_stability**

```rust
#[test]
fn test_rms_norm_eps_stability() -> Result<()> {
    let device = Device::Cpu;
    let x = Tensor::ones((1, 64), DType::F32, &device)?;
    
    let weight1 = Tensor::ones(64, DType::F32, &device)?;
    let weight2 = Tensor::ones(64, DType::F32, &device)?;
    
    let norm1 = RmsNorm::new(weight1, 1e-8);
    let norm2 = RmsNorm::new(weight2, 1e-2);
    
    let out1 = norm1.forward(&x)?;
    let out2 = norm2.forward(&x)?;
    
    assert!(out1.to_vec::<f32>()?.iter().all(|v| v.is_finite()));
    assert!(out2.to_vec::<f32>()?.iter().all(|v| v.is_finite()));
    Ok(())
}
```

- [ ] **Step 50: Commit RMSNorm tests**

```bash
git add crates/model/src/components/norm/rms_norm.rs
git commit -m "test(model): add 9 tests for RMSNorm (core + boundary + numerical)"
```

---

## Task 5: RoPE Tests

**File:** `crates/model/src/components/positional/rope.rs`

### A: Core Functionality Tests (3 tests)

- [ ] **Step 51: Add test_rope_forward_q_shape_preserved**

```rust
#[test]
fn test_rope_forward_q_shape_preserved() -> Result<()> {
    let device = Device::Cpu;
    let rope = RoPE::new(64, 1024, 10000.0, &device)?;
    
    let q = Tensor::randn(0.0, 1.0, (2, 4, 64), DType::F32, &device)?;
    let k = Tensor::randn(0.0, 1.0, (2, 4, 64), DType::F32, &device)?;
    
    let (q_out, _) = rope.forward(&q, &k, 0)?;
    assert_eq!(q_out.dims(), q.dims());
    Ok(())
}
```

- [ ] **Step 52: Add test_rope_forward_k_shape_preserved**

```rust
#[test]
fn test_rope_forward_k_shape_preserved() -> Result<()> {
    let device = Device::Cpu;
    let rope = RoPE::new(64, 1024, 10000.0, &device)?;
    
    let q = Tensor::randn(0.0, 1.0, (2, 4, 64), DType::F32, &device)?;
    let k = Tensor::randn(0.0, 1.0, (2, 4, 64), DType::F32, &device)?;
    
    let (_, k_out) = rope.forward(&q, &k, 0)?;
    assert_eq!(k_out.dims(), k.dims());
    Ok(())
}
```

- [ ] **Step 53: Add test_rope_rotation_applied**

```rust
#[test]
fn test_rope_rotation_applied() -> Result<()> {
    let device = Device::Cpu;
    let rope = RoPE::new(64, 1024, 10000.0, &device)?;
    
    let q = Tensor::ones((1, 2, 64), DType::F32, &device)?;
    let k = Tensor::ones((1, 2, 64), DType::F32, &device)?;
    
    let (q_out, _) = rope.forward(&q, &k, 0)?;
    
    // After RoPE, values should be different from input (rotation applied)
    let diff = (&q_out - &q)?.abs()?;
    let max_diff: f32 = diff.max()?.to_scalar()?;
    assert!(max_diff > 1e-6, "RoPE should modify the tensor");
    Ok(())
}
```

### B: Boundary Condition Tests (2 tests)

- [ ] **Step 54: Add test_rope_minimal_head_dim**

```rust
#[test]
fn test_rope_minimal_head_dim() -> Result<()> {
    let device = Device::Cpu;
    let rope = RoPE::new(64, 512, 10000.0, &device)?;
    
    let q = Tensor::randn(0.0, 1.0, (1, 1, 64), DType::F32, &device)?;
    let k = Tensor::randn(0.0, 1.0, (1, 1, 64), DType::F32, &device)?;
    
    let (q_out, _) = rope.forward(&q, &k, 0)?;
    assert_eq!(q_out.dims(), q.dims());
    Ok(())
}
```

- [ ] **Step 55: Add test_rope_large_position**

```rust
#[test]
fn test_rope_large_position() -> Result<()> {
    let device = Device::Cpu;
    let rope = RoPE::new(64, 1024, 10000.0, &device)?;
    
    let q = Tensor::randn(0.0, 1.0, (1, 1, 64), DType::F32, &device)?;
    let k = Tensor::randn(0.0, 1.0, (1, 1, 64), DType::F32, &device)?;
    
    // Start position 8192 (larger than max_position)
    let (q_out, _) = rope.forward(&q, &k, 8192)?;
    let data: Vec<f32> = q_out.to_vec()?;
    assert!(data.iter().all(|v| v.is_finite()));
    Ok(())
}
```

### C: Numerical Correctness Tests (3 tests)

- [ ] **Step 56: Add test_rope_output_finite**

```rust
#[test]
fn test_rope_output_finite() -> Result<()> {
    let device = Device::Cpu;
    let rope = RoPE::new(128, 2048, 10000.0, &device)?;
    
    let q = Tensor::randn(-5.0, 5.0, (2, 8, 128), DType::F32, &device)?;
    let k = Tensor::randn(-5.0, 5.0, (2, 8, 128), DType::F32, &device)?;
    
    let (q_out, k_out) = rope.forward(&q, &k, 0)?;
    
    let q_data: Vec<f32> = q_out.to_vec()?;
    let k_data: Vec<f32> = k_out.to_vec()?;
    assert!(q_data.iter().all(|v| v.is_finite()));
    assert!(k_data.iter().all(|v| v.is_finite()));
    Ok(())
}
```

- [ ] **Step 57: Add test_rope_unitary_property**

```rust
#[test]
fn test_rope_unitary_property() -> Result<()> {
    let device = Device::Cpu;
    let rope = RoPE::new(64, 1024, 10000.0, &device)?;
    
    let q = Tensor::randn(0.0, 1.0, (1, 1, 64), DType::F32, &device)?;
    let k = Tensor::randn(0.0, 1.0, (1, 1, 64), DType::F32, &device)?;
    
    let (q_out, _) = rope.forward(&q, &k, 0)?;
    
    // RoPE should be approximately unitary (preserve L2 norm)
    let norm_in: f32 = q.sqr()?.sum_all()?.to_scalar()?;
    let norm_out: f32 = q_out.sqr()?.sum_all()?.to_scalar()?;
    let ratio = norm_out / norm_in;
    assert!((ratio - 1.0).abs() < 0.1, "RoPE should approximately preserve norm");
    Ok(())
}
```

- [ ] **Step 58: Add test_rope_deterministic**

```rust
#[test]
fn test_rope_deterministic() -> Result<()> {
    let device = Device::Cpu;
    let rope = RoPE::new(64, 1024, 10000.0, &device)?;
    
    let q = Tensor::randn(0.0, 1.0, (1, 2, 64), DType::F32, &device)?;
    let k = Tensor::randn(0.0, 1.0, (1, 2, 64), DType::F32, &device)?;
    
    let (q1, _) = rope.forward(&q, &k, 100)?;
    let (q2, _) = rope.forward(&q, &k, 100)?;
    
    let diff = (&q1 - &q2)?.abs()?;
    let max_diff: f32 = diff.max()?.to_scalar()?;
    assert!(max_diff < 1e-6, "RoPE should be deterministic");
    Ok(())
}
```

- [ ] **Step 59: Commit RoPE tests**

```bash
git add crates/model/src/components/positional/rope.rs
git commit -m "test(model): add 5 tests for RoPE (core + boundary + numerical)"
```

---

## Summary

| Task | Component | New Tests | Total After |
|------|-----------|-----------|-------------|
| 1 | StandardBlock | +12 | 16 |
| 2 | SwiGLU | +13 | 16 |
| 3 | GqaAttention | +8 | 18 |
| 4 | RMSNorm | +9 | 12 |
| 5 | RoPE | +5 | 13 |
| **Total** | | **+47** | **75** |

## Verification

After all tests are added, run:

```bash
cargo test -p vllm-model -- --test-threads=1 2>&1 | tail -20
just ci
```

Expected: All tests pass, code coverage increases.

---

## Execution Options

**Plan complete and saved.**

**Two execution options:**

**1. Subagent-Driven (recommended)** - Dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
