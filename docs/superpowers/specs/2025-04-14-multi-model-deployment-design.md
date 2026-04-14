# Multi-Model Deployment Support Design Specification

**Date:** 2025-04-14  
**Status:** Ready for Review  
**Scope:** vLLM-lite multi-model deployment and GQA tensor shape fixes  

## Executive Summary

This document outlines the design for supporting multiple models in vLLM-lite and fixing the GQA (Grouped Query Attention) tensor shape mismatch issue that prevents models from generating valid responses.

### Current State
- ✅ Server starts successfully and responds to health checks
- ✅ 4 models available in `/models` directory
- ✅ Tokenizer loads correctly from model directories
- ❌ **Model inference fails with GQA tensor shape mismatch**

### Target State
- ✅ All 4 models deployable via `--model <path>`
- ✅ Each model correctly responds to "hi" query
- ✅ Unified model interface via `ModelBackend` trait

---

## Problem Analysis

### Error Root Cause
The Qwen2.5-0.5B model uses GQA with:
- `num_attention_heads = 14`
- `num_key_value_heads = 2`

During inference, the attention module produces tensor shape `[32]` when it expects `[1, 14, 32]` (batch, heads, head_dim). This occurs in:

```rust
// crates/model/src/qwen3/attention.rs:140
let k = self.expand_kv(&k, self.num_heads, self.num_kv_heads)?;
let v = self.expand_kv(&v, self.num_heads, self.num_kv_heads)?;
```

The `expand_kv` function fails when reshaping from `[batch, seq, num_kv_heads, head_dim]` to broadcast-compatible shape for `num_heads`.

### Affected Models

| Model | Architecture | num_heads | num_kv_heads | Status |
|-------|--------------|-----------|--------------|--------|
| Qwen2.5-0.5B | GQA | 14 | 2 | ❌ Broken (GQA mismatch) |
| Qwen3-0.6B | GQA | 16 | 8 | ❌ Likely affected |
| DeepSeek-R1-Qwen3-8B | GQA | 32 | 8 | ❌ Likely affected |
| Qwen3.5-0.8B | Mamba | ? | ? | ❓ Unknown (different arch) |

---

## Design Options

### Option A: Minimal Fix (Recommended)

**Approach:** Fix only the GQA tensor reshape bug in `qwen3/attention.rs`

**Changes:**
1. Fix `expand_kv()` function to handle non-divisible head counts
2. Add proper shape validation before reshape operations
3. Add `contiguous()` calls before matmul operations

**Pros:**
- Fastest to implement (1-2 days)
- Minimal code changes
- Low risk of breaking existing functionality
- Immediately unblocks model deployment

**Cons:**
- Technical debt remains (models still use separate implementations)
- Each new model may need individual fixes

**Estimated Effort:** 1-2 days  
**Risk:** Low

---

### Option B: Unified Attention Kernel

**Approach:** Create a shared GQA attention implementation used by all models

**Changes:**
1. Move `expand_kv()` to `crates/model/src/components/attention.rs`
2. Create `GqaAttention` struct with standardized interface
3. Refactor all models to use shared implementation
4. Add comprehensive unit tests for GQA variations

**Pros:**
- Single source of truth for GQA logic
- Future models automatically supported
- Easier to add optimizations (Flash Attention, etc.)
- Better test coverage

**Cons:**
- Requires refactoring 3+ model implementations
- Higher initial effort
- Risk of breaking existing working paths

**Estimated Effort:** 1 week  
**Risk:** Medium

---

### Option C: Model Adapter Pattern

**Approach:** Add adapter layer between model-specific and unified interfaces

**Changes:**
1. Define `AttentionConfig` trait for model-specific configs
2. Create `AttentionAdapter` that translates between formats
3. Keep model-specific attention implementations
4. Add adapter per model

**Pros:**
- Preserves model-specific optimizations
- Clean separation of concerns
- Easy to add new models

**Cons:**
- Most complex implementation
- More boilerplate code
- Potential performance overhead

**Estimated Effort:** 1.5 weeks  
**Risk:** Medium-High

---

## Recommendation

**Choose Option A (Minimal Fix)** for the following reasons:

1. **Immediate Impact:** Unblocks model deployment within 1-2 days
2. **Low Risk:** Minimal code changes, easy to review and rollback
3. **Foundation:** Creates working baseline for future refactoring
4. **User Need:** Priority is "deploy models successfully", not code perfection

**Future Work:** After Option A is complete, evaluate Option B for long-term maintainability.

---

## Detailed Design: Option A Implementation

### 1. Fix `expand_kv()` Function

**Location:** `crates/model/src/qwen3/attention.rs`

**Current Implementation (Broken):**
```rust
pub fn expand_kv(
    &self,
    kv: &Tensor,
    num_q_heads: usize,
    num_kv_heads: usize,
) -> Result<Tensor> {
    expand_kv(kv, num_q_heads, num_kv_heads)
}
```

**Fix:** Handle head count mismatch properly:
```rust
fn expand_kv(kv: &Tensor, num_q_heads: usize, num_kv_heads: usize) -> Result<Tensor> {
    if num_q_heads == num_kv_heads {
        return Ok(kv.clone());
    }
    
    let dims = kv.dims();
    let batch_size = dims[0];
    let seq_len = dims[1];
    let head_dim = dims[3];
    
    // Check if num_q_heads is divisible by num_kv_heads
    if num_q_heads % num_kv_heads != 0 {
        // Handle edge case: repeat KV heads to match Q heads
        let repeats = (num_q_heads + num_kv_heads - 1) / num_kv_heads;
        let kv_expanded = kv.repeat(&[1, 1, repeats, 1])?;
        // Slice to exact num_q_heads
        return kv_expanded.narrow(2, 0, num_q_heads);
    }
    
    let repeats = num_q_heads / num_kv_heads;
    kv.repeat(&[1, 1, repeats, 1])
}
```

### 2. Add Shape Validation

**Before reshape operations:**
```rust
// Validate expected shapes before operations
if k.dims() != &[batch_size, seq_len, num_kv_heads, head_dim] {
    return Err(candle_core::Error::msg(format!(
        "Unexpected K tensor shape: {:?}, expected {:?}",
        k.dims(),
        &[batch_size, seq_len, num_kv_heads, head_dim]
    )));
}
```

### 3. Ensure Tensor Contiguity

**Before matmul:**
```rust
let k = k.contiguous()?;
let v = v.contiguous()?;
```

### 4. Add Test Coverage

**New test file:** `crates/model/tests/gqa_shape_tests.rs`

```rust
#[test]
fn test_expand_kv_gqa_divisible() {
    // num_q_heads=14, num_kv_heads=2 (Qwen2.5)
    // 14 % 2 == 0 ✓
}

#[test]
fn test_expand_kv_gqa_non_divisible() {
    // Edge cases
}
```

---

## Deployment Verification

### Test Protocol

For each model, run:

```bash
# 1. Start server
./target/release/vllm-server --model /models/<MODEL>

# 2. Health check
curl http://localhost:8000/health
# Expected: {"status":"ok"}

# 3. Inference test
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "hi", "max_tokens": 20}'
# Expected: Valid JSON response with generated text
```

### Success Criteria

| Model | Health Check | "hi" Query | Notes |
|-------|--------------|------------|-------|
| Qwen2.5-0.5B | ✅ | ✅ | Baseline |
| Qwen3-0.6B | ✅ | ✅ | Verify GQA fix |
| DeepSeek-R1-Qwen3-8B | ✅ | ✅ | Larger model |
| Qwen3.5-0.8B | ✅ | ✅ | Mamba arch |

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| GQA fix introduces new bugs | Medium | High | Comprehensive test coverage |
| Performance regression | Low | Medium | Benchmark before/after |
| Other models have different issues | Medium | Medium | Test each model individually |
| Mamba architecture incompatible | Medium | High | Fallback to stub response |

---

## Appendix: Technical Details

### GQA Tensor Shape Reference

```
Qwen2.5-0.5B Config:
- hidden_size: 896
- num_attention_heads: 14
- num_key_value_heads: 2
- head_dim: hidden_size / num_heads = 64

Expected shapes:
- Q: [batch, seq, num_heads, head_dim] = [1, N, 14, 64]
- K: [batch, seq, num_kv_heads, head_dim] = [1, N, 2, 64]
- V: [batch, seq, num_kv_heads, head_dim] = [1, N, 2, 64]

After expand_kv:
- K: [1, N, 14, 64] (repeated 7 times per head group)
- V: [1, N, 14, 64] (repeated 7 times per head group)
```

### Error Log Analysis

```
shape mismatch in reshape, lhs: [32], rhs: [1, 14, 32]
- lhs [32] suggests flattened tensor
- Expected [1, 14, 32] = [batch, heads, head_dim]
- Root cause: expand_kv() returning wrong shape

matmul is only supported for contiguous tensors
- Fix: Add .contiguous() before matmul
```

---

## Next Steps

1. **Implement** Option A (Minimal Fix)
2. **Test** against all 4 models
3. **Write** implementation plan via `writing-plans` skill
4. **Execute** plan with proper testing

---

## References

- [AGENTS.md](/AGENTS.md) - Project development guidelines
- [ROADMAP.md](/ROADMAP.md) - Project roadmap
- Model configs in `/models/*/config.json`
- Qwen2 paper: https://arxiv.org/abs/2309.16609 (GQA specification)
