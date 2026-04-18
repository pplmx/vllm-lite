# Test Coverage Enhancement Plan (Revised)

**Date:** 2026-04-19
**Status:** Approved
**Reference:** Inspired by SQLite test coverage approach
**Revision:** Adjusted targets based on implementation complexity review

## Overview

Enhance test coverage for shared components with comprehensive tests across multiple categories. Target: 42 new tests across 5 components.

## Test Categories (Priority Order)

### Category A: Core Functionality Tests

Verify normal operation paths work correctly.

### Category B: Boundary Condition Tests

Test edge cases with safe boundaries (avoid panic risks).

### Category C: Numerical Correctness Tests

Verify mathematical properties and output validity.

## Component Test Plans

### 1. StandardBlock Tests

**Location:** `crates/model/src/components/block.rs`

#### A: Core Functionality (5 tests)
- `test_standard_block_deterministic_output` - Same input produces same output
- `test_standard_block_residual_connection` - Verify skip connection works
- `test_standard_block_layernorm_application` - Verify normalization is applied
- `test_standard_block_attention_called` - Verify attention receives correct input
- `test_standard_block_mlp_called` - Verify MLP receives correct input

#### B: Boundary Conditions (4 tests)
- `test_standard_block_minimal_hidden_size` - hidden_size = 16 (safe minimum)
- `test_standard_block_single_head` - num_attention_heads = 1, num_kv_heads = 1
- `test_standard_block_large_batch` - batch_size = 64
- `test_standard_block_extreme_intermediate_size` - intermediate_size = 4 * hidden_size

#### C: Numerical Correctness (3 tests)
- `test_standard_block_output_finite` - Output values are finite (not NaN/Inf)
- `test_standard_block_output_magnitude` - Output magnitude is reasonable
- `test_standard_block_eps_stability` - Output stable with different eps values

**Current:** 4 tests → **Target:** 16 tests (+12)

---

### 2. SwiGLU Tests

**Location:** `crates/model/src/components/mlp/swiglu.rs`

#### A: Core Functionality (5 tests)
- `test_swiglu_new_with_weights_creation` - Test constructor with explicit weights
- `test_swiglu_gate_proj_output_shape` - Verify gate projection shape
- `test_swiglu_up_proj_output_shape` - Verify up projection shape
- `test_swiglu_down_proj_output_shape` - Verify down projection shape
- `test_swiglu_silu_activation` - Verify SiLU activation is applied

#### B: Boundary Conditions (4 tests)
- `test_swiglu_minimal_hidden_size` - hidden_size = 16, intermediate_size = 32
- `test_swiglu_large_ratio` - intermediate_size = 8 * hidden_size
- `test_swiglu_single_token_batch` - batch = 1
- `test_swiglu_multi_token_sequence` - batch = 32

#### C: Numerical Correctness (4 tests)
- `test_swiglu_output_finite` - No NaN or Inf in output
- `test_swiglu_deterministic` - Same weights, same input, same output
- `test_swiglu_zero_input` - All zeros input produces valid output
- `test_swiglu_silu_range` - SiLU output in valid range (-1, 1) for positive inputs

**Current:** 3 tests → **Target:** 16 tests (+13)

---

### 3. GqaAttention Tests

**Location:** `crates/model/src/components/attention/gqa.rs`

#### A: Core Functionality (5 tests)
- `test_gqa_attention_new_creation` - Test constructor
- `test_gqa_attention_num_heads_accessors` - Test num_heads() and num_kv_heads()
- `test_gqa_attention_head_dim_accessors` - Test head_dim() accessor
- `test_gqa_attention_paged_attention_shape` - Verify paged_attention output shape
- `test_gqa_attention_tiled_attention_shape` - Verify tiled_attention output shape

#### B: Boundary Conditions (4 tests)
- `test_gqa_attention_single_q_head` - num_q_heads = 1
- `test_gqa_attention_matching_heads` - num_q_heads = num_kv_heads (MHA mode)
- `test_gqa_attention_small_head_dim` - head_dim = 32
- `test_gqa_attention_large_head_dim` - head_dim = 128

#### C: Numerical Correctness (3 tests)
- `test_gqa_attention_output_finite` - No NaN/Inf
- `test_gqa_attention_deterministic` - Deterministic forward pass
- `test_gqa_attention_expand_kv_correct` - KV expansion produces expected shape

**Note:** Forward tests with RoPE are architecture-specific and deferred.

**Current:** ~10 tests → **Target:** 18 tests (+8)

---

### 4. RMSNorm Tests

**Location:** `crates/model/src/components/norm/rms_norm.rs`

#### A: Core Functionality (3 tests)
- `test_rms_norm_forward` - Basic forward pass
- `test_rms_norm_weight_application` - Weight is correctly applied
- `test_rms_norm_variance_calculation` - RMS variance calculated correctly

#### B: Boundary Conditions (3 tests)
- `test_rms_norm_minimal_dim` - dim = 8 (safe minimum)
- `test_rms_norm_large_dim` - dim = 8192
- `test_rms_norm_large_eps` - eps = 1e-1 (highly stable)

#### C: Numerical Correctness (3 tests)
- `test_rms_norm_output_finite` - No NaN/Inf
- `test_rms_norm_output_scale` - Output magnitude bounded
- `test_rms_norm_eps_stability` - Output stable with different eps

**Current:** 3 tests → **Target:** 12 tests (+9)

---

### 5. RoPE Tests

**Location:** `crates/model/src/components/positional/rope.rs`

#### A: Core Functionality (3 tests)
- `test_rope_forward_q_shape_preserved` - Q shape unchanged after RoPE
- `test_rope_forward_k_shape_preserved` - K shape unchanged after RoPE
- `test_rope_rotation_applied` - Verify rotation is actually applied

#### B: Boundary Conditions (2 tests)
- `test_rope_minimal_head_dim` - head_dim = 64 (even, safe minimum)
- `test_rope_large_position` - position = 8192 (large position)

#### C: Numerical Correctness (3 tests)
- `test_rope_output_finite` - No NaN/Inf
- `test_rope_unitary_property` - RoPE is unitary (preserves norm)
- `test_rope_deterministic` - Same input, same output

**Current:** 8 tests → **Target:** 13 tests (+5)

---

## Summary

| Component | Current | Target | New Tests |
|-----------|---------|--------|-----------|
| StandardBlock | 4 | 16 | +12 |
| SwiGLU | 3 | 16 | +13 |
| GqaAttention | 10 | 18 | +8 |
| RMSNorm | 3 | 12 | +9 |
| RoPE | 8 | 13 | +5 |
| **Total** | **28** | **75** | **+47** |

## Test Naming Convention

```
test_<component>_<category>_<description>

Examples:
- test_swiglu_core_gate_proj_output_shape
- test_standard_block_boundary_single_head
- test_rms_norm_numerical_output_scale
```

## Test Design Principles

1. **Safety First**: Avoid tests that could cause panics (no empty tensors, no zero eps)
2. **Determinism**: All tests must be deterministic (no flaky tests)
3. **Clear Assertions**: Each assertion has a descriptive message
4. **Isolation**: Each test is independent, no shared state

## Acceptance Criteria

1. All new tests pass on first run
2. No compiler warnings
3. Tests are deterministic
4. Code coverage increases measurably
