# Test Coverage Enhancement Plan

**Date:** 2026-04-19
**Status:** Approved
**Reference:** Inspired by SQLite test coverage approach

## Overview

Enhance test coverage for shared components with comprehensive tests across multiple categories. Goal:达到接近生产级别的测试覆盖。

## Test Categories (Priority Order)

### Category A: Core Functionality Tests

Verify normal operation paths work correctly.

### Category B: Boundary Condition Tests

Test edge cases: empty inputs, max sizes, zero values, etc.

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

#### B: Boundary Conditions (5 tests)
- `test_standard_block_minimal_hidden_size` - hidden_size = 1
- `test_standard_block_single_head` - num_attention_heads = 1, num_kv_heads = 1
- `test_standard_block_large_batch` - batch_size = 256
- `test_standard_block_long_sequence` - seq_len = 4096
- `test_standard_block_extreme_intermediate_size` - intermediate_size = 4 * hidden_size

#### C: Numerical Correctness (3 tests)
- `test_standard_block_output_range` - Output values are finite (not NaN/Inf)
- `test_standard_block_gradient_flow` - Backward pass works (if supported)
- `test_standard_block_eps_stability` - Output stable with different eps values

**Current:** 4 tests → **Target:** 17 tests

---

### 2. SwiGLU Tests

**Location:** `crates/model/src/components/mlp/swiglu.rs`

#### A: Core Functionality (5 tests)
- `test_swiglu_new_with_weights_creation` - Test constructor with explicit weights
- `test_swiglu_gate_proj_output_shape` - Verify gate projection shape
- `test_swiglu_up_proj_output_shape` - Verify up projection shape
- `test_swiglu_down_proj_output_shape` - Verify down projection shape
- `test_swiglu_silu_activation` - Verify SiLU activation is applied

#### B: Boundary Conditions (5 tests)
- `test_swiglu_minimal_hidden_size` - hidden_size = 1, intermediate_size = 2
- `test_swiglu_zero_intermediate` - intermediate_size = 0 (edge case)
- `test_swiglu_large_ratio` - intermediate_size = 10 * hidden_size
- `test_swiglu_single_token_batch` - batch = 1
- `test_swiglu_empty_tensor` - shape (0, hidden_size) handling

#### C: Numerical Correctness (4 tests)
- `test_swiglu_output_finite` - No NaN or Inf in output
- `test_swiglu_deterministic` - Same weights, same input, same output
- `test_swiglu_zero_input` - All zeros input produces valid output
- `test_swiglu_silu_range` - SiLU output in valid range (-1, 1) for positive inputs

**Current:** 3 tests → **Target:** 17 tests

---

### 3. GqaAttention Tests

**Location:** `crates/model/src/components/attention/gqa.rs`

#### A: Core Functionality (5 tests)
- `test_gqa_attention_forward_basic` - Basic forward pass
- `test_gqa_attention_kv_expansion` - Verify KV heads are correctly expanded
- `test_gqa_attention_o_proj_applied` - Verify output projection is applied
- `test_gqa_attention_num_heads_accessors` - Test num_heads() and num_kv_heads()
- `test_gqa_attention_head_dim_accessors` - Test head_dim() accessor

#### B: Boundary Conditions (5 tests)
- `test_gqa_attention_single_q_head` - num_q_heads = 1
- `test_gqa_attention_matching_heads` - num_q_heads = num_kv_heads (MHA mode)
- `test_gqa_attention_minimal_head_dim` - head_dim = 1
- `test_gqa_attention_large_head_dim` - head_dim = 256
- `test_gqa_attention_batch_size_1` - Single batch element

#### C: Numerical Correctness (4 tests)
- `test_gqa_attention_output_finite` - No NaN/Inf
- `test_gqa_attention_output_scale` - Attention scores in valid range [0, 1]
- `test_gqa_attention_zero_kv` - Zero KV produces zero output contribution
- `test_gqa_attention_deterministic` - Deterministic forward pass

**Current:** ~10 tests → **Target:** 24 tests

---

### 4. RMSNorm Tests

**Location:** `crates/model/src/components/norm/rms_norm.rs`

#### A: Core Functionality (3 tests)
- `test_rms_norm_forward` - Basic forward pass
- `test_rms_norm_weight_application` - Weight is correctly applied
- `test_rms_norm_variance_calculation` - RMS variance calculated correctly

#### B: Boundary Conditions (4 tests)
- `test_rms_norm_minimal_dim` - dim = 1
- `test_rms_norm_large_dim` - dim = 65536
- `test_rms_norm_zero_eps` - eps = 0 (avoid division by zero)
- `test_rms_norm_large_eps` - eps = 1e-1 (highly stable)

#### C: Numerical Correctness (3 tests)
- `test_rms_norm_output_finite` - No NaN/Inf
- `test_rms_norm_output_scale` - Output magnitude bounded
- `test_rms_norm_eps_stability` - Output stable with different eps

**Current:** 3 tests → **Target:** 10 tests

---

### 5. RoPE Tests

**Location:** `crates/model/src/components/positional/rope.rs`

#### A: Core Functionality (4 tests)
- `test_rope_forward_q_shape_preserved` - Q shape unchanged after RoPE
- `test_rope_forward_k_shape_preserved` - K shape unchanged after RoPE
- `test_rope_rotation_applied` - Verify rotation is actually applied
- `test_rope_different_positions_different_output` - Position affects output

#### B: Boundary Conditions (4 tests)
- `test_rope_minimal_head_dim` - head_dim = 2 (minimum for RoPE)
- `test_rope_even_head_dim_required` - Verify odd head_dim handling
- `test_rope_single_position` - positions = [0]
- `test_rope_large_position` - position > max_position (caching)

#### C: Numerical Correctness (3 tests)
- `test_rope_output_finite` - No NaN/Inf
- `test_rope_unitary_property` - RoPE is unitary (preserves norm)
- `test_rope_deterministic` - Same input, same output

**Current:** 8 tests → **Target:** 11 tests

---

## Summary

| Component | Current | Target | New Tests |
|-----------|---------|--------|-----------|
| StandardBlock | 4 | 17 | +13 |
| SwiGLU | 3 | 17 | +14 |
| GqaAttention | 10 | 24 | +14 |
| RMSNorm | 3 | 10 | +7 |
| RoPE | 8 | 11 | +3 |
| **Total** | **28** | **79** | **+51** |

## Test Naming Convention

```
test_<component>_<category>_<description>

Examples:
- test_swiglu_core_gate_proj_output_shape
- test_standard_block_boundary_single_head
- test_rms_norm_numerical_output_scale
```

## Acceptance Criteria

1. All new tests pass on first run
2. No compiler warnings
3. Tests are deterministic (no flaky tests)
4. Each test has clear assertion message on failure
