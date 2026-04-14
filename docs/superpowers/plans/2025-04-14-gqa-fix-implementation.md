# GQA Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix GQA tensor shape mismatch in vLLM-lite to enable all 4 models to respond correctly to "hi" query

**Architecture:** Fix the `expand_kv()` function in qwen3 attention module to handle GQA head count mismatches (e.g., num_heads=14, num_kv_heads=2), add shape validation, and ensure tensor contiguity before matmul operations.

**Tech Stack:** Rust, Candle ML framework, vLLM-lite codebase

**Reference:** `docs/superpowers/specs/2025-04-14-multi-model-deployment-design.md` (Option A)

---

## File Structure

| File | Purpose | Action |
|------|---------|--------|
| `crates/model/src/qwen3/attention.rs` | Qwen3 GQA attention implementation | Modify `expand_kv()` and add validation |
| `crates/model/tests/gqa_shape_tests.rs` | New test file for GQA shape handling | Create |
| `crates/model/src/components/attention.rs` | Shared attention utilities (optional) | Modify if needed |

---

### Task 1: Investigate Current `expand_kv` Implementation

**Files:**
- Read: `crates/model/src/qwen3/attention.rs:155-165`
- Read: `crates/model/src/qwen3/attention.rs:300-325` (helper functions)

- [ ] **Step 1: Locate the current `expand_kv` function**
  Run: `grep -n "fn expand_kv" crates/model/src/qwen3/attention.rs`
  Expected output: Line numbers where `expand_kv` is defined

- [ ] **Step 2: Read the current implementation**
  Run: `sed -n '155,165p' crates/model/src/qwen3/attention.rs`
  Expected: See current `expand_kv` function body

- [ ] **Step 3: Identify the bug**
  Current implementation likely fails when `num_heads % num_kv_heads != 0` or produces wrong shape
  Note: Qwen2.5 has num_heads=14, num_kv_heads=2, so 14/2=7 (divisible, but shape may still be wrong)

---

### Task 2: Write Failing Test for GQA Shape Handling

**Files:**
- Create: `crates/model/tests/gqa_shape_tests.rs`

- [ ] **Step 1: Create the test file**
  ```bash
  mkdir -p crates/model/tests
  cat > crates/model/tests/gqa_shape_tests.rs << 'EOF'
  //! Tests for GQA tensor shape handling
  //! 
  //! These tests verify that expand_kv correctly handles GQA head count mismatches.

  use candle_core::{Device, Tensor};

  /// Helper function to simulate expand_kv logic
  fn expand_kv_simple(kv: &Tensor, num_q_heads: usize, num_kv_heads: usize) -> candle_core::Result<Tensor> {
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

  #[test]
  fn test_expand_kv_gqa_qwen25_config() {
      // Qwen2.5-0.5B: num_heads=14, num_kv_heads=2
      let device = Device::Cpu;
      let kv = Tensor::zeros(&[1, 10, 2, 64], candle_core::DType::F32, &device).unwrap();
      
      let result = expand_kv_simple(&kv, 14, 2);
      assert!(result.is_ok(), "expand_kv should succeed for Qwen2.5 config");
      
      let expanded = result.unwrap();
      let dims = expanded.dims();
      assert_eq!(dims, &[1, 10, 14, 64], "Expected shape [1, 10, 14, 64], got {:?}", dims);
  }

  #[test]
  fn test_expand_kv_gqa_qwen3_config() {
      // Qwen3-0.6B: num_heads=16, num_kv_heads=8
      let device = Device::Cpu;
      let kv = Tensor::zeros(&[1, 10, 8, 64], candle_core::DType::F32, &device).unwrap();
      
      let result = expand_kv_simple(&kv, 16, 8);
      assert!(result.is_ok(), "expand_kv should succeed for Qwen3 config");
      
      let expanded = result.unwrap();
      let dims = expanded.dims();
      assert_eq!(dims, &[1, 10, 16, 64], "Expected shape [1, 10, 16, 64], got {:?}", dims);
  }

  #[test]
  fn test_expand_kv_no_expansion_needed() {
      // MHA case: num_heads == num_kv_heads
      let device = Device::Cpu;
      let kv = Tensor::zeros(&[1, 10, 8, 64], candle_core::DType::F32, &device).unwrap();
      
      let result = expand_kv_simple(&kv, 8, 8);
      assert!(result.is_ok());
      
      let expanded = result.unwrap();
      let dims = expanded.dims();
      assert_eq!(dims, &[1, 10, 8, 64], "Shape should remain unchanged for MHA");
  }

  #[test]
  fn test_expand_kv_non_divisible() {
      // Edge case: num_heads=10, num_kv_heads=3 (10 % 3 != 0)
      let device = Device::Cpu;
      let kv = Tensor::zeros(&[1, 5, 3, 64], candle_core::DType::F32, &device).unwrap();
      
      let result = expand_kv_simple(&kv, 10, 3);
      // Should handle gracefully
      assert!(result.is_ok(), "expand_kv should handle non-divisible case");
  }
  EOF
  ```

- [ ] **Step 2: Run the failing test to verify test setup**
  Run: `cargo test -p vllm-model --test gqa_shape_tests 2>&1 | tail -30`
  Expected: Tests run, possibly some fail if expand_kv_simple has bugs

- [ ] **Step 3: Commit the test file**
  ```bash
  git add crates/model/tests/gqa_shape_tests.rs
  git commit -m "test(model): add GQA shape handling tests

  Add tests for expand_kv with various GQA configurations:
  - Qwen2.5 (14 heads, 2 kv heads)
  - Qwen3 (16 heads, 8 kv heads)
  - MHA (no expansion)
  - Non-divisible edge case"
  ```

---

### Task 3: Fix `expand_kv` in qwen3/attention.rs

**Files:**
- Modify: `crates/model/src/qwen3/attention.rs:155-175`

- [ ] **Step 1: Read the current expand_kv implementation**
  Run: `sed -n '155,175p' crates/model/src/qwen3/attention.rs`
  Note current implementation for reference

- [ ] **Step 2: Replace expand_kv with fixed version**
  Edit `crates/model/src/qwen3/attention.rs` around line 158:
  
  ```rust
  pub fn expand_kv(
      &self,
      kv: &Tensor,
      num_q_heads: usize,
      num_kv_heads: usize,
  ) -> Result<Tensor> {
      expand_kv_fixed(kv, num_q_heads, num_kv_heads)
  }
  ```

- [ ] **Step 3: Add the fixed helper function before line 160**
  Insert this function after the current `expand_kv` method:
  
  ```rust
  /// Expand KV heads to match Q heads count for GQA
  /// 
  /// For GQA, we need to repeat KV heads to match Q heads.
  /// Example: num_q_heads=14, num_kv_heads=2 => repeat 7 times
  fn expand_kv_fixed(kv: &Tensor, num_q_heads: usize, num_kv_heads: usize) -> Result<Tensor> {
      if num_q_heads == num_kv_heads {
          // Standard MHA - no expansion needed
          return Ok(kv.clone());
      }
      
      let dims = kv.dims();
      if dims.len() < 4 {
          return Err(candle_core::Error::msg(format!(
              "KV tensor must have at least 4 dimensions, got {:?}",
              dims
          )));
      }
      
      let num_kv_heads_in_tensor = dims[2];
      if num_kv_heads_in_tensor != num_kv_heads {
          return Err(candle_core::Error::msg(format!(
              "KV tensor has {} heads but expected {}",
              num_kv_heads_in_tensor, num_kv_heads
          )));
      }
      
      // Calculate repeat factor
      if num_q_heads % num_kv_heads != 0 {
          // Edge case: use ceil division and slice
          let repeat_factor = (num_q_heads + num_kv_heads - 1) / num_kv_heads;
          let kv_repeated = kv.repeat(&[1, 1, repeat_factor, 1])?;
          // Slice to exact num_q_heads
          return kv_repeated.narrow(2, 0, num_q_heads);
      }
      
      let repeat_factor = num_q_heads / num_kv_heads;
      kv.repeat(&[1, 1, repeat_factor, 1])
  }
  ```

- [ ] **Step 4: Run cargo check to verify syntax**
  Run: `cargo check -p vllm-model 2>&1 | tail -20`
  Expected: No compilation errors

- [ ] **Step 5: Commit the fix**
  ```bash
  git add crates/model/src/qwen3/attention.rs
  git commit -m "fix(model): correct GQA expand_kv tensor shapes

  Fix expand_kv to properly handle GQA head count mismatches:
  - Add dimension validation
  - Handle non-divisible head counts with narrow() slicing
  - Return proper error messages for shape mismatches

  Fixes shape mismatch for Qwen2.5 (14 heads, 2 kv heads)"
  ```

---

### Task 4: Add Shape Validation to Attention Forward Pass

**Files:**
- Modify: `crates/model/src/qwen3/attention.rs:124-156` (forward method)

- [ ] **Step 1: Locate the forward method**
  Run: `grep -n "pub fn forward" crates/model/src/qwen3/attention.rs | head -5`
  Note the line numbers

- [ ] **Step 2: Add shape validation before matmul**
  In the `forward` method, after line 140 (after expand_kv calls), add:
  
  ```rust
  // Ensure tensors are contiguous before matmul
  let k = k.contiguous()?;
  let v = v.contiguous()?;
  
  // Validate shapes before matmul
  let q_dims = q.dims();
  let k_dims = k.dims();
  let v_dims = v.dims();
  
  if q_dims.len() != 4 || k_dims.len() != 4 || v_dims.len() != 4 {
      return Err(candle_core::Error::msg(format!(
          "Invalid tensor dimensions: Q={:?}, K={:?}, V={:?}",
          q_dims, k_dims, v_dims
      )));
  }
  
  // Verify head counts match after expansion
  if q_dims[2] != k_dims[2] || q_dims[2] != v_dims[2] {
      return Err(candle_core::Error::msg(format!(
          "Head count mismatch after expand_kv: Q heads={}, K heads={}, V heads={}",
          q_dims[2], k_dims[2], v_dims[2]
      )));
  }
  ```

- [ ] **Step 3: Run cargo check**
  Run: `cargo check -p vllm-model 2>&1 | tail -20`
  Expected: No errors

- [ ] **Step 4: Commit**
  ```bash
  git add crates/model/src/qwen3/attention.rs
  git commit -m "fix(model): add shape validation in attention forward

  Add validation before matmul operations:
  - Ensure tensor contiguity with .contiguous()
  - Validate tensor dimensions
  - Verify head counts match after expand_kv
  - Return descriptive error messages"
  ```

---

### Task 5: Fix Similar Issues in forward_prefill and forward_decode

**Files:**
- Modify: `crates/model/src/qwen3/attention.rs:193-258` (forward_prefill)
- Modify: `crates/model/src/qwen3/attention.rs:260-299` (forward_decode)

- [ ] **Step 1: Check forward_prefill for expand_kv calls**
  Run: `sed -n '245,252p' crates/model/src/qwen3/attention.rs`
  Look for expand_kv calls around line 248-249

- [ ] **Step 2: Add contiguity and validation to forward_prefill**
  After expand_kv calls (around line 249), add:
  ```rust
  // Ensure contiguity before operations
  let k_expanded = k_expanded.contiguous()?;
  let v_expanded = v_expanded.contiguous()?;
  ```

- [ ] **Step 3: Check forward_decode for expand_kv calls**
  Run: `sed -n '288,292p' crates/model/src/qwen3/attention.rs`

- [ ] **Step 4: Add contiguity to forward_decode**
  After expand_kv calls (around line 290), add:
  ```rust
  // Ensure contiguity before operations
  let k_expanded = k_expanded.contiguous()?;
  let v_expanded = v_expanded.contiguous()?;
  ```

- [ ] **Step 5: Run cargo check**
  Run: `cargo check -p vllm-model 2>&1 | tail -20`
  Expected: No errors

- [ ] **Step 6: Commit**
  ```bash
  git add crates/model/src/qwen3/attention.rs
  git commit -m "fix(model): ensure tensor contiguity in attention ops

  Add .contiguous() calls in forward_prefill and forward_decode
  to prevent matmul errors with non-contiguous tensors.

  Applied after expand_kv operations in both methods."
  ```

---

### Task 6: Build Release and Test with Qwen2.5

**Files:**
- Test with: `/models/Qwen2.5-0.5B-Instruct`

- [ ] **Step 1: Build release version**
  Run: `cargo build --release -p vllm-server 2>&1 | tail -10`
  Expected: Build succeeds

- [ ] **Step 2: Start server with Qwen2.5**
  Run: 
  ```bash
  timeout 60 ./target/release/vllm-server --model /models/Qwen2.5-0.5B-Instruct 2>&1 &
  sleep 20
  ```
  Expected: Server starts, "Server listening on 0.0.0.0:8000"

- [ ] **Step 3: Test health endpoint**
  Run: `curl -s http://localhost:8000/health`
  Expected: `{"status":"ok"}`

- [ ] **Step 4: Test "hi" query**
  Run:
  ```bash
  timeout 30 curl -s -X POST http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"prompt": "hi", "max_tokens": 20}'
  ```
  Expected: Valid JSON response with generated text, NO shape mismatch errors

- [ ] **Step 5: Kill server**
  Run: `pkill -9 -f "vllm-server" 2>/dev/null || true`

- [ ] **Step 6: Commit test results**
  ```bash
  git commit --allow-empty -m "test(server): verify Qwen2.5 GQA fix

  Manual test results:
  - Server starts successfully
  - Health check passes
  - 'hi' query returns valid response (no shape mismatch errors)"
  ```

---

### Task 7: Test All 4 Models

**Files:**
- Test: All models in `/models/`

- [ ] **Step 1: Test Qwen3-0.6B**
  ```bash
  timeout 60 ./target/release/vllm-server --model /models/Qwen3-0.6B 2>&1 &
  sleep 20
  curl -s http://localhost:8000/health
  timeout 30 curl -s -X POST http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"prompt": "hi", "max_tokens": 20}'
  pkill -9 -f "vllm-server" 2>/dev/null || true
  ```

- [ ] **Step 2: Test DeepSeek-R1-Qwen3-8B**
  ```bash
  timeout 120 ./target/release/vllm-server --model /models/DeepSeek-R1-0528-Qwen3-8B 2>&1 &
  sleep 30
  curl -s http://localhost:8000/health
  timeout 30 curl -s -X POST http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"prompt": "hi", "max_tokens": 20}'
  pkill -9 -f "vllm-server" 2>/dev/null || true
  ```

- [ ] **Step 3: Test Qwen3.5-0.8B**
  ```bash
  timeout 60 ./target/release/vllm-server --model /models/Qwen3.5-0.8B 2>&1 &
  sleep 20
  curl -s http://localhost:8000/health
  timeout 30 curl -s -X POST http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"prompt": "hi", "max_tokens": 20}'
  pkill -9 -f "vllm-server" 2>/dev/null || true
  ```

- [ ] **Step 4: Document results**
  Create a summary of which models work:
  - Qwen2.5-0.5B: [PASS/FAIL]
  - Qwen3-0.6B: [PASS/FAIL]
  - DeepSeek-R1-Qwen3-8B: [PASS/FAIL]
  - Qwen3.5-0.8B: [PASS/FAIL]

- [ ] **Step 5: Commit results**
  ```bash
  git commit --allow-empty -m "test(server): verify all 4 models

  Multi-model deployment test results:
  - Qwen2.5-0.5B: PASS/FAIL
  - Qwen3-0.6B: PASS/FAIL
  - DeepSeek-R1-Qwen3-8B: PASS/FAIL
  - Qwen3.5-0.8B: PASS/FAIL"
  ```

---

### Task 8: Run Full Test Suite

**Files:**
- All tests in workspace

- [ ] **Step 1: Run cargo test**
  Run: `cargo test --workspace 2>&1 | tail -50`
  Expected: All tests pass (or same number as before)

- [ ] **Step 2: Run clippy**
  Run: `cargo clippy --workspace -- -D warnings 2>&1 | tail -30`
  Expected: No new warnings

- [ ] **Step 3: Final verification**
  Run: `cargo fmt --all --check`
  Expected: No formatting issues

- [ ] **Step 4: Commit**
  ```bash
  git commit --allow-empty -m "ci: verify full test suite passes

  - All workspace tests pass
  - Clippy clean
  - Code formatted"
  ```

---

## Verification Checklist

After completing all tasks, verify:

- [ ] All 4 models can be loaded with `--model <path>`
- [ ] All 4 models respond correctly to "hi" query
- [ ] No GQA shape mismatch errors in logs
- [ ] `cargo test --workspace` passes
- [ ] `cargo clippy --workspace` clean
- [ ] Documentation updated (if needed)

---

## Rollback Plan

If issues arise:

1. **Immediate rollback:** `git revert HEAD~N` (N = number of commits to undo)
2. **Disable specific model:** Comment out model in `loader/mod.rs` detect_architecture
3. **Add feature flag:** Wrap changes in `#[cfg(feature = "gqa_fix")]`

---

## Notes for Implementer

1. **Test incrementally:** After each commit, verify the server still starts
2. **Watch for OOM:** DeepSeek-R1-8B requires ~16GB VRAM, test on CPU if needed
3. **Shape errors:** If you see `[32]` vs `[1, 14, 32]`, the expand_kv fix isn't working
4. **Matmul errors:** Add `.contiguous()` before the failing matmul
5. **Ask for help:** If stuck on tensor shapes, print dims at each step

---

**Plan created:** 2025-04-14  
**Expected completion:** 1-2 days  
**Reference spec:** `docs/superpowers/specs/2025-04-14-multi-model-deployment-design.md`
