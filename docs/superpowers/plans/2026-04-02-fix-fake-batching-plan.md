# Fix Fake Batching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement real batching for decode phase in Qwen3Model

**Architecture:** Stack input tokens into a batched tensor, process through embedding and transformer layers in a single forward pass, then extract next tokens using argmax.

**Tech Stack:** Rust, Candle (tensor operations), vllm-traits (ModelBackend trait)

---

## Task 1: Add Helper Function for Token Stacking

**Files:**

- Modify: `crates/model/src/qwen3/model.rs:310-320`

- [ ] **Step 1: Add stack_tokens helper method**

Add this method to `impl Qwen3Model`:

```rust
fn stack_tokens(&self, tokens: &[Vec<TokenId>]) -> EngineResult<Tensor> {
    let batch_size = tokens.len();
    let token_ids: Vec<u32> = tokens
        .iter()
        .map(|t| t.last().copied().unwrap_or(0))
        .collect();

    Tensor::from_slice(&token_ids, [batch_size], &self.device)
        .map_err(|e| EngineError::new(e.to_string()))
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cargo check -p vllm-model`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add crates/model/src/qwen3/model.rs
git commit -m "feat(model): add stack_tokens helper for batching"
```

---

### Task 2: Rewrite forward Method for Batch Processing

**Files:**

- Modify: `crates/model/src/qwen3/model.rs:304-372`

- [ ] **Step 1: Replace the forward implementation**

Replace the entire `fn forward` body in `impl ModelBackend for Qwen3Model`:

```rust
fn forward(
    &self,
    seq_ids: &[SeqId],
    input_tokens: &[Vec<TokenId>],
    _positions: &[Vec<usize>],
) -> EngineResult<BatchOutput> {
    if seq_ids.is_empty() {
        return Ok(BatchOutput {
            seq_ids: vec![],
            next_tokens: vec![],
        });
    }

    // Stack all tokens into a batched tensor
    let batch_size = seq_ids.len();
    let token_ids: Vec<u32> = input_tokens
        .iter()
        .map(|t| t.last().copied().unwrap_or(0))
        .collect();

    // [batch_size, 1]
    let token_tensor = Tensor::from_slice(&token_ids, [batch_size], &self.device)
        .map_err(|e| EngineError::new(e.to_string()))?;

    // Embedding: [batch_size, 1] -> [batch_size, 1, hidden_size]
    let mut hidden_states = self.embed_tokens
        .forward(&token_tensor)
        .map_err(|e| EngineError::new(e.to_string()))?;

    // Transformer layers: process entire batch
    for layer in &self.layers {
        hidden_states = layer
            .forward(&hidden_states)
            .map_err(|e| EngineError::new(e.to_string()))?;
    }

    // Final norm
    hidden_states = self.norm
        .forward(&hidden_states)
        .map_err(|e| EngineError::new(e.to_string()))?;

    // LM head: [batch_size, 1, hidden_size] -> [batch_size, 1, vocab_size]
    let logits = self.lm_head
        .forward(&hidden_states)
        .map_err(|e| EngineError::new(e.to_string()))?;

    // Get argmax: [batch_size, 1, vocab_size] -> [batch_size]
    let next_tokens: Vec<TokenId> = logits
        .argmax(candle_core::D::Minus1)?
        .to_vec1::<u32>()?
        .into_iter()
        .map(|t| t as TokenId)
        .collect();

    Ok(BatchOutput {
        seq_ids: seq_ids.to_vec(),
        next_tokens,
    })
}
```

- [ ] **Step 2: Check for compilation errors**

Run: `cargo check -p vllm-model 2>&1 | head -30`
Expected: Any error messages (to fix in next step)

- [ ] **Step 3: Fix compilation errors**

If there are errors, fix them. Common issues:

- Import `DType` for `candle_core::D::Minus1`
- Ensure tensor shapes are correct

- [ ] **Step 4: Verify it compiles**

Run: `cargo check -p vllm-model`
Expected: No errors

- [ ] **Step 5: Commit**

```bash
git add crates/model/src/qwen3/model.rs
git commit -m "feat(model): implement real batching in forward method"
```

---

### Task 3: Test Batch Processing

**Files:**

- Modify: `crates/model/src/qwen3/model.rs` (add tests in existing test module)

- [ ] **Step 1: Run existing tests**

Run: `cargo test -p vllm-model -- qwen3::model --nocapture 2>&1 | tail -20`
Expected: Tests pass

- [ ] **Step 2: Add batch size test**

Add this test to the existing `#[cfg(test)]` module:

```rust
#[test]
fn test_qwen3_model_batch_forward() {
    use candle_core::Device;
    use crate::config::Qwen3Config;

    let config = Qwen3Config::qwen3_0_6b();
    let device = Device::Cpu;
    let model = Qwen3Model::new(config, device).unwrap();

    // Test with batch size 3
    let seq_ids = vec![1u64, 2, 3];
    let input_tokens = vec![vec![1], vec![2], vec![3]];
    let positions = vec![vec![0], vec![0], vec![0]];

    let output = model.forward(&seq_ids, &input_tokens, &positions).unwrap();

    assert_eq!(output.seq_ids.len(), 3);
    assert_eq!(output.next_tokens.len(), 3);
}
```

- [ ] **Step 3: Run the new test**

Run: `cargo test -p vllm-model -- test_qwen3_model_batch_forward --nocapture`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add crates/model/src/qwen3/model.rs
git commit -m "test(model): add batch forward test"
```

---

### Task 4: Verify CI

- [ ] **Step 1: Run full CI**

Run: `just ci 2>&1 | tail -20`
Expected: All checks pass

- [ ] **Step 2: Commit any fixes**

If there are issues, fix and commit.

---

## Summary

| Task   | Description                  | Status |
| ------ | ---------------------------- | ------ |
| Task 1 | Add stack_tokens helper      | ⏳     |
| Task 2 | Rewrite forward for batching | ⏳     |
| Task 3 | Test batch processing        | ⏳     |
| Task 4 | Verify CI                    | ⏳     |

**Plan complete and saved to `docs/superpowers/plans/2026-04-02-fix-fake-batching-plan.md`. Two execution options:**

1. **Subagent-Driven (recommended)** - dispatch a fresh subagent per task
2. **Inline Execution** - execute tasks in this session

Which approach?
