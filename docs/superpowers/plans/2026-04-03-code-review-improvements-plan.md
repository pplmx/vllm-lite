# Code Review Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve code quality through dead code cleanup, implement missing forward_logits, deduplicate scheduler code, and fix random number generation

**Architecture:** Four-phase approach: Quick fixes → Refactoring → Feature completion → Cleanup

**Tech Stack:** Rust, rand crate (for secure random), existing test infrastructure

---

## Phase 1: Quick Wins

### Task 1: Fix Random Number Generation in sampling.rs

**Files:**
- Modify: `crates/core/src/sampling.rs:1-15`
- Test: Run existing sampling tests

- [ ] **Step 1: Add rand dependency if not present**

Check `crates/core/Cargo.toml` for `rand` crate. If not present, add:
```toml
rand = "0.8"
```

- [ ] **Step 2: Replace random_f32 implementation**

Replace the current implementation in `crates/core/src/sampling.rs`:

```rust
use rand::Rng;

fn random_f32() -> f32 {
    rand::thread_rng().gen::<f32>()
}
```

- [ ] **Step 3: Run sampling tests to verify**

```bash
cargo test -p vllm-core sampling
```

Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add crates/core/src/sampling.rs crates/core/Cargo.toml
git commit -m "fix(core): use secure random number generation"
```

---

### Task 2: Remove Dead Code in server/src/api.rs

**Files:**
- Modify: `crates/server/src/api.rs:19-77` and other dead code locations

- [ ] **Step 1: Identify removable dead code**

Run clippy to find unused code:
```bash
cargo clippy --workspace -- -D warnings 2>&1 | grep -i "dead_code"
```

- [ ] **Step 2: Remove dead code in api.rs**

Review each `#[allow(dead_code)]` function:
- `parse_prompt_template` (line 19)
- `validate_max_tokens` (line 33)
- `calculate_stop_conditions` (line 38)
- etc.

For each unused function, check if it's truly unused, then remove both the function and the `#[allow(dead_code)]` attribute.

- [ ] **Step 3: Run tests**

```bash
cargo test -p vllm-server
```

- [ ] **Step 4: Commit**

```bash
git add crates/server/src/api.rs
git commit -m "refactor(server): remove unused dead_code in api.rs"
```

---

## Phase 2: Refactoring

### Task 3: Deduplicate Scheduler Batch Building

**Files:**
- Modify: `crates/core/src/scheduler.rs:280-373`
- Test: `crates/core/tests/scheduler_refactored.rs`

- [ ] **Step 1: Review duplicated code**

Both `build_batch_with_pd_separation` and `build_batch_mixed` contain:
```rust
// Lines 302-319 (identical in both methods)
let batch_len = seq_ids.len();
let seq_id_to_idx: HashMap<SeqId, usize> =
    seq_ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();

let mut kv_block_ids: Vec<Vec<usize>> = vec![vec![]; batch_len];
let mut num_computed_tokens: Vec<usize> = vec![0; batch_len];
let mut is_prefill: Vec<bool> = vec![false; batch_len];

for seq in &self.running {
    if let Some(&idx) = seq_id_to_idx.get(&seq.id) {
        kv_block_ids[idx] = seq.kv_blocks.as_ref().clone();
        num_computed_tokens[idx] = seq.num_computed_tokens;
        is_prefill[idx] = seq.status == Status::Prefilling;
    }
}
```

- [ ] **Step 2: Extract helper method**

Add to `Scheduler` impl:

```rust
fn finalize_batch(
    seq_ids: Vec<SeqId>,
    input_tokens: Vec<Vec<TokenId>>,
    positions: Vec<Vec<usize>>,
    running: &[Sequence],
) -> Batch {
    let batch_len = seq_ids.len();
    let seq_id_to_idx: HashMap<SeqId, usize> =
        seq_ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();

    let mut kv_block_ids: Vec<Vec<usize>> = vec![vec![]; batch_len];
    let mut num_computed_tokens: Vec<usize> = vec![0; batch_len];
    let mut is_prefill: Vec<bool> = vec![false; batch_len];

    for seq in running {
        if let Some(&idx) = seq_id_to_idx.get(&seq.id) {
            kv_block_ids[idx] = seq.kv_blocks.as_ref().clone();
            num_computed_tokens[idx] = seq.num_computed_tokens;
            is_prefill[idx] = seq.status == Status::Prefilling;
        }
    }

    Batch {
        seq_ids,
        input_tokens,
        positions,
        kv_block_ids,
        num_computed_tokens,
        is_prefill,
    }
}
```

- [ ] **Step 3: Refactor both methods to use helper**

Replace duplicated code with:
```rust
let batch = Self::finalize_batch(seq_ids, input_tokens, positions, &self.running);
return batch;
```

- [ ] **Step 4: Run scheduler tests**

```bash
cargo test -p vllm-core scheduler
```

Expected: All tests pass

- [ ] **Step 5: Commit**

```bash
git add crates/core/src/scheduler.rs
git commit -m "refactor(core): deduplicate batch building in scheduler"
```

---

## Phase 3: Feature Complete

### Task 4: Implement forward_logits Properly

**Files:**
- Modify: `crates/model/src/qwen3/model.rs:487-501`
- Test: Existing model tests + add new test for beam search

- [ ] **Step 1: Add cache field to Qwen3Model**

In `crates/model/src/qwen3/model.rs`, add to struct:
```rust
use std::collections::HashMap;

pub struct Qwen3Model {
    // ... existing fields
    logits_cache: HashMap<SeqId, Vec<f32>>,
}
```

- [ ] **Step 2: Initialize cache in constructors**

In `new`, `new_with_tp`, and `from_weights` methods, add:
```rust
logits_cache: HashMap::new(),
```

- [ ] **Step 3: Implement forward_logits properly**

Replace placeholder with:
```rust
fn forward_logits(
    &mut self,
    seq_ids: &[SeqId],
    input_tokens: &[Vec<TokenId>],
    positions: &[Vec<usize>],
    kv_block_ids: &[Vec<usize>],
    num_computed_tokens: &[usize],
    is_prefill: &[bool],
) -> EngineResult<Vec<Vec<f32>>> {
    if seq_ids.is_empty() {
        return Ok(vec![]);
    }

    // Call forward to get logits, store in cache
    let mut results = Vec::with_capacity(seq_ids.len());
    
    for (i, seq_id) in seq_ids.iter().enumerate() {
        let tokens = &input_tokens[i];
        let pos = &positions[i];
        let blocks = &kv_block_ids[i];
        let computed = num_computed_tokens[i];
        let pf = is_prefill[i];

        let (logits, _) = self.forward_with_cache(
            tokens, computed, blocks, pos, pf
        )?;

        // Extract logits as Vec<f32>
        use candle_core::D;
        let logits_vec = logits
            .squeeze(0)
            .map_err(|e| EngineError::new(e.to_string()))?
            .to_vec1::<f32>()
            .map_err(|e| EngineError::new(e.to_string()))?;

        self.logits_cache.insert(*seq_id, logits_vec.clone());
        results.push(logits_vec);
    }

    Ok(results)
}
```

- [ ] **Step 4: Add test for forward_logits**

In test module:
```rust
#[test]
fn test_forward_logits_returns_actual_logits() {
    let config = Qwen3Config {
        vocab_size: Some(1000),
        hidden_size: Some(128),
        num_hidden_layers: Some(1),
        num_attention_heads: Some(4),
        num_key_value_heads: Some(2),
        intermediate_size: Some(256),
        ..Default::default()
    };

    let device = Device::Cpu;
    let mut model = Qwen3Model::new(config, device, 1024).unwrap();

    let output = model.forward_logits(
        &[1],
        &[vec![42]],
        &[vec![0]],
        &[vec![0]],
        &[0],
        &[true],
    ).unwrap();

    assert_eq!(output.len(), 1);
    assert_eq!(output[0].len(), 1000);
    // Verify it's not all zeros
    assert!(output[0].iter().any(|&x| x != 0.0));
}
```

- [ ] **Step 5: Run tests**

```bash
cargo test -p vllm-model
```

- [ ] **Step 6: Commit**

```bash
git add crates/model/src/qwen3/model.rs
git commit -m "feat(model): implement forward_logits with cache"
```

---

## Phase 4: Cleanup

### Task 5: Remove Remaining Dead Code

**Files:**
- Modify: Multiple files across crates

- [ ] **Step 1: Address remaining dead code**

For each remaining location:
- `model/src/kv_cache.rs` - 3 instances
- `core/src/scheduler.rs` - 1 instance (`pending_tokens`)
- `model/src/qwen3/block.rs` - 1 instance
- `core/src/engine/batch.rs` - 1 instance
- `server/src/logging.rs` - 3 instances

For each:
1. Check if actually unused
2. If truly unused, remove
3. If still needed but warned, add proper doc comments

- [ ] **Step 2: Run full test suite**

```bash
cargo test --workspace
```

- [ ] **Step 3: Run clippy**

```bash
cargo clippy --workspace -- -D warnings
```

- [ ] **Step 4: Commit**

```bash
git add .
git commit -m "refactor: remove remaining dead code"
```

---

## Final Verification

### Run Full CI Check

```bash
just ci
```

Expected output:
- ✅ Format check passes
- ✅ Clippy passes with no warnings
- ✅ Documentation builds
- ✅ All tests pass

---

## Summary

| Task | Description | Risk |
|------|-------------|------|
| 1 | Fix random number generation | Low |
| 2 | Remove dead code in api.rs | Low |
| 3 | Deduplicate scheduler code | Low |
| 4 | Implement forward_logits | Medium |
| 5 | Remove remaining dead code | Low |

**Plan complete and saved to `docs/superpowers/plans/2026-04-03-code-review-improvements-plan.md`. Two execution options:**

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**