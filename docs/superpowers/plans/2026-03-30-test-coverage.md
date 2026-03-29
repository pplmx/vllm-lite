# vLLM-lite Test Coverage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended)

**Goal:** Add comprehensive test coverage for core components: Scheduler, Engine, Sampling, KV Cache, Prefix Cache.

**Architecture:** Expand unit tests in each crate, add integration tests, ensure edge cases covered.

**Tech Stack:** Rust, cargo test

**Spec:** This is a test coverage plan for existing functionality.

---

## Current Test Status

```
vllm-core: 16 tests
vllm-model: 0 tests  
vllm-server: 0 tests
Total: ~20 tests (many in core)
```

---

### Task TC-1: Scheduler Tests

**Files:**
- Modify: `crates/core/src/scheduler.rs`

- [ ] **Step 1: Add more scheduler tests**

Add to `crates/core/src/scheduler.rs`:

```rust
#[test]
fn test_multi_sequence_batch_order() {
    let mut sched = Scheduler::new();
    sched.add_request(Request::new(1, vec![10], 5));
    let batch1 = sched.build_batch();
    sched.update(&batch1.seq_ids, &[99]);
    
    sched.add_request(Request::new(2, vec![20, 30], 5));
    
    let batch = sched.build_batch();
    // Seq1 is decoding (1 token), Seq2 is prefill (2 tokens)
    // Decode should come first
    assert_eq!(batch.seq_ids[0], 1);
}

#[test]
fn test_max_batched_tokens_limit() {
    let config = SchedulerConfig {
        max_num_seqs: 10,
        max_num_batched_tokens: 2,  // Very small
    };
    let mut sched = Scheduler::with_config(config, 100);
    
    // Add sequence with 10 tokens prompt
    sched.add_request(Request::new(1, vec![1,2,3,4,5,6,7,8,9,10], 5));
    
    let batch = sched.build_batch();
    // Should only include 2 tokens due to budget
    let total_tokens: usize = batch.input_tokens.iter().map(|v| v.len()).sum();
    assert!(total_tokens <= 2);
}

#[test]
fn test_empty_prompt_request() {
    let mut sched = Scheduler::new();
    // Empty prompt - should still work
    sched.add_request(Request::new(1, vec![], 5));
    let batch = sched.build_batch();
    assert!(batch.is_empty() || batch.input_tokens[0].is_empty());
}

#[test]
fn test_finishes_at_exact_max_tokens() {
    let mut sched = Scheduler::new();
    sched.add_request(Request::new(1, vec![10, 20], 3)); // max_tokens = prompt + 1 = 3
    
    // prefill step
    let batch = sched.build_batch();
    sched.update(&batch.seq_ids, &[30]); // tokens: [10,20,30] = max_tokens
    
    assert!(!sched.has_pending());
    assert_eq!(sched.finished_sequences().len(), 1);
}
```

- [ ] **Step 2: Run tests**

```bash
cargo test -p vllm-core -- scheduler
```

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "test(core): add scheduler edge case tests"
```

---

### Task TC-2: Sampling Tests

**Files:**
- Modify: `crates/core/src/sampling.rs`

- [ ] **Step 1: Add more sampling tests**

Add to `crates/core/src/sampling.rs`:

```rust
#[test]
fn test_greedy_all_same_logit() {
    // All same logits - should return first (index 0)
    assert_eq!(greedy_sample(&[0.5, 0.5, 0.5]), 0);
}

#[test]
fn test_greedy_negative_logits() {
    // Negative logits should still work
    assert_eq!(greedy_sample(&[-1.0, -0.5, 0.0]), 2);
}

#[test]
fn test_greedy_large_vocab() {
    // Large vocabulary
    let mut logits = vec![0.0; 10000];
    logits[9999] = 1.0;
    assert_eq!(greedy_sample(&logits), 9999);
}

#[test]
fn test_sample_batch_empty() {
    assert_eq!(sample_batch(&[], 0.0), vec![]);
    assert_eq!(sample_batch(&[vec![]], 0.0), vec![0]);
}
```

- [ ] **Step 2: Run tests**

```bash
cargo test -p vllm-core -- sampling
```

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "test(core): add sampling edge case tests"
```

---

### Task TC-3: Engine Integration Tests

**Files:**
- Modify: `crates/core/src/engine.rs`

- [ ] **Step 1: Add engine tests**

Add to `crates/core/src/engine.rs`:

```rust
#[test]
fn test_engine_no_requests() {
    let mut engine = Engine::new(StubModel { token_to_return: 42 });
    let out = engine.step().unwrap();
    assert!(out.is_empty());
}

#[test]
fn test_engine_early_finish() {
    // Request with max_tokens=1 should finish after prefill
    let mut engine = Engine::new(StubModel { token_to_return: 42 });
    let (tx, _rx) = mpsc::unbounded_channel();
    
    engine.add_request(Request::new(1, vec![10], 2), tx);
    
    let out1 = engine.step().unwrap(); // prefill + token
    assert!(!out1.is_empty());
    
    let out2 = engine.step().unwrap(); // token 2 - finishes
    assert!(out2.is_empty() || !engine.has_pending());
}

#[test]
fn test_engine_max_tokens_zero() {
    // Edge case: max_tokens = 0
    let mut engine = Engine::new(StubModel { token_to_return: 42 });
    let (tx, _rx) = mpsc::unbounded_channel();
    
    engine.add_request(Request::new(1, vec![10], 0), tx);
    
    // Should finish immediately
    engine.step().unwrap();
    assert!(!engine.has_pending());
}
```

- [ ] **Step 2: Run tests**

```bash
cargo test -p vllm-core -- engine
```

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "test(core): add engine edge case tests"
```

---

### Task TC-4: Model Tests

**Files:**
- Create: `crates/model/src/lib.rs` tests (if needed)
- Create: `crates/model/tests/model.rs`

- [ ] **Step 1: Add model tests**

Create `crates/model/tests/model.rs`:

```rust
use vllm_model::fake::FakeModel;
use vllm_core::types::{Batch, BatchOutput, TokenId};

#[test]
fn test_fake_model_output_count() {
    let model = FakeModel::new(1000);
    let batch = Batch {
        seq_ids: vec![1, 2, 3],
        input_tokens: vec![vec![1,2], vec![3,4], vec![5,6]],
        positions: vec![vec![0,1], vec![0,1], vec![0,1]],
    };
    
    let output = model.forward(&batch).unwrap();
    
    assert_eq!(output.seq_ids.len(), 3);
    assert_eq!(output.next_tokens.len(), 3);
}

#[test]
fn test_fake_model_vocab_size() {
    let model = FakeModel::new(5000);
    let batch = Batch {
        seq_ids: vec![1],
        input_tokens: vec![vec![1]],
        positions: vec![vec![0]],
    };
    
    let output = model.forward(&batch).unwrap();
    // All tokens should be < 5000
    for &token in &output.next_tokens {
        assert!(token < 5000);
    }
}
```

- [ ] **Step 2: Run tests**

```bash
cargo test -p vllm-model
```

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "test(model): add FakeModel tests"
```

---

## Verification

```bash
# Full test suite
cargo test --workspace

# Count tests
cargo test --workspace -- --list | wc -l
```

## Target

- Increase test count from ~20 to ~40+
- Cover edge cases in scheduler, sampling, engine, model
- All tests pass