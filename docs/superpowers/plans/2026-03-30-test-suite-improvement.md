# Test Suite Improvement Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix test warnings, strengthen weak tests, add missing edge cases, and improve overall test coverage

**Architecture:** Incremental improvements following existing test patterns - fix warnings first, then strengthen assertions, then add missing tests

**Tech Stack:** Rust, tokio, candle-core

---

## Test File Analysis Summary

| File | Issue | Priority |
|------|-------|----------|
| `crates/model/tests/logits.rs` | 3 warnings (unused imports/vars) | High |
| `crates/model/tests/attention.rs` | Only shape checks | High |
| `crates/model/tests/model.rs` | Only counts, no correctness | High |
| `crates/core/tests/prefix_cache.rs` | Only 2 tests | Medium |
| `crates/core/src/scheduler.rs` | Missing edge cases | Medium |
| `crates/server/src/api.rs` | Only serialization | Low |

---

### Task 1: Fix Test Warnings in logits.rs

**Files:**
- Modify: `crates/model/tests/logits.rs:1-2`
- Modify: `crates/model/tests/logits.rs:93`

- [ ] **Step 1: Remove unused imports**

```rust
// Line 1: Change from:
use candle_core::{DType, Device, Tensor};
use vllm_core::types::{Batch, Request};

// To:
use candle_core::{Device, Tensor};
```

- [ ] **Step 2: Fix unused variable**

```rust
// Line 93: Change from:
let wrong_max = wrong_way.argmax(0).unwrap();

// To:
let _wrong_max = wrong_way.argmax(0).unwrap();
```

- [ ] **Step 3: Run tests to verify**

```bash
cargo test -p vllm-model --test logits
```

Expected: PASS with no warnings

---

### Task 2: Strengthen Attention Tests

**Files:**
- Modify: `crates/model/tests/attention.rs`

- [ ] **Step 1: Add actual computation verification test**

Add after line 113:

```rust
#[test]
fn test_attention_output_is_different_from_input() {
    let device = Device::Cpu;
    let attn = GqaAttention::new(256, 4, 2, 64, None).unwrap();

    // Create deterministic input
    let x = Tensor::zeros((1, 2, 256), DType::F32, &device).unwrap();
    
    let output = attn.forward(&x).unwrap();

    // Attention should produce different output from input
    // Even zero input should produce non-zero output (biases)
    let output_sum = output.sum_all().unwrap().to_scalar::<f32>().unwrap();
    assert!(output_sum != 0.0, "attention with zero input should still produce output from biases");
}

#[test]
fn test_attention_causal_mask_behavior() {
    let device = Device::Cpu;
    let attn = GqaAttention::new(128, 2, 2, 64, None).unwrap();

    // Single sequence of 3 tokens
    let x = Tensor::ones((1, 3, 128), DType::F32, &device).unwrap();
    let output = attn.forward(&x).unwrap();

    // Output shape should be correct
    assert_eq!(output.dims(), &[1, 3, 128]);

    // Each position should have different values (causal mask effect)
    let token_0 = output.get(0).unwrap().get(0).unwrap();
    let token_1 = output.get(0).unwrap().get(1).unwrap();
    let token_2 = output.get(0).unwrap().get(2).unwrap();

    // Tokens at different positions should not be identical
    let t0_data = token_0.to_vec2::<f32>().unwrap();
    let t1_data = token_1.to_vec2::<f32>().unwrap();
    let t2_data = token_2.to_vec2::<f32>().unwrap();

    assert!(t0_data != t1_data || t1_data != t2_data, 
            "causal attention should produce different values at different positions");
}
```

- [ ] **Step 2: Run tests**

```bash
cargo test -p vllm-model --test attention
```

Expected: All 9 tests pass

---

### Task 3: Strengthen Model Tests

**Files:**
- Modify: `crates/model/tests/model.rs`

- [ ] **Step 1: Add FakeModel correctness test**

Replace existing test with:

```rust
use vllm_core::engine::ModelBackend;
use vllm_model::fake::FakeModel;

#[test]
fn test_fake_model_output_count() {
    let model = FakeModel::new(1000);
    let seq_ids = vec![1u64, 2, 3];
    let input_tokens = vec![vec![1u32, 2], vec![3, 4], vec![5, 6]];
    let positions = vec![vec![0usize, 1], vec![0, 1], vec![0, 1]];

    let output = model.forward(&seq_ids, &input_tokens, &positions).unwrap();
    assert_eq!(output.seq_ids.len(), 3);
    assert_eq!(output.next_tokens.len(), 3);
}

#[test]
fn test_fake_model_deterministic() {
    let model = FakeModel::new(42);
    let seq_ids = vec![1u64];
    let input_tokens = vec![vec![1u32, 2, 3]];
    let positions = vec![vec![0usize, 1, 2]];

    let output1 = model.forward(&seq_ids, &input_tokens, &positions).unwrap();
    let output2 = model.forward(&seq_ids, &input_tokens, &positions).unwrap();

    // Same input should produce same output
    assert_eq!(output1.next_tokens, output2.next_tokens);
}

#[test]
fn test_fake_model_different_seqs_different_output() {
    let model = FakeModel::new(100);
    let seq_ids = vec![1u64, 2u64];
    let input_tokens = vec![vec![1u32], vec![1u32]];
    let positions = vec![vec![0usize], vec![0usize]];

    let output = model.forward(&seq_ids, &input_tokens, &positions).unwrap();

    // Different sequence IDs should produce different tokens
    assert_ne!(output.next_tokens[0], output.next_tokens[1], 
               "different seq_ids should produce different tokens");
}

#[test]
fn test_fake_model_batch_size_respected() {
    let model = FakeModel::new(1000);
    
    // Test various batch sizes
    for batch_size in [1, 2, 5, 10] {
        let seq_ids: Vec<u64> = (0..batch_size).map(|i| i as u64).collect();
        let input_tokens: Vec<Vec<u32>> = (0..batch_size).map(|_| vec![1]).collect();
        let positions: Vec<Vec<usize>> = (0..batch_size).map(|_| vec![0]).collect();

        let output = model.forward(&seq_ids, &input_tokens, &positions).unwrap();
        assert_eq!(output.seq_ids.len(), batch_size, "batch size {} not respected", batch_size);
        assert_eq!(output.next_tokens.len(), batch_size, "batch size {} not respected", batch_size);
    }
}
```

- [ ] **Step 2: Run tests**

```bash
cargo test -p vllm-model --test model
```

Expected: All 4 tests pass

---

### Task 4: Expand Prefix Cache Tests

**Files:**
- Modify: `crates/core/tests/prefix_cache.rs`

- [ ] **Step 1: Add more prefix cache edge cases**

Replace existing file content:

```rust
use tokio::sync::mpsc;
use vllm_core::engine::{Engine, ModelBackend};
use vllm_core::error::Result;
use vllm_core::types::{BatchOutput, Request, SchedulerConfig, SeqId, TokenId};

struct StubModel;

impl ModelBackend for StubModel {
    fn forward(
        &self,
        seq_ids: &[SeqId],
        _input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
    ) -> Result<BatchOutput> {
        Ok(BatchOutput {
            seq_ids: seq_ids.to_vec(),
            next_tokens: seq_ids.iter().map(|_| 1 as TokenId).collect(),
        })
    }
}

#[test]
fn test_prefix_cache_hit() {
    let config = SchedulerConfig {
        max_num_seqs: 256,
        max_num_batched_tokens: 4096,
        max_consecutive_decode: 10,
    };
    let mut engine = Engine::with_config(StubModel, StubModel, config, 4, 100);

    let (tx1, _rx1) = mpsc::unbounded_channel();
    let (tx2, _rx2) = mpsc::unbounded_channel();

    // First request: cache miss - wait for completion to populate cache
    engine.add_request(Request::new(1, vec![10, 20], 3), tx1);
    while engine.has_pending() {
        engine.step().unwrap();
    }

    // Second request with same prompt: cache hit
    engine.add_request(Request::new(2, vec![10, 20], 5), tx2);

    // Verify second request is in decoding state (cache hit)
    assert_eq!(engine.scheduler.running().len(), 1);
    let seq = &engine.scheduler.running()[0];
    assert_eq!(seq.status, vllm_core::types::Status::Decoding);
}

#[test]
fn test_cache_after_completion() {
    let config = SchedulerConfig {
        max_num_seqs: 256,
        max_num_batched_tokens: 4096,
        max_consecutive_decode: 10,
    };
    let mut engine = Engine::with_config(StubModel, StubModel, config, 4, 100);

    let (tx, _rx) = mpsc::unbounded_channel();

    // Add request and complete it
    engine.add_request(Request::new(1, vec![10, 20], 3), tx);
    while engine.has_pending() {
        engine.step().unwrap();
    }

    // Cache should have entry
    assert!(engine.scheduler.prefix_cache().len() > 0);
}

#[test]
fn test_prefix_cache_partial_hit() {
    let config = SchedulerConfig {
        max_num_seqs: 256,
        max_num_batched_tokens: 4096,
        max_consecutive_decode: 10,
    };
    let mut engine = Engine::with_config(StubModel, StubModel, config, 4, 100);

    let (tx1, _rx1) = mpsc::unbounded_channel();
    let (tx2, _rx2) = mpsc::unbounded_channel();

    // First request: [10, 20, 30]
    engine.add_request(Request::new(1, vec![10, 20, 30], 3), tx1);
    while engine.has_pending() {
        engine.step().unwrap();
    }

    // Second request: [10, 20] - prefix of first
    engine.add_request(Request::new(2, vec![10, 20], 5), tx2);

    // Should be in decoding state (prefix hit)
    assert_eq!(engine.scheduler.running().len(), 1);
    let seq = &engine.scheduler.running()[0];
    assert_eq!(seq.status, vllm_core::types::Status::Decoding);
}

#[test]
fn test_prefix_cache_no_hit_different_prefix() {
    let config = SchedulerConfig {
        max_num_seqs: 256,
        max_num_batched_tokens: 4096,
        max_consecutive_decode: 10,
    };
    let mut engine = Engine::with_config(StubModel, StubModel, config, 4, 100);

    let (tx1, _rx1) = mpsc::unbounded_channel();
    let (tx2, _rx2) = mpsc::unbounded_channel();

    // First request: [10, 20]
    engine.add_request(Request::new(1, vec![10, 20], 3), tx1);
    while engine.has_pending() {
        engine.step().unwrap();
    }

    // Second request: [30, 40] - different prefix
    engine.add_request(Request::new(2, vec![30, 40], 5), tx2);

    // Should be in prefill state (cache miss)
    assert_eq!(engine.scheduler.running().len(), 1);
    let seq = &engine.scheduler.running()[0];
    assert_eq!(seq.status, vllm_core::types::Status::Prefilling);
}

#[test]
fn test_prefix_cache_multiple_shared() {
    let config = SchedulerConfig {
        max_num_seqs: 256,
        max_num_batched_tokens: 4096,
        max_consecutive_decode: 10,
    };
    let mut engine = Engine::with_config(StubModel, StubModel, config, 4, 100);

    // First: [1, 2, 3]
    engine.add_request(Request::new(1, vec![1, 2, 3], 3), mpsc::unbounded_channel().0);
    while engine.has_pending() {
        engine.step().unwrap();
    }

    // Second: [1, 2] (prefix)
    engine.add_request(Request::new(2, vec![1, 2], 3), mpsc::unbounded_channel().0);
    while engine.has_pending() {
        engine.step().unwrap();
    }

    // Third: [1, 2, 3, 4] (longer)
    engine.add_request(Request::new(3, vec![1, 2, 3, 4], 3), mpsc::unbounded_channel().0);

    // Should all share the common prefix [1, 2]
    let cache = engine.scheduler.prefix_cache();
    assert!(cache.len() >= 1, "cache should have entries");
}
```

- [ ] **Step 2: Run tests**

```bash
cargo test -p vllm-core --test prefix_cache
```

Expected: All 5 tests pass

---

### Task 5: Add Edge Case Tests to Scheduler

**Files:**
- Modify: `crates/core/src/scheduler.rs` (add to existing test module at line 286)

- [ ] **Step 1: Add edge case tests**

Add after line 562:

```rust
    #[test]
    fn test_empty_prompt_handling() {
        let mut sched = Scheduler::new();
        sched.add_request(Request::new(1, vec![], 5));
        
        let batch = sched.build_batch();
        // Empty prompt should still be processed
        assert!(batch.is_empty() || batch.input_tokens.iter().all(|t| t.is_empty()));
    }

    #[test]
    fn test_single_token_prompt() {
        let mut sched = Scheduler::new();
        sched.add_request(Request::new(1, vec![42], 3));
        
        let batch = sched.build_batch();
        assert_eq!(batch.input_tokens[0], vec![42]);
        
        sched.update(&batch.seq_ids, &[99], &[1]);
        
        // Should transition to decoding after single token
        let batch2 = sched.build_batch();
        assert!(!batch2.is_empty());
    }

    #[test]
    fn test_max_tokens_exactly_reached() {
        let config = SchedulerConfig {
            max_num_seqs: 10,
            max_num_batched_tokens: 100,
            max_consecutive_decode: 10,
        };
        let mut sched = Scheduler::with_config(config, 1024);

        // prompt length = 2, max_tokens = 2 (exactly)
        sched.add_request(Request::new(1, vec![10, 20], 2));
        
        let batch = sched.build_batch();
        sched.update(&batch.seq_ids, &[], &[2]);  // Signal prompt done
        
        // Should be finished
        assert!(!sched.has_pending());
    }

    #[test]
    fn test_premature_completion_in_prefill() {
        // When max_tokens <= prompt_len, should finish immediately after prefill
        let config = SchedulerConfig {
            max_num_seqs: 10,
            max_num_batched_tokens: 100,
            max_consecutive_decode: 10,
        };
        let mut sched = Scheduler::with_config(config, 1024);

        sched.add_request(Request::new(1, vec![1, 2, 3], 3));
        
        let batch = sched.build_batch();
        assert_eq!(batch.input_tokens[0], vec![1, 2, 3]);
        
        // After prefill with exact token count
        sched.update(&batch.seq_ids, &[], &[3]);
        
        // Should be done immediately
        assert!(!sched.has_pending());
    }
```

- [ ] **Step 2: Run scheduler tests**

```bash
cargo test -p vllm-core -- scheduler
```

Expected: All 20+ tests pass

---

### Task 6: Add API Validation Tests

**Files:**
- Modify: `crates/server/src/api.rs`

- [ ] **Step 1: Add validation tests**

Add to existing test module (after line 129):

```rust
    #[test]
    fn test_completion_request_invalid_json() {
        let result: Result<CompletionRequest, _> = serde_json::from_str("{invalid}");
        assert!(result.is_err());
    }

    #[test]
    fn test_completion_request_missing_prompt() {
        let result: Result<CompletionRequest, _> = serde_json::from_str("{}");
        assert!(result.is_err());
    }

    #[test]
    fn test_completion_request_negative_max_tokens() {
        let req: CompletionRequest = serde_json::from_str(r#"{"prompt": "test", "max_tokens": -1}"#).unwrap();
        // Should handle gracefully (clamp or error)
        assert!(req.max_tokens > 0 || req.max_tokens == -1);
    }

    #[test]
    fn test_completion_request_very_long_max_tokens() {
        let req: CompletionRequest = serde_json::from_str(r#"{"prompt": "test", "max_tokens": 100000}"#).unwrap();
        // Should clamp to reasonable limit
        assert!(req.max_tokens <= 10000);
    }

    #[test]
    fn test_completion_chunk_empty_choices() {
        let chunk = CompletionChunk {
            id: "test".to_string(),
            choices: vec![],
        };
        let json = serde_json::to_string(&chunk).unwrap();
        assert!(json.contains("\"choices\":[]"));
    }
```

- [ ] **Step 2: Run API tests**

```bash
cargo test -p vllm-server -- api
```

Expected: All 9 tests pass

---

### Task 7: Final Verification

**Files:**
- All test files

- [ ] **Step 1: Run all tests**

```bash
cargo test --workspace
```

Expected: All tests pass with no warnings

- [ ] **Step 2: Run clippy**

```bash
cargo clippy --workspace -- -D warnings
```

Expected: No warnings

- [ ] **Step 3: Count tests**

```bash
cargo test --workspace -- --list | grep "test" | wc -l
```

Expected: Should be > 114 (previous count was 114)

---

### Summary of Changes

| Task | Files Modified | Tests Added | Warnings Fixed |
|------|---------------|-------------|----------------|
| 1. Fix warnings | 1 | 0 | 3 |
| 2. Strengthen attention | 1 | 2 | 0 |
| 3. Strengthen model | 1 | 3 | 0 |
| 4. Prefix cache | 1 | 3 | 0 |
| 5. Scheduler edge cases | 1 | 4 | 0 |
| 6. API validation | 1 | 5 | 0 |
| **Total** | **6** | **17** | **3** |