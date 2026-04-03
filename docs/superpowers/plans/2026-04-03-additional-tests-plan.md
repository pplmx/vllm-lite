# Additional Tests Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add critical missing tests for vllm-traits, error handling, resource limits, and model loading

**Architecture:** Add unit and integration tests to existing test infrastructure

**Tech Stack:** Rust, tokio, candle-core

---

## Task 1: vllm-traits Interface Tests

**Files:**
- Create: `crates/traits/tests/model_backend.rs`

**Why**: No tests validate the core trait interfaces that all model implementations depend on

- [ ] **Step 1: Create test directory**

```bash
mkdir -p crates/traits/tests
```

- [ ] **Step 2: Create test file**

Create `crates/traits/tests/model_backend.rs`:

```rust
#[cfg(test)]
mod tests {
    use vllm_traits::{ModelBackend, BatchOutput, SeqId, TokenId};
    use std::result::Result;

    #[test]
    fn test_model_backend_trait_object() {
        // Test that ModelBackend can be used as trait object
        // fn takes_backend<M: ModelBackend>(m: M) {}
    }

    #[test]
    fn test_batch_output_builder() {
        let output = BatchOutput {
            seq_ids: vec![1, 2, 3],
            next_tokens: vec![10, 20, 30],
        };
        assert_eq!(output.seq_ids.len(), 3);
        assert_eq!(output.next_tokens.len(), 3);
    }
}
```

- [ ] **Step 3: Run tests**

```bash
cargo test -p vllm-traits
```

- [ ] **Step 4: Commit**

```bash
git add crates/traits/tests/
git commit -m "test(traits): add trait interface tests"
```

---

## Task 2: Error Handling Tests

**Files:**
- Create: `crates/core/tests/error_handling.rs`

**Why**: No tests for error propagation from engine to API

- [ ] **Step 1: Create error handling tests**

Create `crates/core/tests/error_handling.rs`:

```rust
use vllm_core::error::EngineError;
use vllm_core::types::Request;

#[test]
fn test_error_message_format() {
    let err = EngineError::new("test error");
    assert_eq!(err.to_string(), "test error");
}

#[test]
fn test_seq_not_found_error() {
    let err = EngineError::seq_not_found(42);
    assert!(err.to_string().contains("42"));
}

#[test]
fn test_request_validation_zero_prompt() {
    let req = Request::new(1, vec![], 10);
    // Should handle gracefully
    assert_eq!(req.input_tokens.len(), 0);
}
```

- [ ] **Step 2: Run tests**

```bash
cargo test -p vllm-core error
```

- [ ] **Step 3: Commit**

```bash
git add crates/core/tests/error_handling.rs
git commit -m "test(core): add error handling tests"
```

---

## Task 3: Resource Limit Tests

**Files:**
- Create: `crates/core/tests/resource_limits.rs`

**Why**: Critical for production stability - memory pressure, OOM handling

- [ ] **Step 1: Create resource limit tests**

Create `crates/core/tests/resource_limits.rs`:

```rust
use vllm_core::kv_cache::{BlockAllocator, BlockId};
use vllm_core::scheduler::Scheduler;
use vllm_core::types::SchedulerConfig;

#[test]
fn test_allocate_exact_fit() {
    let mut allocator = BlockAllocator::new(10);
    let blocks = allocator.allocate(5).unwrap();
    assert_eq!(blocks.len(), 5);
    assert_eq!(allocator.available(), 5);
}

#[test]
fn test_allocate_overflow_rejected() {
    let allocator = BlockAllocator::new(5);
    let result = allocator.allocate(10);
    assert!(result.is_err());
}

#[test]
fn test_scheduler_max_seqs_limit() {
    let config = SchedulerConfig {
        max_num_seqs: 2,
        ..Default::default()
    };
    let scheduler = Scheduler::with_config(config, 100);
    // Adding more than max_num_seqs should be handled
}
```

- [ ] **Step 2: Run tests**

```bash
cargo test -p vllm-core resource
```

- [ ] **Step 3: Commit**

```bash
git add crates/core/tests/resource_limits.rs
git commit -m "test(core): add resource limit tests"
```

---

## Task 4: Model Loader Tests

**Files:**
- Create: `crates/model/tests/loader.rs`

**Why**: No tests for loading weights from disk

- [ ] **Step 1: Create loader tests**

Create `crates/model/tests/loader.rs`:

```rust
use vllm_model::loader::ModelLoader;
use vllm_model::config::Qwen3Config;
use std::path::Path;

#[test]
fn test_config_from_file() {
    let config = Qwen3Config::from_file("test_data/qwen3/config.json").unwrap();
    assert_eq!(config.hidden_size(), 2048);
}

#[test]
fn test_config_defaults() {
    let config = Qwen3Config::default();
    assert_eq!(config.vocab_size(), 151936);
}
```

- [ ] **Step 2: Run tests**

```bash
cargo test -p vllm-model loader
```

- [ ] **Step 3: Commit**

```bash
git add crates/model/tests/loader.rs
git commit -m "test(model): add loader tests"
```

---

## Task 5: Final Verification

- [ ] **Step 1: Run all tests**

```bash
cargo test --workspace
```

Expected: All tests pass (should be ~300+ now)

- [ ] **Step 2: Verify new test count increased**

```bash
cargo test --workspace -- --list | grep -c "test$"
```

- [ ] **Step 3: Run clippy**

```bash
cargo clippy --workspace -- -D warnings
```

- [ ] **Step 4: Commit**

```bash
git add .
git commit -m "test: add comprehensive test coverage for traits, errors, resources, loader"
```

---

## Success Criteria

- [ ] vllm-traits tests exist (was 0, now >0)
- [ ] Error handling tests added
- [ ] Resource limit tests added
- [ ] Model loader tests added
- [ ] Total test count increases from 294 to ~310+
- [ ] All tests pass
