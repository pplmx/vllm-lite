# Test Architecture Improvement Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve test coverage by adding missing tests, increasing edge case coverage, and adding concurrency tests

**Architecture:** Add integration tests for server auth, edge case tests for core modules, and concurrency tests for engine

**Tech Stack:** Rust, tokio::test, assert crate

---

## Task 1: Add Auth/RateLimiter Tests

**Files:**
- Create: `crates/server/tests/auth.rs`
- Modify: `crates/server/src/auth.rs` (add test helpers)

- [ ] **Step 1: Write failing tests for RateLimiter**

```rust
use vllm_server::auth::{RateLimiter, AuthMiddleware};
use std::time::{Duration, Instant};
use std::sync::Arc;
use tokio::sync::RwLock;

#[tokio::test]
async fn test_rate_limiter_allows_within_limit() {
    let mut limiter = RateLimiter::new(3, 60);
    assert!(limiter.check_rate_limit("key1").await);
    assert!(limiter.check_rate_limit("key1").await);
    assert!(limiter.check_rate_limit("key1").await);
}

#[tokio::test]
async fn test_rate_limiter_blocks_over_limit() {
    let mut limiter = RateLimiter::new(2, 60);
    assert!(limiter.check_rate_limit("key1").await);
    assert!(limiter.check_rate_limit("key1").await);
    assert!(!limiter.check_rate_limit("key1").await);
}

#[tokio::test]
async fn test_rate_limiter_separate_keys() {
    let mut limiter = RateLimiter::new(1, 60);
    assert!(limiter.check_rate_limit("key1").await);
    assert!(limiter.check_rate_limit("key2").await);
}

#[tokio::test]
async fn test_rate_limiter_window_expiry() {
    let mut limiter = RateLimiter::new(1, 0);
    assert!(limiter.check_rate_limit("key1").await);
    tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
    assert!(limiter.check_rate_limit("key1").await);
}

#[tokio::test]
async fn test_auth_middleware_valid_key() {
    let auth = AuthMiddleware::new(
        vec!["test_key".to_string()],
        10,
        60
    );
    let mut headers = HeaderMap::new();
    headers.insert(AUTHORIZATION, "Bearer test_key".parse().unwrap());
    let result = auth.verify(&headers).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_auth_middleware_invalid_key() {
    let auth = AuthMiddleware::new(
        vec!["test_key".to_string()],
        10,
        60
    );
    let mut headers = HeaderMap::new();
    headers.insert(AUTHORIZATION, "Bearer wrong_key".parse().unwrap());
    let result = auth.verify(&headers).await;
    assert_eq!(result.unwrap_err(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn test_auth_middleware_no_key() {
    let auth = AuthMiddleware::new(
        vec!["test_key".to_string()],
        10,
        60
    );
    let headers = HeaderMap::new();
    let result = auth.verify(&headers).await;
    assert_eq!(result.unwrap_err(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn test_auth_middleware_no_keys_allow_all() {
    let auth = AuthMiddleware::new(vec![], 10, 60);
    let mut headers = HeaderMap::new();
    headers.insert(AUTHORIZATION, "Bearer any_key".parse().unwrap());
    let result = auth.verify(&headers).await;
    assert!(result.is_ok());
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p vllm-server auth -- --nocapture`
Expected: Test file not found (need to create it)

- [ ] **Step 3: Create test file with imports**

```rust
// crates/server/tests/auth.rs
use vllm_server::auth::{AuthMiddleware, RateLimiter};
use axum::http::header::AUTHORIZATION;
use axum::http::HeaderMap;
use axum::http::StatusCode;
use std::sync::Arc;
use tokio::sync::RwLock;

#[tokio::test]
async fn test_rate_limiter_allows_within_limit() {
    let mut limiter = RateLimiter::new(3, 60);
    assert!(limiter.check_rate_limit("key1").await);
    assert!(limiter.check_rate_limit("key1").await);
    assert!(limiter.check_rate_limit("key1").await);
}
// ... rest of tests
```

- [ ] **Step 4: Add test helpers to auth.rs**

Modify: `crates/server/src/auth.rs`

Add test module at end:
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{sleep, Duration};
    
    #[tokio::test]
    async fn test_rate_limiter_allows_within_limit() {
        let mut limiter = RateLimiter::new(3, 60);
        assert!(limiter.check_rate_limit("key1").await);
        assert!(limiter.check_rate_limit("key1").await);
        assert!(limiter.check_rate_limit("key1").await);
    }
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test -p vllm-server -- auth --nocapture`
Expected: All 8 tests pass

- [ ] **Step 6: Commit**

```bash
git add crates/server/src/auth.rs crates/server/tests/auth.rs
git commit -m "test(server): add auth and rate limiter tests"
```

---

## Task 2: Add Error Path Tests for Scheduler

**Files:**
- Modify: `crates/core/src/scheduler.rs`

- [ ] **Step 1: Write failing tests for error paths**

In scheduler.rs, add to #[cfg(test)] module:
```rust
#[test]
fn test_add_request_zero_prompt() {
    let mut sched = Scheduler::new();
    let id = sched.add_request(Request::new(1, vec![], 5));
    assert_eq!(id, 1);
    assert!(sched.has_pending());
}

#[test]
fn test_add_request_duplicate_id() {
    let mut sched = Scheduler::new();
    sched.add_request(Request::new(1, vec![1, 2], 5));
    let id = sched.add_request(Request::new(1, vec![3, 4], 5));
    // Should use new ID since 1 is taken
    assert_eq!(id, 2);
}

#[test]
fn test_build_batch_empty() {
    let sched = Scheduler::new();
    let batch = sched.build_batch();
    assert!(batch.is_empty());
}

#[test]
fn test_update_nonexistent_seq() {
    let mut sched = Scheduler::new();
    sched.update(&[999], &[1], &[1]); // Non-existent seq
    // Should not panic, just ignore
    assert!(!sched.has_pending());
}

#[test]
fn test_running_after_all_finished() {
    let config = SchedulerConfig {
        max_num_seqs: 10,
        max_num_batched_tokens: 100,
        max_consecutive_decode: 10,
        enable_pd_separation: false,
        prefill_chunk_size: 512,
        decode_preference_ratio: 0.7,
        enable_priority_scheduling: false,
        enable_dynamic_batching: false,
        min_batch_size: 1,
        max_batch_size: 256,
    };
    let mut sched = Scheduler::with_config(config, 10);
    sched.add_request(Request::new(1, vec![1], 1));
    let batch = sched.build_batch();
    sched.update(&batch.seq_ids, &[99], &[batch.input_tokens[0].len()]);
    // Now has 1 token, max_tokens=1, should finish
    let batch2 = sched.build_batch();
    // Should be empty or have next token
    assert!(!sched.has_pending() || !batch2.is_empty());
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p vllm-core scheduler -- --test-threads=1`
Expected: Some tests may fail if edge cases not handled

- [ ] **Step 3: Fix any failing tests**

Add proper handling for edge cases:
- Empty prompt: ensure prompt_len = 0 works
- Duplicate ID: ensure next_seq_id increments properly

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p vllm-core scheduler::tests -- --nocapture`

- [ ] **Step 5: Commit**

```bash
git add crates/core/src/scheduler.rs
git commit -m "test(core): add scheduler error path tests"
```

---

## Task 3: Add Engine Concurrency Tests

**Files:**
- Modify: `crates/core/tests/integration.rs`

- [ ] **Step 1: Write failing tests for concurrency**

Add to integration.rs:
```rust
#[tokio::test]
async fn test_concurrent_requests_different_prompts() {
    let (tx1, mut rx1) = mpsc::unbounded_channel();
    let (tx2, mut rx2) = mpsc::unbounded_channel();
    let (tx3, mut rx3) = mpsc::unbounded_channel();
    
    let mut engine = Engine::new(StubModel, StubModel);
    
    engine.add_request(Request::new(0, vec![1, 2, 3], 5), tx1);
    engine.add_request(Request::new(0, vec![4, 5], 5), tx2);
    engine.add_request(Request::new(0, vec![6, 7, 8, 9], 5), tx3);
    
    // Run multiple steps
    for _ in 0..10 {
        let _ = engine.step();
    }
    
    // Collect outputs
    let mut count = 0;
    loop {
        tokio::select! {
            Some(token) = rx1.recv() => count += 1,
            Some(token) = rx2.recv() => count += 1,
            Some(token) = rx3.recv() => count += 1,
            else => break,
        }
    }
    
    assert!(count > 0, "Should have generated tokens");
}

#[tokio::test]
async fn test_rapid_request_addition() {
    let mut engine = Engine::new(StubModel, StubModel);
    let mut txs = Vec::new();
    
    // Add 10 requests rapidly
    for i in 0..10 {
        let (tx, _rx) = mpsc::unbounded_channel();
        engine.add_request(Request::new(0, vec![i as TokenId], 3), tx);
        txs.push(tx);
    }
    
    // Process
    for _ in 0..5 {
        let _ = engine.step();
    }
    
    // All should complete without panic
    assert!(!engine.scheduler.has_pending() || engine.scheduler.running_count() > 0);
}

#[tokio::test]
async fn test_request_cancellation() {
    let (tx1, mut rx1) = mpsc::unbounded_channel();
    let (tx2, _rx2) = mpsc::unbounded_channel();
    
    let mut engine = Engine::new(StubModel, StubModel);
    
    engine.add_request(Request::new(0, vec![1, 2], 10), tx1);
    engine.add_request(Request::new(0, vec![3, 4], 10), tx2);
    
    // Drop tx2 to simulate cancellation
    drop(tx2);
    
    // Run a few steps
    for _ in 0..3 {
        let _ = engine.step();
    }
    
    // Engine should still work
    let batch = engine.scheduler.build_batch();
    assert!(!engine.has_pending() || batch.seq_ids.len() > 0);
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p vllm-core integration -- --nocapture`
Expected: Tests fail (not implemented yet)

- [ ] **Step 3: Fix issues if any**

The tests should pass with current implementation if properly structured

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p vllm-core test_concurrent -- --nocapture`

- [ ] **Step 5: Commit**

```bash
git add crates/core/tests/integration.rs
git commit -m "test(core): add engine concurrency tests"
```

---

## Task 4: Add More Edge Case Tests for Model

**Files:**
- Modify: `crates/model/tests/model.rs`

- [ ] **Step 1: Write failing tests for edge cases**

```rust
#[test]
fn test_model_empty_batch() -> Result<()> {
    let model = FakeModel::new(256, 4, 2, 64);
    let result = model.forward(&[], &[], &[])?;
    assert!(result.next_tokens.is_empty());
    Ok(())
}

#[test]
fn test_model_single_token_batch() -> Result<()> {
    let model = FakeModel::new(256, 4, 2, 64);
    let result = model.forward(
        &[1],
        &[vec![42]],
        &[vec![0]],
    )?;
    assert_eq!(result.next_tokens.len(), 1);
    Ok(())
}

#[test]
fn test_model_large_batch() -> Result<()> {
    let model = FakeModel::new(256, 4, 2, 64);
    let batch_size = 32;
    let seq_ids: Vec<u64> = (0..batch_size).map(|i| i as u64).collect();
    let tokens: Vec<Vec<u32>> = (0..batch_size).map(|i| vec![i as u32]).collect();
    let positions: Vec<Vec<usize>> = (0..batch_size).map(|_| vec![0]).collect();
    
    let result = model.forward(&seq_ids, &tokens, &positions)?;
    assert_eq!(result.next_tokens.len(), batch_size);
    Ok(())
}
```

- [ ] **Step 2: Run tests**

Run: `cargo test -p vllm-model model -- --nocapture`

- [ ] **Step 3: Commit**

```bash
git add crates/model/tests/model.rs
git commit -m "test(model): add edge case tests"
```

---

## Task 5: Add Prefix Cache Stress Test

**Files:**
- Modify: `crates/core/tests/prefix_cache.rs`

- [ ] **Step 1: Write stress test**

```rust
#[test]
fn test_prefix_cache_high_volume() {
    let mut sched = Scheduler::with_config(
        SchedulerConfig::default(),
        100,
    );
    
    // Add 50 different requests
    for i in 0..50 {
        let tokens: Vec<TokenId> = (0..10).map(|j| (i * 100 + j) as TokenId).collect();
        sched.add_request(Request::new(0, tokens, 5), mpsc::unbounded_channel().0);
    }
    
    // Process all
    for _ in 0..20 {
        let batch = sched.build_batch();
        if batch.is_empty() {
            break;
        }
        let next_tokens: Vec<TokenId> = batch.seq_ids.iter().map(|_| 1).collect();
        let counts: Vec<usize> = batch.input_tokens.iter().map(|t| t.len()).collect();
        sched.update(&batch.seq_ids, &next_tokens, &counts);
    }
    
    // All should be finished
    assert!(!sched.has_pending());
}

#[test]
fn test_prefix_cache_many_sequences_same_prefix() {
    let common_prefix = vec![1, 2, 3, 4, 5];
    
    let mut sched = Scheduler::with_config(
        SchedulerConfig::default(),
        50,
    );
    
    // Add 10 requests with same prefix, different completions
    for i in 0..10 {
        let mut tokens = common_prefix.clone();
        tokens.push(i as TokenId);
        sched.add_request(Request::new(0, tokens, 3), mpsc::unbounded_channel().0);
    }
    
    let batch = sched.build_batch();
    assert!(!batch.is_empty());
}
```

- [ ] **Step 2: Run tests**

Run: `cargo test -p vllm-core prefix_cache -- --nocapture`

- [ ] **Step 3: Commit**

```bash
git add crates/core/tests/prefix_cache.rs
git commit -m "test(core): add prefix cache stress tests"
```

---

## Execution Summary

| Task | Tests Added | Priority |
|------|-------------|----------|
| Auth/RateLimiter | 8 | High |
| Scheduler Error Paths | 5 | High |
| Engine Concurrency | 3 | High |
| Model Edge Cases | 3 | Medium |
| Prefix Cache Stress | 2 | Medium |
| **Total** | **21** | |

After all tasks: ~251 tests total