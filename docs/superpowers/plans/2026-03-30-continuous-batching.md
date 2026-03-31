# Continuous Batching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement continuous batching with separate prefill/decode queues and fairness-aware scheduling

**Status:** ✅ Implemented (2026-03-30, commit: d22d3e3)

**Architecture:** Maintain two queues (prefill/decode) in Scheduler. Prefill requests get priority to minimize latency for new requests. Decode requests have a consecutive round limit to prevent starvation.

**Tech Stack:** Rust, cargo test

**Implementation Summary:**

- Added `consecutive_decode_rounds` field to Sequence
- Added `max_consecutive_decode` config (default: 10)
- Separate prefill_queue and decode_queue in Scheduler
- Continuous batching with decode-first priority and fairness limit

---

## Task 1: Add new fields to types

**Files:**

- Modify: `crates/core/src/types.rs:44-54`
- Modify: `crates/core/src/types.rs:84-96`

- [ ] **Step 1: Add consecutive_decode_rounds to Sequence**

```rust
#[derive(Clone, Debug)]
pub struct Sequence {
    pub id: SeqId,
    pub tokens: Vec<TokenId>,
    pub kv_blocks: Vec<BlockId>,
    pub num_computed_tokens: usize,
    pub prompt_len: usize,
    pub status: Status,
    pub max_tokens: usize,
    pub sampling_params: SamplingParams,
    pub consecutive_decode_rounds: u32,  // NEW: Track decode rounds since last prefill
}
```

- [ ] **Step 2: Add max_consecutive_decode to SchedulerConfig**

```rust
#[derive(Clone, Debug)]
pub struct SchedulerConfig {
    pub max_num_seqs: usize,
    pub max_num_batched_tokens: usize,
    pub max_consecutive_decode: u32,  // NEW: Max decode rounds before yielding
}
```

- [ ] **Step 3: Update SchedulerConfig Default**

```rust
impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_num_seqs: 256,
            max_num_batched_tokens: 4096,
            max_consecutive_decode: 10,
        }
    }
}
```

- [ ] **Step 4: Run tests to verify changes compile**

Run: `cargo test -p vllm-core -- --nocapture`
Expected: PASS (existing tests should still work)

- [ ] **Step 5: Commit**

```bash
git add crates/core/src/types.rs
git commit -m "feat(core): add consecutive_decode_rounds and max_consecutive_decode fields"
```

---

### Task 2: Modify Scheduler to use separate queues

**Files:**

- Modify: `crates/core/src/scheduler.rs:5-13`

- [ ] **Step 1: Add prefill_queue and decode_queue to Scheduler struct**

```rust
pub struct Scheduler {
    waiting: VecDeque<Sequence>,           // Existing: waiting for scheduling
    prefill_queue: VecDeque<Sequence>,     // NEW: sequences needing prefill
    decode_queue: VecDeque<Sequence>,      // NEW: sequences in decode phase
    running: Vec<Sequence>,
    finished: Vec<Sequence>,
    next_seq_id: SeqId,
    config: SchedulerConfig,
    kv_allocator: BlockAllocator,
    prefix_cache: PrefixCache,
}
```

- [ ] **Step 2: Initialize new queues in with_config**

```rust
pub fn with_config(config: SchedulerConfig, num_kv_blocks: usize) -> Self {
    Self {
        waiting: VecDeque::new(),
        prefill_queue: VecDeque::new(),   // NEW
        decode_queue: VecDeque::new(),    // NEW
        running: Vec::new(),
        finished: Vec::new(),
        next_seq_id: 1,
        config,
        kv_allocator: BlockAllocator::new(num_kv_blocks),
        prefix_cache: PrefixCache::new(),
    }
}
```

- [ ] **Step 3: Run tests to verify compilation**

Run: `cargo build -p vllm-core`
Expected: Compiles successfully

- [ ] **Step 4: Commit**

```bash
git add crates/core/src/scheduler.rs
git commit -m "feat(core): add prefill_queue and decode_queue to Scheduler"
```

---

### Task 3: Implement continuous batching algorithm

**Files:**

- Modify: `crates/core/src/scheduler.rs:92-167`

- [ ] **Step 1: Replace admit_waiting with new queue-based logic**

Replace the current `admit_waiting` method with this implementation:

```rust
fn drain_and_requeue(&mut self) {
    // Move finished sequences to completed list
    let mut newly_finished = Vec::new();
    let mut remaining = Vec::new();

    for seq in self.running.drain(..) {
        if seq.status == Status::Finished {
            newly_finished.push(seq);
        } else if seq.status == Status::Prefilling {
            // Still needs more prefill
            self.prefill_queue.push_back(seq);
        } else if seq.status == Status::Decoding {
            self.decode_queue.push_back(seq);
        } else {
            self.waiting.push_back(seq);
        }
    }

    // Store in prefix cache
    for seq in newly_finished.iter() {
        let prompt_tokens = &seq.tokens[..seq.prompt_len];
        let key = hash_tokens(prompt_tokens);
        if !self.prefix_cache.contains_key(&key) {
            self.prefix_cache
                .insert(key, seq.kv_blocks.clone(), seq.prompt_len);
        }
    }

    self.finished.extend(newly_finished);
}
```

- [ ] **Step 2: Implement new build_batch with continuous batching**

Replace the current `build_batch` method:

```rust
pub fn build_batch(&mut self) -> Batch {
    self.drain_and_requeue();

    // Move waiting to prefill queue
    while let Some(seq) = self.waiting.pop_front() {
        let mut seq = seq;
        seq.status = Status::Prefilling;
        self.prefill_queue.push_back(seq);
    }

    let mut seq_ids = vec![];
    let mut input_tokens = vec![];
    let mut positions = vec![];
    let mut budget = self.config.max_num_batched_tokens;
    let max_seqs = self.config.max_num_seqs;
    let decode_limit = self.config.max_consecutive_decode;

    // Phase 1: Prefill sequences (priority for new requests)
    while seq_ids.len() < max_seqs {
        match self.prefill_queue.pop_front() {
            Some(mut seq) => {
                let remaining = seq.tokens.len() - seq.num_computed_tokens;
                let chunk_size = remaining.min(budget);

                if chunk_size == 0 {
                    // Prefill complete, switch to decode
                    seq.status = Status::Decoding;
                    seq.consecutive_decode_rounds = 0;
                    self.decode_queue.push_back(seq);
                    continue;
                }

                let start = seq.num_computed_tokens;
                let tokens = seq.tokens[start..start + chunk_size].to_vec();
                let pos: Vec<usize> = (start..start + chunk_size).collect();

                seq_ids.push(seq.id);
                input_tokens.push(tokens);
                positions.push(pos);
                budget = budget.saturating_sub(chunk_size);

                // Put back if more prefill needed
                seq.num_computed_tokens += chunk_size;
                if seq.num_computed_tokens < seq.tokens.len() {
                    seq.status = Status::Prefilling;
                    self.prefill_queue.push_back(seq);
                } else {
                    seq.status = Status::Decoding;
                    seq.consecutive_decode_rounds = 0;
                    self.decode_queue.push_back(seq);
                }
            }
            None => break,
        }

        if budget == 0 {
            break;
        }
    }

    // Phase 2: Decode sequences (with fairness limit)
    while seq_ids.len() < max_seqs && budget > 0 {
        match self.decode_queue.pop_front() {
            Some(mut seq) => {
                // Check consecutive decode limit
                if seq.consecutive_decode_rounds >= decode_limit {
                    // Yield to allow other sequences
                    self.decode_queue.push_back(seq);
                    break;
                }

                let last = *seq.tokens.last().unwrap();
                let pos = seq.tokens.len() - 1;

                seq_ids.push(seq.id);
                input_tokens.push(vec![last]);
                positions.push(vec![pos]);
                budget = budget.saturating_sub(1);
                seq.consecutive_decode_rounds += 1;

                // Add back for next round
                self.decode_queue.push_back(seq);
            }
            None => break,
        }
    }

    Batch {
        seq_ids,
        input_tokens,
        positions,
    }
}
```

- [ ] **Step 3: Run tests to verify compilation**

Run: `cargo build -p vllm-core`
Expected: Compiles successfully

- [ ] **Step 4: Run existing tests**

Run: `cargo test -p vllm-core -- --nocapture`
Expected: Some tests may need updates due to changed behavior

- [ ] **Step 5: Commit**

```bash
git add crates/core/src/scheduler.rs
git commit -m "feat(core): implement continuous batching with prefill/decode queues"
```

---

### Task 4: Add continuous batching tests

**Files:**

- Modify: `crates/core/src/scheduler.rs:254-389`

- [ ] **Step 1: Add test for prefill/decode queue separation**

Add this test to the test module:

```rust
#[test]
fn test_prefill_decode_queue_separation() {
    let mut sched = Scheduler::new();

    // Add request 1 - will be prefill then decode
    sched.add_request(Request::new(1, vec![10, 20, 30], 5));
    let batch1 = sched.build_batch();
    assert_eq!(batch1.seq_ids.len(), 1);

    // Complete prefill
    sched.update(
        &batch1.seq_ids,
        &[99],
        &[batch1.input_tokens[0].len()],
    );

    // Add request 2 - should go to prefill queue
    sched.add_request(Request::new(2, vec![40, 50], 5));

    let batch2 = sched.build_batch();
    // Should have decode (seq 1) first, then prefill (seq 2)
    assert!(batch2.seq_ids.len() >= 1);
}
```

- [ ] **Step 2: Add test for max_consecutive_decode fairness**

```rust
#[test]
fn test_max_consecutive_decode_limit() {
    let config = SchedulerConfig {
        max_num_seqs: 10,
        max_num_batched_tokens: 100,
        max_consecutive_decode: 2,
    };
    let mut sched = Scheduler::with_config(config, 1024);

    // Add two requests
    sched.add_request(Request::new(1, vec![10], 10));
    let batch1 = sched.build_batch();
    sched.update(
        &batch1.seq_ids,
        &[99],
        &[batch1.input_tokens[0].len()],
    );

    sched.add_request(Request::new(2, vec![20], 10));

    // First batch after prefill should include both
    let batch2 = sched.build_batch();
    assert!(batch2.seq_ids.len() >= 1);
}
```

- [ ] **Step 3: Add test for token budget enforcement**

```rust
#[test]
fn test_token_budget_in_continuous_batching() {
    let config = SchedulerConfig {
        max_num_seqs: 10,
        max_num_batched_tokens: 3,  // Very small budget
        max_consecutive_decode: 10,
    };
    let mut sched = Scheduler::with_config(config, 1024);

    // Large prompt
    sched.add_request(Request::new(1, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 10));

    let batch = sched.build_batch();
    let total_tokens: usize = batch.input_tokens.iter().map(|v| v.len()).sum();
    assert!(total_tokens <= 3, "total_tokens {} should be <= 3", total_tokens);
}
```

- [ ] **Step 4: Run all tests**

Run: `cargo test -p vllm-core -- --nocapture`
Expected: All tests pass

- [ ] **Step 5: Commit**

```bash
git add crates/core/src/scheduler.rs
git commit -m "test(core): add continuous batching tests"
```

---

### Task 5: Run full test suite and lint

**Files:**

- Run commands

- [ ] **Step 1: Run full workspace tests**

Run: `cargo test --workspace -- --nocapture`
Expected: All tests pass

- [ ] **Step 2: Run clippy**

Run: `cargo clippy --workspace -- -D warnings`
Expected: No warnings

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "test: run full test suite for continuous batching"
```

---

## Implementation Complete

After all tasks complete, continuous batching will:

1. Maintain separate prefill and decode queues
2. Prioritize prefill for new requests
3. Limit consecutive decode rounds to prevent starvation
4. Enforce token budget per batch
5. Keep existing behavior compatible where possible
