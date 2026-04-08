# Hybrid Event-Driven Scheduler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement hybrid event-driven scheduler with observer pattern while maintaining direct call performance.

**Architecture:** 
- Phase 1: Create observer infrastructure (trait, registry, event enum)
- Phase 2: Integrate observer dispatch into scheduler methods
- Phase 3: Remove ActionExecutor's duplicate BlockAllocator
- Phase 4: Deprecate old event system

**Tech Stack:** Rust, tokio, vllm-core crate

---

## Background

This plan implements the design from `docs/superpowers/specs/2026-04-09-hybrid-event-scheduler-design.md`

Key files to understand:
- `crates/core/src/scheduler/engine.rs` - Main scheduler (modify)
- `crates/core/src/scheduler/action_executor.rs` - Simplify
- `crates/core/src/scheduler/event_handler.rs` - Deprecate
- `crates/core/src/scheduler/mod.rs` - Add observer module

---

## Task 1: Create Observer Infrastructure

**Files:**
- Create: `crates/core/src/scheduler/observer.rs`
- Modify: `crates/core/src/scheduler/mod.rs`
- Test: `crates/core/tests/observer.rs`

- [ ] **Step 1: Create observer.rs with trait and enum**

```rust
// crates/core/src/scheduler/observer.rs

use crate::types::{SeqId, TokenId};
use std::sync::RwLock;

pub trait SchedulerObserver: Send + Sync {
    fn on_request_arrived(&self, seq_id: SeqId, prompt_len: usize);
    fn on_batch_scheduled(&self, seq_ids: &[SeqId], batch_size: usize);
    fn on_decoding(&self, seq_id: SeqId, token: TokenId);
    fn on_sequence_finished(&self, seq_id: SeqId, total_tokens: usize);
    fn on_preemption(&self, seq_id: SeqId, reason: &str);
    fn on_memory_pressure(&self, available_blocks: usize);
}

pub enum ObserverEvent {
    RequestArrived { seq_id: SeqId, prompt_len: usize },
    BatchScheduled { seq_ids: Vec<SeqId>, batch_size: usize },
    Decoding { seq_id: SeqId, token: TokenId },
    SequenceFinished { seq_id: SeqId, total_tokens: usize },
    Preemption { seq_id: SeqId, reason: String },
    MemoryPressure { available_blocks: usize },
}

pub struct SchedulerObservers {
    observers: RwLock<Vec<Box<dyn SchedulerObserver>>>,
}

impl SchedulerObservers {
    pub fn new() -> Self {
        Self { observers: RwLock::new(Vec::new()) }
    }
    
    pub const MAX_OBSERVERS: usize = 16;
    
    pub fn register(&self, observer: Box<dyn SchedulerObserver>) -> Result<(), String> {
        let mut guards = self.observers.write().map_err(|e| e.to_string())?;
        if guards.len() >= Self::MAX_OBSERVERS {
            return Err("Max observers reached".to_string());
        }
        guards.push(observer);
        Ok(())
    }
    
    pub fn dispatch(&self, event: &ObserverEvent) {
        if let Ok(observers) = self.observers.read() {
            for observer in observers.iter() {
                let _ = std::panic::catch_unwind(|| self.notify_one(observer.as_ref(), event));
            }
        }
    }
    
    fn notify_one(&self, observer: &dyn SchedulerObserver, event: &ObserverEvent) {
        match event {
            ObserverEvent::RequestArrived { seq_id, prompt_len } => {
                observer.on_request_arrived(*seq_id, *prompt_len);
            }
            ObserverEvent::BatchScheduled { seq_ids, batch_size } => {
                observer.on_batch_scheduled(seq_ids, *batch_size);
            }
            ObserverEvent::Decoding { seq_id, token } => {
                observer.on_decoding(*seq_id, *token);
            }
            ObserverEvent::SequenceFinished { seq_id, total_tokens } => {
                observer.on_sequence_finished(*seq_id, *total_tokens);
            }
            ObserverEvent::Preemption { seq_id, reason } => {
                observer.on_preemption(*seq_id, reason);
            }
            ObserverEvent::MemoryPressure { available_blocks } => {
                observer.on_memory_pressure(*available_blocks);
            }
        }
    }
}

impl Default for SchedulerObservers {
    fn default() -> Self {
        Self::new()
    }
}
```

- [ ] **Step 2: Add observer module to mod.rs**

```rust
// In crates/core/src/scheduler/mod.rs, add:
pub mod observer;
pub use observer::{SchedulerObserver, SchedulerObservers, ObserverEvent};
```

- [ ] **Step 3: Write test for observer registration**

```rust
// crates/core/tests/observer.rs

use vllm_core::scheduler::{SchedulerEngine, SchedulerObservers, SchedulerObserver, ObserverEvent};
use vllm_core::types::Request;
use std::sync::Mutex;

struct TestObserver {
    events: Mutex<Vec<ObserverEvent>>,
}

impl TestObserver {
    fn new() -> Self {
        Self { events: Mutex::new(Vec::new()) }
    }
}

impl SchedulerObserver for TestObserver {
    fn on_request_arrived(&self, seq_id: u64, prompt_len: usize) {
        self.events.lock().unwrap().push(
            ObserverEvent::RequestArrived { seq_id, prompt_len }
        );
    }
    
    fn on_sequence_finished(&self, seq_id: u64, total_tokens: usize) {
        self.events.lock().unwrap().push(
            ObserverEvent::SequenceFinished { seq_id, total_tokens }
        );
    }
}

#[test]
fn test_observer_registration() {
    let observers = SchedulerObservers::new();
    let observer = Box::new(TestObserver::new());
    assert!(observers.register(observer).is_ok());
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p vllm-core observer -- --nocapture`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/core/src/scheduler/observer.rs crates/core/src/scheduler/mod.rs
git commit -m "feat(scheduler): add observer infrastructure"
```

---

## Task 2: Integrate Observers into SchedulerEngine

**Files:**
- Modify: `crates/core/src/scheduler/engine.rs`

- [ ] **Step 1: Add observers field to SchedulerEngine**

```rust
// In engine.rs, add to SchedulerEngine struct:
observers: SchedulerObservers,
```

Update `new()`:
```rust
observers: SchedulerObservers::new(),
```

- [ ] **Step 2: Add register_observer method**

```rust
pub fn register_observer(&self, observer: Box<dyn SchedulerObserver>) -> Result<(), String> {
    self.observers.register(observer)
}
```

- [ ] **Step 3: Add dispatch_observer_event helper**

```rust
fn dispatch_observer_event(&self, event: ObserverEvent) {
    self.observers.dispatch(&event);
}
```

- [ ] **Step 4: Dispatch RequestArrived in add_request**

After `self.queue_manager.enqueue(seq, priority);` (around line 271), add:
```rust
self.dispatch_observer_event(ObserverEvent::RequestArrived {
    seq_id,
    prompt_len,
});
```

- [ ] **Step 5: Dispatch BatchScheduled in build_batch**

After building batch, before return Batch (around line 349):
```rust
self.dispatch_observer_event(ObserverEvent::BatchScheduled {
    seq_ids: seq_ids.clone(),
    batch_size: seq_ids.len(),
});
```

- [ ] **Step 6: Dispatch events in update**

After processing each token, add:
```rust
self.dispatch_observer_event(ObserverEvent::Decoding {
    seq_id: *seq_id,
    token,
});
```

After sequence finished:
```rust
self.dispatch_observer_event(ObserverEvent::SequenceFinished {
    seq_id: finished_seq.id,
    total_tokens: finished_seq.tokens.len(),
});
```

- [ ] **Step 7: Dispatch Preemption in execute_preemption**

After preempting a sequence (around line 315):
```rust
self.dispatch_observer_event(ObserverEvent::Preemption {
    seq_id,
    reason: "MemoryPressure".to_string(),
});
```

- [ ] **Step 8: Dispatch MemoryPressure in update**

When allocation fails (around line 467):
```rust
self.dispatch_observer_event(ObserverEvent::MemoryPressure {
    available_blocks: self.kv_allocator.available(),
});
```

- [ ] **Step 9: Run tests**

Run: `cargo test -p vllm-core -- --nocapture`
Expected: All tests pass

- [ ] **Step 10: Commit**

```bash
git add crates/core/src/scheduler/engine.rs
git commit -m "feat(scheduler): integrate observers into scheduler"
```

---

## Task 3: Remove ActionExecutor's Independent Allocator

**Files:**
- Modify: `crates/core/src/scheduler/action_executor.rs`
- Modify: `crates/core/src/scheduler/engine.rs` (remove field)

- [ ] **Step 1: Read current ActionExecutor implementation**

```bash
cat crates/core/src/scheduler/action_executor.rs
```

- [ ] **Step 2: Simplify ActionExecutor**

Replace the entire file:
```rust
use super::actions::Action;
use super::events::SchedulerEvent;

/// Simplified ActionExecutor - no longer manages resources
/// Kept for backwards compatibility with old event system
pub struct ActionExecutor {
    _placeholder: (),
}

impl ActionExecutor {
    pub fn new(_num_blocks: usize) -> Self {
        Self { _placeholder: () }
    }
    
    pub fn execute(&mut self, action: Action) -> Vec<SchedulerEvent> {
        // Legacy: return empty, real handling done by SchedulerEngine
        match action {
            Action::Preempt { seq_id, reason } => {
                vec![SchedulerEvent::Preempt {
                    seq_id,
                    reason: format!("{:?}", reason),
                }]
            }
            Action::Finish { seq_id } => {
                vec![SchedulerEvent::SequenceFinished { seq_id }]
            }
            _ => vec![],
        }
    }
    
    #[allow(dead_code)]
    pub fn available_blocks(&self) -> usize {
        0  // No longer valid
    }
    
    #[allow(dead_code)]
    pub fn total_blocks(&self) -> usize {
        0  // No longer valid
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_execute_returns_events() {
        let mut executor = ActionExecutor::new(1024);
        let events = executor.execute(Action::Finish { seq_id: 1 });
        assert!(!events.is_empty());
    }
}
```

- [ ] **Step 3: Verify ActionExecutor is still used**

```bash
grep -n "action_executor" crates/core/src/scheduler/engine.rs
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p vllm-core -- --nocapture`
Expected: All tests pass

- [ ] **Step 5: Commit**

```bash
git add crates/core/src/scheduler/action_executor.rs
git commit -m "refactor(scheduler): simplify ActionExecutor, remove duplicate allocator"
```

---

## Task 4: Deprecate Old Event System

**Files:**
- Modify: `crates/core/src/scheduler/event_handler.rs`
- Modify: `crates/core/src/scheduler/events.rs`
- Modify: `crates/core/src/scheduler/actions.rs`

- [ ] **Step 1: Add deprecation warnings**

In event_handler.rs, add:
```rust
#![deprecated(since = "0.2.0", note = "Use SchedulerObserver instead")]
```

In events.rs, add:
```rust
#![deprecated(since = "0.2.0", note = "Use ObserverEvent instead")]
```

In actions.rs, add:
```rust
#![deprecated(since = "0.2.0", note = "Use SchedulerObserver instead")]
```

- [ ] **Step 2: Update engine.rs to remove unused imports**

Remove any `#![allow(dead_code)]` on event_handler related fields.

- [ ] **Step 3: Run clippy to check deprecation warnings**

Run: `cargo clippy -p vllm-core -- -D warnings`
Expected: Deprecation warnings, but no other warnings

- [ ] **Step 4: Commit**

```bash
git add crates/core/src/scheduler/event_handler.rs crates/core/src/scheduler/events.rs crates/core/src/scheduler/actions.rs
git commit -m "deprec(scheduler): mark old event system as deprecated"
```

---

## Task 5: Full Integration Test

**Files:**
- Test: `crates/core/tests/observer_integration.rs`

- [ ] **Step 1: Write integration test**

```rust
use vllm_core::scheduler::{SchedulerEngine, SchedulerObservers, SchedulerObserver, ObserverEvent};
use vllm_core::types::Request;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
struct TrackingObserver {
    events: Arc<Mutex<Vec<ObserverEvent>>>,
}

impl TrackingObserver {
    fn new() -> Self {
        Self { events: Arc::new(Mutex::new(Vec::new())) }
    }
}

impl SchedulerObserver for TrackingObserver {
    fn on_request_arrived(&self, seq_id: u64, prompt_len: usize) {
        self.events.lock().unwrap().push(
            ObserverEvent::RequestArrived { seq_id, prompt_len }
        );
    }
    
    fn on_sequence_finished(&self, seq_id: u64, total_tokens: usize) {
        self.events.lock().unwrap().push(
            ObserverEvent::SequenceFinished { seq_id, total_tokens }
        );
    }
}

#[test]
fn test_full_observer_integration() {
    let engine = SchedulerEngine::default();
    let observer = TrackingObserver::new();
    engine.register_observer(Box::new(observer.clone())).unwrap();
    
    // Add request
    let seq_id = engine.add_request(Request::new(0, vec![1, 2, 3], 10));
    
    // Verify event was dispatched
    let events = observer.events.lock().unwrap();
    assert!(events.iter().any(|e| matches!(
        e, ObserverEvent::RequestArrived { seq_id: id, prompt_len: 3 } 
        if id == seq_id
    )));
}
```

- [ ] **Step 2: Run integration test**

Run: `cargo test -p vllm-core observer_integration -- --nocapture`
Expected: PASS

- [ ] **Step 3: Run full test suite**

Run: `cargo test --workspace`
Expected: All tests pass

- [ ] **Step 4: Run clippy**

Run: `cargo clippy --workspace -- -D warnings`
Expected: No warnings (except deprecation)

- [ ] **Step 5: Commit**

```bash
git add crates/core/tests/
git commit -m "test(scheduler): add observer integration tests"
```

---

## Execution Summary

| Task | Description | Estimated Time |
|------|-------------|----------------|
| 1 | Create observer infrastructure | 15 min |
| 2 | Integrate observers into scheduler | 20 min |
| 3 | Remove ActionExecutor duplicate | 10 min |
| 4 | Deprecate old event system | 5 min |
| 5 | Full integration test | 10 min |

**Total: ~60 minutes**

---

## Plan Complete

**Plan saved to:** `docs/superpowers/plans/2026-04-09-hybrid-event-scheduler-plan.md`

**Two execution options:**

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
