# Scheduler Aggressive Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the polling-based scheduler with an event-driven, state-machine based system for 2x throughput improvement and 50% reduction in scheduling overhead.

**Architecture:** Event-driven scheduler with separate EventHandler, StateMachine, ActionExecutor, QueueManager, BatchPlanner, and PreemptionManager. Each component handles one responsibility.

**Tech Stack:** Rust, existing vllm-core crate, thiserror for errors

---

## File Structure

```
crates/core/src/scheduler/
├── mod.rs              # Update exports
├── engine.rs           # Replace main Scheduler logic
├── events.rs           # NEW - Event definitions
├── state.rs            # NEW - State machine
├── actions.rs          # NEW - Action definitions
├── event_handler.rs    # NEW - Event dispatching
├── action_executor.rs  # NEW - Action execution
├── queue_manager.rs    # REPLACE queue.rs - multi-level priority
├── batch_planner.rs    # NEW - predictive batch planning
├── preemption.rs       # ENHANCE - cost-aware preemption
└── legacy_adapter.rs   # NEW - backward compatibility
```

---

## Task 1: Core Types and Events

**Files:**
- Create: `crates/core/src/scheduler/events.rs`
- Modify: `crates/core/src/scheduler/mod.rs`

- [ ] **Step 1: Write the failing test**

Create `crates/core/src/scheduler/events.rs`:

```rust
use crate::types::{SeqId, TokenId, Sequence, Request};
use std::time::Instant;

#[derive(Debug, Clone)]
pub enum SchedulerEvent {
    RequestArrived(Request),
    RequestCancelled(SeqId),
    RequestTimeout(SeqId),
    PrefillChunkComplete { seq_id: SeqId, tokens_computed: usize },
    PrefillComplete { seq_id: SeqId },
    DecodeComplete { seq_id: SeqId, new_token: TokenId },
    SequenceFinished { seq_id: SeqId },
    MemoryPressure { available_blocks: usize },
    GPUIdle,
    Tick,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_event_cloning() {
        let event = SchedulerEvent::RequestArrived(Request::new(1, vec![1,2,3], 5));
        let cloned = event.clone();
        assert!(matches!(cloned, SchedulerEvent::RequestArrived(_)));
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p vllm-core scheduler::events -- --nocapture 2>&1 | head -30`
Expected: FAIL - file not found

- [ ] **Step 3: Create the events.rs file**

Write the full `events.rs` with:
- `SchedulerEvent` enum with all variants
- `Event` trait for extensibility
- Basic tests

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p vllm-core scheduler::events`
Expected: PASS

- [ ] **Step 5: Update mod.rs exports**

In `crates/core/src/scheduler/mod.rs`, add:
```rust
pub mod events;
pub use events::SchedulerEvent;
```

- [ ] **Step 6: Commit**

```bash
git add crates/core/src/scheduler/events.rs crates/core/src/scheduler/mod.rs
git commit -m "feat(scheduler): add event types for event-driven design"
```

---

## Task 2: State Machine

**Files:**
- Create: `crates/core/src/scheduler/state.rs`
- Test: `crates/core/src/scheduler/state.rs` (same file)

- [ ] **Step 1: Write the failing test**

Create `crates/core/src/scheduler/state.rs`:

```rust
use super::events::SchedulerEvent;
use crate::types::{Priority, Status};
use std::time::Instant;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SeqState {
    Pending,
    Queued { priority: Priority, queued_at: Instant, prompt_length: usize },
    Prefilling { chunk_idx: usize, total_chunks: usize, started_at: Instant },
    DecodeWaiting,
    Decoding { decode_count: u32, started_at: Instant },
    Preempted { resume_at: usize, preempted_at: Instant, preemption_count: u32 },
    Finished,
    Cancelled,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_pending_to_queued_transition() {
        let state = SeqState::Pending;
        let event = SchedulerEvent::RequestArrived(crate::types::Request::new(
            1, vec![1,2,3], 5
        ));
        let next = state.transition(&event);
        assert!(next.is_some());
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p vllm-core scheduler::state -- --nocapture`
Expected: FAIL - method not implemented

- [ ] **Step 3: Write the state machine implementation**

Add to `state.rs`:
```rust
impl SeqState {
    pub fn transition(&self, event: &SchedulerEvent) -> Option<SeqState> {
        match (self, event) {
            (SeqState::Pending, SchedulerEvent::RequestArrived(req)) => {
                Some(SeqState::Queued {
                    priority: req.priority,
                    queued_at: Instant::now(),
                    prompt_length: req.prompt.len(),
                })
            }
            (SeqState::Queued(_), SchedulerEvent::PrefillChunkComplete { seq_id, tokens_computed }) => {
                Some(SeqState::Prefilling {
                    chunk_idx: 0,
                    total_chunks: 1,
                    started_at: Instant::now(),
                })
            }
            // ... complete implementation for all valid transitions
            _ => None,
        }
    }
    
    pub fn is_active(&self) -> bool {
        matches!(self, SeqState::Prefilling { .. } | SeqState::Decoding { .. })
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p vllm-core scheduler::state`
Expected: PASS

- [ ] **Step 5: Add more transition tests**

Test: Pending→Queued, Queued→Prefilling, Prefilling→Decoding, Decoding→Finished, any→Cancelled

- [ ] **Step 6: Commit**

```bash
git add crates/core/src/scheduler/state.rs crates/core/src/scheduler/mod.rs
git commit -m "feat(scheduler): add state machine for sequence lifecycle"
```

---

## Task 3: Actions System

**Files:**
- Create: `crates/core/src/scheduler/actions.rs`

- [ ] **Step 1: Write the failing test**

Create `crates/core/src/scheduler/actions.rs`:

```rust
use crate::types::{SeqId, TokenId};

#[derive(Debug, Clone)]
pub enum Action {
    ScheduleBatch(Vec<SeqId>),
    ReserveDecodeSlots(Vec<SeqId>),
    ReleaseDecodeSlots(Vec<SeqId>),
    StartPrefill { seq_id: SeqId, chunk: Vec<TokenId> },
    StartDecode { seq_id: SeqId, token: TokenId },
    Preempt { seq_id: SeqId, reason: String },
    Resume { seq_id: SeqId },
    Finish { seq_id: SeqId },
    AllocateBlocks { seq_id: SeqId, count: usize },
    EvictCache { target_size: usize },
    SendToken { seq_id: SeqId, token: TokenId },
    SendFinish { seq_id: SeqId },
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_action_debug() {
        let action = Action::Preempt { 
            seq_id: 1, 
            reason: "memory pressure".to_string() 
        };
        let debug_str = format!("{:?}", action);
        assert!(debug_str.contains("Preempt"));
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p vllm-core scheduler::actions -- --nocapture`
Expected: FAIL - file not found

- [ ] **Step 3: Write the actions implementation**

Complete `actions.rs` with:
- All action variants
- ActionDisplay trait for logging
- Basic tests

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p vllm-core scheduler::actions`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/core/src/scheduler/actions.rs crates/core/src/scheduler/mod.rs
git commit -m "feat(scheduler): add action definitions"
```

---

## Task 4: Event Handler

**Files:**
- Create: `crates/core/src/scheduler/event_handler.rs`

- [ ] **Step 1: Write the failing test**

Create `crates/core/src/scheduler/event_handler.rs`:

```rust
use super::events::SchedulerEvent;
use super::actions::Action;

pub struct EventHandler {
    // config would go here
}

impl EventHandler {
    pub fn new() -> Self {
        Self { }
    }
    
    pub fn dispatch(&mut self, event: SchedulerEvent) -> Vec<Action> {
        todo!("Implement event dispatching")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_request_arrived_generates_actions() {
        let mut handler = EventHandler::new();
        let event = SchedulerEvent::RequestArrived(
            crate::types::Request::new(1, vec![1,2,3], 5)
        );
        let actions = handler.dispatch(event);
        assert!(!actions.is_empty());
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p vllm-core scheduler::event_handler`
Expected: FAIL - panic: not yet implemented

- [ ] **Step 3: Implement basic event dispatching**

```rust
impl EventHandler {
    pub fn dispatch(&mut self, event: SchedulerEvent) -> Vec<Action> {
        match event {
            SchedulerEvent::RequestArrived(req) => {
                vec![Action::AllocateBlocks { 
                    seq_id: req.id, 
                    count: req.prompt.len().div_ceil(16) 
                }]
            }
            SchedulerEvent::PrefillChunkComplete { seq_id, tokens_computed } => {
                vec![Action::StartDecode { 
                    seq_id, 
                    token: 0 // placeholder
                }]
            }
            // ... more handlers
        }
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p vllm-core scheduler::event_handler`
Expected: PASS

- [ ] **Step 5: Add more event handlers**

Handle: DecodeComplete, SequenceFinished, MemoryPressure, RequestCancelled

- [ ] **Step 6: Commit**

```bash
git add crates/core/src/scheduler/event_handler.rs
git commit -m "feat(scheduler): add event handler for dispatching actions"
```

---

## Task 5: Queue Manager (Multi-Level Priority)

**Files:**
- Create: `crates/core/src/scheduler/queue_manager.rs`
- Modify: `crates/core/src/scheduler/mod.rs`

- [ ] **Step 1: Write the failing test**

Create `crates/core/src/scheduler/queue_manager.rs`:

```rust
use crate::types::{Sequence, SeqId, Priority, Status};
use std::collections::VecDeque;

pub struct QueueManager {
    critical: VecDeque<Sequence>,
    normal: VecDeque<Sequence>,
    background: VecDeque<Sequence>,
    preempted: VecDeque<Sequence>,
}

impl QueueManager {
    pub fn new() -> Self {
        todo!("Implement constructor")
    }
    
    pub fn enqueue(&mut self, seq: Sequence, priority: Priority) {
        todo!("Implement enqueue with priority")
    }
    
    pub fn dequeue(&mut self) -> Option<Sequence> {
        todo!("Implement dequeue (FIFO within priority)")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Status;
    
    fn make_seq(id: u64, priority: u8) -> Sequence {
        Sequence {
            id,
            tokens: vec![1,2,3],
            kv_blocks: std::sync::Arc::new(vec![]),
            num_computed_tokens: 0,
            prompt_len: 3,
            status: Status::Waiting,
            max_tokens: 100,
            sampling_params: Default::default(),
            consecutive_decode_rounds: 0,
            priority: Priority(priority),
        }
    }
    
    #[test]
    fn test_priority_ordering() {
        let mut qm = QueueManager::new();
        qm.enqueue(make_seq(1, 50), Priority(50));  // normal
        qm.enqueue(make_seq(2, 10), Priority(10));  // critical
        qm.enqueue(make_seq(3, 100), Priority(100)); // background
        
        // Critical should come first
        assert_eq!(qm.dequeue().map(|s| s.id), Some(2));
        assert_eq!(qm.dequeue().map(|s| s.id), Some(1));
        assert_eq!(qm.dequeue().map(|s| s.id), Some(3));
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p vllm-core scheduler::queue_manager`
Expected: FAIL - not implemented

- [ ] **Step 3: Implement QueueManager**

```rust
impl QueueManager {
    pub fn new() -> Self {
        Self {
            critical: VecDeque::new(),
            normal: VecDeque::new(),
            background: VecDeque::new(),
            preempted: VecDeque::new(),
        }
    }
    
    pub fn enqueue(&mut self, seq: Sequence, priority: Priority) {
        let queue = self.queue_for_priority(priority);
        queue.push_back(seq);
    }
    
    fn queue_for_priority(&mut self, priority: Priority) -> &mut VecDeque<Sequence> {
        if priority.0 <= 10 {
            &mut self.critical
        } else if priority.0 <= 50 {
            &mut self.normal
        } else {
            &mut self.background
        }
    }
    
    pub fn dequeue(&mut self) -> Option<Sequence> {
        self.critical.pop_front()
            .or_else(|| self.normal.pop_front())
            .or_else(|| self.background.pop_front())
    }
    
    pub fn is_empty(&self) -> bool {
        self.critical.is_empty() && self.normal.is_empty() && self.background.is_empty()
    }
    
    pub fn len(&self) -> usize {
        self.critical.len() + self.normal.len() + self.background.len()
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p vllm-core scheduler::queue_manager`
Expected: PASS

- [ ] **Step 5: Add more methods**

Add: `requeue_preempted()`, `get_waiting_count()`, `get_by_status()`

- [ ] **Step 6: Commit**

```bash
git add crates/core/src/scheduler/queue_manager.rs crates/core/src/scheduler/mod.rs
git commit -m "feat(scheduler): add multi-level priority queue manager"
```

---

## Task 6: Batch Planner (Predictive)

**Files:**
- Create: `crates/core/src/scheduler/batch_planner.rs`

- [ ] **Step 1: Write the failing test**

Create `crates/core/src/scheduler/batch_planner.rs`:

```rust
use crate::types::SchedulerConfig;

#[derive(Clone)]
pub struct BatchSnapshot {
    pub timestamp: std::time::Instant,
    pub batch_size: usize,
    pub prefill_count: usize,
    pub decode_count: usize,
    pub total_tokens: usize,
    pub latency_ms: f64,
}

pub struct BatchPlanner {
    history: Vec<BatchSnapshot>,
    config: SchedulerConfig,
}

pub struct BatchPlan {
    pub target_batch_size: usize,
    pub prefill_budget: usize,
    pub decode_budget: usize,
    pub max_concurrent_prefill: usize,
    pub decode_throughput_hint: f64,
}

impl BatchPlanner {
    pub fn new(config: SchedulerConfig) -> Self {
        todo!("Implement constructor")
    }
    
    pub fn plan(&mut self, state: &impl SchedulerStateView) -> BatchPlan {
        todo!("Implement planning")
    }
}

pub trait SchedulerStateView {
    fn waiting_count(&self) -> usize;
    fn running_count(&self) -> usize;
    fn prefill_count(&self) -> usize;
    fn decode_count(&self) -> usize;
    fn available_memory(&self) -> usize;
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_batch_plan_creation() {
        let config = SchedulerConfig::default();
        let planner = BatchPlanner::new(config);
        // Basic construction test
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p vllm-core scheduler::batch_planner`
Expected: FAIL - not implemented

- [ ] **Step 3: Implement BatchPlanner**

```rust
impl BatchPlanner {
    pub fn new(config: SchedulerConfig) -> Self {
        Self {
            history: Vec::with_capacity(100),
            config,
        }
    }
    
    pub fn plan(&mut self, state: &impl SchedulerStateView) -> BatchPlan {
        let adaptive_ratio = self.compute_adaptive_ratio(state);
        let budget = self.config.max_num_batched_tokens;
        
        BatchPlan {
            target_batch_size: self.predict_optimal_size(state),
            prefill_budget: (budget as f32 * (1.0 - adaptive_ratio)) as usize,
            decode_budget: (budget as f32 * adaptive_ratio) as usize,
            max_concurrent_prefill: self.config.max_num_seqs / 2,
            decode_throughput_hint: self.estimate_throughput(),
        }
    }
    
    fn compute_adaptive_ratio(&self, state: &impl SchedulerStateView) -> f32 {
        // If lots of waiting prefills, favor prefill
        // If mostly decoding, favor decode
        0.7 // default
    }
    
    fn predict_optimal_size(&self, state: &impl SchedulerStateView) -> usize {
        self.config.max_num_seqs
    }
    
    fn estimate_throughput(&self) -> f64 {
        // Simple moving average from history
        1000.0 // tokens/sec placeholder
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p vllm-core scheduler::batch_planner`
Expected: PASS

- [ ] **Step 5: Implement history tracking**

Add methods to record batch snapshots and compute statistics

- [ ] **Step 6: Commit**

```bash
git add crates/core/src/scheduler/batch_planner.rs
git commit -m "feat(scheduler): add predictive batch planner"
```

---

## Task 7: Action Executor

**Files:**
- Create: `crates/core/src/scheduler/action_executor.rs`

- [ ] **Step 1: Write the failing test**

Create `crates/core/src/scheduler/action_executor.rs`:

```rust
use super::actions::Action;
use super::events::SchedulerEvent;
use crate::kv_cache::BlockAllocator;
use crate::types::SeqId;

pub struct ActionExecutor {
    kv_allocator: BlockAllocator,
}

impl ActionExecutor {
    pub fn new(num_blocks: usize) -> Self {
        Self {
            kv_allocator: BlockAllocator::new(num_blocks),
        }
    }
    
    pub fn execute(&mut self, action: Action) -> Vec<SchedulerEvent> {
        todo!("Implement action execution")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_allocate_action() {
        let mut executor = ActionExecutor::new(100);
        let action = Action::AllocateBlocks { seq_id: 1, count: 5 };
        let events = executor.execute(action);
        // Should return success event or error
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p vllm-core scheduler::action_executor`
Expected: FAIL - not implemented

- [ ] **Step 3: Implement ActionExecutor**

```rust
impl ActionExecutor {
    pub fn execute(&mut self, action: Action) -> Vec<SchedulerEvent> {
        match action {
            Action::AllocateBlocks { seq_id, count } => {
                if let Some(blocks) = self.kv_allocator.allocate(count) {
                    vec![SchedulerEvent::SequenceFinished { seq_id }] // placeholder
                } else {
                    vec![SchedulerEvent::MemoryPressure { 
                        available_blocks: self.kv_allocator.available() 
                    }]
                }
            }
            Action::EvictCache { target_size } => {
                // trigger eviction
                vec![]
            }
            _ => vec![],
        }
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p vllm-core scheduler::action_executor`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/core/src/scheduler/action_executor.rs
git commit -m "feat(scheduler): add action executor"
```

---

## Task 8: Scheduler Engine (Integration)

**Files:**
- Create: `crates/core/src/scheduler/engine.rs`
- Modify: `crates/core/src/scheduler/mod.rs`

- [ ] **Step 1: Write the failing test**

Create `crates/core/src/scheduler/engine.rs`:

```rust
use super::events::SchedulerEvent;
use super::event_handler::EventHandler;
use super::action_executor::ActionExecutor;
use super::queue_manager::QueueManager;
use super::batch_planner::BatchPlanner;
use crate::types::{SchedulerConfig, Request, Batch};

pub struct SchedulerEngine {
    event_handler: EventHandler,
    action_executor: ActionExecutor,
    queue_manager: QueueManager,
    batch_planner: BatchPlanner,
    config: SchedulerConfig,
}

impl SchedulerEngine {
    pub fn new(config: SchedulerConfig, num_kv_blocks: usize) -> Self {
        Self {
            event_handler: EventHandler::new(),
            action_executor: ActionExecutor::new(num_kv_blocks),
            queue_manager: QueueManager::new(),
            batch_planner: BatchPlanner::new(config.clone()),
            config,
        }
    }
    
    pub fn add_request(&mut self, req: Request) -> SeqId {
        todo!("Implement request handling")
    }
    
    pub fn build_batch(&mut self) -> Batch {
        todo!("Implement batch building")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_engine_basic() {
        let mut engine = SchedulerEngine::new(SchedulerConfig::default(), 1024);
        engine.add_request(Request::new(1, vec![1,2,3], 5));
        assert!(engine.queue_manager.len() > 0);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p vllm-core scheduler::engine -- --nocapture`
Expected: FAIL - not implemented

- [ ] **Step 3: Implement SchedulerEngine**

```rust
impl SchedulerEngine {
    pub fn add_request(&mut self, req: Request) -> SeqId {
        let seq_id = req.id;
        let event = SchedulerEvent::RequestArrived(req);
        let actions = self.event_handler.dispatch(event);
        
        for action in actions {
            self.action_executor.execute(action);
        }
        
        seq_id
    }
    
    pub fn build_batch(&mut self) -> Batch {
        // Use batch planner to determine optimal batch
        // Dequeue from queue manager
        // Build and return Batch
        Batch::default()
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p vllm-core scheduler::engine`
Expected: PASS

- [ ] **Step 5: Add more engine methods**

Add: `step()`, `update()`, `has_pending()`, metrics methods

- [ ] **Step 6: Commit**

```bash
git add crates/core/src/scheduler/engine.rs
git commit -m "feat(scheduler): add scheduler engine as central coordinator"
```

---

## Task 9: Legacy Adapter (Backward Compatibility)

**Files:**
- Create: `crates/core/src/scheduler/legacy_adapter.rs`
- Modify: `crates/core/src/scheduler/mod.rs`

- [ ] **Step 1: Write the failing test**

Create `crates/core/src/scheduler/legacy_adapter.rs`:

```rust
use super::engine::SchedulerEngine;
use crate::types::{Request, SchedulerConfig, Batch, SeqId};

pub struct LegacySchedulerAdapter {
    engine: SchedulerEngine,
}

impl LegacySchedulerAdapter {
    pub fn new(config: SchedulerConfig, num_kv_blocks: usize) -> Self {
        Self {
            engine: SchedulerEngine::new(config, num_kv_blocks),
        }
    }
    
    pub fn add_request(&mut self, req: Request) -> SeqId {
        self.engine.add_request(req)
    }
    
    pub fn build_batch(&mut self) -> Batch {
        self.engine.build_batch()
    }
    
    // Implement all Scheduler methods for compatibility
}

impl Default for LegacySchedulerAdapter {
    fn default() -> Self {
        Self::new(SchedulerConfig::default(), 1024)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_adapter_add_request() {
        let mut adapter = LegacySchedulerAdapter::default();
        let id = adapter.add_request(Request::new(1, vec![1,2,3], 5));
        assert_eq!(id, 1);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p vllm-core scheduler::legacy_adapter`
Expected: FAIL - not implemented

- [ ] **Step 3: Implement LegacyAdapter**

Implement all methods from original Scheduler trait

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p vllm-core scheduler::legacy_adapter`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/core/src/scheduler/legacy_adapter.rs crates/core/src/scheduler/mod.rs
git commit -m "feat(scheduler): add legacy adapter for backward compatibility"
```

---

## Task 10: Integration Tests and Migration

**Files:**
- Modify: Various test files

- [ ] **Step 1: Run existing scheduler tests**

Run: `cargo test -p vllm-core scheduler -- --nocapture 2>&1 | tail -20`
Expected: Some may fail due to API changes

- [ ] **Step 2: Fix test imports**

Update tests to use new module paths

- [ ] **Step 3: Add integration test**

Create integration test that exercises full pipeline:

```rust
#[test]
fn test_full_pipeline() {
    let mut adapter = LegacySchedulerAdapter::default();
    
    // Add requests
    adapter.add_request(Request::new(1, vec![1,2,3], 5));
    adapter.add_request(Request::new(2, vec![4,5,6], 5));
    
    // Build batch
    let batch = adapter.build_batch();
    assert!(!batch.is_empty());
    
    // This tests the full flow
}
```

- [ ] **Step 4: Run full test suite**

Run: `cargo test -p vllm-core`
Expected: All tests pass

- [ ] **Step 5: Benchmark comparison**

Compare performance before/after if possible

- [ ] **Step 6: Final commit**

```bash
git add -A
git commit -m "feat(scheduler): complete event-driven scheduler implementation"
```

---

## Implementation Summary

| Task | Component | Files | Tests |
|------|-----------|-------|-------|
| 1 | Events | events.rs | 1 |
| 2 | State Machine | state.rs | 5 |
| 3 | Actions | actions.rs | 1 |
| 4 | Event Handler | event_handler.rs | 3 |
| 5 | Queue Manager | queue_manager.rs | 3 |
| 6 | Batch Planner | batch_planner.rs | 2 |
| 7 | Action Executor | action_executor.rs | 2 |
| 8 | Engine | engine.rs | 3 |
| 9 | Legacy Adapter | legacy_adapter.rs | 2 |
| 10 | Integration | Various | 5+ |

**Total: ~25 steps, ~20 tests**
