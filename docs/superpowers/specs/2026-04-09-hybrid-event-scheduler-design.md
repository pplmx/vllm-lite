# Hybrid Event-Driven Scheduler Design

## Overview

Redesign the scheduler to use a hybrid architecture: direct method calls for performance-critical paths, with an observer-based event system for observability and extensibility.

## Motivation

Current scheduler uses direct method calls (`add_request` → `build_batch` → `update`) which is fast but:
- Hard to observe internal state
- Difficult to add features like request cancellation, timeout handling
- Complex to test and debug

Pure event-driven adds too much overhead for a latency-sensitive inference engine.

## Architecture

### Core Principle

**Hybrid**: Performance path uses direct calls; side effects use observer events.

```
┌─────────────────────────────────────────────────────────────┐
│                    SchedulerEngine                          │
│                  (Direct Call Layer)                        │
│                                                             │
│  add_request() ──────────────────────────────────────►     │
│      │  1. Direct processing (allocate blocks, enqueue)    │
│      │  2. Fire-and-forget event: RequestArrived           │
│      │  3. Return seq_id                                   │
│                                                             │
│  build_batch() ───────────────────────────────────────►     │
│      │  1. Direct scheduling (select sequences, build)     │
│      │  2. Fire-and-forget event: BatchScheduled           │
│      │  3. Return Batch                                    │
│                                                             │
│  update() ──────────────────────────────────────────────►   │
│      │  1. Direct processing (update state, expand blocks) │
│      │  2. Optional intervention on memory pressure        │
│      │  3. Fire-and-forget events: Decoding, Finished      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Observer System                           │
├─────────────────────────────────────────────────────────────┤
│  trait SchedulerObserver: Send + Sync                       │
│                                                             │
│  fn on_request_arrived(&self, req: &Request, seq_id: u64)   │
│  fn on_batch_scheduled(&self, batch: &Batch)                │
│  fn on_decoding(&self, seq_id: u64, token: u32)             │
│  fn on_sequence_finished(&self, seq: &Sequence)             │
│  fn on_preemption(&self, seq_id: u64, reason: &str)         │
│  fn on_memory_pressure(&self, available: usize)             │
│                                                             │
│  Implementations:                                           │
│  - MetricsObserver: Prometheus metrics                      │
│  - LoggingObserver: Request lifecycle logging               │
│  - CallbackObserver: External callbacks                     │
└─────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

### 1. Fire-and-Forget Events

Events are dispatched asynchronously after the main operation completes. The caller does NOT wait for observers to process.

```rust
// In SchedulerEngine
pub fn add_request(&mut self, req: Request) -> SeqId {
    // ... direct processing ...
    
    // Fire-and-forget: don't wait for observers
    self.dispatch_observer_event(ObserverEvent::RequestArrived {
        seq_id,
        prompt_len: req.prompt.len(),
    });
    
    seq_id
}
```

**Rationale**: Zero latency impact on the critical path.

### 2. Observer Trait with Default No-Op

```rust
pub trait SchedulerObserver: Send + Sync {
    fn on_request_arrived(&self, _req: &Request, _seq_id: SeqId) {}
    fn on_batch_scheduled(&self, _batch: &Batch) {}
    fn on_decoding(&self, _seq_id: SeqId, _token: TokenId) {}
    fn on_sequence_finished(&self, _seq: &Sequence) {}
    fn on_preemption(&self, _seq_id: SeqId, _reason: &str) {}
    fn on_memory_pressure(&self, _available: usize) {}
}
```

**Rationale**: Zero cost when no observers registered.

### 3. Observer Registry

```rust
pub struct SchedulerObservers {
    observers: RwLock<Vec<Box<dyn SchedulerObserver>>>,
}

impl SchedulerObservers {
    pub fn register(&self, observer: Box<dyn SchedulerObserver>) {
        self.observers.write().unwrap().push(observer);
    }
    
    pub fn notify<F>(&self, f: F)
    where
        F: Fn(&dyn SchedulerObserver),
    {
        for observer in self.observers.read().unwrap().iter() {
            f(observer.as_ref());
        }
    }
}
```

### 4. Simplified Event Enum

Unlike the current complex event system, use simple observer events:

```rust
pub enum ObserverEvent {
    RequestArrived { seq_id: SeqId, prompt_len: usize },
    BatchScheduled { seq_ids: Vec<SeqId>, batch_size: usize },
    Decoding { seq_id: SeqId, token: TokenId },
    SequenceFinished { seq_id: SeqId, total_tokens: usize },
    Preemption { seq_id: SeqId, reason: String },
    MemoryPressure { available_blocks: usize },
}
```

**Rationale**: One-way notification, no response expected.

### 5. Remove ActionExecutor's Independent Allocator

Current `ActionExecutor` has its own `BlockAllocator` that duplicates `SchedulerEngine`'s. This causes state inconsistency.

**Solution**: Remove resource management from `ActionExecutor`. It becomes purely observational.

```rust
pub struct ActionExecutor {
    // No more independent allocator
    metrics: SchedulerMetrics,
}
```

## Component Changes

### 1. New: observer.rs

```rust
// crates/core/src/scheduler/observer.rs

pub trait SchedulerObserver: Send + Sync {
    fn on_request_arrived(&self, req: &Request, seq_id: SeqId);
    fn on_batch_scheduled(&self, batch: &Batch);
    fn on_decoding(&self, seq_id: SeqId, token: TokenId);
    fn on_sequence_finished(&self, seq: &Sequence);
    fn on_preemption(&self, seq_id: SeqId, reason: &str);
    fn on_memory_pressure(&self, available: usize);
}

pub struct SchedulerObservers { ... }
pub enum ObserverEvent { ... }
```

### 2. Modified: engine.rs

Changes:
- Add `observers: SchedulerObservers` field
- Add `register_observer()` method
- Replace `process_event_loop()` with simpler `dispatch_observer_event()`
- Remove unused event/action dispatch code
- Keep direct call performance path unchanged

### 3. Modified: action_executor.rs

Changes:
- Remove `BlockAllocator` field
- Simplify to pure metrics/monitoring
- No more resource management

### 4. Modified: event_handler.rs

Changes:
- Keep for potential future use
- Mark as deprecated or convert to observer pattern

## Data Flow

```
Request arrives
    │
    ▼
SchedulerEngine.add_request()
    │
    ├── 1. Allocate blocks (direct)
    ├── 2. Enqueue (direct)
    ├── 3. Return seq_id
    │
    ▼
dispatch_observer_event(RequestArrived)
    │
    ▼
SchedulerObservers.notify()
    │
    ├──► MetricsObserver: record request_count++
    ├──► LoggingObserver: log "request arrived"
    └──► CallbackObserver: trigger webhook (optional)
```

## Performance Expectations

| Operation | Original | Hybrid | Delta |
|-----------|----------|--------|-------|
| add_request | O(1) | O(1) + observer dispatch | ~5% (observer is async) |
| build_batch | O(n) | O(n) + observer dispatch | ~2% |
| update | O(n) | O(n) + observer dispatch | ~2% |

**Note**: Observer dispatch is "fire-and-forget" and doesn't wait for completion.

## Migration Path

1. **Phase 1**: Add `SchedulerObserver` trait and `SchedulerObservers` registry
2. **Phase 2**: Add observer dispatch to existing event points
3. **Phase 3**: Remove `ActionExecutor`'s independent allocator
4. **Phase 4**: Simplify/deprecate old event system

## Testing

```rust
#[test]
fn test_observer_receives_events() {
    let engine = SchedulerEngine::default();
    
    // Register test observer
    let observer = TestObserver::new();
    engine.register_observer(Box::new(observer));
    
    // Add request
    engine.add_request(Request::new(0, vec![1, 2, 3], 10));
    
    // Verify observer was notified
    assert!(observer.has_received(ObserverEvent::RequestArrived { .. }));
}
```

## Open Questions

1. Should we keep the old `SchedulerEvent`/`Action` system for backwards compatibility?
2. Maximum number of observers to prevent abuse?
3. Should observer errors be propagated or silently ignored?

## Related Files

- `crates/core/src/scheduler/engine.rs` - Modified
- `crates/core/src/scheduler/action_executor.rs` - Simplified
- `crates/core/src/scheduler/event_handler.rs` - Deprecated/Converted
- `crates/core/src/scheduler/observer.rs` - New
