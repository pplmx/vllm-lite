# Scheduler Aggressive Refactor Design

## Overview

Completely redesign the scheduler from a polling-based approach to an event-driven, state-machine based system. This is a ground-up重构 targeting maximum performance and flexibility.

## Motivation

Current scheduler issues (cannot be fixed incrementally):
1. **Polling overhead**: Every step iterates all sequences even when nothing changed
2. **Flat queue**: No differentiation between urgent and background work
3. **Coupled concerns**: Batch building mixed with preemption, eviction, stats
4. **Opaque state**: Hard to understand sequence lifecycle and debug issues

## Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────────┐
│                         SchedulerCore                           │
├─────────────────────────────────────────────────────────────────┤
│  EventQueue ──▶ SchedulerEngine ──▶ StateMachine ──▶ Actions  │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ EventHandler │  │ StateMachine │  │ ActionExecutor│        │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    Sub-systems                           │  │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌───────┐  │  │
│  │  │ Queue  │ │ Batch  │ │ Preempt│ │  KV    │ │ Prefix│  │  │
│  │  │ Manager│ │ Planner│ │ Manager│ │ Alloc  │ │ Cache │  │  │
│  │  └────────┘ └────────┘ └────────┘ └────────┘ └───────┘  │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. Event System

```rust
#[derive(Debug, Clone)]
pub enum SchedulerEvent {
    // Request lifecycle
    RequestArrived(Request),
    RequestCancelled(SeqId),
    RequestTimeout(SeqId),
    
    // Sequence state transitions
    PrefillChunkComplete { seq_id: SeqId, tokens_computed: usize },
    PrefillComplete { seq_id: SeqId },
    DecodeComplete { seq_id: SeqId, new_token: TokenId },
    SequenceFinished { seq_id: SeqId },
    
    // Resource events
    MemoryPressure { available_blocks: usize },
    GPUIdle,
    
    // Scheduled events
    Tick,
}
```

#### 2. Event Handler

```rust
pub struct EventHandler {
    pending_events: VecDeque<SchedulerEvent>,
    event_handlers: HashMap<TypeId, Box<dyn EventHandlerTrait>>,
}

impl EventHandler {
    pub fn dispatch(&mut self, event: SchedulerEvent) -> Vec<Action> {
        let mut actions = Vec::new();
        
        match event {
            SchedulerEvent::RequestArrived(req) => {
                actions.extend(self.handle_request_arrived(req));
            }
            SchedulerEvent::PrefillChunkComplete { .. } => {
                actions.extend(self.handle_prefill_complete(...));
            }
            // ...
        }
        
        actions
    }
}
```

#### 3. State Machine

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SeqState {
    /// Newly created, not yet scheduled
    Pending,
    
    /// In waiting queue, awaiting scheduling
    Queued { 
        priority: Priority,
        queued_at: Instant,
        prompt_length: usize,
    },
    
    /// Actively being processed (prefill in progress)
    Prefilling {
        chunk_idx: usize,
        total_chunks: usize,
        started_at: Instant,
    },
    
    /// Waiting for decode slot
    DecodeWaiting,
    
    /// Actively decoding
    Decoding {
        decode_count: u32,
        started_at: Instant,
    },
    
    /// Temporarily paused for resources
    Preempted {
        resume_at: usize,
        preempted_at: Instant,
        preemption_count: u32,
    },
    
    /// Completed successfully
    Finished,
    
    /// Cancelled or failed
    Cancelled,
}

impl SeqState {
    pub fn transition(&self, event: &SchedulerEvent) -> Option<SeqState> {
        match (self, event) {
            (SeqState::Pending, SchedulerEvent::RequestArrived(_)) => {
                Some(SeqState::Queued { ... })
            }
            (SeqState::Queued(_), SchedulerEvent::PrefillChunkComplete { .. }) => {
                Some(SeqState::Prefilling { ... })
            }
            // ...
            _ => None,
        }
    }
}
```

#### 4. Action System

```rust
#[derive(Debug, Clone)]
pub enum Action {
    // Batch actions
    ScheduleBatch(BatchPlan),
    ReserveDecodeSlots(Vec<SeqId>),
    ReleaseDecodeSlots(Vec<SeqId>),
    
    // Sequence actions
    StartPrefill { seq_id: SeqId, chunk: TokenChunk },
    StartDecode { seq_id: SeqId, token: TokenId },
    Preempt { seq_id: SeqId, reason: PreemptReason },
    Resume { seq_id: SeqId },
    Finish { seq_id: SeqId },
    
    // Resource actions
    AllocateBlocks { seq_id: SeqId, count: usize },
    EvictCache { target_size: usize },
    
    // Response actions
    SendToken { seq_id: SeqId, token: TokenId },
    SendFinish { seq_id: SeqId },
}

pub struct ActionExecutor {
    kv_allocator: BlockAllocator,
    prefix_cache: PrefixCache,
    batch_planner: BatchPlanner,
}

impl ActionExecutor {
    pub fn execute(&mut self, action: Action) -> Result<Vec<SchedulerEvent>> {
        match action {
            Action::Preempt { seq_id, reason } => {
                // Generate follow-up events
                Ok(vec![SchedulerEvent::SequencePreempted { seq_id, reason }])
            }
            // ...
        }
    }
}
```

### Queue Manager (Multi-Level Priority)

```rust
pub struct QueueManager {
    // Separate queues for different priority levels
    critical: PriorityQueue,   // Interactive, low latency
    normal: PriorityQueue,     // Standard requests  
    background: PriorityQueue, // Batch, prefetch
    preempted: PriorityQueue,  // Waiting to resume
    
    // Metrics
    queue_wait_times: Histogram,
}

impl QueueManager {
    pub fn enqueue(&mut self, seq: Sequence, priority: Priority) {
        let queue = match priority {
            p if p <= Priority(10) => &mut self.critical,
            p if p <= Priority(50) => &mut self.normal,
            _ => &mut self.background,
        };
        
        queue.push(seq);
        self.notify_changed();
    }
    
    pub fn select_for_scheduling(&mut self, slots: usize, budget: usize) -> Vec<Sequence> {
        // First fill from critical
        // Then normal
        // Finally background if slots remain
    }
}
```

### Batch Planner (Predictive)

```rust
pub struct BatchPlanner {
    history: RingBuffer<BatchSnapshot>,
    config: PlannerConfig,
}

#[derive(Clone)]
pub struct BatchSnapshot {
    timestamp: Instant,
    batch_size: usize,
    prefill_count: usize,
    decode_count: usize,
    total_tokens: usize,
    latency_ms: f64,
}

impl BatchPlanner {
    pub fn plan(&mut self, state: &SchedulerState) -> BatchPlan {
        let adaptive_policy = self.compute_adaptive_policy(state);
        
        BatchPlan {
            target_batch_size: self.predict_optimal_size(state),
            prefill_budget: adaptive_policy.prefill_ratio * state.token_budget,
            decode_budget: adaptive_policy.decode_ratio * state.token_budget,
            max_concurrent_prefill: adaptive_policy.max_prefill_parallelism,
            decode_throughput_hint: self.estimate_decode_throughput(),
        }
    }
    
    fn compute_adaptive_policy(&self, state: &SchedulerState) -> AdaptivePolicy {
        // Analyze current state and historical patterns
        // Return optimal ratios and limits
    }
}
```

### Preemption Manager (Cost-Aware)

```rust
pub struct PreemptionManager {
    config: PreemptionConfig,
    cost_model: PreemptionCostModel,
}

#[derive(Clone)]
pub struct PreemptionCost {
    pub compute_cost: f32,      // Tokens to recompute
    pub memory_benefit: usize,  // Blocks freed
    pub queue_delay: f32,       // Other requests affected
    pub progress_loss: f32,     // Decode rounds lost
}

impl PreemptionManager {
    pub fn should_preempt(&self, state: &SchedulerState, seq: &Sequence) -> PreemptionDecision {
        let cost = self.cost_model.estimate_cost(seq);
        let benefit = self.estimate_benefit(seq);
        
        let score = benefit / (cost + self.config.epsilon);
        
        if score > self.config.threshold {
            PreemptionDecision::Preempt { score, cost, benefit }
        } else {
            PreemptionDecision::Wait
        }
    }
    
    pub fn select_victims(&self, running: &[Sequence], blocks_needed: usize) -> Vec<Sequence> {
        // Score all candidates
        // Select best combination to meet block requirement
        // Consider: preemption chains (preempting a preempted sequence is worse)
    }
}
```

### KV Allocator (Segmented with Defragmentation)

```rust
pub struct SegmentedBlockAllocator {
    // Size-segregated free lists
    free_by_size: HashMap<usize, Vec<usize>>,
    // Allocation metadata
    allocations: HashMap<SeqId, Vec<usize>>,
    // Defragmentation state
    fragmentation_score: f32,
}

impl SegmentedBlockAllocator {
    pub fn allocate(&mut self, num_blocks: usize) -> Result<Vec<usize>> {
        // First try exact size match
        // Then try larger block + split
        // Finally fallback to first-fit
        
        if self.fragmentation_score > 0.3 {
            self.defragment();
        }
    }
    
    pub fn defragment(&mut self) {
        // Copy blocks to compact memory
        // Update all allocation pointers
    }
}
```

## Data Flow

```
                    ┌─────────────────┐
                    │   Incoming      │
                    │   Request       │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  EventHandler   │
                    │  .dispatch()    │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
       ┌──────────┐   ┌───────────┐  ┌──────────┐
       │ Validate │   │   State   │  │  Check   │
       │ Request  │   │ Transition│  │ Resources│
       └────┬─────┘   └─────┬─────┘  └────┬─────┘
            │               │              │
            └───────────────┼──────────────┘
                            ▼
                    ┌─────────────────┐
                    │   ActionQueue   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ ActionExecutor  │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
       ┌──────────┐   ┌───────────┐  ┌──────────┐
       │  Batch   │   │  Update   │  │  Send    │
       │  Planner │   │    KV     │  │ Response │
       └──────────┘   └───────────┘  └──────────┘
                            │
                            ▼
                    ┌─────────────────┐
                    │ New SchedulerEvent │
                    │   (for next loop)  │
                    └─────────────────┘
```

## Event Loop

```rust
pub struct SchedulerEngine {
    state: SchedulerState,
    event_handler: EventHandler,
    action_executor: ActionExecutor,
    queue_manager: QueueManager,
    batch_planner: BatchPlanner,
    preemption_manager: PreemptionManager,
}

impl SchedulerEngine {
    pub fn step(&mut self) -> Result<Vec<(SeqId, TokenId)>> {
        // 1. Process any pending events
        while let Some(event) = self.state.pending_events.pop_front() {
            let actions = self.event_handler.dispatch(event);
            self.execute_actions(actions);
        }
        
        // 2. Check if we should schedule new work
        if self.should_schedule() {
            self.schedule_next_batch()?;
        }
        
        // 3. Process completed work
        self.process_completions()
    }
    
    fn should_schedule(&self) -> bool {
        !self.queue_manager.is_empty() 
            && self.state.running.len() < self.config.max_concurrent
            && self.state.has_available_memory()
    }
}
```

## Configuration

```rust
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    // Queue settings
    pub max_queue_size: usize,
    pub queue_timeout_ms: u64,
    pub priority_levels: usize,
    
    // Batch settings
    pub max_batch_size: usize,
    pub max_token_budget: usize,
    pub prefill_chunk_size: usize,
    
    // Preemption settings
    pub enable_preemption: bool,
    pub preemption_threshold: f32,
    pub max_preemption_chain: usize,
    
    // Memory settings
    pub max_kv_blocks: usize,
    pub eviction_threshold: f32,
    pub enable_defragmentation: bool,
    
    // Adaptive settings
    pub enable_adaptive_batching: bool,
    pub enable_predictive_scheduling: bool,
    pub history_window_size: usize,
}
```

## Backward Compatibility

Provide compatibility layer:

```rust
pub struct LegacySchedulerAdapter {
    engine: SchedulerEngine,
}

impl LegacySchedulerAdapter {
    pub fn add_request(&mut self, req: Request) -> SeqId {
        // Convert to event and process
        let event = SchedulerEvent::RequestArrived(req);
        self.engine.dispatch(event)
    }
    
    pub fn build_batch(&mut self) -> Batch {
        // Delegate to engine
        self.engine.build_batch()
    }
}
```

## Testing Strategy

### Unit Tests
1. State machine transitions (all valid paths)
2. Event dispatching (all event types)
3. Action execution (side effects)
4. Queue ordering (priority behavior)

### Integration Tests
1. Full request lifecycle (arrive → queue → schedule → complete)
2. Preemption under pressure
3. Queue priority ordering
4. Memory eviction integration

### Chaos Tests
1. Rapid request arrival
2. Cancellation during execution
3. Memory pressure scenarios
4. Priority inversion

## Migration Path

1. **Phase 1**: Implement core (EventHandler, StateMachine, ActionExecutor)
2. **Phase 2**: Implement QueueManager
3. **Phase 3**: Implement BatchPlanner
4. **Phase 4**: Implement PreemptionManager
5. **Phase 5**: Wire together, add tests
6. **Phase 6**: Benchmark and tune
7. **Phase 7**: Remove old scheduler (optional: keep adapter)

## Success Metrics

- **Latency**: 50% reduction in P99 scheduling overhead
- **Throughput**: 2x improvement in requests/second
- **Memory**: 20% better utilization under load
- **Preemption**: 30% fewer unnecessary preemptions
- **Debuggability**: State machine makes issues traceable

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Complex state machine bugs | Extensive property-based testing |
| Event loop performance | Profile early, optimize hot paths |
| Backward compatibility | Keep adapter until validated |
| Preemption regression | A/B test against production |
