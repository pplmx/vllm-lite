//! # Scheduler Module
//!
//! The scheduler orchestrates LLM inference requests through a componentized architecture.
//! It handles request queuing, batch building, memory allocation, and scheduling policies.
//!
//! ## Architecture Overview
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                        SchedulerEngine                              │
//! │                     (Orchestration Layer)                           │
//! ├─────────────────┬─────────────────┬─────────────────┬───────────────┤
//! │   RequestQueue  │  PhaseScheduler │  BatchComposer  │ MemoryManager │
//! │  (Request Mgmt) │ (P/D Separation)│ (Batch Building)│(Block Alloc)  │
//! └────────┬────────┴────────┬────────┴────────┬────────┴───────┬───────┘
//!          │                 │                 │                │
//!          ▼                 ▼                 ▼                ▼
//!    ┌──────────┐      ┌───────────┐    ┌───────────┐    ┌────────────┐
//!    │  policy/ │      │  radix_   │    │  packing/ │    │   memory/  │
//!    │  (FCFS,  │      │  cache/   │    │  (Sequence│    │  allocator,│
//!    │  SJF,    │      │  (Prefix  │    │   Packing)│    │  eviction  │
//!    │  Priority)│     │   Cache)  │    │           │    │            │
//!    └──────────┘      └───────────┘    └───────────┘    └────────────┘
//! ```
//!
//! ## Module Responsibilities
//!
//! ### Core Engine
//! - `engine` - Main scheduler orchestration
//!   - Coordinates all sub-components
//!   - Manages request lifecycle
//!   - Implements scheduling loop
//!
//! ### Request Management
//! - `request_queue` - O(1) request queue with phase-aware indexing
//!   - Fast insertion/removal by sequence ID
//!   - Tracks waiting, running, and finished sequences
//!   - Supports FCFS and priority ordering
//!
//! ### Scheduling Policies
//! - `policy` - Interchangeable scheduling algorithms
//!   - `FcfsPolicy` - First-Come-First-Served
//!   - `SjfPolicy` - Shortest Job First
//!   - `PriorityPolicy` - Priority-based scheduling
//!   - `SchedulingPolicy` trait for custom policies
//!
//! ### Batch Construction
//! - `batch_composer` - Builds phase-specific batches
//!   - Combines sequences into model-friendly batches
//!   - Respects token budgets and batch size limits
//! - `batch_planner` - Adaptive batch planning
//!   - Analyzes history for optimal batching
//!   - Computes adaptive ratios for P/D phases
//! - `packing` - Sequence packing utilities
//! - `batch` - Batch data structures
//!
//! ### Phase Management
//! - `phase_scheduler` - Prefill/Decode separation (P/D Separation)
//!   - `PhaseScheduler` manages phase switching
//!   - Configurable via `PhaseSwitchPolicy`
//!   - Tracks consecutive decode rounds
//!
//! ### Memory Management
//! - `memory` - Block allocation and eviction
//!   - `MemoryManager` - Main interface
//!   - `allocator` - Block allocation with free list
//!   - `eviction` - LRU-based eviction policies
//! - `cache` - KV cache management
//!   - `prefix_cache` - Prefix-based cache lookup
//!
//! ### Prefix Caching
//! - `radix_cache` - Radix tree for O(k) prefix lookup
//!   - `RadixTree` - Prefix matching
//!   - `PrefixMatchResult` for match info
//!   - Enables prompt reuse across requests
//!
//! ### Resource Management
//! - `preemption` - Request preemption when memory is tight
//!   - `PreemptionManager` decides victims
//!   - Considers running/waiting ratio and memory shortage
//! - `cuda_graph` - CUDA graph capture/replay optimization
//!   - `SchedulerCudaGraphConfig`
//!
//! ### Observability
//! - `observer` - Event observation system
//!   - `SchedulerObserver` trait
//!   - Events: request arrival, batch scheduling, decoding, completion, preemption
//! - `stats` - Scheduling statistics
//!   - `SchedulerStats` for metrics
//!
//! ## Data Flow
//!
//! 1. **Request Arrival** → `RequestQueue`
//! 2. **Scheduling Decision** → `SchedulingPolicy`
//! 3. **Phase Selection** → `PhaseScheduler`
//! 4. **Batch Construction** → `BatchComposer`
//! 5. **Memory Allocation** → `MemoryManager`
//! 6. **Prefix Lookup** → `RadixTree`
//! 7. **Execution** → Model forward pass
//! 8. **Completion/Eviction** → Update queues and free memory

// === Module Declarations ===

pub mod cache;
pub mod cuda_graph;
pub mod packing;
pub mod phase_scheduler;
pub mod policy;
pub mod preemption;
pub mod radix_cache;
pub mod request_queue;

pub mod memory;

pub mod batch;
pub mod batch_composer;
pub mod batch_planner;
pub mod engine;
pub mod observer;
pub mod stats;

// === Public Re-exports ===

pub use batch_composer::{BatchComposer, BatchCompositionConfig};
pub use cuda_graph::{GraphBatch, GraphPreparedBatch, SchedulerCudaGraphConfig};
pub use engine::SchedulerEngine;
pub use memory::MemoryManager;
pub use observer::{ObserverEvent, SchedulerObserver, SchedulerObservers};
pub use packing::{PackedBatch, SequencePacker};
pub use phase_scheduler::{PhaseScheduler, PhaseSwitchPolicy, SchedulerState};
pub use radix_cache::{PrefixMatchResult, RadixNode, RadixTree};
pub use request_queue::RequestQueue;
pub use stats::SchedulerStats;
