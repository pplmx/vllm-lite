//! `SchedulerEngine` struct definition, the `Debug` / `Default` impls,
//! the `new()` constructor, and small read-only state accessors.
//!
////! The big lifecycle methods (`add_request`, `build_batch`, `schedule`)
//! live in sibling modules under [`self`] so this file stays focused on
//! the struct layout + ctor + cheap getters.
//!
//! Module layout:
//!
//! - [`self`] (`mod.rs`) — struct + `Debug` + `Default` + `new()` + `set_policy` + small getters + observer registration
//! - [`request`] — `add_request` (enqueue + prefix-cache check + metrics + observer dispatch)
//! - [`batch`] — `build_batch` + `schedule` (phase selection + composition + preemption trigger)

mod batch;
mod request;

use std::sync::Arc;

use vllm_traits::{Batch, SeqId};

use crate::metrics::EnhancedMetricsCollector;
use crate::scheduler::cuda_graph::SchedulerCudaGraphConfig;
use crate::scheduler::observer::{
    SchedulerObserver, SchedulerObserverError, SchedulerObservers,
};
use crate::scheduler::policy::{FcfsPolicy, SchedulingPolicy};
use crate::scheduler::{
    BatchComposer, BatchCompositionConfig, MemoryManager, PhaseScheduler, PhaseSwitchPolicy,
    RadixTree, RequestQueue,
};
use crate::types::{SchedulerConfig, Sequence};

/// `SchedulerEngine` - Componentized scheduler architecture
///
/// This engine combines multiple specialized components:
/// - `RequestQueue`: O(1) lookup and removal with phase-aware indexing
/// - `PhaseScheduler`: Strict prefill/decode separation with configurable policies
/// - `BatchComposer`: Phase-specific batch construction
/// - `MemoryManager`: Block allocation and eviction
/// - `RadixTree`: Prefix caching for prompt reuse
pub struct SchedulerEngine {
    pub(super) request_queue: RequestQueue,
    pub(super) phase_scheduler: PhaseScheduler,
    pub(super) batch_composer: BatchComposer,
    pub(super) memory: MemoryManager,
    pub(super) prefix_cache: RadixTree,
    pub(super) policy: Box<dyn SchedulingPolicy>,
    pub(super) running: Vec<Sequence>,
    pub(super) finished: Vec<Sequence>,
    pub(super) next_seq_id: SeqId,
    pub(super) observers: SchedulerObservers,
    /// CUDA Graph configuration for decode optimization
    pub(super) cuda_graph: SchedulerCudaGraphConfig,
    /// Metrics collector for tracking engine performance
    pub metrics: Arc<EnhancedMetricsCollector>,
}

impl std::fmt::Debug for SchedulerEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SchedulerEngine")
            .field("request_queue", &self.request_queue)
            .field("phase_scheduler", &self.phase_scheduler)
            .field("batch_composer", &self.batch_composer)
            .field("memory", &self.memory)
            .field("prefix_cache", &self.prefix_cache)
            .field("policy", &"<dyn SchedulingPolicy>")
            .field("running_count", &self.running.len())
            .field("finished_count", &self.finished.len())
            .field("next_seq_id", &self.next_seq_id)
            .field("observers", &self.observers)
            .field("cuda_graph", &self.cuda_graph)
            .finish_non_exhaustive()
    }
}

impl SchedulerEngine {
    /// Create a new `SchedulerEngine` with the given configuration
    ///
    /// # Arguments
    /// * `config` - Scheduler configuration
    /// * `num_kv_blocks` - Number of KV cache blocks available
    /// * `metrics` - Metrics collector for tracking engine performance
    pub fn new(
        config: SchedulerConfig,
        num_kv_blocks: usize,
        metrics: Arc<EnhancedMetricsCollector>,
    ) -> Self {
        let phase_switch_policy = PhaseSwitchPolicy {
            max_consecutive_decode: config.max_consecutive_decode,
            prefill_priority_threshold: 5,
            min_decode_batch_size: config.min_batch_size,
        };

        let batch_config = BatchCompositionConfig {
            max_batch_size: config.max_num_seqs,
            max_token_budget: config.max_num_batched_tokens,
            enable_similarity_grouping: false,
        };

        // Initialize CUDA Graph config from scheduler config
        let cuda_graph = SchedulerCudaGraphConfig {
            enabled: config.cuda_graph.enabled,
            batch_sizes: config.cuda_graph.batch_sizes.clone(),
        };

        // Initialize metrics with current state
        metrics.set_active_sequences(0);
        metrics.set_queue_depth(0);

        Self {
            request_queue: RequestQueue::new(),
            phase_scheduler: PhaseScheduler::new(phase_switch_policy),
            batch_composer: BatchComposer::with_packing(batch_config, config.packing.clone()),
            memory: MemoryManager::new(config, num_kv_blocks),
            prefix_cache: RadixTree::new(),
            policy: Box::new(FcfsPolicy::new()),
            running: Vec::new(),
            finished: Vec::new(),
            next_seq_id: 1,
            observers: SchedulerObservers::new(),
            cuda_graph,
            metrics,
        }
    }

    /// Set a custom scheduling policy
    pub fn set_policy(&mut self, policy: Box<dyn SchedulingPolicy>) {
        self.policy = policy;
    }

    /// Make scheduling decision and build batch
    #[must_use]
    pub fn schedule(&mut self) -> Option<Batch> {
        let waiting = self.request_queue.len();
        let running = self.running.len();
        let free_blocks = self.memory.available_blocks();

        tracing::debug!(
            waiting = waiting,
            running = running,
            free_blocks = free_blocks,
            "Scheduling decision"
        );

        let batch = self.build_batch();
        if batch.is_empty() { None } else { Some(batch) }
    }

    /// Check if there are pending requests or running sequences
    #[must_use]
    pub fn has_pending(&self) -> bool {
        !self.request_queue.is_empty() || !self.running.is_empty()
    }

    /// Get the number of running sequences
    #[must_use]
    pub const fn running_count(&self) -> usize {
        self.running.len()
    }

    /// Get the number of waiting sequences
    #[must_use]
    pub fn waiting_count(&self) -> usize {
        self.request_queue.len()
    }

    /// Get prefix cache hit rate (hits / requests); returns 0.0 when no requests.
    // invariant: hits/requests are bounded counters; u64 -> f64 precision loss
    // is acceptable for the hit-rate metric.
    #[allow(clippy::cast_precision_loss)]
    pub fn prefix_cache_hit_rate(&self) -> f64 {
        let hits = self.metrics.prefix_cache_hits();
        let requests = self.metrics.prefix_cache_requests();
        if requests == 0 {
            0.0
        } else {
            hits as f64 / requests as f64
        }
    }

    /// Get running sequences
    pub fn running(&self) -> Vec<Sequence> {
        self.running.clone()
    }

    /// Look up a running sequence by id. Returns `Some(&Sequence)` if found,
    /// `None` if not in the running set. v18.0 — used by the per-seq draft
    /// dispatch to read `Sequence.degraded_draft` and `draft_model_id` without
    /// cloning the whole running vec.
    pub fn get_sequence(&self, seq_id: SeqId) -> Option<&Sequence> {
        self.running.iter().find(|s| s.id == seq_id)
    }

    /// Mutable variant of [`Self::get_sequence`]. v18.0 — used by the per-seq
    /// draft dispatch to set `Sequence.degraded_draft = true` when a draft
    /// runtime error occurs (FALL-02).
    pub fn get_sequence_mut(&mut self, seq_id: SeqId) -> Option<&mut Sequence> {
        self.running.iter_mut().find(|s| s.id == seq_id)
    }

    /// Get finished sequences
    pub fn finished_sequences(&self) -> Vec<Sequence> {
        self.finished.clone()
    }

    /// Clear finished sequences
    pub fn clear_finished(&mut self) {
        self.finished.clear();
    }

    /// # Errors
    ///
    /// Returns `Err` if registration fails (e.g. duplicate name or invalid input).
    /// Register an observer
    pub fn register_observer(
        &mut self,
        observer: Box<dyn SchedulerObserver>,
    ) -> Result<(), SchedulerObserverError> {
        self.observers.register(observer)
    }
}

impl Default for SchedulerEngine {
    fn default() -> Self {
        Self::new(
            SchedulerConfig::default(),
            1024,
            Arc::new(EnhancedMetricsCollector::new()),
        )
    }
}
