//! `SchedulerEngine` struct definition, constructor, and state accessors.
//!
//! This module owns the `SchedulerEngine` data layout and the lifecycle
//! methods that touch overall engine state (`new`, `add_request`,
//! `build_batch`, `schedule`, `set_policy`) plus the read-only and minor
//! mutating accessors used by the rest of the engine and the HTTP layer.

use std::sync::Arc;
use std::time::Instant;

use vllm_traits::{Batch, SeqId};

use crate::metrics::EnhancedMetricsCollector;
use crate::scheduler::cuda_graph::SchedulerCudaGraphConfig;
use crate::scheduler::observer::{
    ObserverEvent, SchedulerObserver, SchedulerObserverError, SchedulerObservers,
};
use crate::scheduler::policy::{FcfsPolicy, SchedulingContext, SchedulingPolicy};
use crate::scheduler::{
    BatchComposer, BatchCompositionConfig, MemoryManager, PhaseScheduler, PhaseSwitchPolicy,
    RadixTree, RequestQueue,
};
use crate::types::{Phase, Request, SchedulerConfig, Sequence, Status};

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
            .finish()
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

    /// Add a new request to the scheduler
    ///
    /// Checks the prefix cache for matching prompts and creates a sequence.
    /// Returns the assigned sequence ID.
    pub fn add_request(&mut self, mut req: Request) -> SeqId {
        let _span = tracing::info_span!(
            "scheduler.add_request",
            request_id = req.id,
            prompt_len = req.prompt.len(),
            max_tokens = req.max_tokens
        )
        .entered();

        // Record metrics: request received
        self.metrics.record_request();

        if req.id == 0 {
            req.id = self.next_seq_id;
            self.next_seq_id += 1;
        }

        // Check prefix cache for prompt reuse
        let (tokens, kv_blocks, num_computed) =
            if let Some(result) = self.prefix_cache.longest_prefix_match(&req.prompt) {
                tracing::trace!(
                    request_id = req.id,
                    matched_tokens = result.matched_tokens,
                    "Prefix cache hit"
                );
                (
                    req.prompt.clone(),
                    result.blocks.clone(),
                    result.matched_tokens,
                )
            } else {
                tracing::trace!(request_id = req.id, "Prefix cache miss");
                (req.prompt.clone(), Arc::new(vec![]), 0)
            };

        let seq = Sequence {
            id: req.id,
            tokens,
            kv_blocks,
            num_computed_tokens: num_computed,
            prompt_len: req.prompt.len(),
            status: if num_computed >= req.prompt.len() {
                Status::Waiting
            } else {
                Status::Prefilling
            },
            max_tokens: req.max_tokens,
            sampling_params: req.sampling_params,
            consecutive_decode_rounds: 0,
            priority: req.priority,
            degraded_draft: false,
            draft_model_id: req.draft_model_id.clone(),
        };

        let ctx = SchedulingContext {
            current_time: Instant::now(),
            queue_length: self.request_queue.len(),
            running_count: self.running.len(),
            memory_pressure: self.get_memory_pressure(),
        };

        self.request_queue.enqueue(seq, self.policy.as_ref(), &ctx);

        // Update metrics: queue depth
        self.metrics
            .set_queue_depth(self.request_queue.len() as u64);

        // Dispatch observer event
        self.observers.dispatch(&ObserverEvent::RequestArrived {
            seq_id: req.id,
            prompt_len: req.prompt.len(),
        });

        tracing::info!(
            request_id = req.id,
            queue_depth = self.request_queue.len(),
            "Request added"
        );
        req.id
    }

    /// Build the next batch of sequences to process
    ///
    /// Uses the phase scheduler to determine whether to build a prefill or decode batch,
    /// then composes the batch according to memory constraints.
    #[must_use]
    pub fn build_batch(&mut self) -> Batch {
        let _span = tracing::info_span!(
            "scheduler.build_batch",
            waiting = self.request_queue.len(),
            running = self.running.len()
        )
        .entered();

        let start_time = Instant::now();

        // Get current scheduler state
        let state = crate::scheduler::SchedulerState {
            waiting_count: self.request_queue.len(),
            running_count: self.running.len(),
            prefill_queue_len: self.request_queue.phase_len(Phase::Prefill),
            decode_queue_len: self.request_queue.phase_len(Phase::Decode),
            available_memory: self.memory.available_blocks(),
            consecutive_decode_rounds: 0,
        };

        let phase = self.phase_scheduler.select_phase(&state);

        // Only include running decode sequences when in Decode phase
        let mut sequences: Vec<Sequence> = if phase == Phase::Decode {
            self.running
                .iter()
                .filter(|s| s.status == Status::Decoding)
                .cloned()
                .collect()
        } else {
            Vec::new()
        };

        // Get sequences for this phase from the queue
        let new_sequences = self.request_queue.drain_by_phase(phase);

        // Update metrics: queue depth after draining
        self.metrics
            .set_queue_depth(self.request_queue.len() as u64);

        // If no running decode sequences and no new sequences, return empty
        if sequences.is_empty() && new_sequences.is_empty() {
            return Batch::empty();
        }

        // Add new sequences to the batch
        sequences.extend(new_sequences.iter().cloned());

        // Sort by policy priority
        let ctx = SchedulingContext {
            current_time: Instant::now(),
            queue_length: self.request_queue.len(),
            running_count: self.running.len(),
            memory_pressure: self.get_memory_pressure(),
        };

        sequences.sort_by(|a, b| {
            let pa = self.policy.compute_priority(a, &ctx);
            let pb = self.policy.compute_priority(b, &ctx);
            pa.cmp(&pb)
        });

        // Check memory and preempt if needed
        for seq in &sequences {
            let blocks_needed = seq.tokens.len().div_ceil(vllm_traits::BLOCK_SIZE);
            if blocks_needed > self.memory.available_blocks() {
                self.execute_preemption(blocks_needed);
            }
        }

        // Move new sequences to running
        self.running.extend(new_sequences);

        // Update metrics: active sequences
        self.metrics.set_active_sequences(self.running.len() as u64);

        // Build the batch
        let batch = self.batch_composer.compose(sequences.clone(), phase);

        // Record CUDA Graph metrics if applicable
        if phase == Phase::Decode && self.cuda_graph.enabled {
            let batch_size = batch.seq_ids.len();
            if self.cuda_graph.supports_batch_size(batch_size) {
                self.metrics.record_cuda_graph_hit();
            } else {
                self.metrics.record_cuda_graph_miss();
            }
        }

        // Dispatch observer event
        if !batch.seq_ids.is_empty() {
            self.observers.dispatch(&ObserverEvent::BatchScheduled {
                seq_ids: batch.seq_ids.clone(),
                batch_size: batch.seq_ids.len(),
            });
        }

        // Record batch scheduling latency
        let duration = start_time.elapsed();
        self.metrics
            .record_inference_latency(duration.as_nanos() as u64);

        let prefill_count = batch.is_prefill.iter().filter(|&&x| x).count();
        tracing::debug!(
            batch_size = batch.seq_ids.len(),
            prefill_count = prefill_count,
            total_tokens = batch.total_tokens,
            phase = ?batch.phase,
            "Batch built"
        );

        batch
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
    pub fn running_count(&self) -> usize {
        self.running.len()
    }

    /// Get the number of waiting sequences
    #[must_use]
    pub fn waiting_count(&self) -> usize {
        self.request_queue.len()
    }

    /// Get prefix cache hit rate (hits / requests); returns 0.0 when no requests.
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
