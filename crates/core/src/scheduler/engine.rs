use std::sync::Arc;
use std::time::Instant;

use vllm_traits::{Batch, BlockId, SeqId, TokenId};

use crate::metrics::EnhancedMetricsCollector;
use crate::scheduler::cuda_graph::{GraphBatch, GraphPreparedBatch, SchedulerCudaGraphConfig};
use crate::scheduler::observer::{ObserverEvent, SchedulerObservers};
use crate::scheduler::policy::{FcfsPolicy, SchedulingContext, SchedulingPolicy};
use crate::scheduler::{
    BatchComposer, BatchCompositionConfig, MemoryManager, PhaseScheduler, PhaseSwitchPolicy,
    RadixTree, RequestQueue,
};
use crate::types::{Phase, Request, SchedulerConfig, Sequence, Status};

/// SchedulerEngine - Componentized scheduler architecture
///
/// This engine combines multiple specialized components:
/// - RequestQueue: O(1) lookup and removal with phase-aware indexing
/// - PhaseScheduler: Strict prefill/decode separation with configurable policies
/// - BatchComposer: Phase-specific batch construction
/// - MemoryManager: Block allocation and eviction
/// - RadixTree: Prefix caching for prompt reuse
pub struct SchedulerEngine {
    request_queue: RequestQueue,
    phase_scheduler: PhaseScheduler,
    batch_composer: BatchComposer,
    memory: MemoryManager,
    prefix_cache: RadixTree,
    policy: Box<dyn SchedulingPolicy>,
    #[allow(dead_code)]
    config: SchedulerConfig,
    running: Vec<Sequence>,
    finished: Vec<Sequence>,
    next_seq_id: SeqId,
    observers: SchedulerObservers,
    /// CUDA Graph configuration for decode optimization
    cuda_graph: SchedulerCudaGraphConfig,
    /// Metrics collector for tracking engine performance
    metrics: Arc<EnhancedMetricsCollector>,
}

impl SchedulerEngine {
    /// Create a new SchedulerEngine with the given configuration
    ///
    /// # Arguments
    /// * `config` - Scheduler configuration
    /// * `num_kv_blocks` - Number of KV cache blocks available
    /// * `metrics` - Metrics collector for tracking performance
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
            memory: MemoryManager::new(config.clone(), num_kv_blocks),
            prefix_cache: RadixTree::new(),
            policy: Box::new(FcfsPolicy::new()),
            config,
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
        };

        let ctx = SchedulingContext {
            current_time: Instant::now(),
            queue_length: self.request_queue.len(),
            running_count: self.running.len(),
            memory_pressure: self.get_memory_pressure(),
        };

        self.request_queue
            .enqueue(seq.clone(), self.policy.as_ref(), &ctx);

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

    /// Build batch with potential CUDA Graph routing
    pub fn build_batch_with_graph(&mut self) -> GraphBatch {
        let phase = self
            .phase_scheduler
            .select_phase(&self.get_scheduler_state());
        let sequences = self.select_sequences_for_phase(phase);

        if sequences.is_empty() {
            return GraphBatch::Regular(Batch::empty());
        }

        let batch = self.batch_composer.compose(sequences.clone(), phase);

        tracing::debug!(
            phase = ?phase,
            sequences_count = sequences.len(),
            batch_seq_ids = ?batch.seq_ids,
            batch_input_tokens_count = batch.input_tokens.len(),
            batch_total_tokens = batch.input_tokens.iter().map(|t| t.len()).sum::<usize>(),
            "build_batch_with_graph: built batch"
        );

        // Only use CUDA Graph for decode phase
        match phase {
            Phase::Prefill => GraphBatch::Regular(batch),
            Phase::Decode => {
                let batch_size = batch.seq_ids.len();
                if self.cuda_graph.enabled && self.cuda_graph.supports_batch_size(batch_size) {
                    self.metrics.record_cuda_graph_hit();
                    GraphBatch::Graph(GraphPreparedBatch::new(batch))
                } else {
                    self.metrics.record_cuda_graph_miss();
                    GraphBatch::Regular(batch)
                }
            }
        }
    }

    /// Get current scheduler state for phase selection
    fn get_scheduler_state(&self) -> crate::scheduler::SchedulerState {
        crate::scheduler::SchedulerState {
            waiting_count: self.request_queue.len(),
            running_count: self.running.len(),
            prefill_queue_len: self.request_queue.phase_len(Phase::Prefill),
            decode_queue_len: self.request_queue.phase_len(Phase::Decode),
            available_memory: self.memory.available_blocks(),
            consecutive_decode_rounds: 0,
        }
    }

    /// Select sequences for the given phase
    fn select_sequences_for_phase(&mut self, phase: Phase) -> Vec<Sequence> {
        let mut sequences: Vec<Sequence> = self
            .running
            .iter()
            .filter(|s| s.status == Status::Decoding)
            .cloned()
            .collect();

        let new_sequences = self.request_queue.drain_by_phase(phase);
        sequences.extend(new_sequences.iter().cloned());
        self.running.extend(new_sequences);

        sequences
    }

    /// Update the scheduler after model forward pass
    ///
    /// Processes output tokens, updates sequence status, handles completions,
    /// and adds finished sequences to the prefix cache.
    pub fn update(
        &mut self,
        seq_ids: &[SeqId],
        next_tokens: &[TokenId],
        input_token_counts: &[usize],
    ) {
        let _span = tracing::info_span!(
            "scheduler.update",
            seq_count = seq_ids.len(),
            token_count = next_tokens.len()
        )
        .entered();

        tracing::debug!(
            seq_ids_len = seq_ids.len(),
            next_tokens_len = next_tokens.len(),
            input_counts_len = input_token_counts.len(),
            "Scheduler update"
        );
        for ((&seq_id, &token), &input_count) in
            seq_ids.iter().zip(next_tokens).zip(input_token_counts)
        {
            let _token_span =
                tracing::trace_span!("scheduler.decode_token", seq_id = seq_id, token = token)
                    .entered();

            if let Some(seq) = self.running.iter_mut().find(|s| s.id == seq_id) {
                tracing::debug!(
                    seq_id = seq_id,
                    tokens_len = seq.tokens.len(),
                    status = ?seq.status,
                    max_tokens = seq.max_tokens,
                    "Scheduler update: processing sequence"
                );
                // Update status based on progress
                if seq.status == Status::Waiting || seq.status == Status::Prefilling {
                    seq.num_computed_tokens += input_count;
                    if seq.num_computed_tokens >= seq.prompt_len {
                        seq.status = Status::Decoding;
                        tracing::info!(seq_id = seq_id, "Sequence transitioned to Decode phase");
                    } else {
                        seq.status = Status::Prefilling;
                    }
                }

                seq.tokens.push(token);
                seq.consecutive_decode_rounds += 1;

                // Dispatch observer event for token generation
                self.observers
                    .dispatch(&ObserverEvent::Decoding { seq_id, token });

                // Allocate more blocks if needed
                let blocks_needed = seq.tokens.len().div_ceil(vllm_traits::BLOCK_SIZE);
                while seq.kv_blocks.len() < blocks_needed {
                    if let Some(new_blocks) = self.memory.allocate(1) {
                        let mut blocks = (*seq.kv_blocks).clone();
                        blocks.extend(new_blocks);
                        seq.kv_blocks = Arc::new(blocks);
                    } else {
                        break;
                    }
                }

                // Check completion
                if seq.tokens.len() >= seq.max_tokens {
                    seq.status = Status::Finished;
                    // Add to prefix cache
                    let prompt_tokens = &seq.tokens[..seq.prompt_len];
                    let blocks: Vec<BlockId> = seq.kv_blocks.as_ref().clone();
                    self.prefix_cache.insert(prompt_tokens, blocks);
                }
            }
        }

        // Collect finished sequences and dispatch observer events
        let finished_seqs: Vec<_> = self
            .running
            .iter()
            .filter(|s| s.status == Status::Finished)
            .cloned()
            .collect();

        for seq in &finished_seqs {
            self.observers.dispatch(&ObserverEvent::SequenceFinished {
                seq_id: seq.id,
                total_tokens: seq.tokens.len(),
            });
        }

        for seq in finished_seqs {
            self.memory.release_blocks(seq.kv_blocks.as_ref());
            self.finished.push(seq);
        }

        self.running.retain(|s| s.status != Status::Finished);
    }

    /// Execute preemption to free up memory blocks
    fn execute_preemption(&mut self, blocks_needed: usize) {
        let mut preemptable: Vec<_> = self
            .running
            .iter()
            .filter(|s| s.status == Status::Decoding)
            .cloned()
            .collect();

        preemptable.sort_by(|a, b| {
            b.consecutive_decode_rounds
                .cmp(&a.consecutive_decode_rounds)
        });

        let mut blocks_freed = 0;
        for mut seq in preemptable {
            if blocks_freed >= blocks_needed {
                break;
            }

            let block_count = seq.kv_blocks.len();
            self.memory.release_blocks(seq.kv_blocks.as_ref());
            self.running.retain(|s| s.id != seq.id);

            // Re-queue the preempted sequence
            seq.kv_blocks = Arc::new(vec![]);
            seq.status = Status::Waiting;
            seq.num_computed_tokens = 0;

            let ctx = SchedulingContext {
                current_time: Instant::now(),
                queue_length: self.request_queue.len(),
                running_count: self.running.len(),
                memory_pressure: self.get_memory_pressure(),
            };

            self.request_queue.enqueue(seq, self.policy.as_ref(), &ctx);
            blocks_freed += block_count;
        }
    }

    /// Calculate current memory pressure (0.0 to 1.0)
    fn get_memory_pressure(&self) -> f32 {
        let total = self.memory.total_blocks() as f32;
        let available = self.memory.available_blocks() as f32;
        1.0 - (available / total)
    }

    /// Check if there are pending requests or running sequences
    #[must_use]
    pub fn has_pending(&self) -> bool {
        !self.request_queue.is_empty() || !self.running.is_empty()
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

    /// Get prefix cache hit rate (placeholder - would need stats tracking)
    pub fn prefix_cache_hit_rate(&self) -> f64 {
        // Placeholder - would need stats tracking
        0.0
    }

    /// Cancel a request by sequence ID
    pub fn cancel_request(&mut self, seq_id: SeqId) -> bool {
        if let Some(seq) = self.request_queue.remove(seq_id) {
            // Release blocks if allocated
            if !seq.kv_blocks.is_empty() {
                self.memory.release_blocks(seq.kv_blocks.as_ref());
            }
            return true;
        }
        // Check if it's running
        if let Some(pos) = self.running.iter().position(|s| s.id == seq_id) {
            let seq = self.running.remove(pos);
            if !seq.kv_blocks.is_empty() {
                self.memory.release_blocks(seq.kv_blocks.as_ref());
            }
            return true;
        }
        false
    }

    /// Get KV cache usage statistics
    pub fn get_kv_cache_usage(&self) -> (u64, u64) {
        let total = self.memory.total_blocks() as u64;
        let available = self.memory.available_blocks() as u64;
        let used = total.saturating_sub(available);
        (used, total)
    }

    /// Get running sequences
    pub fn running(&self) -> Vec<Sequence> {
        self.running.clone()
    }

    /// Get finished sequences
    pub fn finished_sequences(&self) -> Vec<Sequence> {
        self.finished.clone()
    }

    /// Clear finished sequences
    pub fn clear_finished(&mut self) {
        self.finished.clear();
    }

    /// Get access to the prefix cache (RadixTree)
    pub fn prefix_cache(&self) -> &RadixTree {
        &self.prefix_cache
    }

    /// Register an observer
    pub fn register_observer(
        &mut self,
        observer: Box<dyn crate::scheduler::observer::SchedulerObserver>,
    ) -> Result<(), String> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Request;
    use vllm_traits::BatchPhase;

    fn create_test_engine(config: SchedulerConfig, num_kv_blocks: usize) -> SchedulerEngine {
        let metrics = Arc::new(EnhancedMetricsCollector::new());
        SchedulerEngine::new(config, num_kv_blocks, metrics)
    }

    #[test]
    fn test_engine_add_request() {
        let config = SchedulerConfig::default();
        let mut engine = create_test_engine(config, 1024);
        let id = engine.add_request(Request::new(0, vec![1, 2, 3], 5));
        assert!(id > 0);
        assert!(engine.has_pending());
        assert_eq!(engine.waiting_count(), 1);
    }

    #[test]
    fn test_engine_build_batch() {
        let config = SchedulerConfig::default();
        let mut engine = create_test_engine(config, 1024);
        engine.add_request(Request::new(0, vec![1, 2, 3], 5));
        let batch = engine.build_batch();
        assert!(!batch.is_empty());
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_engine_batch_phase_is_prefill() {
        let config = SchedulerConfig::default();
        let mut engine = create_test_engine(config, 1024);
        engine.add_request(Request::new(0, vec![1, 2, 3], 5));
        let batch = engine.build_batch();
        assert_eq!(batch.phase, BatchPhase::Prefill);
    }

    #[test]
    fn test_engine_update_sequence() {
        let config = SchedulerConfig::default();
        let mut engine = create_test_engine(config, 1024);
        let id = engine.add_request(Request::new(0, vec![1, 2, 3], 5));
        let _batch = engine.build_batch();

        // Simulate model output: one token generated
        engine.update(&[id], &[100], &[3]); // 3 input tokens processed

        // The sequence should still be in running (not finished yet)
        assert_eq!(engine.running_count(), 1);
    }

    #[test]
    fn test_engine_multiple_requests() {
        let config = SchedulerConfig::default();
        let mut engine = create_test_engine(config, 1024);

        // Add multiple requests
        let id1 = engine.add_request(Request::new(0, vec![1, 2], 5));
        let id2 = engine.add_request(Request::new(0, vec![3, 4], 5));

        assert_eq!(engine.waiting_count(), 2);

        let batch = engine.build_batch();
        assert_eq!(batch.seq_ids.len(), 2);
        assert!(batch.seq_ids.contains(&id1));
        assert!(batch.seq_ids.contains(&id2));
    }

    #[test]
    fn test_engine_memory_pressure() {
        let config = SchedulerConfig::default();
        let mut engine = create_test_engine(config, 10); // Small memory

        // Memory pressure should be 0.0 with all blocks free
        assert_eq!(engine.get_memory_pressure(), 0.0);

        // Add a request
        engine.add_request(Request::new(0, vec![1, 2, 3, 4, 5], 5));

        // After building batch, memory pressure may increase
        let _batch = engine.build_batch();

        // Pressure should be between 0 and 1
        let pressure = engine.get_memory_pressure();
        assert!((0.0..=1.0).contains(&pressure));
    }

    #[test]
    fn test_engine_prefix_cache_hit() {
        let config = SchedulerConfig::default();
        let mut engine = create_test_engine(config, 1024);

        // Add first request
        let prompt = vec![1, 2, 3, 4, 5];
        let id1 = engine.add_request(Request::new(0, prompt.clone(), 5));

        // Build batch and process
        let _batch = engine.build_batch();
        engine.update(&[id1], &[100], &[5]);

        // Complete the sequence to add to cache
        // Update until max_tokens reached
        for i in 0..5 {
            engine.update(&[id1], &[100 + i as u32], &[0]);
        }

        // Add second request with same prefix
        let _id2 = engine.add_request(Request::new(0, vec![1, 2, 3, 6, 7], 5));

        // Second request should be enqueued
        assert!(engine.waiting_count() > 0 || engine.running_count() > 0);
    }

    #[test]
    fn test_engine_metrics_tracking() {
        let config = SchedulerConfig::default();
        let metrics = Arc::new(EnhancedMetricsCollector::new());
        let mut engine = SchedulerEngine::new(config, 1024, metrics.clone());

        // Initially metrics should be zero
        assert_eq!(metrics.get_counter("requests_total"), 0);

        // Add a request
        let _id = engine.add_request(Request::new(0, vec![1, 2, 3], 5));

        // Check metrics were updated
        assert_eq!(metrics.get_counter("requests_total"), 1);

        // Build batch to trigger latency recording
        let _batch = engine.build_batch();

        // Metrics should still track request count
        assert_eq!(metrics.get_counter("requests_total"), 1);
    }
}
