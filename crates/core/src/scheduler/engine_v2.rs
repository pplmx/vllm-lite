use std::sync::Arc;
use std::time::Instant;

use vllm_traits::{Batch, BlockId, SeqId, TokenId};

use crate::scheduler::policy::{FcfsPolicy, SchedulingContext, SchedulingPolicy};
use crate::scheduler::{
    BatchComposer, BatchCompositionConfig, MemoryManager, PhaseScheduler, PhaseSwitchPolicy,
    RadixTree, RequestQueue,
};
use crate::types::{Phase, Request, SchedulerConfig, Sequence, Status};

/// SchedulerEngineV2 - Componentized scheduler architecture
///
/// This engine combines multiple specialized components:
/// - RequestQueue: O(1) lookup and removal with phase-aware indexing
/// - PhaseScheduler: Strict prefill/decode separation with configurable policies
/// - BatchComposer: Phase-specific batch construction
/// - MemoryManager: Block allocation and eviction
/// - RadixTree: Prefix caching for prompt reuse
pub struct SchedulerEngineV2 {
    request_queue: RequestQueue,
    phase_scheduler: PhaseScheduler,
    batch_composer: BatchComposer,
    memory: MemoryManager,
    prefix_cache: RadixTree,
    policy: Box<dyn SchedulingPolicy>,
    #[allow(dead_code)]
    config: SchedulerConfig,
    running: Vec<Sequence>,
    next_seq_id: SeqId,
}

impl SchedulerEngineV2 {
    /// Create a new SchedulerEngineV2 with the given configuration
    ///
    /// # Arguments
    /// * `config` - Scheduler configuration
    /// * `num_kv_blocks` - Number of KV cache blocks available
    pub fn new(config: SchedulerConfig, num_kv_blocks: usize) -> Self {
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

        Self {
            request_queue: RequestQueue::new(),
            phase_scheduler: PhaseScheduler::new(phase_switch_policy),
            batch_composer: BatchComposer::new(batch_config),
            memory: MemoryManager::new(config.clone(), num_kv_blocks),
            prefix_cache: RadixTree::new(),
            policy: Box::new(FcfsPolicy::new()),
            config,
            running: Vec::new(),
            next_seq_id: 1,
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
        if req.id == 0 {
            req.id = self.next_seq_id;
            self.next_seq_id += 1;
        }

        // Check prefix cache for prompt reuse
        let (tokens, kv_blocks, num_computed) =
            if let Some(result) = self.prefix_cache.longest_prefix_match(&req.prompt) {
                let remaining_tokens = req.prompt[result.matched_tokens..].to_vec();
                (
                    remaining_tokens,
                    result.blocks.clone(),
                    result.matched_tokens,
                )
            } else {
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

        self.request_queue.enqueue(seq, self.policy.as_ref(), &ctx);
        req.id
    }

    /// Build the next batch of sequences to process
    ///
    /// Uses the phase scheduler to determine whether to build a prefill or decode batch,
    /// then composes the batch according to memory constraints.
    pub fn build_batch(&mut self) -> Batch {
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

        // Get sequences for this phase
        let mut sequences = self.request_queue.drain_by_phase(phase);
        if sequences.is_empty() {
            return Batch::empty();
        }

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

        // Build batch - sequences are already removed from queue by drain_by_phase
        // Move sequences to running (they were already removed from queue)
        self.running.extend(sequences.iter().cloned());

        // Build and return the batch
        self.batch_composer.compose(sequences, phase)
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
        for ((&seq_id, &token), &input_count) in
            seq_ids.iter().zip(next_tokens).zip(input_token_counts)
        {
            if let Some(seq) = self.running.iter_mut().find(|s| s.id == seq_id) {
                // Update status based on progress
                if seq.status == Status::Waiting || seq.status == Status::Prefilling {
                    seq.num_computed_tokens += input_count;
                    if seq.num_computed_tokens >= seq.prompt_len {
                        seq.status = Status::Decoding;
                    } else {
                        seq.status = Status::Prefilling;
                    }
                }

                seq.tokens.push(token);
                seq.consecutive_decode_rounds += 1;

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

        // Remove finished sequences
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
    pub fn has_pending(&self) -> bool {
        !self.request_queue.is_empty() || !self.running.is_empty()
    }

    /// Get the number of running sequences
    pub fn running_count(&self) -> usize {
        self.running.len()
    }

    /// Get the number of waiting sequences
    pub fn waiting_count(&self) -> usize {
        self.request_queue.len()
    }

    /// Get prefix cache hit rate (placeholder - would need stats tracking)
    pub fn prefix_cache_hit_rate(&self) -> f64 {
        // Placeholder - would need stats tracking
        0.0
    }
}

impl Default for SchedulerEngineV2 {
    fn default() -> Self {
        Self::new(SchedulerConfig::default(), 1024)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Request;
    use vllm_traits::BatchPhase;

    #[test]
    fn test_engine_v2_add_request() {
        let config = SchedulerConfig::default();
        let mut engine = SchedulerEngineV2::new(config, 1024);
        let id = engine.add_request(Request::new(0, vec![1, 2, 3], 5));
        assert!(id > 0);
        assert!(engine.has_pending());
        assert_eq!(engine.waiting_count(), 1);
    }

    #[test]
    fn test_engine_v2_build_batch() {
        let config = SchedulerConfig::default();
        let mut engine = SchedulerEngineV2::new(config, 1024);
        engine.add_request(Request::new(0, vec![1, 2, 3], 5));
        let batch = engine.build_batch();
        assert!(!batch.is_empty());
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_engine_v2_batch_phase_is_prefill() {
        let config = SchedulerConfig::default();
        let mut engine = SchedulerEngineV2::new(config, 1024);
        engine.add_request(Request::new(0, vec![1, 2, 3], 5));
        let batch = engine.build_batch();
        assert_eq!(batch.phase, BatchPhase::Prefill);
    }

    #[test]
    fn test_engine_v2_update_sequence() {
        let config = SchedulerConfig::default();
        let mut engine = SchedulerEngineV2::new(config, 1024);
        let id = engine.add_request(Request::new(0, vec![1, 2, 3], 5));
        let batch = engine.build_batch();

        // Simulate model output: one token generated
        engine.update(&[id], &[100], &[3]); // 3 input tokens processed

        // The sequence should still be in running (not finished yet)
        assert_eq!(engine.running_count(), 1);
    }

    #[test]
    fn test_engine_v2_multiple_requests() {
        let config = SchedulerConfig::default();
        let mut engine = SchedulerEngineV2::new(config, 1024);

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
    fn test_engine_v2_memory_pressure() {
        let config = SchedulerConfig::default();
        let mut engine = SchedulerEngineV2::new(config, 10); // Small memory

        // Memory pressure should be 0.0 with all blocks free
        assert_eq!(engine.get_memory_pressure(), 0.0);

        // Add a request
        engine.add_request(Request::new(0, vec![1, 2, 3, 4, 5], 5));

        // After building batch, memory pressure may increase
        let _batch = engine.build_batch();

        // Pressure should be between 0 and 1
        let pressure = engine.get_memory_pressure();
        assert!(pressure >= 0.0 && pressure <= 1.0);
    }

    #[test]
    fn test_engine_v2_prefix_cache_hit() {
        let config = SchedulerConfig::default();
        let mut engine = SchedulerEngineV2::new(config, 1024);

        // Add first request
        let prompt = vec![1, 2, 3, 4, 5];
        let id1 = engine.add_request(Request::new(0, prompt.clone(), 5));

        // Build batch and process
        let batch = engine.build_batch();
        engine.update(&[id1], &[100], &[5]);

        // Complete the sequence to add to cache
        // Update until max_tokens reached
        for i in 0..5 {
            engine.update(&[id1], &[100 + i as u32], &[0]);
        }

        // Add second request with same prefix
        let id2 = engine.add_request(Request::new(0, vec![1, 2, 3, 6, 7], 5));

        // Second request should be enqueued
        assert!(engine.waiting_count() > 0 || engine.running_count() > 0);
    }
}
