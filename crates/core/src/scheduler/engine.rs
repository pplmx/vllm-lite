use crate::kv_cache::{BlockAllocator, PrefixCache, hash_tokens};
use crate::scheduler::action_executor::ActionExecutor;
use crate::scheduler::batch_planner::{BatchPlanner, SchedulerStateView};
use crate::scheduler::event_handler::EventHandler;
use crate::scheduler::eviction::EvictionPolicy;
use crate::scheduler::preemption::PreemptionManager;
use crate::scheduler::queue_manager::QueueManager;
use crate::types::{BLOCK_SIZE, Batch, Request, SchedulerConfig, SeqId, Sequence, Status};
use std::sync::Arc;
use std::time::Instant;

#[derive(Clone)]
pub struct SchedulerStats {
    pub total_batches: usize,
    pub total_prefill_requests: usize,
    pub total_decode_requests: usize,
    pub total_preemptions: usize,
    pub total_evictions: usize,
    pub avg_batch_size: f64,
    pub last_batch_size: usize,
    pub batch_size_sum: u64,
    pub last_update: Instant,
}

impl Default for SchedulerStats {
    fn default() -> Self {
        Self::new()
    }
}

impl SchedulerStats {
    pub fn new() -> Self {
        Self {
            total_batches: 0,
            total_prefill_requests: 0,
            total_decode_requests: 0,
            total_preemptions: 0,
            total_evictions: 0,
            avg_batch_size: 0.0,
            last_batch_size: 0,
            batch_size_sum: 0,
            last_update: Instant::now(),
        }
    }

    pub fn record_batch(&mut self, batch_size: usize) {
        self.total_batches += 1;
        self.last_batch_size = batch_size;
        self.batch_size_sum += batch_size as u64;
        self.avg_batch_size = self.batch_size_sum as f64 / self.total_batches as f64;
        self.last_update = Instant::now();
    }

    pub fn record_prefill(&mut self) {
        self.total_prefill_requests += 1;
    }

    pub fn record_decode(&mut self) {
        self.total_decode_requests += 1;
    }

    pub fn record_preemption(&mut self) {
        self.total_preemptions += 1;
    }

    pub fn record_eviction(&mut self) {
        self.total_evictions += 1;
    }
}

pub struct SchedulerEngine {
    #[allow(dead_code)]
    event_handler: EventHandler,
    #[allow(dead_code)]
    action_executor: ActionExecutor,
    queue_manager: QueueManager,
    batch_planner: BatchPlanner,
    kv_allocator: BlockAllocator,
    prefix_cache: PrefixCache,
    eviction_policy: EvictionPolicy,
    preemption_manager: PreemptionManager,
    config: SchedulerConfig,
    stats: SchedulerStats,
    next_seq_id: SeqId,
    running: Vec<Sequence>,
    finished: Vec<Sequence>,
}

impl SchedulerEngine {
    pub fn new(config: SchedulerConfig, num_kv_blocks: usize) -> Self {
        Self {
            event_handler: EventHandler::new(),
            action_executor: ActionExecutor::new(num_kv_blocks),
            queue_manager: QueueManager::new(),
            batch_planner: BatchPlanner::new(config.clone()),
            kv_allocator: BlockAllocator::new(num_kv_blocks),
            prefix_cache: PrefixCache::new(),
            eviction_policy: EvictionPolicy::new(),
            preemption_manager: PreemptionManager::new(config.clone()),
            config,
            stats: SchedulerStats::new(),
            next_seq_id: 1,
            running: Vec::new(),
            finished: Vec::new(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn create_sequence(
        seq_id: SeqId,
        prompt: Vec<u32>,
        kv_blocks: Arc<Vec<usize>>,
        num_computed_tokens: usize,
        prompt_len: usize,
        max_tokens: usize,
        sampling_params: crate::types::SamplingParams,
        priority: crate::types::Priority,
        status: Status,
    ) -> Sequence {
        Sequence {
            id: seq_id,
            tokens: prompt,
            kv_blocks,
            num_computed_tokens,
            prompt_len,
            status,
            max_tokens,
            sampling_params,
            consecutive_decode_rounds: 0,
            priority,
        }
    }

    fn check_prefix_cache(&mut self, prompt: &[u32]) -> Option<(Arc<Vec<usize>>, usize)> {
        let key = hash_tokens(prompt);
        if let Some(entry) = self.prefix_cache.get(key) {
            return Some((entry.blocks.clone(), entry.token_count));
        }
        if let Some(entry) = self.prefix_cache.find_prefix_match(prompt) {
            return Some((entry.blocks.clone(), entry.token_count));
        }
        if let Some((blocks, cached_tokens)) = self.prefix_cache.find_reverse_prefix_match(prompt) {
            return Some((blocks, cached_tokens));
        }
        None
    }

    fn insert_into_prefix_cache(&mut self, seq: Sequence) {
        let prompt_tokens = &seq.tokens[..seq.prompt_len];
        let key = hash_tokens(prompt_tokens);
        if !self.prefix_cache.contains_key(&key) {
            self.prefix_cache
                .insert(key, seq.kv_blocks.to_vec(), seq.prompt_len);
        }
        self.finished.push(seq);
    }

    pub fn add_request(&mut self, mut req: Request) -> SeqId {
        if req.id == 0 {
            req.id = self.next_seq_id;
            self.next_seq_id += 1;
        }

        let seq_id = req.id;
        let priority = req.priority.clone();
        let prompt_len = req.prompt.len();

        if let Some((blocks, token_count)) = self.check_prefix_cache(&req.prompt) {
            let status = if token_count >= prompt_len {
                Status::Waiting
            } else {
                Status::Prefilling
            };

            let seq = Self::create_sequence(
                seq_id,
                req.prompt,
                blocks,
                token_count,
                prompt_len,
                req.max_tokens,
                req.sampling_params,
                priority.clone(),
                status,
            );

            self.queue_manager.enqueue(seq.clone(), priority);
            self.running.push(seq);
            return seq_id;
        }

        let blocks_needed = prompt_len.div_ceil(BLOCK_SIZE);

        if blocks_needed > self.kv_allocator.available() {
            let should_preempt = self.preemption_manager.should_preempt(
                self.running.len(),
                self.queue_manager.len(),
                blocks_needed,
                self.kv_allocator.available(),
            );

            if should_preempt {
                self.execute_preemption(blocks_needed);
            }
        }

        let blocks = self
            .kv_allocator
            .allocate(blocks_needed)
            .unwrap_or_default();

        let seq = Self::create_sequence(
            seq_id,
            req.prompt,
            Arc::new(blocks),
            0,
            prompt_len,
            req.max_tokens,
            req.sampling_params,
            priority.clone(),
            Status::Waiting,
        );

        self.queue_manager.enqueue(seq, priority);

        seq_id
    }

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
        for seq in preemptable.iter() {
            if blocks_freed >= blocks_needed {
                break;
            }

            let block_count = seq.kv_blocks.len();

            self.eviction_policy.release_blocks(seq.kv_blocks.as_ref());
            self.kv_allocator.free(seq.kv_blocks.as_ref());

            if let Some(running_seq) = self.running.iter_mut().find(|s| s.id == seq.id) {
                running_seq.status = Status::Waiting;
                running_seq.kv_blocks = Arc::new(vec![]);
            }

            if let Some(running_seq) = self.running.iter().find(|s| s.id == seq.id) {
                self.queue_manager.enqueue_preempted(running_seq.clone());
                self.running.retain(|s| s.id != seq.id);
            }

            blocks_freed += block_count;
            self.stats.record_preemption();
        }
    }

    pub fn build_batch(&mut self) -> Batch {
        if self.queue_manager.is_empty() {
            return Batch {
                seq_ids: Vec::new(),
                input_tokens: Vec::new(),
                positions: Vec::new(),
                kv_block_ids: Vec::new(),
                num_computed_tokens: Vec::new(),
                is_prefill: Vec::new(),
            };
        }

        let state_view = SchedulerStateViewImpl {
            queue_manager: &self.queue_manager,
            kv_allocator: &self.kv_allocator,
            running: &self.running,
        };

        let plan = self.batch_planner.plan(&state_view);

        let mut seq_ids = Vec::new();
        let mut input_tokens = Vec::new();
        let mut positions = Vec::new();
        let mut kv_block_ids = Vec::new();
        let mut num_computed = Vec::new();
        let mut is_prefill = Vec::new();

        let batch_size = plan.target_batch_size.min(self.queue_manager.len());
        let max_tokens = self.config.max_num_batched_tokens;

        // Get sequences from queue for batch (peek, don't remove)
        let all_seqs: Vec<_> = {
            let all = self.queue_manager.all_waiting();
            all.into_iter().take(batch_size).cloned().collect()
        };

        // Apply token budget: collect sequences until we hit max_tokens limit
        let mut current_tokens = 0;
        let mut batch_seqs = Vec::new();

        for seq in all_seqs {
            let is_prefilling = seq.status == Status::Waiting;
            let seq_tokens = if is_prefilling {
                seq.tokens.len().saturating_sub(seq.num_computed_tokens)
            } else {
                1 // decode is 1 token
            };

            // Check if adding this sequence would exceed token budget
            if current_tokens + seq_tokens > max_tokens {
                break;
            }

            current_tokens += seq_tokens;
            batch_seqs.push(seq);
        }

        for seq in batch_seqs {
            let is_prefilling = seq.status == Status::Waiting;

            // Update status and track running sequence
            let mut running_seq = seq.clone();
            if is_prefilling {
                running_seq.status = Status::Prefilling;
            }
            self.running.push(running_seq);

            self.eviction_policy.record_blocks(&seq.kv_blocks);

            let start = seq.num_computed_tokens;
            let tokens: Vec<_> = if is_prefilling {
                seq.tokens[start..].to_vec()
            } else {
                seq.tokens.last().map(|t| vec![*t]).unwrap_or_default()
            };

            let pos: Vec<usize> = (start..start + tokens.len()).collect();

            seq_ids.push(seq.id);
            input_tokens.push(tokens);
            positions.push(pos);
            kv_block_ids.push(seq.kv_blocks.as_ref().clone());
            num_computed.push(seq.num_computed_tokens);
            is_prefill.push(is_prefilling);

            self.stats.record_batch(1);
            if is_prefilling {
                self.stats.record_prefill();
            } else {
                self.stats.record_decode();
            }
        }

        Batch {
            seq_ids,
            input_tokens,
            positions,
            kv_block_ids,
            num_computed_tokens: num_computed,
            is_prefill,
        }
    }

    pub fn update(&mut self, seq_ids: &[SeqId], next_tokens: &[u32], input_token_counts: &[usize]) {
        for ((seq_id, &token), &input_count) in
            seq_ids.iter().zip(next_tokens).zip(input_token_counts)
        {
            let mut updated_seq: Option<Sequence> = None;

            let running_seqs: Vec<_> = self
                .running
                .iter()
                .filter(|s| s.id != *seq_id && s.status != Status::Finished)
                .cloned()
                .collect();

            if let Some(seq) = self.running.iter_mut().find(|s| s.id == *seq_id) {
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

                if seq.tokens.len() >= seq.max_tokens {
                    seq.status = Status::Finished;
                }

                let blocks_needed = seq.tokens.len().div_ceil(BLOCK_SIZE);
                while seq.kv_blocks.len() < blocks_needed {
                    if let Some(new_blocks) = self.kv_allocator.allocate(1) {
                        let mut blocks = (*seq.kv_blocks).clone();
                        blocks.extend(new_blocks);
                        seq.kv_blocks = Arc::new(blocks);
                    } else {
                        let victims = self.eviction_policy.select_victims(&running_seqs, 1);

                        if victims.is_empty() {
                            self.stats.record_preemption();
                            break;
                        }

                        self.eviction_policy.release_blocks(&victims);
                        self.kv_allocator.free(&victims);
                        self.stats.record_eviction();

                        if let Some(new_blocks) = self.kv_allocator.allocate(1) {
                            let mut blocks = (*seq.kv_blocks).clone();
                            blocks.extend(new_blocks);
                            seq.kv_blocks = Arc::new(blocks);
                        } else {
                            self.stats.record_preemption();
                            break;
                        }
                    }
                }

                if seq.status == Status::Finished {
                    updated_seq = Some(seq.clone());
                }
            }

            // Also check queue_manager for any sequences that didn't go through running
            if let Some(seq) = self.queue_manager.get_mut(*seq_id) {
                if seq.status == Status::Waiting {
                    seq.num_computed_tokens += input_count;
                    if seq.num_computed_tokens >= seq.prompt_len {
                        seq.status = Status::Decoding;
                    } else {
                        seq.status = Status::Prefilling;
                    }
                }

                seq.tokens.push(token);
                seq.consecutive_decode_rounds += 1;

                if seq.tokens.len() >= seq.max_tokens {
                    seq.status = Status::Finished;
                }

                if seq.status == Status::Finished {
                    updated_seq = Some(seq.clone());
                }
            }

            // Handle finished sequence
            if let Some(finished_seq) = updated_seq {
                self.running.retain(|s| s.id != finished_seq.id);
                if let Some(seq) = self.queue_manager.remove(finished_seq.id) {
                    self.insert_into_prefix_cache(seq);
                }
            }
        }

        let finished_ids: Vec<_> = self
            .queue_manager
            .all_waiting()
            .into_iter()
            .filter(|s| s.status == Status::Finished)
            .map(|s| s.id)
            .collect();

        for seq_id in finished_ids {
            if let Some(seq) = self.queue_manager.remove(seq_id) {
                self.insert_into_prefix_cache(seq);
            }
            self.running.retain(|s| s.id != seq_id);
        }
    }

    pub fn has_pending(&self) -> bool {
        !self.queue_manager.is_empty() || !self.running.is_empty()
    }

    pub fn running(&self) -> Vec<Sequence> {
        self.running.clone()
    }

    pub fn finished_sequences(&self) -> Vec<Sequence> {
        self.finished.clone()
    }

    pub fn waiting_count(&self) -> usize {
        self.queue_manager.len()
    }

    pub fn running_count(&self) -> usize {
        self.running.len()
    }

    pub fn prefix_cache(&self) -> &PrefixCache {
        &self.prefix_cache
    }

    pub fn eviction(&self) -> crate::scheduler::eviction::EvictionPolicy {
        crate::scheduler::eviction::EvictionPolicy::new()
    }

    pub fn get_stats(&self) -> SchedulerStats {
        self.stats.clone()
    }

    pub fn get_kv_cache_usage(&self) -> (u64, u64) {
        let total = self.kv_allocator.total() as u64;
        let available = self.kv_allocator.available() as u64;
        let used = total.saturating_sub(available);
        (used, total)
    }
}

struct SchedulerStateViewImpl<'a> {
    queue_manager: &'a QueueManager,
    kv_allocator: &'a BlockAllocator,
    running: &'a Vec<Sequence>,
}

impl<'a> SchedulerStateView for SchedulerStateViewImpl<'a> {
    fn waiting_count(&self) -> usize {
        self.queue_manager.len()
    }

    fn running_count(&self) -> usize {
        self.running.len()
    }

    fn prefill_count(&self) -> usize {
        self.queue_manager
            .filter_by_status(Status::Prefilling)
            .len()
    }

    fn decode_count(&self) -> usize {
        self.queue_manager.filter_by_status(Status::Decoding).len()
    }

    fn available_memory(&self) -> usize {
        self.kv_allocator.available()
    }
}

impl Default for SchedulerEngine {
    fn default() -> Self {
        Self::new(SchedulerConfig::default(), 1024)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_add_request() {
        let mut engine = SchedulerEngine::default();
        let id = engine.add_request(Request::new(0, vec![1, 2, 3], 5));
        assert!(id > 0);
        assert!(engine.has_pending());
    }

    #[test]
    fn test_engine_build_batch() {
        let mut engine = SchedulerEngine::default();
        engine.add_request(Request::new(0, vec![1, 2, 3], 5));
        let batch = engine.build_batch();
        assert!(!batch.is_empty() || !engine.has_pending());
    }

    #[test]
    fn test_engine_has_pending() {
        let mut engine = SchedulerEngine::default();
        assert!(!engine.has_pending());

        engine.add_request(Request::new(0, vec![1, 2, 3], 5));
        assert!(engine.has_pending());
    }
}
