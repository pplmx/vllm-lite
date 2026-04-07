use crate::kv_cache::BlockAllocator;
use crate::scheduler::SchedulerStats;
use crate::scheduler::action_executor::ActionExecutor;
use crate::scheduler::batch_planner::{BatchPlanner, SchedulerStateView};
use crate::scheduler::event_handler::EventHandler;
use crate::scheduler::queue_manager::QueueManager;
use crate::types::{BLOCK_SIZE, Batch, Request, SchedulerConfig, SeqId, Sequence, Status};

#[allow(dead_code)]
pub struct SchedulerEngine {
    event_handler: EventHandler,
    action_executor: ActionExecutor,
    queue_manager: QueueManager,
    batch_planner: BatchPlanner,
    kv_allocator: BlockAllocator,
    config: SchedulerConfig,
    stats: SchedulerStats,
    next_seq_id: SeqId,
}

impl SchedulerEngine {
    pub fn new(config: SchedulerConfig, num_kv_blocks: usize) -> Self {
        Self {
            event_handler: EventHandler::new(),
            action_executor: ActionExecutor::new(num_kv_blocks),
            queue_manager: QueueManager::new(),
            batch_planner: BatchPlanner::new(config.clone()),
            kv_allocator: BlockAllocator::new(num_kv_blocks),
            config,
            stats: SchedulerStats::default(),
            next_seq_id: 1,
        }
    }

    pub fn add_request(&mut self, mut req: Request) -> SeqId {
        if req.id == 0 {
            req.id = self.next_seq_id;
            self.next_seq_id += 1;
        }

        let seq_id = req.id;
        let priority = req.priority.clone();

        let prompt_len = req.prompt.len();
        let blocks_needed = prompt_len.div_ceil(BLOCK_SIZE);

        let blocks = self
            .kv_allocator
            .allocate(blocks_needed)
            .unwrap_or_default();

        let seq = Sequence {
            id: seq_id,
            tokens: req.prompt,
            kv_blocks: std::sync::Arc::new(blocks),
            num_computed_tokens: 0,
            prompt_len,
            status: Status::Waiting,
            max_tokens: req.max_tokens,
            sampling_params: req.sampling_params,
            consecutive_decode_rounds: 0,
            priority: priority.clone(),
        };

        self.queue_manager.enqueue(seq, priority);

        seq_id
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
        };

        let plan = self.batch_planner.plan(&state_view);

        let mut seq_ids = Vec::new();
        let mut input_tokens = Vec::new();
        let mut positions = Vec::new();
        let mut kv_block_ids = Vec::new();
        let mut num_computed = Vec::new();
        let mut is_prefill = Vec::new();

        let batch_size = plan.target_batch_size.min(self.queue_manager.len());

        for _ in 0..batch_size {
            if let Some(seq) = self.queue_manager.dequeue() {
                let is_prefilling = seq.status == Status::Waiting;

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

                let blocks_needed = seq.tokens.len().div_ceil(BLOCK_SIZE);
                while seq.kv_blocks.len() < blocks_needed {
                    if let Some(new_blocks) = self.kv_allocator.allocate(1) {
                        let mut blocks = (*seq.kv_blocks).clone();
                        blocks.extend(new_blocks);
                        seq.kv_blocks = std::sync::Arc::new(blocks);
                    } else {
                        self.stats.record_preemption();
                        break;
                    }
                }
            }
        }

        let finished: Vec<_> = self
            .queue_manager
            .all_waiting()
            .into_iter()
            .filter(|s| s.status == Status::Finished)
            .map(|s| s.id)
            .collect();

        for seq_id in finished {
            self.queue_manager.remove(seq_id);
        }
    }

    pub fn has_pending(&self) -> bool {
        !self.queue_manager.is_empty()
    }

    pub fn waiting_count(&self) -> usize {
        self.queue_manager.len()
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
}

impl<'a> SchedulerStateView for SchedulerStateViewImpl<'a> {
    fn waiting_count(&self) -> usize {
        self.queue_manager.len()
    }

    fn running_count(&self) -> usize {
        0
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
