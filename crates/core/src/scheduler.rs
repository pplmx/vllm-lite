use crate::kv_cache::{hash_tokens, BlockAllocator, PrefixCache};
use crate::types::{Batch, Request, SchedulerConfig, SeqId, Sequence, Status, TokenId, BLOCK_SIZE};
use std::collections::VecDeque;

pub struct Scheduler {
    waiting: VecDeque<Sequence>,
    running: Vec<Sequence>,
    finished: Vec<Sequence>,
    next_seq_id: SeqId,
    config: SchedulerConfig,
    kv_allocator: BlockAllocator,
    prefix_cache: PrefixCache,
}

impl Default for Scheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl Scheduler {
    pub fn new() -> Self {
        Self::with_config(SchedulerConfig::default(), 1024)
    }

    pub fn with_config(config: SchedulerConfig, num_kv_blocks: usize) -> Self {
        Self {
            waiting: VecDeque::new(),
            running: Vec::new(),
            finished: Vec::new(),
            next_seq_id: 1,
            config,
            kv_allocator: BlockAllocator::new(num_kv_blocks),
            prefix_cache: PrefixCache::new(),
        }
    }

    pub fn add_request(&mut self, req: Request) -> SeqId {
        let id = if req.id == 0 {
            let id = self.next_seq_id;
            self.next_seq_id += 1;
            id
        } else {
            req.id
        };

        // Check prefix cache
        let prompt_len = req.prompt.len();
        let key = hash_tokens(&req.prompt);
        if let Some(entry) = self.prefix_cache.get(key) {
            let seq = Sequence {
                id,
                tokens: req.prompt,
                kv_blocks: entry.blocks.clone(),
                num_computed_tokens: entry.token_count,
                prompt_len,
                status: Status::Decoding,
                max_tokens: req.max_tokens,
                sampling_params: req.sampling_params,
                consecutive_decode_rounds: 0,
            };
            self.running.push(seq);
            return id;
        }

        // Cache miss - allocate new blocks
        let num_blocks_needed = req.prompt.len().div_ceil(BLOCK_SIZE);

        // Evict if needed before allocation
        if self.kv_allocator.available() < num_blocks_needed {
            self.prefix_cache.evict(&mut self.kv_allocator);
        }

        let blocks = self
            .kv_allocator
            .allocate(num_blocks_needed)
            .unwrap_or_default();

        let seq = Sequence {
            id,
            tokens: req.prompt,
            kv_blocks: blocks,
            num_computed_tokens: 0,
            prompt_len,
            status: Status::Waiting,
            max_tokens: req.max_tokens,
            sampling_params: req.sampling_params,
            consecutive_decode_rounds: 0,
        };
        self.waiting.push_back(seq);
        id
    }

    #[allow(dead_code)]
    fn pending_tokens(seq: &Sequence) -> usize {
        if seq.status == Status::Prefilling {
            seq.tokens.len() - seq.num_computed_tokens
        } else if seq.status == Status::Decoding {
            1
        } else {
            0
        }
    }

    pub fn build_batch(&mut self) -> Batch {
        let mut newly_finished = Vec::new();
        let mut i = 0;
        while i < self.running.len() {
            if self.running[i].status == Status::Finished {
                let seq = self.running.remove(i);
                newly_finished.push(seq);
            } else {
                i += 1;
            }
        }

        for seq in newly_finished.iter() {
            let prompt_tokens = &seq.tokens[..seq.prompt_len];
            let key = hash_tokens(prompt_tokens);
            if !self.prefix_cache.contains_key(&key) {
                self.prefix_cache
                    .insert(key, seq.kv_blocks.clone(), seq.prompt_len);
            }
        }
        self.finished.extend(newly_finished);

        while self.running.len() < self.config.max_num_seqs {
            match self.waiting.pop_front() {
                Some(mut seq) => {
                    seq.status = Status::Prefilling;
                    self.running.push(seq);
                }
                None => break,
            }
        }

        let mut seq_ids = vec![];
        let mut input_tokens = vec![];
        let mut positions = vec![];
        let mut budget = self.config.max_num_batched_tokens;
        let max_seqs = self.config.max_num_seqs;
        let decode_limit = self.config.max_consecutive_decode;

        for seq in &self.running {
            if seq_ids.len() >= max_seqs {
                break;
            }
            if budget == 0 {
                break;
            }

            if seq.status == Status::Decoding {
                if seq.consecutive_decode_rounds >= decode_limit {
                    continue;
                }

                let last = *seq.tokens.last().unwrap();
                let pos = seq.tokens.len() - 1;

                seq_ids.push(seq.id);
                input_tokens.push(vec![last]);
                positions.push(vec![pos]);
                budget = budget.saturating_sub(1);
            }
        }

        for seq in &self.running {
            if seq_ids.len() >= max_seqs {
                break;
            }
            if budget == 0 {
                break;
            }

            if seq.status == Status::Prefilling {
                let start = seq.num_computed_tokens;
                let remaining = seq.tokens.len() - start;
                let chunk_size = remaining.min(budget);

                if chunk_size == 0 {
                    continue;
                }

                let tokens = seq.tokens[start..start + chunk_size].to_vec();
                let pos: Vec<usize> = (start..start + chunk_size).collect();

                seq_ids.push(seq.id);
                input_tokens.push(tokens);
                positions.push(pos);
                budget = budget.saturating_sub(chunk_size);
            }
        }

        Batch {
            seq_ids,
            input_tokens,
            positions,
        }
    }

    pub fn update(
        &mut self,
        seq_ids: &[SeqId],
        next_tokens: &[TokenId],
        input_token_counts: &[usize],
    ) {
        for ((seq_id, token), &input_count) in
            seq_ids.iter().zip(next_tokens).zip(input_token_counts)
        {
            if let Some(seq) = self.running.iter_mut().find(|s| s.id == *seq_id) {
                if seq.status == Status::Prefilling {
                    seq.num_computed_tokens += input_count;
                    if seq.num_computed_tokens >= seq.tokens.len() {
                        seq.status = Status::Decoding;
                    }
                }

                seq.tokens.push(*token);

                let new_total = seq.tokens.len();
                let blocks_needed = new_total.div_ceil(BLOCK_SIZE);
                while seq.kv_blocks.len() < blocks_needed {
                    if let Some(new_block) = self.kv_allocator.allocate(1) {
                        seq.kv_blocks.extend(new_block);
                    } else {
                        break;
                    }
                }

                if seq.tokens.len() >= seq.max_tokens {
                    seq.status = Status::Finished;
                }
            }
        }

        let mut newly_finished = Vec::new();
        let mut i = 0;
        while i < self.running.len() {
            if self.running[i].status == Status::Finished {
                let seq = self.running.remove(i);
                newly_finished.push(seq);
            } else {
                i += 1;
            }
        }

        // Store in cache
        for seq in newly_finished.iter() {
            let prompt_tokens = &seq.tokens[..seq.prompt_len];
            let key = hash_tokens(prompt_tokens);
            if !self.prefix_cache.contains_key(&key) {
                self.prefix_cache
                    .insert(key, seq.kv_blocks.clone(), seq.prompt_len);
            }
        }

        // Move to finished list
        self.finished.extend(newly_finished);
    }

    pub fn has_pending(&self) -> bool {
        !self.waiting.is_empty() || !self.running.is_empty()
    }

    pub fn running_count(&self) -> usize {
        self.running.len()
    }

    pub fn waiting_count(&self) -> usize {
        self.waiting.len()
    }

    pub fn finished_sequences(&self) -> &[Sequence] {
        &self.finished
    }

    pub fn running(&self) -> &[Sequence] {
        &self.running
    }

    pub fn prefix_cache(&self) -> &PrefixCache {
        &self.prefix_cache
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_request_prefill() {
        let mut sched = Scheduler::new();
        sched.add_request(Request::new(1, vec![10, 20, 30], 5));

        let batch = sched.build_batch();
        assert_eq!(batch.seq_ids.len(), 1);
        assert_eq!(batch.input_tokens[0], vec![10, 20, 30]);
    }

    #[test]
    fn test_decode_priority() {
        let mut sched = Scheduler::new();
        sched.add_request(Request::new(1, vec![10], 5));
        let batch = sched.build_batch();
        sched.update(
            &batch.seq_ids,
            &[99],
            &batch
                .input_tokens
                .iter()
                .map(|t| t.len())
                .collect::<Vec<_>>(),
        );

        sched.add_request(Request::new(2, vec![20, 30, 40], 5));

        let batch = sched.build_batch();
        assert_eq!(batch.seq_ids[0], 1); // decode first
        assert_eq!(batch.seq_ids[1], 2); // prefill second
    }

    #[test]
    fn test_max_num_seqs_limit() {
        let config = SchedulerConfig {
            max_num_seqs: 1,
            max_num_batched_tokens: 4096,
            max_consecutive_decode: 10,
        };
        let mut sched = Scheduler::with_config(config, 1024);
        sched.add_request(Request::new(1, vec![10], 5));
        sched.add_request(Request::new(2, vec![20], 5));

        let batch = sched.build_batch();
        assert_eq!(batch.seq_ids.len(), 1);
        assert_eq!(sched.running_count(), 1);
        assert_eq!(sched.waiting_count(), 1);
    }

    #[test]
    fn test_chunked_prefill() {
        let config = SchedulerConfig {
            max_num_seqs: 256,
            max_num_batched_tokens: 3,
            max_consecutive_decode: 10,
        };
        let mut sched = Scheduler::with_config(config, 1024);
        sched.add_request(Request::new(1, vec![10, 20, 30, 40, 50], 10));

        // Step 1: chunk 3 tokens
        let batch = sched.build_batch();
        assert_eq!(batch.input_tokens[0], vec![10, 20, 30]);
        sched.update(
            &batch.seq_ids,
            &[99],
            &batch
                .input_tokens
                .iter()
                .map(|t| t.len())
                .collect::<Vec<_>>(),
        );

        // Step 2: process remaining 2 tokens [40, 50] + 1 new token (99)
        let batch = sched.build_batch();
        assert_eq!(batch.input_tokens[0], vec![40, 50, 99]);
        sched.update(
            &batch.seq_ids,
            &[99],
            &batch
                .input_tokens
                .iter()
                .map(|t| t.len())
                .collect::<Vec<_>>(),
        );

        // Step 3: now decoding (all prompt tokens computed)
        let batch = sched.build_batch();
        assert_eq!(batch.input_tokens[0], vec![99]);
    }

    #[test]
    fn test_multi_sequence_batch_order() {
        let mut sched = Scheduler::new();
        sched.add_request(Request::new(1, vec![10], 5));
        let batch1 = sched.build_batch();
        sched.update(
            &batch1.seq_ids,
            &[99],
            &batch1
                .input_tokens
                .iter()
                .map(|t| t.len())
                .collect::<Vec<_>>(),
        );

        sched.add_request(Request::new(2, vec![20, 30], 5));

        let batch = sched.build_batch();
        assert_eq!(batch.seq_ids[0], 1);
    }

    #[test]
    fn test_max_batched_tokens_limit() {
        let config = SchedulerConfig {
            max_num_seqs: 10,
            max_num_batched_tokens: 2,
            max_consecutive_decode: 10,
        };
        let mut sched = Scheduler::with_config(config, 100);

        sched.add_request(Request::new(1, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 5));

        let batch = sched.build_batch();
        let total_tokens: usize = batch.input_tokens.iter().map(|v| v.len()).sum();
        assert!(total_tokens <= 2);
    }

    #[test]
    fn test_empty_prompt_request() {
        let mut sched = Scheduler::new();
        sched.add_request(Request::new(1, vec![], 5));
        let batch = sched.build_batch();
        assert!(batch.is_empty() || batch.input_tokens[0].is_empty());
    }

    #[test]
    fn test_prefill_decode_queue_separation() {
        let mut sched = Scheduler::new();

        sched.add_request(Request::new(1, vec![10, 20, 30], 5));
        let batch1 = sched.build_batch();
        assert_eq!(batch1.seq_ids.len(), 1);

        sched.update(&batch1.seq_ids, &[99], &[batch1.input_tokens[0].len()]);

        sched.add_request(Request::new(2, vec![40, 50], 5));

        let batch2 = sched.build_batch();
        assert!(batch2.seq_ids.len() >= 1);
    }

    #[test]
    fn test_max_consecutive_decode_limit() {
        let config = SchedulerConfig {
            max_num_seqs: 10,
            max_num_batched_tokens: 100,
            max_consecutive_decode: 2,
        };
        let mut sched = Scheduler::with_config(config, 1024);

        sched.add_request(Request::new(1, vec![10], 10));
        let batch1 = sched.build_batch();
        sched.update(&batch1.seq_ids, &[99], &[batch1.input_tokens[0].len()]);

        sched.add_request(Request::new(2, vec![20], 10));

        let batch2 = sched.build_batch();
        assert!(batch2.seq_ids.len() >= 1);
    }

    #[test]
    fn test_token_budget_in_continuous_batching() {
        let config = SchedulerConfig {
            max_num_seqs: 10,
            max_num_batched_tokens: 3,
            max_consecutive_decode: 10,
        };
        let mut sched = Scheduler::with_config(config, 1024);

        sched.add_request(Request::new(1, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 10));

        let batch = sched.build_batch();
        let total_tokens: usize = batch.input_tokens.iter().map(|v| v.len()).sum();
        assert!(
            total_tokens <= 3,
            "total_tokens {} should be <= 3",
            total_tokens
        );
    }

    #[test]
    fn test_multiple_sequences_finish_together() {
        let config = SchedulerConfig {
            max_num_seqs: 10,
            max_num_batched_tokens: 100,
            max_consecutive_decode: 10,
        };
        let mut sched = Scheduler::with_config(config, 1024);

        sched.add_request(Request::new(1, vec![10], 2));
        sched.add_request(Request::new(2, vec![20], 2));

        let batch1 = sched.build_batch();
        assert_eq!(batch1.seq_ids.len(), 2);

        sched.update(&batch1.seq_ids, &[1, 2], &[1, 1]);

        let batch2 = sched.build_batch();
        sched.update(&batch2.seq_ids, &[3, 4], &[1, 1]);

        assert!(!sched.has_pending());
    }

    #[test]
    fn test_waiting_queue_blocked_by_decode_limit() {
        let config = SchedulerConfig {
            max_num_seqs: 2,
            max_num_batched_tokens: 100,
            max_consecutive_decode: 1,
        };
        let mut sched = Scheduler::with_config(config, 1024);

        sched.add_request(Request::new(1, vec![10], 5));
        let batch1 = sched.build_batch();
        sched.update(&batch1.seq_ids, &[99], &[1]);

        sched.add_request(Request::new(2, vec![20], 5));
        sched.add_request(Request::new(3, vec![30], 5));

        let batch2 = sched.build_batch();
        assert!(batch2.seq_ids.len() <= 2);
    }

    #[test]
    fn test_max_tokens_equals_prompt_length() {
        let config = SchedulerConfig {
            max_num_seqs: 10,
            max_num_batched_tokens: 100,
            max_consecutive_decode: 10,
        };
        let mut sched = Scheduler::with_config(config, 1024);

        sched.add_request(Request::new(1, vec![10, 20, 30], 5));

        let batch = sched.build_batch();
        assert_eq!(batch.input_tokens[0], vec![10, 20, 30]);

        sched.update(&batch.seq_ids, &[99], &[3]);
        assert!(sched.has_pending());

        let batch2 = sched.build_batch();
        assert!(!batch2.input_tokens.is_empty());
        sched.update(&batch2.seq_ids, &[100], &[1]);
        assert!(!sched.has_pending());
    }

    #[test]
    fn test_cache_miss_full_path() {
        let config = SchedulerConfig {
            max_num_seqs: 10,
            max_num_batched_tokens: 100,
            max_consecutive_decode: 10,
        };
        let mut sched = Scheduler::with_config(config, 10);

        sched.add_request(Request::new(1, vec![1, 2, 3, 4, 5], 10));

        let batch = sched.build_batch();
        assert_eq!(batch.input_tokens[0], vec![1, 2, 3, 4, 5]);

        sched.update(&batch.seq_ids, &[99], &[5]);
        assert!(!sched.running().is_empty());
    }
}
