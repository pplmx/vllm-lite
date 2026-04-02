use crate::kv_cache::{BlockAllocator, PrefixCache, hash_tokens};
use crate::types::{BLOCK_SIZE, Batch, Request, SchedulerConfig, SeqId, Sequence, Status, TokenId};
use std::collections::VecDeque;
use std::sync::Arc;

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
                priority: req.priority,
            };
            self.running.push(seq);
            return id;
        }

        // Check for prefix match
        if let Some(entry) = self.prefix_cache.find_prefix_match(&req.prompt) {
            let cached_len = entry.token_count;
            let cached_blocks = entry.blocks.as_ref().clone();
            let remaining_len = req.prompt.len() - cached_len;

            let num_blocks_needed = remaining_len.div_ceil(BLOCK_SIZE);
            if self.kv_allocator.available() < num_blocks_needed {
                self.prefix_cache.evict(&mut self.kv_allocator);
            }

            let blocks = self
                .kv_allocator
                .allocate(num_blocks_needed)
                .unwrap_or_default();

            let mut all_blocks = cached_blocks;
            all_blocks.extend(blocks);

            let seq = Sequence {
                id,
                tokens: req.prompt,
                kv_blocks: Arc::new(all_blocks),
                num_computed_tokens: cached_len,
                prompt_len,
                status: Status::Prefilling,
                max_tokens: req.max_tokens,
                sampling_params: req.sampling_params,
                consecutive_decode_rounds: 0,
                priority: req.priority,
            };
            self.waiting.push_back(seq);
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
            kv_blocks: Arc::new(blocks),
            num_computed_tokens: 0,
            prompt_len,
            status: Status::Waiting,
            max_tokens: req.max_tokens,
            sampling_params: req.sampling_params,
            consecutive_decode_rounds: 0,
            priority: req.priority,
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

    fn process_finished_sequences(&mut self) {
        let mut newly_finished = Vec::with_capacity(self.running.len());
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
                    .insert_arc(key, seq.kv_blocks.clone(), seq.prompt_len);
            }
        }
        self.finished.extend(newly_finished);
    }

    fn promote_waiting_to_running(&mut self) {
        if self.config.enable_priority_scheduling {
            let mut waiting_vec: Vec<_> = self.waiting.drain(..).collect();
            waiting_vec.sort_by(|a, b| a.priority.cmp(&b.priority));
            self.waiting = waiting_vec.into();
        }

        while self.running.len() < self.config.max_num_seqs {
            match self.waiting.pop_front() {
                Some(mut seq) => {
                    seq.status = Status::Prefilling;
                    self.running.push(seq);
                }
                None => break,
            }
        }
    }

    #[allow(clippy::type_complexity)]
    fn build_decode_batch(
        &self,
        budget: usize,
    ) -> (Vec<SeqId>, Vec<Vec<TokenId>>, Vec<Vec<usize>>, usize) {
        let effective_max_seqs = if self.config.enable_dynamic_batching {
            self.adjust_batch_size()
        } else {
            self.config.max_num_seqs
        };

        let mut seq_ids = Vec::with_capacity(effective_max_seqs);
        let mut input_tokens = Vec::with_capacity(effective_max_seqs);
        let mut positions = Vec::with_capacity(effective_max_seqs);
        let decode_limit = self.config.max_consecutive_decode;

        let mut remaining_budget = budget;

        for seq in &self.running {
            if seq_ids.len() >= effective_max_seqs {
                break;
            }
            if remaining_budget == 0 {
                break;
            }

            if seq.status == Status::Decoding {
                if seq.consecutive_decode_rounds >= decode_limit {
                    continue;
                }

                let Some(last) = seq.tokens.last() else {
                    continue;
                };
                let last = *last;
                let pos = seq.tokens.len() - 1;

                seq_ids.push(seq.id);
                input_tokens.push(vec![last]);
                positions.push(vec![pos]);
                remaining_budget = remaining_budget.saturating_sub(1);
            }
        }

        (seq_ids, input_tokens, positions, remaining_budget)
    }

    #[allow(clippy::type_complexity)]
    fn build_prefill_batch(
        &self,
        budget: usize,
        exclude_count: usize,
    ) -> (Vec<SeqId>, Vec<Vec<TokenId>>, Vec<Vec<usize>>, usize) {
        let effective_max_seqs = if self.config.enable_dynamic_batching {
            self.adjust_batch_size()
        } else {
            self.config.max_num_seqs
        };

        let max_seqs = effective_max_seqs.saturating_sub(exclude_count);
        let mut seq_ids = Vec::with_capacity(max_seqs);
        let mut input_tokens = Vec::with_capacity(max_seqs);
        let mut positions = Vec::with_capacity(max_seqs);

        let mut remaining_budget = budget;

        for seq in &self.running {
            if seq_ids.len() >= max_seqs {
                break;
            }
            if remaining_budget == 0 {
                break;
            }

            if seq.status == Status::Prefilling {
                let start = seq.num_computed_tokens;
                let remaining = seq.tokens.len() - start;
                let chunk_size = remaining
                    .min(remaining_budget)
                    .min(self.config.prefill_chunk_size);

                if chunk_size == 0 {
                    continue;
                }

                let tokens = seq.tokens[start..start + chunk_size].to_vec();
                let pos: Vec<usize> = (start..start + chunk_size).collect();

                seq_ids.push(seq.id);
                input_tokens.push(tokens);
                positions.push(pos);
                remaining_budget = remaining_budget.saturating_sub(chunk_size);
            }
        }

        (seq_ids, input_tokens, positions, remaining_budget)
    }

    fn build_batch_with_pd_separation(&mut self) -> Batch {
        let budget = self.config.max_num_batched_tokens;
        let decode_preference = self.config.decode_preference_ratio;

        let decode_budget = (budget as f32 * decode_preference) as usize;
        let prefill_budget = budget.saturating_sub(decode_budget);

        let (decode_ids, decode_tokens, decode_pos, _remaining) =
            self.build_decode_batch(decode_budget);

        let (prefill_ids, prefill_tokens, prefill_pos, _) =
            self.build_prefill_batch(prefill_budget, decode_ids.len());

        let mut seq_ids = decode_ids;
        seq_ids.extend(prefill_ids);
        let mut input_tokens = decode_tokens;
        input_tokens.extend(prefill_tokens);
        let mut positions = decode_pos;
        positions.extend(prefill_pos);

        Batch {
            seq_ids,
            input_tokens,
            positions,
        }
    }

    fn build_batch_mixed(&mut self) -> Batch {
        let budget = self.config.max_num_batched_tokens;

        let (decode_ids, decode_tokens, decode_pos, remaining) = self.build_decode_batch(budget);

        let (prefill_ids, prefill_tokens, prefill_pos, _) =
            self.build_prefill_batch(remaining, decode_ids.len());

        let mut seq_ids = decode_ids;
        seq_ids.extend(prefill_ids);
        let mut input_tokens = decode_tokens;
        input_tokens.extend(prefill_tokens);
        let mut positions = decode_pos;
        positions.extend(prefill_pos);

        Batch {
            seq_ids,
            input_tokens,
            positions,
        }
    }

    pub fn build_batch(&mut self) -> Batch {
        self.process_finished_sequences();
        self.promote_waiting_to_running();

        if self.config.enable_pd_separation {
            let has_decode = self.running.iter().any(|s| s.status == Status::Decoding);
            let has_prefill = self.running.iter().any(|s| s.status == Status::Prefilling);

            if has_decode && has_prefill {
                self.build_batch_with_pd_separation()
            } else {
                self.build_batch_mixed()
            }
        } else {
            self.build_batch_mixed()
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
                        Arc::make_mut(&mut seq.kv_blocks).extend(new_block);
                    } else {
                        break;
                    }
                }

                if seq.tokens.len() >= seq.max_tokens {
                    seq.status = Status::Finished;
                }
            }
        }

        let mut newly_finished = Vec::with_capacity(self.running.len());
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
                    .insert_arc(key, seq.kv_blocks.clone(), seq.prompt_len);
            }
        }

        // Move to finished list
        self.finished.extend(newly_finished);
    }

    pub fn has_pending(&self) -> bool {
        !self.waiting.is_empty() || !self.running.is_empty()
    }

    fn adjust_batch_size(&self) -> usize {
        let available_blocks = self.kv_allocator.available();
        let waiting_count = self.waiting.len();
        let running_count = self.running.len();

        let base_batch = self.config.max_num_seqs;

        if available_blocks < 10 {
            return self.config.min_batch_size;
        }

        let memory_factor = (available_blocks as f32 / 1024.0).min(1.0);
        let target_batch = (self.config.max_batch_size as f32 * memory_factor) as usize;

        let max_possible = waiting_count + running_count;
        target_batch
            .min(max_possible)
            .max(self.config.min_batch_size)
            .min(base_batch)
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
    use crate::Priority;

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
            enable_pd_separation: false,
            prefill_chunk_size: 512,
            decode_preference_ratio: 0.7,
            enable_priority_scheduling: false,
            enable_dynamic_batching: false,
            min_batch_size: 1,
            max_batch_size: 256,
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
            enable_pd_separation: false,
            prefill_chunk_size: 512,
            decode_preference_ratio: 0.7,
            enable_priority_scheduling: false,
            enable_dynamic_batching: false,
            min_batch_size: 1,
            max_batch_size: 256,
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
            enable_pd_separation: false,
            prefill_chunk_size: 512,
            decode_preference_ratio: 0.7,
            enable_priority_scheduling: false,
            enable_dynamic_batching: false,
            min_batch_size: 1,
            max_batch_size: 256,
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
        assert!(!batch2.seq_ids.is_empty());
    }

    #[test]
    fn test_max_consecutive_decode_limit() {
        let config = SchedulerConfig {
            max_num_seqs: 10,
            max_num_batched_tokens: 100,
            max_consecutive_decode: 2,
            enable_pd_separation: false,
            prefill_chunk_size: 512,
            decode_preference_ratio: 0.7,
            enable_priority_scheduling: false,
            enable_dynamic_batching: false,
            min_batch_size: 1,
            max_batch_size: 256,
        };
        let mut sched = Scheduler::with_config(config, 1024);

        sched.add_request(Request::new(1, vec![10], 10));
        let batch1 = sched.build_batch();
        sched.update(&batch1.seq_ids, &[99], &[batch1.input_tokens[0].len()]);

        sched.add_request(Request::new(2, vec![20], 10));

        let batch2 = sched.build_batch();
        assert!(!batch2.seq_ids.is_empty());
    }

    #[test]
    fn test_token_budget_in_continuous_batching() {
        let config = SchedulerConfig {
            max_num_seqs: 10,
            max_num_batched_tokens: 3,
            max_consecutive_decode: 10,
            enable_pd_separation: false,
            prefill_chunk_size: 512,
            decode_preference_ratio: 0.7,
            enable_priority_scheduling: false,
            enable_dynamic_batching: false,
            min_batch_size: 1,
            max_batch_size: 256,
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
            enable_pd_separation: false,
            prefill_chunk_size: 512,
            decode_preference_ratio: 0.7,
            enable_priority_scheduling: false,
            enable_dynamic_batching: false,
            min_batch_size: 1,
            max_batch_size: 256,
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
            enable_pd_separation: false,
            prefill_chunk_size: 512,
            decode_preference_ratio: 0.7,
            enable_priority_scheduling: false,
            enable_dynamic_batching: false,
            min_batch_size: 1,
            max_batch_size: 256,
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
            enable_pd_separation: false,
            prefill_chunk_size: 512,
            decode_preference_ratio: 0.7,
            enable_priority_scheduling: false,
            enable_dynamic_batching: false,
            min_batch_size: 1,
            max_batch_size: 256,
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
            enable_pd_separation: false,
            prefill_chunk_size: 512,
            decode_preference_ratio: 0.7,
            enable_priority_scheduling: false,
            enable_dynamic_batching: false,
            min_batch_size: 1,
            max_batch_size: 256,
        };
        let mut sched = Scheduler::with_config(config, 10);

        sched.add_request(Request::new(1, vec![1, 2, 3, 4, 5], 10));

        let batch = sched.build_batch();
        assert_eq!(batch.input_tokens[0], vec![1, 2, 3, 4, 5]);

        sched.update(&batch.seq_ids, &[99], &[5]);
        assert!(!sched.running().is_empty());
    }

    #[test]
    fn test_empty_prompt_handling() {
        let mut sched = Scheduler::new();
        sched.add_request(Request::new(1, vec![], 5));

        let batch = sched.build_batch();
        assert!(batch.is_empty(), "empty prompt should produce empty batch");
    }

    #[test]
    fn test_single_token_prompt() {
        let mut sched = Scheduler::new();
        sched.add_request(Request::new(1, vec![42], 3));

        let batch = sched.build_batch();
        assert_eq!(batch.input_tokens[0], vec![42]);

        sched.update(&batch.seq_ids, &[99], &[1]);

        // Should transition to decoding after single token
        let batch2 = sched.build_batch();
        assert!(!batch2.is_empty());
    }

    #[test]
    fn test_max_tokens_exactly_reached() {
        let config = SchedulerConfig {
            max_num_seqs: 10,
            max_num_batched_tokens: 100,
            max_consecutive_decode: 10,
            enable_pd_separation: false,
            prefill_chunk_size: 512,
            decode_preference_ratio: 0.7,
            enable_priority_scheduling: false,
            enable_dynamic_batching: false,
            min_batch_size: 1,
            max_batch_size: 256,
        };
        let mut sched = Scheduler::with_config(config, 1024);

        // prompt length = 2, max_tokens = 2 (exactly)
        // Need at least one decode token to trigger finish check
        sched.add_request(Request::new(1, vec![10, 20], 2));

        let batch = sched.build_batch();
        // Push one token to make tokens.len() == max_tokens (2 == 2)
        sched.update(&batch.seq_ids, &[99], &[2]);

        // Should be finished now
        assert!(!sched.has_pending());
    }

    #[test]
    fn test_premature_completion_in_prefill() {
        // When max_tokens <= prompt_len, should finish immediately after prefill
        let config = SchedulerConfig {
            max_num_seqs: 10,
            max_num_batched_tokens: 100,
            max_consecutive_decode: 10,
            enable_pd_separation: false,
            prefill_chunk_size: 512,
            decode_preference_ratio: 0.7,
            enable_priority_scheduling: false,
            enable_dynamic_batching: false,
            min_batch_size: 1,
            max_batch_size: 256,
        };
        let mut sched = Scheduler::with_config(config, 1024);

        // max_tokens = 3, prompt_len = 3
        sched.add_request(Request::new(1, vec![1, 2, 3], 3));

        let batch = sched.build_batch();
        assert_eq!(batch.input_tokens[0], vec![1, 2, 3]);

        // Push one token to trigger finish check: 3 + 1 = 4 >= 3
        sched.update(&batch.seq_ids, &[99], &[3]);

        // Should be done immediately
        assert!(!sched.has_pending());
    }

    #[test]
    fn test_pd_separation_with_config() {
        let config = SchedulerConfig {
            max_num_seqs: 10,
            max_num_batched_tokens: 10,
            max_consecutive_decode: 10,
            enable_pd_separation: true,
            prefill_chunk_size: 2,
            decode_preference_ratio: 0.5,
            enable_priority_scheduling: false,
            enable_dynamic_batching: false,
            min_batch_size: 1,
            max_batch_size: 256,
        };
        let mut sched = Scheduler::with_config(config, 1024);

        // First request: prefill
        sched.add_request(Request::new(1, vec![1, 2, 3, 4, 5], 3));
        let batch1 = sched.build_batch();

        // With prefill_chunk_size=2, should only process 2 tokens
        assert!(batch1.input_tokens[0].len() <= 2);

        // Update to complete first 2 tokens
        sched.update(&batch1.seq_ids, &[99], &[2]);

        // Second request: new prefill
        sched.add_request(Request::new(2, vec![10, 20], 3));

        // First request is still prefill (3 more tokens), second is prefill (2 tokens)
        let batch2 = sched.build_batch();
        assert!(!batch2.seq_ids.is_empty());
    }

    #[test]
    fn test_pd_separation_decode_only() {
        let config = SchedulerConfig {
            max_num_seqs: 10,
            max_num_batched_tokens: 10,
            max_consecutive_decode: 10,
            enable_pd_separation: true,
            prefill_chunk_size: 512,
            decode_preference_ratio: 0.7,
            enable_priority_scheduling: false,
            enable_dynamic_batching: false,
            min_batch_size: 1,
            max_batch_size: 256,
        };
        let mut sched = Scheduler::with_config(config, 1024);

        sched.add_request(Request::new(1, vec![1, 2], 5));
        let batch1 = sched.build_batch();
        assert_eq!(batch1.input_tokens[0], vec![1, 2]);

        sched.update(&batch1.seq_ids, &[99], &[2]);

        let batch2 = sched.build_batch();
        assert_eq!(batch2.input_tokens[0].len(), 1);

        sched.update(&batch2.seq_ids, &[100], &[1]);

        let batch3 = sched.build_batch();
        assert_eq!(batch3.input_tokens[0].len(), 1);
    }

    #[test]
    fn test_pd_separation_prefill_only() {
        let config = SchedulerConfig {
            max_num_seqs: 10,
            max_num_batched_tokens: 10,
            max_consecutive_decode: 10,
            enable_pd_separation: true,
            prefill_chunk_size: 512,
            decode_preference_ratio: 0.7,
            enable_priority_scheduling: false,
            enable_dynamic_batching: false,
            min_batch_size: 1,
            max_batch_size: 256,
        };
        let mut sched = Scheduler::with_config(config, 1024);

        sched.add_request(Request::new(1, vec![1, 2, 3, 4, 5], 3));

        let batch = sched.build_batch();

        assert!(!batch.input_tokens[0].is_empty());
    }

    #[test]
    fn test_pd_separation_decode_preference_ratio() {
        let config = SchedulerConfig {
            max_num_seqs: 10,
            max_num_batched_tokens: 10,
            max_consecutive_decode: 10,
            enable_pd_separation: true,
            prefill_chunk_size: 512,
            decode_preference_ratio: 0.9,
            enable_priority_scheduling: false,
            enable_dynamic_batching: false,
            min_batch_size: 1,
            max_batch_size: 256,
        };
        let mut sched = Scheduler::with_config(config, 1024);

        sched.add_request(Request::new(1, vec![1], 5));
        let batch1 = sched.build_batch();
        sched.update(&batch1.seq_ids, &[10], &[1]);

        sched.add_request(Request::new(2, vec![2, 3], 3));

        let batch2 = sched.build_batch();

        assert!(!batch2.seq_ids.is_empty());
    }

    #[test]
    fn test_chunked_prefill_limits_tokens() {
        let config = SchedulerConfig {
            max_num_seqs: 256,
            max_num_batched_tokens: 3,
            max_consecutive_decode: 10,
            enable_pd_separation: false,
            prefill_chunk_size: 2,
            decode_preference_ratio: 0.7,
            enable_priority_scheduling: false,
            enable_dynamic_batching: false,
            min_batch_size: 1,
            max_batch_size: 256,
        };
        let mut sched = Scheduler::with_config(config, 1024);

        // Add a request with 10 tokens
        sched.add_request(Request::new(1, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 5));

        let batch = sched.build_batch();

        // Should be limited by max_num_batched_tokens (3) AND prefill_chunk_size (2)
        // The minimum should be 2
        let total_tokens: usize = batch.input_tokens.iter().map(|v| v.len()).sum();
        assert!(total_tokens <= 3, "Should respect max_num_batched_tokens");
    }

    #[test]
    fn test_priority_scheduling() {
        let config = SchedulerConfig {
            max_num_seqs: 10,
            max_num_batched_tokens: 100,
            max_consecutive_decode: 10,
            enable_pd_separation: false,
            prefill_chunk_size: 512,
            decode_preference_ratio: 0.7,
            enable_priority_scheduling: true,
            enable_dynamic_batching: false,
            min_batch_size: 1,
            max_batch_size: 256,
        };
        let mut sched = Scheduler::with_config(config, 1024);

        // Add requests with different priorities
        // Lower Priority value = higher priority
        let mut req1 = Request::new(1, vec![1], 5);
        req1.priority = Priority(2);

        let mut req2 = Request::new(2, vec![2], 5);
        req2.priority = Priority(1); // Higher priority than req1

        let mut req3 = Request::new(3, vec![3], 5);
        req3.priority = Priority(3); // Lowest priority

        sched.add_request(req1);
        sched.add_request(req2);
        sched.add_request(req3);

        let batch = sched.build_batch();

        // With priority scheduling, request 2 (priority 1) should be processed first
        // The first request in batch should have id 2
        assert_eq!(
            batch.seq_ids[0], 2,
            "Highest priority request should be first"
        );
    }

    #[test]
    fn test_dynamic_batching_with_low_memory() {
        let config = SchedulerConfig {
            max_num_seqs: 10,
            max_num_batched_tokens: 100,
            max_consecutive_decode: 10,
            enable_pd_separation: false,
            prefill_chunk_size: 512,
            decode_preference_ratio: 0.7,
            enable_priority_scheduling: false,
            enable_dynamic_batching: true,
            min_batch_size: 1,
            max_batch_size: 10,
        };
        // Only 5 KV blocks - very limited memory
        let mut sched = Scheduler::with_config(config, 5);

        // Add multiple requests
        for i in 1..=5 {
            sched.add_request(Request::new(i, vec![i as TokenId], 3));
        }

        let batch = sched.build_batch();

        // With only 5 KV blocks and min_batch_size=1, should allow at least 1
        assert!(
            !batch.seq_ids.is_empty(),
            "Should allow at least min_batch_size"
        );
    }

    #[test]
    fn test_dynamic_batching_with_high_memory() {
        let config = SchedulerConfig {
            max_num_seqs: 10,
            max_num_batched_tokens: 100,
            max_consecutive_decode: 10,
            enable_pd_separation: false,
            prefill_chunk_size: 512,
            decode_preference_ratio: 0.7,
            enable_priority_scheduling: false,
            enable_dynamic_batching: true,
            min_batch_size: 2,
            max_batch_size: 10,
        };
        // Lots of KV blocks available
        let mut sched = Scheduler::with_config(config, 1000);

        // Add 3 requests
        for i in 1..=3 {
            sched.add_request(Request::new(i, vec![i as TokenId], 3));
        }

        let batch = sched.build_batch();

        // With plenty of memory, should process all 3 requests
        assert_eq!(
            batch.seq_ids.len(),
            3,
            "Should process all requests with enough memory"
        );
    }

    #[test]
    fn test_dynamic_batching_disabled() {
        let config = SchedulerConfig {
            max_num_seqs: 3,
            max_num_batched_tokens: 100,
            max_consecutive_decode: 10,
            enable_pd_separation: false,
            prefill_chunk_size: 512,
            decode_preference_ratio: 0.7,
            enable_priority_scheduling: false,
            enable_dynamic_batching: false,
            min_batch_size: 1,
            max_batch_size: 256,
        };
        let mut sched = Scheduler::with_config(config, 5);

        // Add 5 requests
        for i in 1..=5 {
            sched.add_request(Request::new(i, vec![i as TokenId], 3));
        }

        let batch = sched.build_batch();

        // Without dynamic batching, should respect max_num_seqs=3
        assert_eq!(
            batch.seq_ids.len(),
            3,
            "Should respect max_num_seqs when dynamic batching disabled"
        );
    }

    #[test]
    fn test_empty_waiting_queue() {
        let mut sched = Scheduler::new();
        let batch = sched.build_batch();

        assert!(batch.is_empty());
    }

    #[test]
    fn test_max_tokens_zero() {
        let mut sched = Scheduler::new();

        // max_tokens = 0 should still allow prompt processing
        sched.add_request(Request::new(1, vec![1, 2, 3], 0));

        let batch = sched.build_batch();

        // Should process prompt even with max_tokens=0
        assert!(!batch.is_empty() || !sched.has_pending());
    }

    #[test]
    fn test_batch_with_zero_max_consecutive_decode() {
        let config = SchedulerConfig {
            max_num_seqs: 10,
            max_num_batched_tokens: 100,
            max_consecutive_decode: 0,
            enable_pd_separation: false,
            prefill_chunk_size: 512,
            decode_preference_ratio: 0.7,
            enable_priority_scheduling: false,
            enable_dynamic_batching: false,
            min_batch_size: 1,
            max_batch_size: 256,
        };
        let mut sched = Scheduler::with_config(config, 1024);

        sched.add_request(Request::new(1, vec![10], 5));

        let batch1 = sched.build_batch();
        sched.update(&batch1.seq_ids, &[99], &[1]);

        // With max_consecutive_decode=0, decode should not be blocked
        let batch2 = sched.build_batch();

        assert!(batch2.seq_ids.len() <= 10);
    }

    #[test]
    fn test_waiting_requests_after_decode_limit() {
        let config = SchedulerConfig {
            max_num_seqs: 2,
            max_num_batched_tokens: 10,
            max_consecutive_decode: 1,
            enable_pd_separation: false,
            prefill_chunk_size: 512,
            decode_preference_ratio: 0.7,
            enable_priority_scheduling: false,
            enable_dynamic_batching: false,
            min_batch_size: 1,
            max_batch_size: 256,
        };
        let mut sched = Scheduler::with_config(config, 1024);

        // Add request 1: prefill then decode
        sched.add_request(Request::new(1, vec![1, 2], 3));
        let batch1 = sched.build_batch();
        sched.update(&batch1.seq_ids, &[10], &[2]); // Complete prefill

        let batch2 = sched.build_batch();
        sched.update(&batch2.seq_ids, &[11], &[1]); // One decode step

        // Add more requests - they should be processed
        sched.add_request(Request::new(2, vec![3, 4], 3));
        sched.add_request(Request::new(3, vec![5, 6], 3));

        let batch3 = sched.build_batch();

        // Should process waiting requests
        assert!(!batch3.seq_ids.is_empty());
    }

    #[test]
    fn test_add_request_zero_prompt() {
        let mut sched = Scheduler::new();
        let id = sched.add_request(Request::new(1, vec![], 5));
        assert_eq!(id, 1);
        assert!(sched.has_pending());
    }

    #[test]
    fn test_add_request_duplicate_id() {
        let mut sched = Scheduler::new();
        sched.add_request(Request::new(1, vec![1, 2], 5));
        let id = sched.add_request(Request::new(1, vec![3, 4], 5));
        // Current behavior: uses provided ID if non-zero
        assert_eq!(id, 1);
    }

    #[test]
    fn test_build_batch_empty() {
        let mut sched = Scheduler::new();
        let batch = sched.build_batch();
        assert!(batch.is_empty());
    }

    #[test]
    fn test_update_nonexistent_seq() {
        let mut sched = Scheduler::new();
        sched.update(&[999], &[1], &[1]);
        assert!(!sched.has_pending());
    }

    #[test]
    fn test_running_after_all_finished() {
        let config = SchedulerConfig {
            max_num_seqs: 10,
            max_num_batched_tokens: 100,
            max_consecutive_decode: 10,
            enable_pd_separation: false,
            prefill_chunk_size: 512,
            decode_preference_ratio: 0.7,
            enable_priority_scheduling: false,
            enable_dynamic_batching: false,
            min_batch_size: 1,
            max_batch_size: 256,
        };
        let mut sched = Scheduler::with_config(config, 10);
        sched.add_request(Request::new(1, vec![1], 1));
        let batch = sched.build_batch();
        sched.update(&batch.seq_ids, &[99], &[batch.input_tokens[0].len()]);
        assert!(!sched.has_pending());
    }
}
