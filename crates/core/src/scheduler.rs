use crate::kv_cache::BlockAllocator;
use crate::types::{Batch, Request, SchedulerConfig, SeqId, Sequence, Status, TokenId, BLOCK_SIZE};
use std::collections::VecDeque;

pub struct Scheduler {
    waiting: VecDeque<Sequence>,
    running: Vec<Sequence>,
    finished: Vec<Sequence>,
    next_seq_id: SeqId,
    config: SchedulerConfig,
    kv_allocator: BlockAllocator,
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

        let num_blocks_needed = (req.prompt.len() + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let blocks = self
            .kv_allocator
            .allocate(num_blocks_needed)
            .unwrap_or_default();

        let seq = Sequence {
            id,
            tokens: req.prompt,
            kv_blocks: blocks,
            num_computed_tokens: 0,
            status: Status::Waiting,
            max_tokens: req.max_tokens,
            sampling_params: req.sampling_params,
        };
        self.waiting.push_back(seq);
        id
    }

    fn admit_waiting(&mut self) {
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
        self.admit_waiting();

        let mut seq_ids = vec![];
        let mut input_tokens = vec![];
        let mut positions = vec![];
        let mut budget = self.config.max_num_batched_tokens;

        // Phase 1: Decode sequences first (1 token each)
        for seq in &self.running {
            if seq.status != Status::Decoding {
                continue;
            }
            if budget == 0 {
                break;
            }

            let last = *seq.tokens.last().unwrap();
            let pos = seq.tokens.len() - 1;
            seq_ids.push(seq.id);
            input_tokens.push(vec![last]);
            positions.push(vec![pos]);
            budget = budget.saturating_sub(1);
        }

        // Phase 2: Prefill sequences with remaining budget
        for seq in &self.running {
            if seq.status != Status::Prefilling {
                continue;
            }
            if budget == 0 {
                break;
            }

            let start = seq.num_computed_tokens;
            let remaining = seq.tokens.len() - start;
            let chunk_size = remaining.min(budget);

            let tokens = seq.tokens[start..start + chunk_size].to_vec();
            let pos: Vec<usize> = (start..start + chunk_size).collect();

            seq_ids.push(seq.id);
            input_tokens.push(tokens);
            positions.push(pos);
            budget = budget.saturating_sub(chunk_size);
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
                let blocks_needed = (new_total + BLOCK_SIZE - 1) / BLOCK_SIZE;
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

        let mut i = 0;
        while i < self.running.len() {
            if self.running[i].status == Status::Finished {
                let seq = self.running.remove(i);
                self.finished.push(seq);
            } else {
                i += 1;
            }
        }
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
}
