use crate::types::{Batch, Request, SeqId, Sequence, Status, TokenId};
use std::collections::VecDeque;

pub struct Scheduler {
    waiting: VecDeque<Sequence>,
    running: Vec<Sequence>,
    finished: Vec<Sequence>,
    next_seq_id: SeqId,
}

impl Default for Scheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl Scheduler {
    pub fn new() -> Self {
        Self {
            waiting: VecDeque::new(),
            running: Vec::new(),
            finished: Vec::new(),
            next_seq_id: 1,
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

        let seq = Sequence {
            id,
            tokens: req.prompt,
            num_computed_tokens: 0,
            status: Status::Waiting,
            max_tokens: req.max_tokens,
            sampling_params: req.sampling_params,
        };
        self.waiting.push_back(seq);
        id
    }

    pub fn build_batch(&mut self) -> Batch {
        // Move waiting → running
        while let Some(mut seq) = self.waiting.pop_front() {
            if seq.status == Status::Waiting {
                seq.status = Status::Prefilling;
            }
            self.running.push(seq);
        }

        let mut seq_ids = vec![];
        let mut input_tokens = vec![];
        let mut positions = vec![];

        for seq in &self.running {
            if seq.status == Status::Finished {
                continue;
            }

            let (tokens, pos) = if seq.status == Status::Prefilling {
                // Prefill: process all tokens not yet computed
                let start = seq.num_computed_tokens;
                let tokens = seq.tokens[start..].to_vec();
                let pos: Vec<usize> = (start..seq.tokens.len()).collect();
                (tokens, pos)
            } else {
                // Decode: only the last token
                let last = *seq.tokens.last().unwrap();
                let pos = seq.tokens.len() - 1;
                (vec![last], vec![pos])
            };

            seq_ids.push(seq.id);
            input_tokens.push(tokens);
            positions.push(pos);
        }

        Batch {
            seq_ids,
            input_tokens,
            positions,
        }
    }

    pub fn update(&mut self, seq_ids: &[SeqId], next_tokens: &[TokenId]) {
        for (seq_id, token) in seq_ids.iter().zip(next_tokens) {
            if let Some(seq) = self.running.iter_mut().find(|s| s.id == *seq_id) {
                if seq.status == Status::Prefilling {
                    seq.num_computed_tokens = seq.tokens.len();
                    seq.status = Status::Decoding;
                }

                seq.tokens.push(*token);

                if seq.tokens.len() >= seq.max_tokens {
                    seq.status = Status::Finished;
                }
            }
        }

        // Move finished → finished list
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

    pub fn finished_sequences(&self) -> &[Sequence] {
        &self.finished
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_build_batch_prefill() {
        let mut sched = Scheduler::new();
        sched.add_request(Request::new(1, vec![10, 20, 30], 5));

        let batch = sched.build_batch();
        assert_eq!(batch.seq_ids.len(), 1);
        assert_eq!(batch.input_tokens[0], vec![10, 20, 30]);
        assert_eq!(batch.positions[0], vec![0, 1, 2]);
    }

    #[test]
    fn test_update_transitions_to_decode() {
        let mut sched = Scheduler::new();
        sched.add_request(Request::new(1, vec![10, 20, 30], 5));

        let batch = sched.build_batch();
        sched.update(&batch.seq_ids, &[99]);

        // After prefill update, next batch should be decode (last token only)
        let batch2 = sched.build_batch();
        assert_eq!(batch2.input_tokens[0], vec![99]);
        assert_eq!(batch2.positions[0], vec![3]); // tokens now [10,20,30,99], position 3
    }

    #[test]
    fn test_finished_after_max_tokens() {
        let mut sched = Scheduler::new();
        sched.add_request(Request::new(1, vec![10], 3)); // max_tokens=3, already 1 token

        // Step 1: prefill
        let batch = sched.build_batch();
        sched.update(&batch.seq_ids, &[20]); // tokens: [10, 20]

        // Step 2: decode
        let batch = sched.build_batch();
        sched.update(&batch.seq_ids, &[30]); // tokens: [10, 20, 30] → finished

        assert!(!sched.has_pending());
        assert_eq!(sched.finished_sequences().len(), 1);
    }

    #[test]
    fn test_empty_batch_when_no_requests() {
        let mut sched = Scheduler::new();
        let batch = sched.build_batch();
        assert!(batch.is_empty());
    }
}
