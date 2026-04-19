use std::collections::{BinaryHeap, HashMap, HashSet};
use std::time::Instant;

use crate::scheduler::policy::{PriorityScore, SchedulingContext, SchedulingPolicy};
use crate::types::{Phase, SeqId, Sequence, Status};

#[derive(Clone, Debug)]
struct ScheduledSequence {
    seq_id: SeqId,
    priority: PriorityScore,
    arrival_time: Instant,
}

impl PartialEq for ScheduledSequence {
    fn eq(&self, other: &Self) -> bool {
        self.seq_id == other.seq_id
    }
}

impl Eq for ScheduledSequence {}

impl PartialOrd for ScheduledSequence {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScheduledSequence {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.priority
            .cmp(&other.priority)
            .reverse()
            .then_with(|| self.arrival_time.cmp(&other.arrival_time))
    }
}

pub struct RequestQueue {
    sequences: HashMap<SeqId, Sequence>,
    priority_queue: BinaryHeap<ScheduledSequence>,
    phase_index: HashMap<Phase, HashSet<SeqId>>,
    in_queue: HashSet<SeqId>,
}

impl RequestQueue {
    pub fn new() -> Self {
        Self {
            sequences: HashMap::new(),
            priority_queue: BinaryHeap::new(),
            phase_index: HashMap::new(),
            in_queue: HashSet::new(),
        }
    }

    pub fn enqueue(
        &mut self,
        seq: Sequence,
        policy: &dyn SchedulingPolicy,
        ctx: &SchedulingContext,
    ) {
        tracing::debug!(
            request_id = seq.id,
            prompt_tokens = seq.prompt_len,
            max_tokens = seq.max_tokens,
            queue_size = self.in_queue.len(),
            "Request enqueued"
        );

        if self.in_queue.contains(&seq.id) {
            return;
        }

        let seq_id = seq.id;
        let phase = self.determine_phase(&seq);
        let priority = policy.compute_priority(&seq, ctx);

        let scheduled = ScheduledSequence {
            seq_id,
            priority,
            arrival_time: Instant::now(),
        };

        self.sequences.insert(seq_id, seq);
        self.priority_queue.push(scheduled);
        self.phase_index.entry(phase).or_default().insert(seq_id);
        self.in_queue.insert(seq_id);
    }

    pub fn dequeue(&mut self) -> Option<Sequence> {
        while let Some(scheduled) = self.priority_queue.pop() {
            let seq_id = scheduled.seq_id;
            if let Some(seq) = self.sequences.remove(&seq_id) {
                let phase = self.determine_phase(&seq);
                if let Some(set) = self.phase_index.get_mut(&phase) {
                    set.remove(&seq_id);
                }
                self.in_queue.remove(&seq_id);
                return Some(seq);
            }
        }
        None
    }

    /// O(1) lookup by sequence ID.
    #[must_use]
    pub fn get(&self, seq_id: SeqId) -> Option<&Sequence> {
        self.sequences.get(&seq_id)
    }

    pub fn get_mut(&mut self, seq_id: SeqId) -> Option<&mut Sequence> {
        self.sequences.get_mut(&seq_id)
    }

    /// O(1) removal by sequence ID.
    #[must_use]
    pub fn remove(&mut self, seq_id: SeqId) -> Option<Sequence> {
        if let Some(seq) = self.sequences.remove(&seq_id) {
            let phase = self.determine_phase(&seq);
            if let Some(set) = self.phase_index.get_mut(&phase) {
                set.remove(&seq_id);
            }
            self.in_queue.remove(&seq_id);
            Some(seq)
        } else {
            None
        }
    }

    pub fn drain_by_phase(&mut self, phase: Phase) -> Vec<Sequence> {
        let ids: Vec<_> = self
            .phase_index
            .remove(&phase)
            .unwrap_or_default()
            .into_iter()
            .collect();
        let mut result = Vec::with_capacity(ids.len());

        for id in ids {
            if let Some(seq) = self.sequences.remove(&id) {
                self.in_queue.remove(&id);
                result.push(seq);
            }
        }

        self.cleanup_priority_queue();
        result
    }

    /// Get the number of sequences in the queue.
    #[must_use]
    pub fn len(&self) -> usize {
        self.sequences.len()
    }

    /// Check if the queue is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.sequences.is_empty()
    }

    pub fn phase_len(&self, phase: Phase) -> usize {
        self.phase_index.get(&phase).map(|s| s.len()).unwrap_or(0)
    }

    fn determine_phase(&self, seq: &Sequence) -> Phase {
        match seq.status {
            Status::Waiting | Status::Prefilling => Phase::Prefill,
            Status::Decoding => Phase::Decode,
            _ => Phase::Prefill,
        }
    }

    fn cleanup_priority_queue(&mut self) {
        let mut new_queue = BinaryHeap::new();
        for scheduled in self.priority_queue.drain() {
            if self.sequences.contains_key(&scheduled.seq_id) {
                new_queue.push(scheduled);
            }
        }
        self.priority_queue = new_queue;
    }
}

impl Default for RequestQueue {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::scheduler::policy::FcfsPolicy;
    use crate::types::{Priority, SamplingParams, Status};

    fn make_sequence(id: u64, status: Status) -> Sequence {
        Sequence {
            id,
            tokens: vec![1, 2, 3],
            kv_blocks: Arc::new(vec![]),
            num_computed_tokens: 0,
            prompt_len: 3,
            status,
            max_tokens: 10,
            sampling_params: SamplingParams::default(),
            consecutive_decode_rounds: 0,
            priority: Priority::default(),
        }
    }

    #[test]
    fn test_enqueue_and_dequeue() {
        let mut queue = RequestQueue::new();
        let policy = FcfsPolicy::new();
        let ctx = SchedulingContext {
            current_time: Instant::now(),
            queue_length: 0,
            running_count: 0,
            memory_pressure: 0.0,
        };

        let seq1 = make_sequence(1, Status::Waiting);
        let seq2 = make_sequence(2, Status::Waiting);

        queue.enqueue(seq1.clone(), &policy, &ctx);
        queue.enqueue(seq2.clone(), &policy, &ctx);

        assert_eq!(queue.len(), 2);

        let dequeued = queue.dequeue().unwrap();
        assert_eq!(dequeued.id, 1);
        let dequeued = queue.dequeue().unwrap();
        assert_eq!(dequeued.id, 2);
        assert!(queue.is_empty());
    }

    #[test]
    fn test_get_o1() {
        let mut queue = RequestQueue::new();
        let policy = FcfsPolicy::new();
        let ctx = SchedulingContext {
            current_time: Instant::now(),
            queue_length: 0,
            running_count: 0,
            memory_pressure: 0.0,
        };

        let seq = make_sequence(42, Status::Waiting);
        queue.enqueue(seq.clone(), &policy, &ctx);

        let retrieved = queue.get(42);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().id, 42);
    }

    #[test]
    fn test_remove_o1() {
        let mut queue = RequestQueue::new();
        let policy = FcfsPolicy::new();
        let ctx = SchedulingContext {
            current_time: Instant::now(),
            queue_length: 0,
            running_count: 0,
            memory_pressure: 0.0,
        };

        let seq = make_sequence(42, Status::Waiting);
        queue.enqueue(seq.clone(), &policy, &ctx);

        let removed = queue.remove(42);
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().id, 42);
        assert!(queue.get(42).is_none());
    }

    #[test]
    fn test_drain_by_phase() {
        let mut queue = RequestQueue::new();
        let policy = FcfsPolicy::new();
        let ctx = SchedulingContext {
            current_time: Instant::now(),
            queue_length: 0,
            running_count: 0,
            memory_pressure: 0.0,
        };

        let prefill_seq = make_sequence(1, Status::Waiting);
        let decode_seq = make_sequence(2, Status::Decoding);

        queue.enqueue(prefill_seq, &policy, &ctx);
        queue.enqueue(decode_seq, &policy, &ctx);

        let prefill_seqs = queue.drain_by_phase(Phase::Prefill);
        assert_eq!(prefill_seqs.len(), 1);
        assert_eq!(prefill_seqs[0].id, 1);
        assert_eq!(queue.phase_len(Phase::Prefill), 0);
        assert_eq!(queue.phase_len(Phase::Decode), 1);
    }
}
