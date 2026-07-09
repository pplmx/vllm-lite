//! Request waiting queue: priority heap + insertion-order FIFO for fair scheduling.
//!
//! Sequences sit here until the scheduler promotes them to `running`.
//! `RequestQueue` exposes `push`, `pop_next`, `peek`, and `remove` for the
//! preemption path. Pure data; no I/O.
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

#[derive(Debug)]
/// Thread-safe FIFO queue of pending inference requests. New requests are pushed on arrival and popped by the scheduler.
pub struct RequestQueue {
    sequences: HashMap<SeqId, Sequence>,
    priority_queue: BinaryHeap<ScheduledSequence>,
    phase_index: HashMap<Phase, HashSet<SeqId>>,
    in_queue: HashSet<SeqId>,
}

impl RequestQueue {
    /// Construct an empty queue.
    #[must_use]
    pub fn new() -> Self {
        Self {
            sequences: HashMap::new(),
            priority_queue: BinaryHeap::new(),
            phase_index: HashMap::new(),
            in_queue: HashSet::new(),
        }
    }

    /// Insert `seq` into the queue. Computes its [`PriorityScore`] using
    /// `policy` and records its current phase. If the sequence is already
    /// queued (same `seq.id`), the call is a no-op.
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
        let phase = Self::determine_phase(&seq);
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

    /// Pop the highest-priority sequence from the queue and return it.
    /// Stale heap entries (sequences removed out-of-band) are skipped.
    pub fn dequeue(&mut self) -> Option<Sequence> {
        while let Some(scheduled) = self.priority_queue.pop() {
            let seq_id = scheduled.seq_id;
            if let Some(seq) = self.sequences.remove(&seq_id) {
                let phase = Self::determine_phase(&seq);
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

    /// O(1) mutable lookup by sequence ID.
    pub fn get_mut(&mut self, seq_id: SeqId) -> Option<&mut Sequence> {
        self.sequences.get_mut(&seq_id)
    }

    /// O(1) removal by sequence ID.
    #[must_use]
    pub fn remove(&mut self, seq_id: SeqId) -> Option<Sequence> {
        if let Some(seq) = self.sequences.remove(&seq_id) {
            let phase = Self::determine_phase(&seq);
            if let Some(set) = self.phase_index.get_mut(&phase) {
                set.remove(&seq_id);
            }
            self.in_queue.remove(&seq_id);
            Some(seq)
        } else {
            None
        }
    }

    /// Remove and return every sequence currently in `phase`, in arbitrary
    /// order. Lazy-cleans the priority heap afterwards to reclaim stale
    /// entries.
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

    /// Number of sequences currently classified as `phase`.
    #[must_use]
    pub fn phase_len(&self, phase: Phase) -> usize {
        self.phase_index
            .get(&phase)
            .map_or(0, std::collections::HashSet::len)
    }

    const fn determine_phase(seq: &Sequence) -> Phase {
        #[allow(clippy::match_same_arms)]
        match seq.status {
            Status::Waiting | Status::Prefilling => Phase::Prefill,
            Status::Decoding => Phase::Decode,
            Status::Finished => Phase::Prefill,
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

// Unit tests are extracted to `tests.rs` and `prop_tests.rs` to keep
// this file under the 800-line soft cap. See those siblings for the
// test surface (enqueue/dequeue/get/remove/drain_by_phase; plus
// proptest invariants for add-remove round-trip, get-after-enqueue,
// FIFO dequeue, and phase-index consistency).
#[cfg(test)]
mod prop_tests;
#[cfg(test)]
mod tests;
