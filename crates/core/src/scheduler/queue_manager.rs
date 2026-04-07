use crate::types::{Priority, SeqId, Sequence, Status};
use std::collections::VecDeque;

pub struct QueueManager {
    critical: VecDeque<Sequence>,
    normal: VecDeque<Sequence>,
    background: VecDeque<Sequence>,
    preempted: VecDeque<Sequence>,
}

impl QueueManager {
    pub fn new() -> Self {
        Self {
            critical: VecDeque::new(),
            normal: VecDeque::new(),
            background: VecDeque::new(),
            preempted: VecDeque::new(),
        }
    }

    pub fn enqueue(&mut self, seq: Sequence, priority: Priority) {
        let queue = self.queue_for_priority(priority);
        queue.push_back(seq);
    }

    fn queue_for_priority(&mut self, priority: Priority) -> &mut VecDeque<Sequence> {
        if priority.0 <= 10 {
            &mut self.critical
        } else if priority.0 <= 50 {
            &mut self.normal
        } else {
            &mut self.background
        }
    }

    pub fn dequeue(&mut self) -> Option<Sequence> {
        self.critical
            .pop_front()
            .or_else(|| self.normal.pop_front())
            .or_else(|| self.background.pop_front())
    }

    pub fn dequeue_batch(&mut self, count: usize) -> Vec<Sequence> {
        let mut batch = Vec::with_capacity(count);
        for _ in 0..count {
            if let Some(seq) = self.dequeue() {
                batch.push(seq);
            } else {
                break;
            }
        }
        batch
    }

    pub fn is_empty(&self) -> bool {
        self.critical.is_empty() && self.normal.is_empty() && self.background.is_empty()
    }

    pub fn len(&self) -> usize {
        self.critical.len() + self.normal.len() + self.background.len()
    }

    pub fn len_by_priority(&self) -> (usize, usize, usize, usize) {
        (
            self.critical.len(),
            self.normal.len(),
            self.background.len(),
            self.preempted.len(),
        )
    }

    pub fn enqueue_preempted(&mut self, seq: Sequence) {
        self.preempted.push_back(seq);
    }

    pub fn dequeue_preempted(&mut self) -> Option<Sequence> {
        self.preempted.pop_front()
    }

    pub fn get(&self, seq_id: SeqId) -> Option<&Sequence> {
        self.critical
            .iter()
            .find(|s| s.id == seq_id)
            .or_else(|| self.normal.iter().find(|s| s.id == seq_id))
            .or_else(|| self.background.iter().find(|s| s.id == seq_id))
    }

    pub fn get_mut(&mut self, seq_id: SeqId) -> Option<&mut Sequence> {
        self.critical
            .iter_mut()
            .find(|s| s.id == seq_id)
            .or_else(|| self.normal.iter_mut().find(|s| s.id == seq_id))
            .or_else(|| self.background.iter_mut().find(|s| s.id == seq_id))
    }

    pub fn remove(&mut self, seq_id: SeqId) -> Option<Sequence> {
        if let Some(pos) = self.critical.iter().position(|s| s.id == seq_id) {
            return self.critical.remove(pos);
        }
        if let Some(pos) = self.normal.iter().position(|s| s.id == seq_id) {
            return self.normal.remove(pos);
        }
        if let Some(pos) = self.background.iter().position(|s| s.id == seq_id) {
            return self.background.remove(pos);
        }
        None
    }

    pub fn all_waiting(&self) -> Vec<&Sequence> {
        let mut result = Vec::new();
        result.extend(self.critical.iter());
        result.extend(self.normal.iter());
        result.extend(self.background.iter());
        result
    }

    pub fn filter_by_status(&self, status: Status) -> Vec<&Sequence> {
        self.all_waiting()
            .into_iter()
            .filter(|s| s.status == status)
            .collect()
    }
}

impl Default for QueueManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Status;
    use std::sync::Arc;

    fn make_seq(id: SeqId, priority_val: u8) -> Sequence {
        Sequence {
            id,
            tokens: vec![1, 2, 3],
            kv_blocks: Arc::new(vec![]),
            num_computed_tokens: 0,
            prompt_len: 3,
            status: Status::Waiting,
            max_tokens: 100,
            sampling_params: Default::default(),
            consecutive_decode_rounds: 0,
            priority: Priority(priority_val),
        }
    }

    #[test]
    fn test_priority_ordering() {
        let mut qm = QueueManager::new();
        qm.enqueue(make_seq(1, 50), Priority(50));
        qm.enqueue(make_seq(2, 10), Priority(10));
        qm.enqueue(make_seq(3, 100), Priority(100));

        assert_eq!(qm.dequeue().map(|s| s.id), Some(2));
        assert_eq!(qm.dequeue().map(|s| s.id), Some(1));
        assert_eq!(qm.dequeue().map(|s| s.id), Some(3));
    }

    #[test]
    fn test_dequeue_batch() {
        let mut qm = QueueManager::new();
        for i in 1..=5 {
            qm.enqueue(make_seq(i, 30), Priority(30));
        }

        let batch = qm.dequeue_batch(3);
        assert_eq!(batch.len(), 3);
        assert_eq!(batch[0].id, 1);
    }

    #[test]
    fn test_preempted_queue() {
        let mut qm = QueueManager::new();
        qm.enqueue_preempted(make_seq(1, 10));
        qm.enqueue_preempted(make_seq(2, 10));

        assert_eq!(qm.dequeue_preempted().map(|s| s.id), Some(1));
        assert_eq!(qm.dequeue_preempted().map(|s| s.id), Some(2));
    }

    #[test]
    fn test_len_by_priority() {
        let mut qm = QueueManager::new();
        qm.enqueue(make_seq(1, 5), Priority(5));
        qm.enqueue(make_seq(2, 5), Priority(5));
        qm.enqueue(make_seq(3, 30), Priority(30));
        qm.enqueue_preempted(make_seq(4, 10));

        let (c, n, b, p) = qm.len_by_priority();
        assert_eq!(c, 2);
        assert_eq!(n, 1);
        assert_eq!(b, 0);
        assert_eq!(p, 1);
    }
}
