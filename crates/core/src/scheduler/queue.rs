use crate::types::SeqId;
use crate::types::Sequence;
use std::collections::VecDeque;

pub struct RequestQueue {
    waiting: VecDeque<Sequence>,
    running: Vec<Sequence>,
    finished: Vec<Sequence>,
    next_seq_id: SeqId,
}

impl Default for RequestQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl RequestQueue {
    pub fn new() -> Self {
        Self {
            waiting: VecDeque::new(),
            running: Vec::new(),
            finished: Vec::new(),
            next_seq_id: 1,
        }
    }

    pub fn push_waiting(&mut self, seq: Sequence) {
        self.waiting.push_back(seq);
    }

    pub fn pop_waiting(&mut self) -> Option<Sequence> {
        self.waiting.pop_front()
    }

    pub fn add_running(&mut self, seq: Sequence) {
        self.running.push(seq);
    }

    pub fn remove_running(&mut self, id: SeqId) -> Option<Sequence> {
        if let Some(pos) = self.running.iter().position(|s| s.id == id) {
            Some(self.running.remove(pos))
        } else {
            None
        }
    }

    pub fn get_running(&self) -> &[Sequence] {
        &self.running
    }

    pub fn get_running_mut(&mut self) -> &mut Vec<Sequence> {
        &mut self.running
    }

    pub fn mark_finished(&mut self, seq: Sequence) {
        self.finished.push(seq);
    }

    pub fn drain_finished(&mut self) -> Vec<Sequence> {
        std::mem::take(&mut self.finished)
    }

    pub fn waiting_count(&self) -> usize {
        self.waiting.len()
    }

    pub fn running_count(&self) -> usize {
        self.running.len()
    }

    pub fn next_seq_id(&mut self) -> SeqId {
        let id = self.next_seq_id;
        self.next_seq_id += 1;
        id
    }

    pub fn waiting_push_back(&mut self, seq: Sequence) {
        self.waiting.push_back(seq);
    }

    pub fn waiting_pop_front(&mut self) -> Option<Sequence> {
        self.waiting.pop_front()
    }

    pub fn waiting_is_empty(&self) -> bool {
        self.waiting.is_empty()
    }

    pub fn waiting_len(&self) -> usize {
        self.waiting.len()
    }

    pub fn waiting_sort_by_priority(&mut self) {
        let mut waiting_vec: Vec<_> = self.waiting.drain(..).collect();
        waiting_vec.sort_by(|a, b| a.priority.cmp(&b.priority));
        self.waiting = waiting_vec.into();
    }

    pub fn waiting_drain(&mut self) -> std::collections::VecDeque<Sequence> {
        self.waiting.drain(..).collect::<Vec<_>>().into()
    }

    pub fn running_push(&mut self, seq: Sequence) {
        self.running.push(seq);
    }

    pub fn running_len(&self) -> usize {
        self.running.len()
    }

    pub fn running_retain<F>(&mut self, f: F)
    where
        F: FnMut(&Sequence) -> bool,
    {
        self.running.retain(f);
    }

    pub fn running_iter(&self) -> impl Iterator<Item = &Sequence> {
        self.running.iter()
    }

    pub fn running_iter_mut(&mut self) -> impl Iterator<Item = &mut Sequence> {
        self.running.iter_mut()
    }

    pub fn running_find(&self, id: SeqId) -> Option<&Sequence> {
        self.running.iter().find(|s| s.id == id)
    }

    pub fn running_find_mut(&mut self, id: SeqId) -> Option<&mut Sequence> {
        self.running.iter_mut().find(|s| s.id == id)
    }

    pub fn finished_extend(&mut self, seqs: Vec<Sequence>) {
        self.finished.extend(seqs);
    }

    pub fn finished(&self) -> &Vec<Sequence> {
        &self.finished
    }
}
