use crate::types::{SeqId, Sequence};
use std::collections::VecDeque;

pub struct RequestQueue {
    waiting: VecDeque<Sequence>,
    running: Vec<Sequence>,
    finished: Vec<Sequence>,
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
}
