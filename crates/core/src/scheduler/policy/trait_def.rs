use crate::types::Sequence;
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct SchedulingContext {
    pub current_time: Instant,
    pub queue_length: usize,
    pub running_count: usize,
    pub memory_pressure: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct PriorityScore(pub u64);

pub trait SchedulingPolicy: Send + Sync {
    fn compute_priority(&self, seq: &Sequence, context: &SchedulingContext) -> PriorityScore;
    fn name(&self) -> &'static str;
}
