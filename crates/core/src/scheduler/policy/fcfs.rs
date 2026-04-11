use super::trait_def::{PriorityScore, SchedulingContext, SchedulingPolicy};
use crate::types::Sequence;

pub struct FcfsPolicy;

impl FcfsPolicy {
    pub fn new() -> Self {
        Self
    }
}

impl Default for FcfsPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl SchedulingPolicy for FcfsPolicy {
    fn compute_priority(&self, seq: &Sequence, _ctx: &SchedulingContext) -> PriorityScore {
        PriorityScore(seq.id)
    }

    fn name(&self) -> &'static str {
        "FCFS"
    }
}
