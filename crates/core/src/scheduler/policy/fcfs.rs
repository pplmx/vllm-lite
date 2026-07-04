#![allow(clippy::module_name_repetitions)]
use super::trait_def::{PriorityScore, SchedulingContext, SchedulingPolicy};
use crate::types::Sequence;

#[derive(Debug)]
/// First-Come-First-Served scheduling: sequences are batched in arrival order. Used as the baseline / fallback policy when fairness is the dominant constraint.
pub struct FcfsPolicy;

impl FcfsPolicy {
    #[must_use]
    pub const fn new() -> Self {
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
