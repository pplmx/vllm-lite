//! Priority scheduling policy: scores sequences by their explicit priority field, breaking ties by arrival order.
//!
//! `Sequence::priority` is set by the API layer from the `OpenAI`
//! `priority` parameter (or via the `/admin/priorities` endpoint).
#![allow(clippy::module_name_repetitions)]
use super::trait_def::{PriorityScore, SchedulingContext, SchedulingPolicy};
use crate::types::Sequence;

#[derive(Debug)]
/// Priority-based scheduling: sequences are batched in descending priority order, then by arrival time as tiebreaker. Use this when you have SLA tiers (premium vs. free).
pub struct PriorityPolicy {
    priority_aging_factor: f32,
    _priority_levels: u8,
}

impl PriorityPolicy {
    #[must_use]
    pub const fn new(priority_aging_factor: f32, priority_levels: u8) -> Self {
        Self {
            priority_aging_factor,
            _priority_levels: priority_levels,
        }
    }
}

impl Default for PriorityPolicy {
    fn default() -> Self {
        Self::new(0.1, 10)
    }
}

impl SchedulingPolicy for PriorityPolicy {
    fn compute_priority(&self, seq: &Sequence, _ctx: &SchedulingContext) -> PriorityScore {
        // invariant: seq.id is bounded by request count, far below 2^24; f32
        // precision loss is acceptable for the wait-factor. wait_factor and
        // the product are non-negative, so the f32 -> u64 conversion is sign-safe.
        #[allow(
            clippy::cast_precision_loss,
            clippy::cast_sign_loss,
            clippy::cast_possible_truncation
        )]
        let aging_bonus: u64 = {
            let wait_factor = seq.id.saturating_sub(1) as f32;
            (wait_factor * self.priority_aging_factor) as u64
        };
        let base_priority = u64::from(seq.priority.0);
        let effective_priority = base_priority.saturating_sub(aging_bonus);
        PriorityScore(effective_priority)
    }

    fn name(&self) -> &'static str {
        "Priority"
    }
}

// Unit tests are extracted to `tests.rs` and `prop_tests.rs` to keep
// this file under the 800-line soft cap. See those siblings for the
// test surface (priority respects user priority; plus proptest
// invariants for higher-priority → lower score, aging reduces score,
// and score stays bounded for arbitrary u8/u64 inputs).
#[cfg(test)]
mod prop_tests;
#[cfg(test)]
mod tests;
