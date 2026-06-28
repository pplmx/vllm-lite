#![allow(clippy::module_name_repetitions)]
use super::trait_def::{PriorityScore, SchedulingContext, SchedulingPolicy};
use crate::types::Sequence;

#[derive(Debug)]
/// `PriorityPolicy`: priority policy.
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
        let wait_factor = seq.id.saturating_sub(1) as f32;
        let aging_bonus = (wait_factor * self.priority_aging_factor) as u64;
        let base_priority = u64::from(seq.priority.0);
        let effective_priority = base_priority.saturating_sub(aging_bonus);
        PriorityScore(effective_priority)
    }

    fn name(&self) -> &'static str {
        "Priority"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Priority, SamplingParams, Sequence, Status};
    use std::sync::Arc;

    fn make_sequence(id: u64, priority: u8) -> Sequence {
        Sequence {
            id,
            tokens: vec![1, 2, 3],
            kv_blocks: Arc::new(vec![]),
            num_computed_tokens: 0,
            prompt_len: 3,
            status: Status::Waiting,
            max_tokens: 10,
            sampling_params: SamplingParams::default(),
            consecutive_decode_rounds: 0,
            priority: Priority(priority),
            degraded_draft: false,
            draft_model_id: None,
        }
    }

    #[test]
    fn test_priority_respects_user_priority() {
        let policy = PriorityPolicy::default();
        let ctx = SchedulingContext {
            current_time: std::time::Instant::now(),
            queue_length: 2,
            running_count: 0,
            memory_pressure: 0.0,
        };
        let high_priority = make_sequence(1, 10);
        let low_priority = make_sequence(2, 50);

        let p1 = policy.compute_priority(&high_priority, &ctx);
        let p2 = policy.compute_priority(&low_priority, &ctx);

        assert!(p1 < p2, "Higher user priority should have lower score");
    }
}

#[cfg(test)]
mod prop_tests {
    use super::*;
    use crate::types::{Priority, SamplingParams, Status};
    use proptest::prelude::*;
    use std::sync::Arc;
    use std::time::Instant;

    fn make_seq(id: u64, priority: u8) -> Sequence {
        Sequence {
            id,
            tokens: vec![1, 2, 3],
            kv_blocks: Arc::new(vec![]),
            num_computed_tokens: 0,
            prompt_len: 3,
            status: Status::Waiting,
            max_tokens: 10,
            sampling_params: SamplingParams::default(),
            consecutive_decode_rounds: 0,
            priority: Priority(priority),
            degraded_draft: false,
            draft_model_id: None,
        }
    }

    fn ctx() -> SchedulingContext {
        SchedulingContext {
            current_time: Instant::now(),
            queue_length: 0,
            running_count: 0,
            memory_pressure: 0.0,
        }
    }

    // Invariant 1: higher user priority (smaller `Priority`) yields a
    //              strictly lower (or equal) `PriorityScore`.
    // Invariant 2: aging — for the same priority, an older sequence
    //              (higher id) gets a lower score (more urgent).
    // Invariant 3: PriorityScore is non-negative (u64 saturating math).

    proptest! {
        /// Higher user priority → lower (or equal) PriorityScore.
        #[test]
        fn prop_higher_priority_lower_score(
            high in 0u8..128,
            low in 0u8..128,
        ) {
            let policy = PriorityPolicy::default();
            let seq_high = make_seq(1, high);
            let seq_low = make_seq(1, low);
            let p_high = policy.compute_priority(&seq_high, &ctx());
            let p_low = policy.compute_priority(&seq_low, &ctx());
            // `high` is the user-facing priority value where smaller
            // means more urgent; the score must reflect that ordering.
            if high < low {
                prop_assert!(p_high.0 <= p_low.0, "high priority (value={}) gave higher score {} vs low priority {} score {}", high, p_high.0, low, p_low.0);
            } else if high > low {
                prop_assert!(p_high.0 >= p_low.0);
            } else {
                prop_assert_eq!(p_high.0, p_low.0);
            }
        }

        /// Aging: for sequences with the same user priority, a higher
        /// `seq.id` (older) produces a lower (or equal) score because
        /// of the aging bonus.
        #[test]
        fn prop_aging_reduces_score(
            id_a in 1u64..1000,
            id_b in 1u64..1000,
            priority in 0u8..128,
        ) {
            let policy = PriorityPolicy::default();
            let seq_a = make_seq(id_a, priority);
            let seq_b = make_seq(id_b, priority);
            let score_a = policy.compute_priority(&seq_a, &ctx());
            let score_b = policy.compute_priority(&seq_b, &ctx());
            if id_a > id_b {
                prop_assert!(score_a.0 <= score_b.0, "older seq id={} should have score <= newer seq id={}, got {} vs {}", id_a, id_b, score_a.0, score_b.0);
            } else if id_a < id_b {
                prop_assert!(score_a.0 >= score_b.0);
            } else {
                prop_assert_eq!(score_a.0, score_b.0);
            }
        }

        /// PriorityScore never panics or overflows: for any u8 priority
        /// and any u64 id, the score is a valid u64.
        #[test]
        fn prop_score_in_bounds(
            priority in proptest::num::u8::ANY,
            id in proptest::num::u64::ANY,
        ) {
            let policy = PriorityPolicy::default();
            let seq = make_seq(id, priority);
            let score = policy.compute_priority(&seq, &ctx());
            // Just verify we get a result; saturating arithmetic
            // guarantees no overflow panic regardless of input range.
            let _ = score.0;
        }
    }
}
