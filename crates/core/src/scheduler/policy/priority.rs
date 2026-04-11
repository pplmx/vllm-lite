use super::trait_def::{PriorityScore, SchedulingContext, SchedulingPolicy};
use crate::types::Sequence;

pub struct PriorityPolicy {
    priority_aging_factor: f32,
    _priority_levels: u8,
}

impl PriorityPolicy {
    pub fn new(priority_aging_factor: f32, priority_levels: u8) -> Self {
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
        let base_priority = seq.priority.0 as u64;
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
