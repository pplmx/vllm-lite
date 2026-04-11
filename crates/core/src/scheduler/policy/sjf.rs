use super::trait_def::{PriorityScore, SchedulingContext, SchedulingPolicy};
use crate::types::Sequence;

pub struct SjfPolicy {
    sjf_priority_weight: f32,
    sjf_remaining_work_weight: f32,
}

impl SjfPolicy {
    pub fn new(sjf_priority_weight: f32, sjf_remaining_work_weight: f32) -> Self {
        Self {
            sjf_priority_weight,
            sjf_remaining_work_weight,
        }
    }
}

impl Default for SjfPolicy {
    fn default() -> Self {
        Self::new(0.3, 0.7)
    }
}

impl SchedulingPolicy for SjfPolicy {
    fn compute_priority(&self, seq: &Sequence, _ctx: &SchedulingContext) -> PriorityScore {
        let remaining_tokens = seq.max_tokens.saturating_sub(seq.tokens.len());
        let user_priority = seq.priority.0 as u64;
        let score = (self.sjf_priority_weight * user_priority as f32
            + self.sjf_remaining_work_weight * remaining_tokens as f32) as u64;
        PriorityScore(score)
    }

    fn name(&self) -> &'static str {
        "SJF"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Priority, SamplingParams, Sequence, Status};
    use std::sync::Arc;

    fn make_sequence(id: u64, tokens_len: usize, max_tokens: usize) -> Sequence {
        Sequence {
            id,
            tokens: vec![1; tokens_len],
            kv_blocks: Arc::new(vec![]),
            num_computed_tokens: 0,
            prompt_len: tokens_len,
            status: Status::Waiting,
            max_tokens,
            sampling_params: SamplingParams::default(),
            consecutive_decode_rounds: 0,
            priority: Priority::default(),
        }
    }

    #[test]
    fn test_sjf_prefers_shorter_jobs() {
        let policy = SjfPolicy::default();
        let ctx = SchedulingContext {
            current_time: std::time::Instant::now(),
            queue_length: 2,
            running_count: 0,
            memory_pressure: 0.0,
        };
        let seq1 = make_sequence(1, 10, 100); // 90 tokens remaining
        let seq2 = make_sequence(2, 10, 60); // 50 tokens remaining

        let priority1 = policy.compute_priority(&seq1, &ctx);
        let priority2 = policy.compute_priority(&seq2, &ctx);

        assert!(
            priority2 < priority1,
            "SJF should prioritize shorter remaining work"
        );
    }
}
