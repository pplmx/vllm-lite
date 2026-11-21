//! Shortest-job-first scheduling policy: scores sequences by remaining token count, shortest first.
//!
//! Minimises average completion time for mixed-length workloads at
//! the cost of starving long sequences — typically paired with
//! priority-policy for tiered fairness.
#![allow(clippy::module_name_repetitions)]
use super::trait_def::{PriorityScore, SchedulingContext, SchedulingPolicy};
use crate::types::Sequence;

#[derive(Debug)]
/// Shortest-Job-First scheduling: sequences with the fewest remaining tokens are batched first. Optimizes tail latency but can starve long-running sequences.
pub struct SjfPolicy {
    sjf_priority_weight: f32,
    sjf_remaining_work_weight: f32,
}

impl SjfPolicy {
    #[must_use]
    pub const fn new(sjf_priority_weight: f32, sjf_remaining_work_weight: f32) -> Self {
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
        // `max_tokens` is the upper bound on *generated* tokens (prompt not
        // included), so remaining work = max_tokens - already-generated
        // (tokens.len() - prompt_len). Using saturating_sub on both subtractions
        // guards against drift where prompt_len > tokens.len().
        let remaining_tokens = seq
            .max_tokens
            .saturating_sub(seq.tokens.len().saturating_sub(seq.prompt_len));
        let user_priority = u64::from(seq.priority.0);
        // invariant: user_priority and remaining_tokens are bounded; f32
        // precision loss is acceptable for the SJF score. The f32 score is
        // non-negative (positive weights × non-negative counts), so the
        // f32 -> u64 conversion is sign-safe.
        #[allow(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss
        )]
        let score: u64 = self.sjf_priority_weight.mul_add(
            user_priority as f32,
            self.sjf_remaining_work_weight * remaining_tokens as f32,
        ) as u64;
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
            degraded_draft: false,
            draft_model_id: None,
        }
    }

    /// The remaining-work term must be a PRODUCT, not a sum.
    ///
    /// The score is `pw * user_priority + rw * remaining_tokens`. `cargo
    /// mutants` replaced the `*` with `+` and the existing test still passed:
    /// with the default weights both orderings agree on that fixture. The
    /// weights below make them disagree — a long low-priority sequence must
    /// beat a short high-priority one under the real formula, and lose to it
    /// once `rw * remaining` becomes `rw + remaining`.
    #[test]
    fn test_sjf_remaining_work_is_weighted_not_merely_added() {
        // pw = 10, rw = 0.5 -> product score: A = 0*10 + 0.5*100 = 50,
        // B = 9*10 + 0.5*2 = 91, so A (the long, low-priority one) wins.
        // Summed score: A = 0 + (0.5 + 100) = 100.5, B = 90 + 2.5 = 92.5,
        // which flips the order to B.
        let policy = SjfPolicy::new(10.0, 0.5);
        let ctx = SchedulingContext {
            current_time: std::time::Instant::now(),
            queue_length: 2,
            running_count: 0,
            memory_pressure: 0.0,
        };
        let long_low_priority = make_sequence(1, 10, 100);
        let mut short_high_priority = make_sequence(2, 10, 2);
        short_high_priority.priority = Priority(9);

        let long_score = policy.compute_priority(&long_low_priority, &ctx);
        let short_score = policy.compute_priority(&short_high_priority, &ctx);

        assert!(
            long_score < short_score,
            "the remaining-work term must scale with the token count (product), \
             not merely offset the priority term (sum): got long={long_score:?} \
             short={short_score:?}"
        );
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
        let seq1 = make_sequence(1, 10, 100); // 100 generated tokens remaining
        let seq2 = make_sequence(2, 10, 60); // 60 generated tokens remaining

        let priority1 = policy.compute_priority(&seq1, &ctx);
        let priority2 = policy.compute_priority(&seq2, &ctx);

        assert!(
            priority2 < priority1,
            "SJF should prioritize shorter remaining work"
        );
    }
}
