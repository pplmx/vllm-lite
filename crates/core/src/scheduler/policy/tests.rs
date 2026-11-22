#[cfg(test)]
mod policy_tests {
    use super::super::*;
    use crate::types::{Priority, SamplingParams, Sequence, Status};
    use std::sync::Arc;
    use std::time::Instant;

    fn make_sequence(id: u64) -> Sequence {
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
            priority: Priority::default(),
            degraded_draft: false,
            draft_model_id: None,
        }
    }

    #[test]
    fn test_fcfs_priority_ordering() {
        let policy = FcfsPolicy::new();
        let ctx = SchedulingContext {
            current_time: Instant::now(),
            queue_length: 2,
            running_count: 0,
            memory_pressure: 0.0,
        };

        let seq1 = make_sequence(1);
        let seq2 = make_sequence(2);

        let priority1 = policy.compute_priority(&seq1, &ctx);
        let priority2 = policy.compute_priority(&seq2, &ctx);

        assert!(priority1 < priority2, "FCFS should prioritize lower ID");
    }

    #[test]
    fn test_fcfs_name() {
        let policy = FcfsPolicy::new();
        assert_eq!(policy.name(), "FCFS");
    }

    /// Every policy's `name()` must report its own identity.
    ///
    /// `cargo mutants` replaced `PriorityPolicy::name` and `SjfPolicy::name`
    /// with `""` and `"xyzzy"` and nothing failed: only `FcfsPolicy` had a name
    /// assertion. The name is what logs and metrics label a policy with, so a
    /// wrong string silently mis-attributes every scheduling decision in
    /// production telemetry.
    #[test]
    fn every_policy_reports_a_distinct_name() {
        assert_eq!(PriorityPolicy::default().name(), "Priority");
        assert_eq!(SjfPolicy::default().name(), "SJF");
        assert_eq!(FcfsPolicy::new().name(), "FCFS");

        // Distinct, so a log line cannot be read as another policy.
        let names = [
            PriorityPolicy::default().name(),
            SjfPolicy::default().name(),
            FcfsPolicy::new().name(),
        ];
        for (i, left) in names.iter().enumerate() {
            assert!(!left.is_empty(), "policy names must not be empty");
            for right in &names[i + 1..] {
                assert_ne!(left, right, "policy names must be distinct");
            }
        }
    }

    #[test]
    fn scheduling_policy_default_arc_is_fcfs() {
        let policy: Arc<dyn SchedulingPolicy> = <dyn SchedulingPolicy>::default_arc();
        assert_eq!(policy.name(), "FCFS");
    }
    /// `PriorityPolicy` must age a sequence by
    /// `(seq_id - 1) * priority_aging_factor`, subtracted from its base
    /// priority.
    ///
    /// `cargo mutants` turned that `*` into `+` and `/`, both surviving: the
    /// existing tests only assert `name()`, so the aging formula itself was never
    /// checked. Aging is what keeps a low-priority sequence from starving — with
    /// `+` the bonus becomes 100.1 for any factor, and with `/` it collapses
    /// toward zero for any factor below 1.
    #[test]
    fn test_priority_policy_ages_by_the_configured_factor() {
        let ctx = SchedulingContext {
            current_time: Instant::now(),
            queue_length: 0,
            running_count: 0,
            memory_pressure: 0.0,
        };

        // aging_factor 0.1, seq id 101 -> wait_factor 100 -> bonus 10.
        let policy = PriorityPolicy::new(0.1, 10);
        let mut seq = make_sequence(101);
        seq.priority = Priority(50);
        assert_eq!(
            policy.compute_priority(&seq, &ctx),
            PriorityScore(40),
            "50 - (100 * 0.1) must be 40"
        );

        // A larger factor ages faster: 1.0 * 100 = 100 > 50, so the score
        // saturates at 0 rather than wrapping.
        let aggressive = PriorityPolicy::new(1.0, 10);
        assert_eq!(
            aggressive.compute_priority(&seq, &ctx),
            PriorityScore(0),
            "an aging bonus above the base priority must saturate at 0"
        );

        // The first sequence never ages: wait_factor is id-1 == 0.
        let mut senior = make_sequence(1);
        senior.priority = Priority(7);
        assert_eq!(
            policy.compute_priority(&senior, &ctx),
            PriorityScore(7),
            "seq id 1 has no wait and must keep its base priority exactly"
        );
    }
}
