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
}
