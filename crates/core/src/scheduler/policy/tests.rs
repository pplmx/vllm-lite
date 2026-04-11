#[cfg(test)]
mod tests {
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
}
