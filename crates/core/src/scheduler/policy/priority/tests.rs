//! Unit tests for the `PriorityPolicy` (priority + aging scheduler).
//!
//! Extracted from `priority.rs` to keep the implementation file under
//! the project's 800-line soft cap. Exercises:
//!
//! - `PriorityPolicy::compute_priority` reflects user priority
//!   (smaller Priority value → smaller score).
//! - Property-based tests (proptest) in the sibling `prop_tests`
//!   module: monotonic priority ordering, aging reduces score for
//!   older sequences, score remains a valid u64 for arbitrary inputs.

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
