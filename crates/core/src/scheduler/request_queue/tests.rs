//! Unit tests for the `RequestQueue` priority-aware scheduling queue.
//!
//! Extracted from `request_queue.rs` to keep the implementation file
//! under the project's 800-line soft cap. Exercises:
//!
//! - `enqueue` + `dequeue` (FIFO ordering under FCFS)
//! - `get(id)` O(1) lookup after enqueue
//! - `remove(id)` O(1) removal
//! - `drain_by_phase(Phase::Prefill)` separation of prefill vs. decode
//!   sequences
//! - Property-based tests (proptest) in the sibling `prop_tests` module:
//!   add/remove round-trip, get-after-enqueue, FIFO dequeue ordering,
//!   phase-index consistency

use std::sync::Arc;

use super::*;
use crate::scheduler::policy::FcfsPolicy;
use crate::types::{Priority, SamplingParams, Status};

fn make_sequence(id: u64, status: Status) -> Sequence {
    Sequence {
        id,
        tokens: vec![1, 2, 3],
        kv_blocks: Arc::new(vec![]),
        num_computed_tokens: 0,
        prompt_len: 3,
        status,
        max_tokens: 10,
        sampling_params: SamplingParams::default(),
        consecutive_decode_rounds: 0,
        priority: Priority::default(),
        degraded_draft: false,
        draft_model_id: None,
    }
}

#[test]
fn test_enqueue_and_dequeue() {
    let mut queue = RequestQueue::new();
    let policy = FcfsPolicy::new();
    let ctx = SchedulingContext {
        current_time: Instant::now(),
        queue_length: 0,
        running_count: 0,
        memory_pressure: 0.0,
    };

    let seq1 = make_sequence(1, Status::Waiting);
    let seq2 = make_sequence(2, Status::Waiting);

    queue.enqueue(seq1, &policy, &ctx);
    queue.enqueue(seq2, &policy, &ctx);

    assert_eq!(queue.len(), 2);

    let dequeued = queue.dequeue().unwrap();
    assert_eq!(dequeued.id, 1);
    let dequeued = queue.dequeue().unwrap();
    assert_eq!(dequeued.id, 2);
    assert!(queue.is_empty());
}

#[test]
fn test_get_o1() {
    let mut queue = RequestQueue::new();
    let policy = FcfsPolicy::new();
    let ctx = SchedulingContext {
        current_time: Instant::now(),
        queue_length: 0,
        running_count: 0,
        memory_pressure: 0.0,
    };

    let seq = make_sequence(42, Status::Waiting);
    queue.enqueue(seq, &policy, &ctx);

    let retrieved = queue.get(42);
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().id, 42);
}

#[test]
fn test_remove_o1() {
    let mut queue = RequestQueue::new();
    let policy = FcfsPolicy::new();
    let ctx = SchedulingContext {
        current_time: Instant::now(),
        queue_length: 0,
        running_count: 0,
        memory_pressure: 0.0,
    };

    let seq = make_sequence(42, Status::Waiting);
    queue.enqueue(seq, &policy, &ctx);

    let removed = queue.remove(42);
    assert!(removed.is_some());
    assert_eq!(removed.unwrap().id, 42);
    assert!(queue.get(42).is_none());
}

#[test]
fn test_drain_by_phase() {
    let mut queue = RequestQueue::new();
    let policy = FcfsPolicy::new();
    let ctx = SchedulingContext {
        current_time: Instant::now(),
        queue_length: 0,
        running_count: 0,
        memory_pressure: 0.0,
    };

    let prefill_seq = make_sequence(1, Status::Waiting);
    let decode_seq = make_sequence(2, Status::Decoding);

    queue.enqueue(prefill_seq, &policy, &ctx);
    queue.enqueue(decode_seq, &policy, &ctx);

    let prefill_seqs = queue.drain_by_phase(Phase::Prefill);
    assert_eq!(prefill_seqs.len(), 1);
    assert_eq!(prefill_seqs[0].id, 1);
    assert_eq!(queue.phase_len(Phase::Prefill), 0);
    assert_eq!(queue.phase_len(Phase::Decode), 1);
}
