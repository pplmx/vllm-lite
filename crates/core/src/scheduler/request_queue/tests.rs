#![allow(clippy::doc_markdown)]

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
use crate::scheduler::policy::{FcfsPolicy, PriorityPolicy};
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

fn make_sequence_with_priority(id: u64, priority: u8, status: Status) -> Sequence {
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
        priority: Priority(priority),
        degraded_draft: false,
        draft_model_id: None,
    }
}

fn test_ctx() -> SchedulingContext {
    SchedulingContext {
        current_time: Instant::now(),
        queue_length: 0,
        running_count: 0,
        memory_pressure: 0.0,
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

/// PriorityPolicy: `dequeue` must return sequences ordered by their explicit
/// `Priority` value — lower `Priority(n)` means higher scheduling priority and
/// must be dequeued first. With small seq_ids (< 10) the aging bonus is 0,
/// so the ordering is determined purely by the user-supplied priority.
#[test]
fn test_priority_policy_dequeue_ordering() {
    let mut queue = RequestQueue::new();
    let policy = PriorityPolicy::default();
    let ctx = test_ctx();

    // Enqueue out of priority order to ensure the heap actually orders them.
    queue.enqueue(
        make_sequence_with_priority(1, 100, Status::Waiting),
        &policy,
        &ctx,
    );
    queue.enqueue(
        make_sequence_with_priority(2, 0, Status::Waiting),
        &policy,
        &ctx,
    );
    queue.enqueue(
        make_sequence_with_priority(3, 50, Status::Waiting),
        &policy,
        &ctx,
    );

    assert_eq!(queue.len(), 3);

    // Lower Priority value → lower PriorityScore → popped first.
    let first = queue.dequeue().expect("should have a first");
    assert_eq!(first.id, 2, "Priority(0) should dequeue first");

    let second = queue.dequeue().expect("should have a second");
    assert_eq!(second.id, 3, "Priority(50) should dequeue second");

    let third = queue.dequeue().expect("should have a third");
    assert_eq!(third.id, 1, "Priority(100) should dequeue last");

    assert!(queue.is_empty());
}

/// PriorityPolicy: `dequeue` must respect priority even when sequences arrive
/// in strict FIFO order (i.e. insertion order does not determine dequeue order).
#[test]
fn test_priority_policy_overrides_fifo_insertion() {
    let mut queue = RequestQueue::new();
    let policy = PriorityPolicy::default();
    let ctx = test_ctx();

    // Insert 5 sequences with descending priority values — FIFO would
    // dequeue them in insertion order (1, 2, 3, 4, 5), but PriorityPolicy
    // must reorder so Priority(10) is first and Priority(50) is last.
    for (i, prio) in [(1u64, 50u8), (2, 40), (3, 30), (4, 20), (5, 10)] {
        queue.enqueue(
            make_sequence_with_priority(i, prio, Status::Waiting),
            &policy,
            &ctx,
        );
    }

    // Dequeue and collect — should get them in priority order, not insertion order.
    let dequeued_ids: Vec<SeqId> = (0..5).map(|_| queue.dequeue().unwrap().id).collect();

    assert_eq!(
        dequeued_ids,
        vec![5, 4, 3, 2, 1],
        "dequeue must return highest-priority (lowest Priority value) first, got {dequeued_ids:?}"
    );
}

/// PriorityPolicy: tie-breaking — when two sequences have the same explicit
/// `Priority` value, the one that arrived earlier (lower seq_id) must be
/// dequeued first (FIFO within the same priority tier).
#[test]
fn test_priority_policy_tiebreak_is_fifo() {
    let mut queue = RequestQueue::new();
    let policy = PriorityPolicy::default();
    let ctx = test_ctx();

    // All three have Priority(25) — tie-break should be by arrival time.
    // Lower seq_id = arrived earlier (in these tests, seq_id ≈ arrival order).
    queue.enqueue(
        make_sequence_with_priority(1, 25, Status::Waiting),
        &policy,
        &ctx,
    );
    queue.enqueue(
        make_sequence_with_priority(2, 25, Status::Waiting),
        &policy,
        &ctx,
    );
    queue.enqueue(
        make_sequence_with_priority(3, 25, Status::Waiting),
        &policy,
        &ctx,
    );

    assert_eq!(
        queue.dequeue().unwrap().id,
        1,
        "tie-break: earliest arrival first"
    );
    assert_eq!(queue.dequeue().unwrap().id, 2);
    assert_eq!(queue.dequeue().unwrap().id, 3);
}
