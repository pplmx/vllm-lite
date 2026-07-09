//! Property-based tests (proptest) for `RequestQueue`.
//! Companion to `tests.rs`; both extracted from `request_queue.rs` to
//! keep the implementation file under the project's 800-line soft cap.
//!
//! Invariants under test:
//! - Add then remove: queue is empty and removed sequence matches
//! - Add then `get(id)` returns Some with the same id
//! - Under FCFS, `dequeue` returns sequences in ascending id order (FIFO)
//! - `len()` equals the sum of `phase_len` across all phases, and
//!   `is_empty` is consistent with `len`

use std::sync::Arc;

use proptest::prelude::*;

use super::*;
use crate::scheduler::policy::FcfsPolicy;
use crate::types::{Priority, SamplingParams};

fn make_context() -> SchedulingContext {
    SchedulingContext {
        current_time: Instant::now(),
        queue_length: 0,
        running_count: 0,
        memory_pressure: 0.0,
    }
}

fn arb_sequence() -> impl Strategy<Value = Sequence> {
    (
        1u64..10_000,
        1usize..50,
        1usize..200,
        any::<u8>(),
        prop_oneof![Just(Status::Waiting), Just(Status::Decoding)],
    )
        .prop_map(
            |(id, prompt_len, max_tokens, priority_value, status)| Sequence {
                id,
                tokens: vec![0; prompt_len],
                kv_blocks: Arc::new(vec![]),
                num_computed_tokens: 0,
                prompt_len,
                status,
                max_tokens,
                sampling_params: SamplingParams::default(),
                consecutive_decode_rounds: 0,
                priority: Priority(priority_value),
                degraded_draft: false,
                draft_model_id: None,
            },
        )
}

fn arb_unique_sequences(max_len: usize) -> impl Strategy<Value = Vec<Sequence>> {
    proptest::collection::vec(arb_sequence(), 0..=max_len).prop_map(|mut seqs| {
        seqs.sort_by_key(|s| s.id);
        seqs.dedup_by_key(|s| s.id);
        seqs
    })
}

proptest! {
    /// Add then remove: queue is empty and removed sequence matches.
    #[test]
    fn prop_add_remove_empty(seq in arb_sequence()) {
        let mut queue = RequestQueue::new();
        let policy = FcfsPolicy::new();
        let ctx = make_context();

        queue.enqueue(seq.clone(), &policy, &ctx);
        prop_assert_eq!(queue.len(), 1);

        let removed = queue.remove(seq.id);
        prop_assert_eq!(removed.map(|s| s.id), Some(seq.id));
        prop_assert!(queue.is_empty());
        prop_assert_eq!(queue.len(), 0);
    }

    /// Add then get(id) returns Some with the same id.
    #[test]
    fn prop_get_after_enqueue(seq in arb_sequence()) {
        let mut queue = RequestQueue::new();
        let policy = FcfsPolicy::new();
        let ctx = make_context();

        queue.enqueue(seq.clone(), &policy, &ctx);

        let got = queue.get(seq.id);
        prop_assert!(got.is_some());
        // invariant: pre-conditions make this infallible at this call site.
        prop_assert_eq!(got.expect("just enqueued").id, seq.id);
    }

    /// Under FCFS, `dequeue` returns sequences in ascending id order (FIFO).
    #[test]
    fn prop_dequeue_is_fifo(seqs in arb_unique_sequences(40)) {
        let mut queue = RequestQueue::new();
        let policy = FcfsPolicy::new();
        let ctx = make_context();

        for seq in &seqs {
            queue.enqueue(seq.clone(), &policy, &ctx);
        }

        let mut popped = Vec::with_capacity(seqs.len());
        while let Some(s) = queue.dequeue() {
            popped.push(s.id);
        }

        prop_assert_eq!(popped, seqs.iter().map(|s| s.id).collect::<Vec<_>>());
        prop_assert!(queue.is_empty());
    }

    /// `len()` equals the sum of `phase_len` across all phases after
    /// enqueue/dequeue, and `is_empty` is consistent with `len`.
    #[test]
    fn prop_phase_index_consistent(seqs in arb_unique_sequences(30)) {
        let mut queue = RequestQueue::new();
        let policy = FcfsPolicy::new();
        let ctx = make_context();

        for seq in &seqs {
            queue.enqueue(seq.clone(), &policy, &ctx);
        }

        let total_via_phase =
            queue.phase_len(Phase::Prefill) + queue.phase_len(Phase::Decode);
        prop_assert_eq!(queue.len(), total_via_phase);
        prop_assert_eq!(queue.is_empty(), seqs.is_empty());

        while queue.dequeue().is_some() {}
        prop_assert!(queue.is_empty());
        prop_assert_eq!(queue.len(), 0);
        prop_assert_eq!(queue.phase_len(Phase::Prefill), 0);
        prop_assert_eq!(queue.phase_len(Phase::Decode), 0);
    }
}
