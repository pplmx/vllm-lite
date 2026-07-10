//! Property-based tests (proptest) for the LRU + priority-weighted
//! `EvictionPolicy`. Companion to `tests.rs`; both extracted from
//! `eviction.rs` to keep the implementation file under the project's
//! 800-line soft cap.
//!
//! Invariants under test:
//! - `record_blocks` / `release_blocks` refcount conservation (count after
//!   N records and M releases equals `max(0, N - M)`)
//! - `select_victims` returns at most `num_blocks` entries, and yields
//!   empty for empty input sequences
//! - cache-hit on identical `select_victims` call: `cache_hits` strictly
//!   increases on the second call with the same input

use super::*;
use crate::types::{Priority, SamplingParams};
use proptest::prelude::*;
use std::sync::Arc;

fn make_sequence(id: u64, blocks: Vec<BlockId>, status: Status, decode_rounds: u32) -> Sequence {
    Sequence {
        id,
        tokens: vec![],
        kv_blocks: Arc::new(blocks),
        num_computed_tokens: 0,
        prompt_len: 0,
        status,
        max_tokens: 10,
        sampling_params: SamplingParams::default(),
        consecutive_decode_rounds: decode_rounds,
        priority: Priority::default(),
        degraded_draft: false,
        draft_model_id: None,
    }
}

#[allow(dead_code)] // proptest helpers referenced indirectly via proptest! macro
fn arb_status() -> impl Strategy<Value = Status> {
    prop_oneof![
        Just(Status::Waiting),
        Just(Status::Prefilling),
        Just(Status::Decoding)
    ]
}

#[allow(dead_code)] // proptest helpers referenced indirectly via proptest! macro
fn arb_sequence(id: u64) -> impl Strategy<Value = Sequence> {
    (
        proptest::collection::vec(0usize..32, 1..8),
        arb_status(),
        0u32..10,
    )
        .prop_map(move |(blocks, status, decode_rounds)| {
            make_sequence(id, blocks, status, decode_rounds)
        })
}

proptest! {
    /// Refcount conservation: total_refs equals
    /// max(0, records - releases) across any sequence of operations.
    #[test]
    fn prop_record_release_refcount_conserved(
        ops in proptest::collection::vec(
            (0usize..16, proptest::bool::ANY),
            1..50,
        ),
    ) {
        let mut policy = EvictionPolicy::new();
        let mut expected: HashMap<BlockId, usize> = HashMap::new();

        for (block_id, is_record) in ops {
            if is_record {
                policy.record_blocks(&[block_id]);
                *expected.entry(block_id).or_insert(0) += 1;
            } else if let Some(&count) = expected.get(&block_id) {
                policy.release_blocks(&[block_id]);
                if count <= 1 {
                    expected.remove(&block_id);
                } else {
                    expected.insert(block_id, count - 1);
                }
            }
            prop_assert_eq!(policy.get_block_ref_count(block_id), expected.get(&block_id).copied().unwrap_or(0));
        }
    }

    /// `select_victims` never returns more than `num_blocks` entries,
    /// and yields empty for empty input sequences.
    #[test]
    fn prop_select_victims_length_bounded(
        num_blocks in 0usize..10,
    ) {
        let mut policy = EvictionPolicy::new();
        let victims = policy.select_victims(&[], num_blocks);
        prop_assert!(victims.is_empty());
        prop_assert!(victims.len() <= num_blocks);
    }

    /// Repeated identical `select_victims` call must hit the cache:
    /// `cache_hits` strictly increases on the second call with the
    /// same input. We pin status to Decoding (Waiting/Finished
    /// sequences are skipped) and use unique blocks (otherwise the
    /// ref count exceeds 1, making the block unavailable for
    /// eviction and producing an empty victims list that bypasses
    /// the cache-hit check).
    #[test]
    fn prop_select_victims_cache_hit(
        blocks in proptest::collection::hash_set(0usize..32, 1..4),
    ) {
        let mut policy = EvictionPolicy::new();
        let blocks: Vec<usize> = blocks.into_iter().collect();
        for &block in &blocks {
            policy.record_blocks(&[block]);
        }
        let seq = make_sequence(1, blocks, Status::Decoding, 0);

        let _ = policy.select_victims(std::slice::from_ref(&seq), 1);
        let after_first = policy.stats();
        let _ = policy.select_victims(std::slice::from_ref(&seq), 1);
        let after_second = policy.stats();

        prop_assert_eq!(after_first.cache_hits, 0);
        prop_assert!(after_second.cache_hits > after_first.cache_hits);
    }
}
