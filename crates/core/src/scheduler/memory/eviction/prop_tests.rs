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

/// Build a sequence with a unique block id derived from the sequence id,
/// so distinct sequences never share blocks (ref_count stays 1).
fn make_seq(id: u64, decode_rounds: u32, status: Status) -> Sequence {
    Sequence {
        id,
        tokens: vec![],
        kv_blocks: Arc::new(vec![id as usize]),
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

proptest! {
    /// When sequences have blocks with different eviction priorities,
    /// `select_victims` must return blocks from priority-3 sequences
    /// ("new" decode, ≤5 rounds) before priority-1 blocks ("long-running"
    /// decode, >5 rounds). Prefill (priority 2) falls between.
    ///
    /// We construct 1 sequence per priority tier, each owning a unique
    /// block (ref_count = 1, so all are eviction-eligible), then request
    /// 1 victim and verify it came from the highest-priority block.
    #[test]
    fn prop_eviction_prefers_higher_priority_blocks(
        long_running_rounds in 6u32..100,
        new_rounds in 0u32..5,
    ) {
        let mut policy = EvictionPolicy::new();

        // 3 sequences, each with a unique block, in different priority tiers.
        let long_seq = make_seq(1, long_running_rounds, Status::Decoding); // priority 1
        let new_seq = make_seq(2, new_rounds, Status::Decoding);           // priority 3
        let prefill_seq = make_seq(3, 0, Status::Prefilling);              // priority 2

        // Record each block so ref_count = 1 (eligible for eviction).
        policy.record_blocks(&[1]);
        policy.record_blocks(&[2]);
        policy.record_blocks(&[3]);

        let running = vec![long_seq, new_seq, prefill_seq];

        // Request 1 victim — should come from block 2 (priority 3, "new" decode).
        let victims = policy.select_victims(&running, 1);
        prop_assert_eq!(
            victims.len(),
            1,
            "exactly one victim should be returned for num_blocks=1"
        );
        prop_assert_eq!(
            victims[0], 2,
            "priority-3 block (new decode, seq 2) should be evicted first, got block {}",
            victims[0]
        );

        // Request 3 victims — should be sorted by priority descending.
        // We need a fresh policy because the cache may hit on the
        // second call with the same sequences.
        let mut policy2 = EvictionPolicy::new();
        policy2.record_blocks(&[1]);
        policy2.record_blocks(&[2]);
        policy2.record_blocks(&[3]);

        let all_victims = policy2.select_victims(&running, 3);
        prop_assert_eq!(all_victims.len(), 3, "should return all 3 eligible blocks");
        // Order: priority 3 (block 2) → priority 2 (block 3) → priority 1 (block 1)
        prop_assert!(
            all_victims == vec![2, 3, 1],
            "eviction order should be priority-3 → priority-2 → priority-1, got {:?}",
            all_victims
        );
    }

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
