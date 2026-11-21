//! Unit tests for the LRU + priority-weighted `EvictionPolicy`.
//!
//! Extracted from `eviction.rs` to keep the implementation file under
//! the project's 800-line soft cap. Exercises:
//!
//! - `EvictionPolicy::new` initial state
//! - `record_blocks` / `release_blocks` refcount bookkeeping
//!   (single, repeated, and zero-ref removal)
//! - `touch_blocks` access-order updates
//! - `select_victims` edge cases (empty sequences, zero blocks,
//!   prefill-vs-decode priority, only zero-ref blocks)
//! - `stats()` counters (`total_selections`, cache hit behavior)
//! - Property-based tests (proptest) in the sibling `prop_tests` module:
//!   refcount conservation, `select_victims` length bound, cache-hit
//!   on identical inputs.

use super::*;
use crate::types::{Priority, SamplingParams};
use std::sync::Arc;

fn create_test_sequence(id: u64, blocks: Vec<BlockId>, status: Status) -> Sequence {
    Sequence {
        id,
        tokens: vec![],
        kv_blocks: Arc::new(blocks),
        num_computed_tokens: 0,
        prompt_len: 0,
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
fn test_eviction_policy_new() {
    let policy = EvictionPolicy::new();
    assert!(policy.block_access_order.is_empty());
    assert!(policy.block_ref_count.is_empty());
}

#[test]
fn test_record_blocks() {
    let mut policy = EvictionPolicy::new();
    policy.record_blocks(&[1, 2, 3]);

    assert_eq!(policy.get_block_ref_count(1), 1);
    assert_eq!(policy.get_block_ref_count(2), 1);
    assert_eq!(policy.get_block_ref_count(3), 1);
}

#[test]
fn test_record_blocks_increments_ref_count() {
    let mut policy = EvictionPolicy::new();
    policy.record_blocks(&[1, 2]);
    policy.record_blocks(&[2, 3]);

    assert_eq!(policy.get_block_ref_count(1), 1);
    assert_eq!(policy.get_block_ref_count(2), 2);
    assert_eq!(policy.get_block_ref_count(3), 1);
}

#[test]
fn test_release_blocks() {
    let mut policy = EvictionPolicy::new();
    policy.record_blocks(&[1, 2, 3]);
    policy.release_blocks(&[2]);

    assert_eq!(policy.get_block_ref_count(1), 1);
    assert_eq!(policy.get_block_ref_count(2), 0);
    assert_eq!(policy.get_block_ref_count(3), 1);
}

#[test]
fn test_release_blocks_removes_zero_refs() {
    let mut policy = EvictionPolicy::new();
    policy.record_blocks(&[1]);
    policy.release_blocks(&[1]);

    assert_eq!(policy.get_block_ref_count(1), 0);
}

/// Releasing a block this policy never held must not report it as freed.
///
/// `MemoryManager::release_blocks` forwards `freed` to the allocator, so an
/// unknown block counted as released would hand a block that still has a live
/// owner back to the free pool — two sequences would then share one KV block.
#[test]
fn test_release_unrecorded_block_reports_nothing() {
    let mut policy = EvictionPolicy::new();
    let freed = policy.release_blocks(&[42]);
    assert!(
        freed.is_empty(),
        "a block that was never recorded must not be freed, got {freed:?}"
    );
    assert_eq!(policy.get_block_ref_count(42), 0);
}

/// Only the release that drops the count to zero reports the block.
#[test]
fn test_release_reports_only_the_final_reference() {
    let mut policy = EvictionPolicy::new();
    policy.record_blocks(&[5, 5]);

    let first = policy.release_blocks(&[5]);
    assert!(
        first.is_empty(),
        "one reference remains, so nothing is freed, got {first:?}"
    );

    let second = policy.release_blocks(&[5]);
    assert_eq!(second, vec![5], "the last reference frees the block");
}

#[test]
fn test_touch_blocks_updates_order() {
    let mut policy = EvictionPolicy::new();
    policy.record_blocks(&[1, 2, 3]);
    policy.touch_blocks(&[1]);

    let front = policy.block_access_order.front();
    assert_eq!(front, Some(&1));
}

#[test]
fn test_select_victims_empty_sequences() {
    let mut policy = EvictionPolicy::new();
    let victims = policy.select_victims(&[], 5);
    assert!(victims.is_empty());
}

#[test]
fn test_select_victims_zero_blocks() {
    let mut policy = EvictionPolicy::new();
    let seq = create_test_sequence(1, vec![1, 2], Status::Decoding);
    let victims = policy.select_victims(&[seq], 0);
    assert!(victims.is_empty());
}

#[test]
fn test_select_victims_prefilling_priority() {
    let mut policy = EvictionPolicy::new();
    policy.record_blocks(&[1, 2]);

    let prefill_seq = create_test_sequence(1, vec![1], Status::Prefilling);
    let decode_seq = create_test_sequence(2, vec![2], Status::Decoding);

    let victims = policy.select_victims(&[prefill_seq, decode_seq], 1);
    assert_eq!(victims.len(), 1);
}

#[test]
fn test_select_victims_only_zero_ref_blocks() {
    let mut policy = EvictionPolicy::new();
    policy.record_blocks(&[1]);
    policy.record_blocks(&[1]);

    let seq = create_test_sequence(1, vec![1], Status::Decoding);
    let victims = policy.select_victims(&[seq], 1);

    assert!(victims.is_empty());
}

#[test]
fn test_stats() {
    let mut policy = EvictionPolicy::new();

    assert_eq!(policy.stats().total_selections, 0);

    let seq = create_test_sequence(1, vec![1], Status::Decoding);
    policy.select_victims(&[seq], 1);

    assert_eq!(policy.stats().total_selections, 1);
}

#[test]
fn test_cache_invalidation() {
    let mut policy = EvictionPolicy::new();

    let seq = create_test_sequence(1, vec![1], Status::Decoding);
    let victims1 = policy.select_victims(std::slice::from_ref(&seq), 1);

    policy.record_blocks(&[2]);
    let victims2 = policy.select_victims(&[seq], 1);

    assert_eq!(victims1.len(), victims2.len());
}

/// Regression: the victim-cache hash must cover
/// `consecutive_decode_rounds`, because `compute_priority` flips when a
/// sequence crosses the 5-round threshold. Without it, a cached victim
/// order computed before the crossing is served unchanged afterwards.
#[test]
fn test_victim_cache_invalidates_on_decode_rounds_change() {
    let mut policy = EvictionPolicy::new();
    policy.record_blocks(&[1]);
    policy.record_blocks(&[2]);

    // seq 1: new decode (rounds 0) -> priority 3; seq 2: long-running
    // (rounds 6) -> priority 1. Higher priority values evict first.
    let mut seq_a = create_test_sequence(1, vec![1], Status::Decoding);
    let mut seq_b = create_test_sequence(2, vec![2], Status::Decoding);
    seq_b.consecutive_decode_rounds = 6;

    let victims = policy.select_victims(&[seq_a.clone(), seq_b.clone()], 2);
    assert_eq!(victims, vec![1, 2], "priority-3 block evicts first");

    // Cross the threshold on both sequences without changing id, status,
    // or block counts: priorities flip (a -> 1, b -> 3).
    seq_a.consecutive_decode_rounds = 7;
    seq_b.consecutive_decode_rounds = 0;

    let victims = policy.select_victims(&[seq_a, seq_b], 2);
    assert_eq!(
        victims,
        vec![2, 1],
        "cache must invalidate when decode rounds cross the priority threshold"
    );
}

/// Regression: `touch_blocks` reorders the LRU
/// deque that the victim order's tiebreak embeds, so it must invalidate
/// the cached victim set.
#[test]
fn test_touch_blocks_invalidates_victim_cache() {
    let mut policy = EvictionPolicy::new();
    // Record order 1 then 2 -> deque [2, 1]; block 1 is the LRU victim.
    policy.record_blocks(&[1, 2]);

    let seq = create_test_sequence(1, vec![1, 2], Status::Decoding);
    let victims = policy.select_victims(std::slice::from_ref(&seq), 2);
    assert_eq!(victims, vec![1, 2], "LRU block 1 evicts first");

    let hits_before = policy.stats().cache_hits;
    let cached = policy.select_victims(std::slice::from_ref(&seq), 2);
    assert_eq!(cached, vec![1, 2]);
    assert_eq!(
        policy.stats().cache_hits,
        hits_before + 1,
        "identical second call must hit the cache"
    );

    // Touch block 1 (now MRU): block 2 becomes the LRU victim.
    policy.touch_blocks(&[1]);
    let victims = policy.select_victims(std::slice::from_ref(&seq), 2);
    assert_eq!(
        victims,
        vec![2, 1],
        "touch must invalidate the cached victim order"
    );
    assert_eq!(
        policy.stats().cache_hits,
        hits_before + 1,
        "post-touch call must recompute, not reuse the stale cache"
    );
}

/// `select_victims` must skip Finished and Waiting sequences entirely.
///
/// The filter is `if seq.status == Status::Finished || seq.status ==
/// Status::Waiting { continue; }`. `cargo mutants` turned that `||` into `&&`,
/// which can never be true (a sequence holds one status), so NOTHING is
/// skipped — finished sequences then compete to become eviction victims, and a
/// block still owned by a live sequence can be chosen for reclaim because a
/// dead one co-owns it at a lower priority.
#[test]
fn test_select_victims_skips_finished_and_waiting_sequences() {
    let mut policy = EvictionPolicy::new();
    policy.record_blocks(&[1, 2, 3]);

    // Only excluded sequences own these blocks.
    let finished = create_test_sequence(1, vec![1], Status::Finished);
    let waiting = create_test_sequence(2, vec![2], Status::Waiting);
    let victims = policy.select_victims(&[finished, waiting], 3);
    assert!(
        victims.is_empty(),
        "blocks owned solely by finished/waiting sequences must not be chosen; \
         got {victims:?}"
    );

    // Control: the SAME block owned by a running sequence IS eligible, so the
    // assertion above cannot pass by the policy simply never returning
    // anything.
    let decoding = create_test_sequence(3, vec![3], Status::Decoding);
    let victims = policy.select_victims(&[decoding], 3);
    assert_eq!(
        victims,
        vec![3],
        "a decoding sequence's block must be eligible for eviction"
    );
}

/// Touching a block must REORDER the LRU list, not reduce it to that block.
///
/// `record_blocks` / `release_blocks` / `touch_blocks` all re-insert a block at
/// the front after `block_access_order.retain(|&b| b != block)`. `cargo mutants`
/// flipped that `!=` to `==`, which keeps ONLY the touched block and discards
/// every other entry. The existing test asserts only that the front is the
/// touched block, which the broken version also satisfies — so the whole LRU
/// list could collapse to one entry unnoticed.
#[test]
fn test_touch_blocks_preserves_the_other_entries() {
    let mut policy = EvictionPolicy::new();
    policy.record_blocks(&[1, 2, 3]);
    // After recording, the most recent is at the front.
    assert_eq!(
        policy
            .block_access_order
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        vec![3, 2, 1],
        "setup: record order must put the newest first"
    );

    policy.touch_blocks(&[1]);

    assert_eq!(
        policy
            .block_access_order
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        vec![1, 3, 2],
        "a touch must move block 1 to the front and keep 3 and 2 in LRU order"
    );

    // Re-recording an existing block must also move it, not prune the list.
    policy.record_blocks(&[2]);
    assert_eq!(
        policy
            .block_access_order
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        vec![2, 1, 3],
        "re-recording must move block 2 to the front and keep the rest"
    );
}

/// `compute_priority` must flip at EXACTLY `> 5` decode rounds.
///
/// A long-running decode sequence drops from priority 3 to 1 so the policy
/// prefers evicting it. `cargo mutants` relaxed `> 5` to `>=`, which moves the
/// boundary one round earlier; no test inspected the boundary itself, only
/// values on either side of the old fixture.
#[test]
fn test_compute_priority_flips_above_five_decode_rounds() {
    let mut at_five = create_test_sequence(1, vec![1], Status::Decoding);
    at_five.consecutive_decode_rounds = 5;
    assert_eq!(
        EvictionPolicy::compute_priority(&at_five),
        3,
        "5 rounds is NOT above the threshold, so the sequence keeps priority 3"
    );

    let mut at_six = create_test_sequence(2, vec![2], Status::Decoding);
    at_six.consecutive_decode_rounds = 6;
    assert_eq!(
        EvictionPolicy::compute_priority(&at_six),
        1,
        "6 rounds is above the threshold, so the priority drops to 1"
    );

    // Prefilling is a flat 2 regardless of rounds, and the excluded statuses
    // are 0 — both arms pinned so a deleted match arm cannot go unnoticed.
    let mut prefilling = create_test_sequence(3, vec![3], Status::Prefilling);
    prefilling.consecutive_decode_rounds = 99;
    assert_eq!(EvictionPolicy::compute_priority(&prefilling), 2);

    let finished = create_test_sequence(4, vec![4], Status::Finished);
    assert_eq!(EvictionPolicy::compute_priority(&finished), 0);
}

/// Freeing one block must leave the OTHER blocks in the LRU list.
///
/// `release_blocks` removes a block from `block_access_order` with
/// `retain(|&b| b != block)` once its count hits zero. `cargo mutants` flipped
/// that `!=` to `==`, which keeps only the freed block and prunes every other
/// entry. The existing final-reference test uses a SINGLE block, where the two
/// forms coincide; with several tracked blocks the difference is total.
#[test]
fn test_release_blocks_keeps_the_other_entries() {
    let mut policy = EvictionPolicy::new();
    policy.record_blocks(&[1, 2, 3]);
    assert_eq!(
        policy
            .block_access_order
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        vec![3, 2, 1]
    );

    // Block 2 has one reference; releasing it frees and removes it.
    let freed = policy.release_blocks(&[2]);
    assert_eq!(freed, vec![2], "block 2 was the last reference");

    assert_eq!(
        policy
            .block_access_order
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        vec![3, 1],
        "the freed block must leave and blocks 3 and 1 must survive in order"
    );

    // The surviving blocks must still be individually releasable, proving the
    // bookkeeping stayed consistent after the removal.
    assert_eq!(policy.release_blocks(&[3]), vec![3]);
    assert_eq!(policy.release_blocks(&[1]), vec![1]);
    assert!(
        policy.block_access_order.is_empty(),
        "all blocks are released, so the order must be empty"
    );
}
