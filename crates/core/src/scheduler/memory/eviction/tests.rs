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

/// Regression (RIL TASK-002 / ISS-002): the victim-cache hash must cover
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

/// Regression (RIL TASK-002 / ISS-002): `touch_blocks` reorders the LRU
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
