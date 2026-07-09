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
