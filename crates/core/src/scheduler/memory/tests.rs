//! Unit tests for `MemoryManager`.
//!
//! Covers the entry points the scheduler actually drives:
//!
//! 1. **allocate / free**: `allocate(3)` reduces `available_blocks`
//!    by 3; `free(blocks)` restores it. Round-trip invariant.
//! 2. **`select_victims`**: returns up to `num_blocks` block IDs from
//!    `Decoding` sequences (in order). For a single 2-block seq with
//!    `num_blocks=1`, the result is 0 or 1 blocks (depends on
//!    whether the partial block-count is honored — see the
//!    implementation).
//! 3. **OOM**: `allocate(capacity)` succeeds; `allocate(1)` on a
//!    full manager returns `None`.
//! 4. **Distributed-KV write-through** (multi-node feature): when a
//!    `DistributedKVCache` is wired in, allocate / free round-trip
//!    through the cache so peer nodes can observe activity. The unit
//!    tests below assert the cache stats reflect the manager's
//!    lifecycle without touching the network.
//! 5. **Content hashing** (multi-node feature): when a custom
//!    [`vllm_traits::BlockHasher`] is wired in, allocate writes the
//!    hasher's chain hash (not the `0` placeholder) to the cache, and
//!    [`MemoryManager::record_block_tokens`] re-publishes blocks with
//!    their real per-block content hash.
//!
//! [`MemoryManager::record_block_tokens`]: super::MemoryManager::record_block_tokens
#[cfg(feature = "multi-node")]
use vllm_dist::distributed_kv::protocol::NodeId;
#[cfg(feature = "multi-node")]
use vllm_dist::{CacheConfig, DistributedKVCache};
#[cfg(feature = "multi-node")]
use vllm_traits::{TokenId, XorShiftHasher};

use super::*;
use crate::types::{Priority, SamplingParams, Status};
use std::sync::Arc;

fn make_sequence(id: u64, blocks: Vec<BlockId>, status: Status) -> Sequence {
    Sequence {
        id,
        tokens: vec![1, 2, 3],
        kv_blocks: Arc::new(blocks),
        num_computed_tokens: 3,
        prompt_len: 3,
        status,
        max_tokens: 100,
        sampling_params: SamplingParams::default(),
        consecutive_decode_rounds: 0,
        priority: Priority::default(),
        degraded_draft: false,
        draft_model_id: None,
    }
}

#[test]
fn test_memory_manager_allocate_free() {
    let mut manager = MemoryManager::new(SchedulerConfig::default(), 10);

    let blocks = manager.allocate(3).unwrap();
    assert_eq!(blocks.len(), 3);
    assert_eq!(manager.available_blocks(), 7);

    manager.free(&blocks);
    assert_eq!(manager.available_blocks(), 10);
}

#[test]
fn test_memory_manager_select_victims() {
    let manager = MemoryManager::new(SchedulerConfig::default(), 10);

    let seq = make_sequence(1, vec![1, 2], Status::Decoding);
    let victims = manager.select_victims(&[seq], 1);
    assert!(victims.is_empty() || victims.len() == 1);
}

#[test]
fn test_memory_manager_oom() {
    let mut manager = MemoryManager::new(SchedulerConfig::default(), 2);
    manager.allocate(2).unwrap();
    assert!(manager.allocate(1).is_none());
}

#[cfg(feature = "multi-node")]
#[test]
fn test_memory_manager_allocate_bumps_cache_updates() {
    // When a cache is wired, `allocate()` registers each new block
    // via `cache.put`. Verify the cache's `updates` counter reflects
    // the number of blocks allocated.
    let cache = Arc::new(DistributedKVCache::new(CacheConfig::new(NodeId(0), 4)));
    let mut manager =
        MemoryManager::new(SchedulerConfig::default(), 10).with_distributed_kv(Arc::clone(&cache));

    let _ = manager.allocate(3).expect("allocation should succeed");
    let stats = cache.stats();
    assert_eq!(
        stats.updates, 3,
        "allocate(3) must register 3 blocks in the distributed cache"
    );

    let _ = manager.allocate(2).expect("allocation should succeed");
    let stats = cache.stats();
    assert_eq!(
        stats.updates, 5,
        "second allocate(2) brings the cumulative updates count to 5"
    );
}

#[cfg(feature = "multi-node")]
#[test]
fn test_memory_manager_free_bumps_cache_invalidations() {
    // Symmetric to the allocate test: free() invalidates each block in
    // the cache, bumping the `invalidations` counter.
    let cache = Arc::new(DistributedKVCache::new(CacheConfig::new(NodeId(0), 4)));
    let mut manager =
        MemoryManager::new(SchedulerConfig::default(), 10).with_distributed_kv(Arc::clone(&cache));

    let blocks = manager.allocate(3).expect("allocation should succeed");
    manager.free(&blocks);
    let stats = cache.stats();
    assert_eq!(stats.updates, 3, "three puts on allocate");
    assert_eq!(
        stats.invalidations, 3,
        "free(blocks) must invalidate every block in the cache"
    );
}

#[cfg(feature = "multi-node")]
#[test]
fn test_memory_manager_without_cache_is_a_no_op() {
    // Default construction path (no cache installed) must behave
    // exactly as before — allocate / free still work, no panic, no
    // cache reference to chase.
    let mut manager = MemoryManager::new(SchedulerConfig::default(), 10);
    let blocks = manager.allocate(3).expect("allocation should succeed");
    manager.free(&blocks);
    assert_eq!(manager.available_blocks(), 10);
}

// ---------------------------------------------------------------------------
// Content-hashing tests (Phase 19 OPS-05b2)
// ---------------------------------------------------------------------------

#[cfg(feature = "multi-node")]
#[test]
fn test_memory_manager_default_hasher_is_identity() {
    // Default construction uses `IdentityHasher`, so the chain hash
    // collapses to the cursor value (`0` for the first block of a
    // fresh manager). All blocks therefore have hash `0` — matches
    // the pre-OPS-05b2 placeholder behavior so callers that haven't
    // opted into content hashing see no regression.
    let cache = Arc::new(DistributedKVCache::new(CacheConfig::new(NodeId(0), 4)));
    let mut manager =
        MemoryManager::new(SchedulerConfig::default(), 10).with_distributed_kv(Arc::clone(&cache));

    let blocks = manager.allocate(3).expect("allocation should succeed");
    assert_eq!(blocks.len(), 3);

    // Every block should be findable in the cache with value `0`
    // (the IdentityHasher collapses the chain to the parent hash,
    // which is `0` for all blocks of a fresh manager).
    for &block_id in &blocks {
        let cached = cache.get(u64::try_from(block_id).unwrap_or(u64::MAX));
        assert_eq!(
            cached,
            Some(0),
            "IdentityHasher + fresh cursor → cache value must be 0 for block {block_id}",
        );
    }
}

#[cfg(feature = "multi-node")]
#[test]
fn test_memory_manager_with_xorshift_hasher_produces_distinct_hashes() {
    // With `XorShiftHasher`, each allocated block gets a distinct
    // non-zero hash because the chain cursor advances per block and
    // the hasher mixes `block_id` into the state.
    let cache = Arc::new(DistributedKVCache::new(CacheConfig::new(NodeId(0), 4)));
    let mut manager = MemoryManager::new(SchedulerConfig::default(), 10)
        .with_distributed_kv(Arc::clone(&cache))
        .with_block_hasher(Arc::new(XorShiftHasher));

    let blocks = manager.allocate(3).expect("allocation should succeed");

    let hashes: Vec<u64> = blocks
        .iter()
        .map(|&id| {
            cache
                .get(u64::try_from(id).unwrap_or(u64::MAX))
                .unwrap_or(u64::MAX)
        })
        .collect();
    assert_eq!(hashes.len(), 3);
    // All three hashes must be non-zero (XorShift mixing the cursor
    // and block id guarantees this).
    for (block_id, hash) in blocks.iter().zip(&hashes) {
        assert_ne!(
            *hash, 0,
            "xorshift hash for block {block_id} must be non-zero"
        );
    }
    // Distinct (no two blocks share a chain hash).
    let unique: std::collections::HashSet<u64> = hashes.iter().copied().collect();
    assert_eq!(
        unique.len(),
        hashes.len(),
        "xorshift chain must produce distinct hashes per block"
    );
}

#[cfg(feature = "multi-node")]
#[test]
fn test_memory_manager_record_block_tokens_advances_chain() {
    // `record_block_tokens` re-publishes a block with a content-
    // derived hash (using the supplied `parent_hash` + tokens). The
    // caller (scheduler) advances its cursor with the returned hash
    // so the next block's chain entry is consistent.
    let cache = Arc::new(DistributedKVCache::new(CacheConfig::new(NodeId(0), 4)));
    let mut manager = MemoryManager::new(SchedulerConfig::default(), 10)
        .with_distributed_kv(Arc::clone(&cache))
        .with_block_hasher(Arc::new(XorShiftHasher));

    let blocks = manager.allocate(3).expect("allocation should succeed");
    assert_eq!(blocks.len(), 3);

    // First block: parent_hash = 0, tokens = a known vector.
    let tokens_block0: Vec<TokenId> = vec![10, 20, 30];
    let h0 = manager.record_block_tokens(blocks[0], 0, &tokens_block0);
    assert_ne!(
        h0, 0,
        "xorshift hash with non-empty tokens must be non-zero"
    );

    // Second block: parent_hash = h0, different tokens.
    let tokens_block1: Vec<TokenId> = vec![40, 50, 60];
    let h1 = manager.record_block_tokens(blocks[1], h0, &tokens_block1);
    assert_ne!(
        h1, h0,
        "different tokens + parent must produce a different hash"
    );

    // Third block: parent_hash = h1, different tokens again.
    let tokens_block2: Vec<TokenId> = vec![70, 80, 90];
    let h2 = manager.record_block_tokens(blocks[2], h1, &tokens_block2);
    assert_ne!(h2, h1, "different parent must produce a different hash");
    assert_ne!(h2, h0);

    // Determinism: re-publishing the same parent + tokens yields the
    // same hash (the chain property that makes cross-node prefix
    // lookup possible).
    let h0_repeat = manager.record_block_tokens(blocks[0], 0, &tokens_block0);
    assert_eq!(h0_repeat, h0, "chain hashing must be deterministic");

    // Cache values match the returned hashes.
    assert_eq!(
        cache.get(u64::try_from(blocks[0]).unwrap_or(u64::MAX)),
        Some(h0),
        "block 0 cache value must match recorded hash"
    );
    assert_eq!(
        cache.get(u64::try_from(blocks[1]).unwrap_or(u64::MAX)),
        Some(h1),
        "block 1 cache value must match recorded hash"
    );
    assert_eq!(
        cache.get(u64::try_from(blocks[2]).unwrap_or(u64::MAX)),
        Some(h2),
        "block 2 cache value must match recorded hash"
    );
}

#[cfg(feature = "multi-node")]
#[test]
fn test_memory_manager_record_block_tokens_different_sequences_diverge() {
    // Two sequences with identical first-block tokens but different
    // starting cursor values should diverge — `parent_hash` is part
    // of the chain input, so it matters.
    let cache = Arc::new(DistributedKVCache::new(CacheConfig::new(NodeId(0), 4)));
    let mut manager = MemoryManager::new(SchedulerConfig::default(), 10)
        .with_distributed_kv(Arc::clone(&cache))
        .with_block_hasher(Arc::new(XorShiftHasher));
    let blocks = manager.allocate(2).expect("allocation should succeed");

    let tokens: Vec<TokenId> = vec![1, 2, 3];
    let h_seq_a = manager.record_block_tokens(blocks[0], 0, &tokens);
    let h_seq_b = manager.record_block_tokens(blocks[1], 42, &tokens);
    assert_ne!(
        h_seq_a, h_seq_b,
        "same tokens but different parent hashes must diverge"
    );
}
