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
use vllm_traits::{BLOCK_SIZE, TokenId, XorShiftHasher};

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

    // Cache entries keyed by content hash (not block_id) — see
    // `record_block_tokens` docs. Each block is findable via its
    // chain hash, which is what `lookup_distributed_prefix` walks.
    assert_eq!(
        cache.get(h0),
        Some(u64::try_from(blocks[0]).unwrap_or(u64::MAX)),
        "block 0 cache entry must be keyed by h0"
    );
    assert_eq!(
        cache.get(h1),
        Some(u64::try_from(blocks[1]).unwrap_or(u64::MAX)),
        "block 1 cache entry must be keyed by h1"
    );
    assert_eq!(
        cache.get(h2),
        Some(u64::try_from(blocks[2]).unwrap_or(u64::MAX)),
        "block 2 cache entry must be keyed by h2"
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

// ---------------------------------------------------------------------------
// Distributed prefix-lookup tests (Phase 19 OPS-05b3)
// ---------------------------------------------------------------------------

#[cfg(feature = "multi-node")]
#[test]
fn test_memory_manager_lookup_distributed_prefix_full_hit() {
    // Allocate 2 blocks, record known tokens, then look up a 3-block
    // prompt whose first 2 blocks match — the 3rd block's chain
    // hash is unknown so the match is partial (2/3).
    //
    // To make the test exercise a full match, we publish 3 hashes
    // for a 3-block prompt and look up the same 3-block prompt.
    let cache = Arc::new(DistributedKVCache::new(CacheConfig::new(NodeId(0), 4)));
    let manager = MemoryManager::new(SchedulerConfig::default(), 10)
        .with_distributed_kv(Arc::clone(&cache))
        .with_block_hasher(Arc::new(XorShiftHasher));

    // Build a 3-block prompt (BLOCK_SIZE = 16) and pre-publish each
    // block's chain hash into the cache directly. We compute the
    // hashes the same way the lookup would.
    let prompt: Vec<TokenId> = (0..(3 * BLOCK_SIZE)).map(|i| i as TokenId).collect();
    let chain_hashes = {
        let mut hashes = Vec::new();
        let mut parent = 0u64;
        for chunk in prompt.chunks(BLOCK_SIZE) {
            let h = manager.hasher().hash_block(parent, chunk);
            hashes.push(h);
            parent = h;
        }
        hashes
    };
    for &h in &chain_hashes {
        // Use the cache directly; key is arbitrary as long as the
        // lookup uses the same keys.
        cache.put(h, 0xCAFE);
    }

    let result = manager
        .lookup_distributed_prefix(&prompt)
        .expect("full prefix should hit");
    assert_eq!(result.matched_blocks, 3);
    assert_eq!(result.matched_tokens, 3 * BLOCK_SIZE);
    assert_eq!(result.hasher_name, "xorshift");
}

#[cfg(feature = "multi-node")]
#[test]
fn test_memory_manager_lookup_distributed_prefix_partial_match() {
    // Two-block prefix prompt; only the first block's hash is in
    // the cache. The lookup should return matched_blocks = 1.
    let cache = Arc::new(DistributedKVCache::new(CacheConfig::new(NodeId(0), 4)));
    let manager = MemoryManager::new(SchedulerConfig::default(), 10)
        .with_distributed_kv(Arc::clone(&cache))
        .with_block_hasher(Arc::new(XorShiftHasher));

    let prompt: Vec<TokenId> = (0..(2 * BLOCK_SIZE)).map(|i| i as TokenId).collect();
    let first_hash = manager.hasher().hash_block(0, &prompt[..BLOCK_SIZE]);
    cache.put(first_hash, 0xCAFE);
    // second_hash is NOT in the cache.

    let result = manager
        .lookup_distributed_prefix(&prompt)
        .expect("first block should hit");
    assert_eq!(result.matched_blocks, 1, "only first block should match");
    assert_eq!(result.matched_tokens, BLOCK_SIZE);
}

#[cfg(feature = "multi-node")]
#[test]
fn test_memory_manager_lookup_distributed_prefix_no_match() {
    // Empty cache: any prompt must return None.
    let cache = Arc::new(DistributedKVCache::new(CacheConfig::new(NodeId(0), 4)));
    let manager = MemoryManager::new(SchedulerConfig::default(), 10)
        .with_distributed_kv(Arc::clone(&cache))
        .with_block_hasher(Arc::new(XorShiftHasher));

    let prompt: Vec<TokenId> = (0..BLOCK_SIZE).map(|i| i as TokenId).collect();
    assert!(manager.lookup_distributed_prefix(&prompt).is_none());
}

#[cfg(feature = "multi-node")]
#[test]
fn test_memory_manager_lookup_distributed_prefix_empty_prompt() {
    let cache = Arc::new(DistributedKVCache::new(CacheConfig::new(NodeId(0), 4)));
    let manager =
        MemoryManager::new(SchedulerConfig::default(), 10).with_distributed_kv(Arc::clone(&cache));
    assert!(manager.lookup_distributed_prefix(&[]).is_none());
}

#[cfg(feature = "multi-node")]
#[test]
fn test_memory_manager_lookup_distributed_prefix_no_cache_returns_none() {
    // No cache wired in: lookup is a no-op that returns None
    // (mirrors the "default construction" semantics — there's no
    // cache to ask).
    let manager = MemoryManager::new(SchedulerConfig::default(), 10)
        .with_block_hasher(Arc::new(XorShiftHasher));
    let prompt: Vec<TokenId> = (0..BLOCK_SIZE).map(|i| i as TokenId).collect();
    assert!(manager.lookup_distributed_prefix(&prompt).is_none());
}

#[cfg(feature = "multi-node")]
#[test]
fn test_memory_manager_lookup_distributed_prefix_round_trip_with_record() {
    // Allocate blocks via the manager, record tokens (using a real
    // chain cursor), then look up the same prompt — every block's
    // chain hash is in the cache (it was just put there by
    // record_block_tokens), so the lookup returns a full match.
    //
    // This mirrors what the production scheduler does in
    // `SchedulerEngine::update` (allocate → record with parent =
    // previous block's hash → advance cursor).
    let cache = Arc::new(DistributedKVCache::new(CacheConfig::new(NodeId(0), 4)));
    let mut manager = MemoryManager::new(SchedulerConfig::default(), 10)
        .with_distributed_kv(Arc::clone(&cache))
        .with_block_hasher(Arc::new(XorShiftHasher));

    // Build a 2-block prompt.
    let prompt: Vec<TokenId> = (0..(2 * BLOCK_SIZE))
        .map(|i| (i + 100) as TokenId)
        .collect();

    // Allocate 2 blocks and record tokens for each, threading the
    // chain cursor (parent_hash = previous block's hash).
    let blocks = manager.allocate(2).expect("allocation should succeed");
    let h0 = manager.record_block_tokens(blocks[0], 0, &prompt[..BLOCK_SIZE]);
    let h1 = manager.record_block_tokens(blocks[1], h0, &prompt[BLOCK_SIZE..]);
    // Silence the unused-warning for h1 (it's the cursor the
    // scheduler would store; the lookup is the real assertion).
    let _ = h1;

    // Now look up the same prompt — every chain hash is in the
    // cache (recorded via the per-block puts above).
    let result = manager
        .lookup_distributed_prefix(&prompt)
        .expect("all blocks recorded → all should hit on lookup");
    assert_eq!(result.matched_blocks, 2);
    assert_eq!(result.matched_tokens, 2 * BLOCK_SIZE);
}

// ──────────────────────────────────────────────────────────────────────
// block_data_source tests (P41 T3). The field + setters/getter live
// in `mod.rs::impl MemoryManager` above.
// ──────────────────────────────────────────────────────────────────────

#[cfg(feature = "multi-node")]
#[test]
fn memory_manager_with_block_data_source_stores_it() {
    use vllm_dist::distributed_kv::block_data_source::MockBlockDataSource;
    let source: Arc<dyn vllm_dist::BlockDataSource + Send + Sync> =
        Arc::new(MockBlockDataSource::new());
    let mgr = MemoryManager::new(SchedulerConfig::default(), 4)
        .with_block_data_source(Arc::clone(&source));
    let retrieved = mgr.block_data_source().expect("source stored");
    assert!(Arc::ptr_eq(&retrieved, &source));
}

#[cfg(feature = "multi-node")]
#[test]
fn memory_manager_default_has_no_block_data_source() {
    let mgr = MemoryManager::new(SchedulerConfig::default(), 4);
    assert!(mgr.block_data_source().is_none());
}

#[cfg(feature = "multi-node")]
#[test]
fn memory_manager_set_block_data_source_replaces_existing() {
    use vllm_dist::distributed_kv::block_data_source::MockBlockDataSource;
    let source1: Arc<dyn vllm_dist::BlockDataSource + Send + Sync> =
        Arc::new(MockBlockDataSource::new());
    let source2: Arc<dyn vllm_dist::BlockDataSource + Send + Sync> =
        Arc::new(MockBlockDataSource::new());
    let mut mgr = MemoryManager::new(SchedulerConfig::default(), 4);
    mgr.set_block_data_source(source1);
    mgr.set_block_data_source(Arc::clone(&source2));
    let retrieved = mgr.block_data_source().unwrap();
    assert!(
        Arc::ptr_eq(&retrieved, &source2),
        "second setter should replace the first"
    );
}

#[cfg(feature = "multi-node")]
#[test]
fn memory_manager_block_data_source_clone_is_independent() {
    use vllm_dist::distributed_kv::block_data_source::MockBlockDataSource;
    let source: Arc<dyn vllm_dist::BlockDataSource + Send + Sync> =
        Arc::new(MockBlockDataSource::new());
    let mgr = MemoryManager::new(SchedulerConfig::default(), 4)
        .with_block_data_source(Arc::clone(&source));
    let c1 = mgr.block_data_source().unwrap();
    let c2 = mgr.block_data_source().unwrap();
    assert!(
        Arc::ptr_eq(&c1, &c2),
        "two getter calls should return Arc-clones of the same source"
    );
    assert!(
        Arc::ptr_eq(&c1, &source),
        "both clones should point to the originally-stored source"
    );
}
