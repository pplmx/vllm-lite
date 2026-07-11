//! KV-cache memory subsystem: block allocator + eviction policy.
//!
//! Owns the paged KV blocks (`allocator`) and the policy that decides
//! which block to free under pressure (`eviction`). The scheduler asks
//! this module for `n_blocks` and gets either the blocks back or a
//! preemption signal.
//!
//! ## Multi-node cache coherence
//!
//! When a [`vllm_dist::DistributedKVCache`] is wired in via
//! [`MemoryManager::with_distributed_kv`] (or the corresponding setter),
//! `allocate` / `free` write through it. The cache key is the block id
//! and the cache value is the per-block hash computed by the
//! [`vllm_traits::BlockHasher`] (a chain hash that depends on the
//! previous block's hash and the tokens stored in this block — see
//! [`MemoryManager::record_block_tokens`]).
//!
//! For backward compatibility, `allocate` writes a deterministic
//! placeholder hash that depends only on the block id (no tokens).
//! The scheduler can overwrite this with a content-derived hash via
//! [`MemoryManager::record_block_tokens`] once it knows the tokens for
//! the block. Cross-node prefix lookup is OPS-05b3.
#![allow(clippy::module_name_repetitions)]
pub mod allocator;
pub mod eviction;

pub use allocator::{BlockAllocator, BlockAllocatorStats};
pub use eviction::{EvictionPolicy, EvictionPolicyStats};

use crate::scheduler::preemption::PreemptionManager;
use crate::types::{BlockId, SchedulerConfig, Sequence, Status};

#[cfg(feature = "multi-node")]
use std::sync::Arc;

#[cfg(feature = "multi-node")]
use vllm_dist::DistributedKVCache;

#[cfg(feature = "multi-node")]
use vllm_traits::{BlockHasher, IdentityHasher, TokenId};

#[derive(Debug)]
/// Top-level KV-cache memory coordinator. Composes a [`BlockAllocator`] and an [`EvictionPolicy`], and exposes the high-level `allocate` / `free` API the scheduler calls.
pub struct MemoryManager {
    allocator: BlockAllocator,
    eviction_policy: EvictionPolicy,
    preemption_manager: PreemptionManager,
    /// Optional distributed KV-cache; when `Some`, every `allocate`
    /// registers new blocks and every `free` invalidates them so peer
    /// nodes can observe activity.
    #[cfg(feature = "multi-node")]
    distributed_kv: Option<Arc<DistributedKVCache>>,
    /// Content hasher used to compute the value side of
    /// `DistributedKVCache::put`. Default is
    /// [`vllm_traits::IdentityHasher`] (collapses every block to its
    /// parent's hash — fine for block-existence tracking, useless for
    /// content addressing).
    #[cfg(feature = "multi-node")]
    hasher: Arc<dyn BlockHasher>,
    /// Chain cursor: hash of the most-recently allocated block.
    /// Advances on every [`Self::allocate`] and is fed back as the
    /// `parent_hash` argument to [`BlockHasher::hash_allocated_block`].
    /// `0` until the first allocate.
    ///
    /// Single-cursor design (one chain per `MemoryManager`) matches
    /// OPS-05b2 §1-§2 exactly. Per-sequence cursors for content-derived
    /// hashing live at the scheduler level via
    /// [`Self::record_block_tokens`] (which receives `parent_hash` from
    /// the caller) — that's the path step 3 ("prefix-cache lookup
    /// through distributed cache") builds on.
    #[cfg(feature = "multi-node")]
    chain_cursor: u64,
}

impl Default for MemoryManager {
    fn default() -> Self {
        Self::new(SchedulerConfig::default(), 1000)
    }
}

impl MemoryManager {
    /// Creates a new `MemoryManager` with the given scheduler configuration and number of blocks.
    #[must_use]
    pub fn new(config: SchedulerConfig, num_blocks: usize) -> Self {
        Self {
            allocator: BlockAllocator::new(num_blocks),
            eviction_policy: EvictionPolicy::new(),
            preemption_manager: PreemptionManager::new(config),
            #[cfg(feature = "multi-node")]
            distributed_kv: None,
            #[cfg(feature = "multi-node")]
            hasher: Arc::new(IdentityHasher),
            #[cfg(feature = "multi-node")]
            chain_cursor: 0,
        }
    }

    /// Wire a [`vllm_dist::DistributedKVCache`] into the memory coordinator.
    ///
    /// Called once during engine construction (or later via
    /// [`Self::set_distributed_kv`]) so that every subsequent
    /// `allocate` / `free` round-trips through the cache.
    #[cfg(feature = "multi-node")]
    #[must_use]
    pub fn with_distributed_kv(mut self, cache: Arc<DistributedKVCache>) -> Self {
        self.distributed_kv = Some(cache);
        self
    }

    /// Install a distributed KV-cache after construction. Equivalent to
    /// [`Self::with_distributed_kv`] but usable when the manager is
    /// already owned by another struct.
    #[cfg(feature = "multi-node")]
    pub fn set_distributed_kv(&mut self, cache: Arc<DistributedKVCache>) {
        self.distributed_kv = Some(cache);
    }

    /// Install a [`vllm_traits::BlockHasher`] for content-derived
    /// block hashing.
    ///
    /// Default (no call to this method) is
    /// [`vllm_traits::IdentityHasher`] — every block collapses to its
    /// parent hash, which preserves block-existence semantics but
    /// produces no useful content address. Production deployments
    /// should pass [`vllm_traits::XorShiftHasher`] or a custom
    /// hasher (blake3, xxhash, …).
    #[cfg(feature = "multi-node")]
    #[must_use]
    pub fn with_block_hasher(mut self, hasher: Arc<dyn BlockHasher>) -> Self {
        self.hasher = hasher;
        self
    }

    /// Replace the block hasher after construction. Equivalent to
    /// [`Self::with_block_hasher`] but usable when the manager is
    /// already owned by another struct.
    #[cfg(feature = "multi-node")]
    pub fn set_block_hasher(&mut self, hasher: Arc<dyn BlockHasher>) {
        self.hasher = hasher;
    }

    /// Allocates the specified number of blocks.
    /// Returns None if not enough blocks are available.
    pub fn allocate(&mut self, num_blocks: usize) -> Option<Vec<BlockId>> {
        let blocks = self.allocator.allocate(num_blocks);
        #[cfg(feature = "multi-node")]
        if let (Some(cache), Some(blocks)) = (self.distributed_kv.as_ref(), blocks.as_ref()) {
            // Per-block chain hash: parent_hash is the cursor, which
            // advances as we publish each block. The hash is content-
            // free at this point (tokens aren't known to the
            // MemoryManager — see `record_block_tokens` for the
            // content-aware path) but is still deterministic given
            // the block id and the cursor.
            for &block_id in blocks {
                let hash = self
                    .hasher
                    .hash_allocated_block(block_id, self.chain_cursor, &[]);
                cache.put(u64::try_from(block_id).unwrap_or(u64::MAX), hash);
                self.chain_cursor = hash;
            }
        }
        blocks
    }

    /// Re-publish a block's content-derived hash to the cache.
    ///
    /// Called by the scheduler after prefill, when the tokens for the
    /// block are known. The chain property requires the caller to
    /// supply the previous block's hash as `parent_hash` — the
    /// scheduler maintains a per-sequence cursor.
    ///
    /// Returns the new hash so the caller can advance its cursor for
    /// the next block in the sequence.
    ///
    /// # Multi-node feature
    ///
    /// Only meaningful when a [`DistributedKVCache`] is wired in (no-op
    /// otherwise). Each call bumps the cache's `updates` counter
    /// (re-publishing the value overwrites the placeholder from
    /// `allocate`).
    #[cfg(feature = "multi-node")]
    pub fn record_block_tokens(
        &mut self,
        block_id: BlockId,
        parent_hash: u64,
        tokens: &[TokenId],
    ) -> u64 {
        let hash = self.hasher.hash_block(parent_hash, tokens);
        // Also update the per-block cursor so subsequent
        // `allocate(block_id + 1)` keeps the chain property (best-
        // effort — only holds when allocate-id order matches the
        // physical sequence order; for true per-sequence chains use
        // the scheduler-side cursor and call this method only).
        self.chain_cursor = hash;
        if let Some(cache) = self.distributed_kv.as_ref() {
            cache.put(u64::try_from(block_id).unwrap_or(u64::MAX), hash);
        }
        hash
    }

    /// Frees the given blocks without updating eviction policy.
    /// Use `release_blocks` if you want to also update reference counts.
    pub fn free(&mut self, blocks: &[BlockId]) {
        #[cfg(feature = "multi-node")]
        if let Some(cache) = self.distributed_kv.as_ref() {
            for &block_id in blocks {
                cache.invalidate(u64::try_from(block_id).unwrap_or(u64::MAX));
            }
        }
        self.allocator.free(blocks);
    }

    /// Releases blocks, updating eviction policy reference counts and freeing the blocks.
    pub fn release_blocks(&mut self, blocks: &[BlockId]) {
        self.eviction_policy.release_blocks(blocks);
        self.allocator.free(blocks);
    }

    /// Selects victim blocks from running sequences to free up the requested number of blocks.
    #[must_use]
    pub fn select_victims(
        &self,
        running_sequences: &[Sequence],
        num_blocks: usize,
    ) -> Vec<BlockId> {
        let mut result = Vec::new();
        for seq in running_sequences
            .iter()
            .filter(|s| s.status == Status::Decoding)
        {
            for &block in seq.kv_blocks.iter() {
                if result.len() >= num_blocks {
                    return result;
                }
                result.push(block);
            }
        }
        result
    }

    /// Records blocks for eviction policy tracking.
    pub fn record_blocks(&mut self, blocks: &[BlockId]) {
        self.eviction_policy.record_blocks(blocks);
    }

    /// Updates the access time for blocks in the eviction policy.
    pub fn touch_blocks(&mut self, blocks: &[BlockId]) {
        self.eviction_policy.touch_blocks(blocks);
    }

    /// Returns the number of currently available (free) blocks.
    #[must_use]
    pub const fn available_blocks(&self) -> usize {
        self.allocator.available()
    }

    /// Returns the total number of blocks managed by this `MemoryManager`.
    #[must_use]
    pub const fn total_blocks(&self) -> usize {
        self.allocator.total()
    }

    /// Determines whether preemption should be triggered based on current system state.
    #[must_use]
    pub fn should_preempt(
        &self,
        running_len: usize,
        waiting_len: usize,
        blocks_needed: usize,
        blocks_available: usize,
    ) -> bool {
        self.preemption_manager.should_preempt(
            running_len,
            waiting_len,
            blocks_needed,
            blocks_available,
        )
    }

    /// Executes preemption by selecting sequences to evict and freeing their blocks.
    /// Returns the list of preempted sequences.
    pub fn execute_preemption(
        &mut self,
        running: &mut Vec<Sequence>,
        blocks_needed: usize,
    ) -> Vec<Sequence> {
        let mut preemptable_indices: Vec<usize> = running
            .iter()
            .enumerate()
            .filter(|(_, s)| s.status == Status::Decoding)
            .map(|(i, _)| i)
            .collect();

        preemptable_indices.sort_by(|&a, &b| {
            running[b]
                .consecutive_decode_rounds
                .cmp(&running[a].consecutive_decode_rounds)
        });

        let mut blocks_freed = 0;
        let mut preempted = Vec::new();
        let mut preempted_indices: Vec<usize> = Vec::new();

        for &idx in &preemptable_indices {
            if blocks_freed >= blocks_needed {
                break;
            }

            let seq = &running[idx];
            let block_count = seq.kv_blocks.len();
            self.free(seq.kv_blocks.as_ref());
            preempted_indices.push(idx);
            blocks_freed += block_count;
        }

        preempted_indices.sort_by(|a, b| b.cmp(a));
        for idx in preempted_indices {
            let seq = running.remove(idx);
            preempted.push(seq);
        }

        preempted
    }

    /// Rollback KV cache blocks for rejected draft tokens (Plan 17.1-D).
    ///
    /// Computes how many blocks to free based on `num_tokens` and block size,
    /// then returns freed blocks to the free pool.
    ///
    /// # Safety Invariant
    ///
    /// Physical KV cache tensor store entries for freed blocks are NOT cleared.
    /// Any attention implementation that reads KV entries without position-based
    /// bounds (e.g., tile-based flash attention reading full tile rows) MUST
    /// respect `num_computed_tokens` to avoid consuming stale data from
    /// rolled-back positions.
    pub fn rollback(&mut self, seq: &mut Sequence, num_tokens: usize) {
        if num_tokens == 0 {
            return;
        }
        let block_size = vllm_traits::BLOCK_SIZE;
        let tokens_after_rollback = seq.num_computed_tokens.saturating_sub(num_tokens);
        let blocks_after = tokens_after_rollback.div_ceil(block_size);
        let blocks_before = seq.num_computed_tokens.div_ceil(block_size);

        if blocks_after < blocks_before {
            let blocks_to_free: Vec<BlockId> = seq.kv_blocks[blocks_after..blocks_before].to_vec();
            self.free(&blocks_to_free);
            let mut new_blocks = (*seq.kv_blocks).clone();
            new_blocks.truncate(blocks_after);
            seq.kv_blocks = std::sync::Arc::new(new_blocks);
        }

        seq.num_computed_tokens = tokens_after_rollback;
    }

    /// Returns preemption statistics: (`preempted_count`, `rejected_count`).
    #[must_use]
    pub const fn preemption_stats(&self) -> (u64, u64) {
        (
            self.preemption_manager.preempted_count(),
            self.preemption_manager.rejected_count(),
        )
    }
}

// Unit tests are extracted to `tests.rs` (sibling) to keep this
// memory-coordinator module under the 800-line soft cap. They cover
// the allocate/free round-trip, select_victims boundary behavior,
// and OOM detection.
#[cfg(test)]
mod tests;
