//! KV-cache memory subsystem: block allocator + eviction policy.
//!
//! Owns the paged KV blocks (`allocator`) and the policy that decides
//! which block to free under pressure (`eviction`). The scheduler asks
//! this module for `n_blocks` and gets either the blocks back or a
//! preemption signal.
//!
//! ## Multi-node cache coherence
//!
//! All items in this section require the `multi-node` Cargo feature.
//! When a `vllm_dist::DistributedKVCache` is wired in via
//! `MemoryManager::with_distributed_kv` (or the corresponding setter
//! `set_distributed_kv`), `allocate` / `free` write through it. The
//! cache key is the block id and the cache value is the per-block hash
//! computed by the `vllm_traits::BlockHasher` (a chain hash that depends
//! on the previous block's hash and the tokens stored in this block —
//! see `MemoryManager::record_block_tokens`).
//!
//! For backward compatibility, `allocate` writes a deterministic
//! placeholder hash that depends only on the block id (no tokens).
//! The scheduler can overwrite this with a content-derived hash via
//! `MemoryManager::record_block_tokens` once it knows the tokens for
//! the block.
//!
//! ### Cross-node prefix lookup
//!
//! `MemoryManager::lookup_distributed_prefix` computes the chain
//! hash for each block of a prompt and asks the cache. The cache's
//! `lookup_prefix` returns the longest matched prefix length; this is
//! what the scheduler surfaces to operators / metrics consumers
//! before deciding whether to recompute the prefix locally. Phase 19
//! OPS-05b3.
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

#[cfg(feature = "multi-node")]
use vllm_traits::BLOCK_SIZE;

/// Outcome of a distributed prefix-cache lookup.
///
/// Returned by [`MemoryManager::lookup_distributed_prefix`] when the
/// cache has at least one block's chain hash present (i.e., *some*
/// node — local or peer, post OPS-05c — has KV for the prefix).
///
/// # Why no block IDs?
///
/// The distributed cache stores *content hashes*, not local block
/// ids. A peer node's KV blocks live in the peer's allocator; we
/// can't reuse them as-is without a block transfer protocol (OPS-05c
/// follow-up). Until then, the only useful information is "how many
/// blocks of this prefix are cached *somewhere*", which is what
/// [`Self::matched_blocks`] / [`Self::matched_tokens`] report.
#[cfg(feature = "multi-node")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DistributedPrefixMatch {
    /// Number of consecutive blocks (from the start of the prompt)
    /// whose chain hash was present in the distributed cache.
    pub matched_blocks: usize,
    /// Number of tokens covered by the matched prefix (always
    /// `matched_blocks * BLOCK_SIZE`, capped at `prompt.len()`).
    pub matched_tokens: usize,
    /// Name of the [`vllm_traits::BlockHasher`] used to compute the
    /// chain — recorded so observers can verify which hash space
    /// the match is in.
    pub hasher_name: &'static str,
}

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
    /// Optional `BlockDataSource` that produces the actual K/V tensor
    /// bytes when a peer requests them via `TransferKVBlock`. Set via
    /// `EngineBuilder::with_paged_kv_cache(...)` (which constructs a
    /// `PagedKvCacheWrapper` from the loader's `Arc<PagedKvCache>` and
    /// passes it through here). Phase 41 OPS-32a second-half.
    #[cfg(feature = "multi-node")]
    block_data_source: Option<Arc<dyn vllm_dist::BlockDataSource + Send + Sync>>,
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
            block_data_source: None,
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

    /// Wire a `BlockDataSource` into the memory coordinator (Phase 41
    /// OPS-32a second-half). Called once during engine construction
    /// (via `EngineBuilder::with_paged_kv_cache` → `Engine::set_paged_kv_cache`
    /// → `SchedulerEngine::set_block_data_source` → here) so that
    /// subsequent gRPC `TransferKVBlock` calls can resolve to real
    /// K/V bytes.
    #[cfg(feature = "multi-node")]
    #[must_use]
    pub fn with_block_data_source(
        mut self,
        source: Arc<dyn vllm_dist::BlockDataSource + Send + Sync>,
    ) -> Self {
        self.block_data_source = Some(source);
        self
    }

    /// Install a `BlockDataSource` after construction. Equivalent to
    /// [`Self::with_block_data_source`] but usable when the manager is
    /// already owned by another struct (the post-construction setter
    /// used by `SchedulerEngine::set_block_data_source`).
    #[cfg(feature = "multi-node")]
    pub fn set_block_data_source(
        &mut self,
        source: Arc<dyn vllm_dist::BlockDataSource + Send + Sync>,
    ) {
        self.block_data_source = Some(source);
    }

    /// Returns a clone of the wired `BlockDataSource` (or `None`).
    /// The gRPC server bootstrap (`crates/server/src/bootstrap/grpc.rs`)
    /// uses this to populate `start_grpc_server_with_listener`'s
    /// `block_data_source` parameter without going through the
    /// engine.
    #[cfg(feature = "multi-node")]
    #[must_use]
    pub fn block_data_source(&self) -> Option<Arc<dyn vllm_dist::BlockDataSource + Send + Sync>> {
        self.block_data_source.as_ref().map(Arc::clone)
    }

    /// Borrow the active [`vllm_traits::BlockHasher`].
    ///
    /// Useful for diagnostics (logging which hash space is in use)
    /// and for tests that need to compute chain hashes the same
    /// way the manager would.
    #[cfg(feature = "multi-node")]
    #[must_use]
    pub fn hasher(&self) -> &dyn vllm_traits::BlockHasher {
        self.hasher.as_ref()
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
    /// # Cache layout
    ///
    /// Puts `(content_hash, block_id)` — the *content hash is the
    /// key*, so [`Self::lookup_distributed_prefix`] can find this
    /// entry by walking the chain. Distinct from `allocate`, which
    /// puts `(block_id, placeholder_hash)` for block-id-keyed
    /// existence tracking; both entries coexist in the cache.
    ///
    /// # Multi-node feature
    ///
    /// Only meaningful when a [`DistributedKVCache`] is wired in
    /// (no-op otherwise). Each call bumps the cache's `updates`
    /// counter.
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
            // Key = content hash, value = block_id. Reverses the
            // `allocate` direction so `lookup_distributed_prefix`
            // (which queries by hash) can find this entry.
            cache.put(hash, u64::try_from(block_id).unwrap_or(u64::MAX));
        }
        hash
    }

    /// Look up a prompt prefix in the distributed cache.
    ///
    /// Computes the chain hash for each `BLOCK_SIZE`-token chunk of
    /// `prompt_tokens` and asks the cache. Returns the longest
    /// matched prefix length, or `None` if no chain hash is present
    /// (no blocks cached anywhere — local or peer, post OPS-05c).
    ///
    /// Returns `None` when no [`DistributedKVCache`] is wired in —
    /// a no-cache manager has nothing to look up.
    ///
    /// # Phase 19 OPS-05b3
    ///
    /// Establishes the API for cross-node prefix-cache hits.
    /// Actual KV block reuse from peers requires a transfer
    /// protocol; that lands with OPS-05c (gRPC plumbing).
    /// Until then the result is informational — observers can use
    /// it to report "X% of incoming prompts have a remote prefix
    /// hit", which is useful for tuning eviction policy or
    /// deciding when to enable cross-node sync.
    #[cfg(feature = "multi-node")]
    #[must_use]
    pub fn lookup_distributed_prefix(
        &self,
        prompt_tokens: &[TokenId],
    ) -> Option<DistributedPrefixMatch> {
        let cache = self.distributed_kv.as_ref()?;

        if prompt_tokens.is_empty() {
            return None;
        }

        // Compute the chain hash for each block in `prompt_tokens`.
        let mut chain_hashes = Vec::with_capacity(prompt_tokens.len().div_ceil(BLOCK_SIZE));
        let mut parent = 0u64;
        for chunk in prompt_tokens.chunks(BLOCK_SIZE) {
            let h = self.hasher.hash_block(parent, chunk);
            chain_hashes.push(h);
            parent = h;
        }

        let matched = cache.lookup_prefix(&chain_hashes);
        if matched == 0 {
            return None;
        }

        // matched_tokens is `min(matched * BLOCK_SIZE, prompt.len())`
        // so a partial last block still reports the right count.
        let matched_tokens = (matched * BLOCK_SIZE).min(prompt_tokens.len());
        Some(DistributedPrefixMatch {
            matched_blocks: matched,
            matched_tokens,
            hasher_name: self.hasher.name(),
        })
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

    /// Releases blocks, updating eviction policy reference counts and
    /// freeing ONLY the blocks whose refcount just reached zero.
    ///
    /// ARCH-01 (technical due diligence): the previous implementation
    /// always freed every released block, which corrupted shared
    /// prefix-cache entries. The fixed contract:
    ///
    /// - `record_blocks` is the *only* way to claim ownership of a
    ///   block. The caller MUST pair every `record_blocks` with one
    ///   `release_blocks`.
    /// - `release_blocks` decrements the per-block refcount and only
    ///   returns blocks to the allocator when no live owner remains.
    /// - When two sequences share a prefix, both `record_blocks`
    ///   against the same blocks. The first sequence to finish
    ///   triggers a release that drops the refcount to 1 (the second
    ///   sequence still owns it); the second sequence's release drops
    ///   it to 0 and only then is the block returned to the allocator.
    pub fn release_blocks(&mut self, blocks: &[BlockId]) {
        let freed = self.eviction_policy.release_blocks(blocks);
        if freed.is_empty() {
            return;
        }
        #[cfg(feature = "multi-node")]
        if let Some(cache) = self.distributed_kv.as_ref() {
            for &block_id in &freed {
                cache.invalidate(u64::try_from(block_id).unwrap_or(u64::MAX));
            }
        }
        self.allocator.free(&freed);
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
