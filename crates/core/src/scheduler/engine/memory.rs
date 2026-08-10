//! Memory and preemption-related methods for `SchedulerEngine`.
//!
//! This sub-module owns everything that touches block allocation,
//! pressure reporting, and KV cache rollback:
//! - `execute_preemption`: re-queue running sequences when block
//!   demand exceeds supply.
//! - `get_memory_pressure`: ratio of allocated blocks to total.
//! - `memory_rollback`: undo speculative-decoding block growth.
//! - `cancel_request`: drop a request from queue or running set,
//!   releasing any blocks it held.
//! - `get_kv_cache_usage`: snapshot of (used, total) block counts.
//! - `prefix_cache`: expose the underlying `RadixTree` so callers
//!   can inspect or prime prefix state.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use vllm_traits::SeqId;

use crate::scheduler::RadixTree;
use crate::scheduler::memory::MemoryManager;
use crate::scheduler::policy::SchedulingContext;
use crate::types::Status;

use super::state::SchedulerEngine;

impl SchedulerEngine {
    /// Execute preemption to free up memory blocks.
    ///
    /// First selects victim blocks via the eviction policy
    /// (`EvictionPolicy::select_victims`), which ranks individual blocks
    /// with ref-count ≤ 1 in priority order. A sequence's block table is
    /// positional — block `i` holds tokens `i * BLOCK_SIZE..(i + 1) *
    /// BLOCK_SIZE` — so losing an interior block would shift every later
    /// position onto the wrong physical block and silently corrupt
    /// attention reads. Every sequence that owns *any* victim block is
    /// therefore preempted wholesale: all of its blocks are released,
    /// its state is reset (`Waiting`, zero computed tokens, tokens
    /// preserved), and it is re-queued for full recompute.
    ///
    /// If the eviction policy can't free enough blocks (e.g. all blocks
    /// are prefix-cache shared with ref-count > 1), falls back to
    /// **sequence-level preemption** — the legacy behaviour that releases
    /// every block from the most-decodable sequences.
    pub(super) fn execute_preemption(&mut self, blocks_needed: usize) {
        // Phase 0: release blocks pinned by the prefix cache. Finished
        // sequences insert their KV blocks into the cache and the cache
        // holds a refcount on each — without eviction those blocks never
        // return to the allocator, so a long-running server exhausts the
        // block pool even with zero running sequences (verified: with a
        // 16-block pool, 16 distinct finished prompts pin 16/16 blocks).
        // Under memory pressure, clear the cache (dropping its refs) so
        // the allocator can reuse the blocks. Sequences still running on
        // a shared prefix keep their own refcount and are unaffected.
        if self.memory.available_blocks() < blocks_needed {
            for blocks in self.prefix_cache.drain_blocks() {
                self.memory.release_blocks(&blocks);
            }
        }
        // Phase 1: priority-weighted victim selection, applied at
        // sequence granularity (see the function docs: a positional
        // block table cannot tolerate interior holes, so any sequence
        // owning a victim block is preempted wholesale).
        let victims = self.memory.select_victims(&self.running, blocks_needed);
        let victim_set: HashSet<_> = victims.iter().copied().collect();

        if !victim_set.is_empty() {
            let preempted: Vec<SeqId> = self
                .running
                .iter()
                .filter(|s| {
                    (s.status == Status::Decoding || s.status == Status::Prefilling)
                        && s.kv_blocks.iter().any(|b| victim_set.contains(b))
                })
                .map(|s| s.id)
                .collect();

            let mut blocks_freed = 0usize;
            for seq_id in preempted {
                if let Some(pos) = self.running.iter().position(|s| s.id == seq_id) {
                    let mut seq = self.running.remove(pos);
                    blocks_freed += seq.kv_blocks.len();
                    self.memory.release_blocks(seq.kv_blocks.as_ref());
                    seq.kv_blocks = Arc::new(vec![]);
                    seq.status = Status::Waiting;
                    seq.num_computed_tokens = 0;
                    let ctx = SchedulingContext {
                        current_time: Instant::now(),
                        queue_length: self.request_queue.len(),
                        running_count: self.running.len(),
                        memory_pressure: self.get_memory_pressure(),
                    };
                    self.request_queue.enqueue(seq, self.policy.as_ref(), &ctx);
                }
            }

            if blocks_freed >= blocks_needed {
                return;
            }
        }

        // Phase 2: fallback — sequence-level preemption (release all blocks
        // from the most-decodable sequences) when block-level eviction
        // couldn't free enough blocks (e.g. all blocks are shared).
        let mut preemptable: Vec<_> = self
            .running
            .iter()
            .filter(|s| s.status == Status::Decoding)
            .cloned()
            .collect();

        preemptable.sort_by(|a, b| {
            b.consecutive_decode_rounds
                .cmp(&a.consecutive_decode_rounds)
        });

        let mut blocks_freed = victim_set.len();
        for mut seq in preemptable {
            if blocks_freed >= blocks_needed {
                break;
            }

            let block_count = seq.kv_blocks.len();
            self.memory.release_blocks(seq.kv_blocks.as_ref());
            self.running.retain(|s| s.id != seq.id);

            // Re-queue the preempted sequence
            seq.kv_blocks = Arc::new(vec![]);
            seq.status = Status::Waiting;
            seq.num_computed_tokens = 0;

            let ctx = SchedulingContext {
                current_time: Instant::now(),
                queue_length: self.request_queue.len(),
                running_count: self.running.len(),
                memory_pressure: self.get_memory_pressure(),
            };

            self.request_queue.enqueue(seq, self.policy.as_ref(), &ctx);
            blocks_freed += block_count;
        }
    }

    /// Calculate current memory pressure (0.0 to 1.0)
    // invariant: block counts are bounded by available memory; f32 precision
    // loss is acceptable for the pressure ratio (always 0..=1).
    #[allow(clippy::cast_precision_loss)]
    pub(super) fn get_memory_pressure(&self) -> f32 {
        let total = self.memory.total_blocks();
        if total == 0 {
            // Degenerate config (zero blocks): report maximum pressure so
            // the scheduler triggers preemption rather than dividing by zero.
            return 1.0;
        }
        let available = self.memory.available_blocks() as f32;
        1.0 - (available / total as f32)
    }

    /// Rollback KV cache for rejected draft tokens (Plan 17.1-D).
    pub fn memory_rollback(&mut self, seq_id: SeqId, num_tokens: usize) {
        if let Some(seq) = self.running.iter_mut().find(|s| s.id == seq_id) {
            self.memory.rollback(seq, num_tokens);
        }
    }

    /// Grow a running sequence's KV block table to hold at least
    /// `min_blocks` blocks, allocating + refcounting each new block exactly
    /// like the post-step growth in `update_running_sequence`.
    ///
    /// Used by the speculative step to cover the full verification span
    /// (input tokens + drafts) BEFORE the target model writes draft KV:
    /// without the extra blocks, `write_prefill_kv`'s
    /// `block_ids.get(block_idx).unwrap_or(0)` fallback silently writes the
    /// overflow into block 0, corrupting the prompt's first-block KV (RIL
    /// ISS-026).
    ///
    /// Returns the number of blocks the sequence holds afterwards.
    pub fn ensure_blocks_for_tokens(&mut self, seq_id: SeqId, min_blocks: usize) -> usize {
        let Some(idx) = self.running.iter().position(|s| s.id == seq_id) else {
            return 0;
        };
        while self.running[idx].kv_blocks.len() < min_blocks {
            let Some(new_blocks) = self.memory.allocate(1) else {
                break;
            };
            // ARCH-01: record the freshly allocated blocks so the refcount
            // matches the number of live owners (this sequence = 1).
            self.memory.record_blocks(&new_blocks);
            #[cfg(feature = "multi-node")]
            {
                // Feed the tokens for each newly-allocated block back to the
                // MemoryManager so the chain hash advances with real
                // content (same contract as build_batch / update).
                let block_idx = self.running[idx].kv_blocks.len();
                let start = block_idx * vllm_traits::BLOCK_SIZE;
                let end = (start + vllm_traits::BLOCK_SIZE).min(self.running[idx].tokens.len());
                let parent_hash = self.chain_cursors.get(&seq_id).copied().unwrap_or(0);
                for &block_id in &new_blocks {
                    let hash = self.memory.record_block_tokens(
                        block_id,
                        parent_hash,
                        &self.running[idx].tokens[start..end],
                    );
                    self.chain_cursors.insert(seq_id, hash);
                }
            }
            let mut blocks = (*self.running[idx].kv_blocks).clone();
            blocks.extend(new_blocks);
            self.running[idx].kv_blocks = Arc::new(blocks);
        }
        self.running[idx].kv_blocks.len()
    }

    /// Cancel a request by sequence ID
    pub fn cancel_request(&mut self, seq_id: SeqId) -> bool {
        if let Some(seq) = self.request_queue.remove(seq_id) {
            // Release blocks if allocated
            if !seq.kv_blocks.is_empty() {
                self.memory.release_blocks(seq.kv_blocks.as_ref());
            }
            return true;
        }
        // Check if it's running
        if let Some(pos) = self.running.iter().position(|s| s.id == seq_id) {
            let seq = self.running.remove(pos);
            if !seq.kv_blocks.is_empty() {
                self.memory.release_blocks(seq.kv_blocks.as_ref());
            }
            return true;
        }
        false
    }

    /// Get KV cache usage statistics
    pub const fn get_kv_cache_usage(&self) -> (u64, u64) {
        let total = self.memory.total_blocks() as u64;
        let available = self.memory.available_blocks() as u64;
        let used = total.saturating_sub(available);
        (used, total)
    }

    /// Get access to the prefix cache (`RadixTree`)
    pub const fn prefix_cache(&self) -> &RadixTree {
        &self.prefix_cache
    }

    /// Mutable accessor for the underlying [`MemoryManager`].
    ///
    /// Used by tests and integration code that needs to drive block
    /// allocation directly (the scheduler's public methods only call
    /// `allocate` indirectly during `add_request` / `build_batch`).
    /// Production code should use the higher-level request lifecycle.
    pub const fn memory_mut(&mut self) -> &mut MemoryManager {
        &mut self.memory
    }

    /// Install a distributed KV-cache so every subsequent block allocate
    /// / free round-trips through the cache.
    ///
    /// Idempotent — re-installing just replaces the cache reference.
    /// Existing tracked blocks are NOT migrated; future work will
    /// provide a snapshot-and-replay path for live migration if needed.
    #[cfg(feature = "multi-node")]
    pub fn set_distributed_kv(&mut self, cache: Arc<vllm_dist::DistributedKVCache>) {
        self.memory.set_distributed_kv(cache);
    }

    /// Propagate a `BlockDataSource` to the underlying [`MemoryManager`]
    /// (Phase 41 OPS-32a second-half).
    ///
    /// Mirrors [`Self::set_distributed_kv`] — used by
    /// `Engine::set_paged_kv_cache` (private) to thread the
    /// `PagedKvCacheWrapper` from the engine down to the memory layer so
    /// every subsequent gRPC `TransferKVBlock` call resolves to the
    /// wrapper.
    ///
    /// Idempotent — re-installing just replaces the source reference.
    #[cfg(feature = "multi-node")]
    pub fn set_block_data_source(
        &mut self,
        source: Arc<dyn vllm_dist::BlockDataSource + Send + Sync>,
    ) {
        self.memory.set_block_data_source(source);
    }

    /// Mutable accessor for the per-sequence chain cursors
    /// (see `super::state::SchedulerEngine::chain_cursors`).
    ///
    /// Used by `super::update::SchedulerEngine::update` to advance
    /// the cursor after each block allocation, and by tests / ops
    /// tools to inspect the chain state without poking at private
    /// fields. Production code should let the request lifecycle
    /// (allocate → record → advance) drive the cursors; this is
    /// exposed so test code can seed cursors and so the prefix-cache
    /// lookup (OPS-05b3) can read them.
    #[cfg(feature = "multi-node")]
    pub const fn chain_cursors_mut(&mut self) -> &mut std::collections::HashMap<SeqId, u64> {
        &mut self.chain_cursors
    }

    /// Look up `prompt_tokens` in the distributed KV cache.
    ///
    /// Thin wrapper around [`crate::scheduler::memory::MemoryManager::lookup_distributed_prefix`]
    /// so callers don't need to reach into the manager directly.
    /// Returns `None` when no cache is wired in.
    ///
    /// Phase 19 OPS-05b3.
    #[cfg(feature = "multi-node")]
    #[must_use]
    pub fn lookup_distributed_prefix(
        &self,
        prompt_tokens: &[vllm_traits::TokenId],
    ) -> Option<crate::scheduler::memory::DistributedPrefixMatch> {
        self.memory.lookup_distributed_prefix(prompt_tokens)
    }
}
