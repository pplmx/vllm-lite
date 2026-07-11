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

use std::sync::Arc;
use std::time::Instant;

use vllm_traits::SeqId;

use crate::scheduler::RadixTree;
use crate::scheduler::memory::MemoryManager;
use crate::scheduler::policy::SchedulingContext;
use crate::types::Status;

use super::state::SchedulerEngine;

impl SchedulerEngine {
    /// Execute preemption to free up memory blocks
    pub(super) fn execute_preemption(&mut self, blocks_needed: usize) {
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

        let mut blocks_freed = 0;
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
        let total = self.memory.total_blocks() as f32;
        let available = self.memory.available_blocks() as f32;
        1.0 - (available / total)
    }

    /// Rollback KV cache for rejected draft tokens (Plan 17.1-D).
    pub fn memory_rollback(&mut self, seq_id: SeqId, num_tokens: usize) {
        if let Some(seq) = self.running.iter_mut().find(|s| s.id == seq_id) {
            self.memory.rollback(seq, num_tokens);
        }
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

    /// Mutable accessor for the per-sequence chain cursors
    /// ([`super::state::SchedulerEngine::chain_cursors`]).
    ///
    /// Used by [`super::update::SchedulerEngine::update`] to advance
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
