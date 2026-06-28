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
}
