pub mod allocator;
pub mod eviction;

pub use allocator::{BlockAllocator, BlockAllocatorStats};
pub use eviction::{EvictionPolicy, EvictionPolicyStats};

use crate::scheduler::preemption::PreemptionManager;
use crate::types::{BlockId, SchedulerConfig, Sequence, Status};

pub struct MemoryManager {
    allocator: BlockAllocator,
    eviction_policy: EvictionPolicy,
    preemption_manager: PreemptionManager,
}

impl Default for MemoryManager {
    fn default() -> Self {
        Self::new(SchedulerConfig::default(), 1000)
    }
}

impl MemoryManager {
    /// Creates a new MemoryManager with the given scheduler configuration and number of blocks.
    pub fn new(config: SchedulerConfig, num_blocks: usize) -> Self {
        Self {
            allocator: BlockAllocator::new(num_blocks),
            eviction_policy: EvictionPolicy::new(),
            preemption_manager: PreemptionManager::new(config),
        }
    }

    /// Allocates the specified number of blocks.
    /// Returns None if not enough blocks are available.
    pub fn allocate(&mut self, num_blocks: usize) -> Option<Vec<BlockId>> {
        self.allocator.allocate(num_blocks)
    }

    /// Frees the given blocks without updating eviction policy.
    /// Use release_blocks if you want to also update reference counts.
    pub fn free(&mut self, blocks: &[BlockId]) {
        self.allocator.free(blocks);
    }

    /// Releases blocks, updating eviction policy reference counts and freeing the blocks.
    pub fn release_blocks(&mut self, blocks: &[BlockId]) {
        self.eviction_policy.release_blocks(blocks);
        self.allocator.free(blocks);
    }

    /// Selects victim blocks from running sequences to free up the requested number of blocks.
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
    pub fn available_blocks(&self) -> usize {
        self.allocator.available()
    }

    /// Returns the total number of blocks managed by this MemoryManager.
    pub fn total_blocks(&self) -> usize {
        self.allocator.total()
    }

    /// Returns statistics about the block allocator.
    pub fn allocator_stats(&self) -> BlockAllocatorStats {
        self.allocator.stats()
    }

    /// Determines whether preemption should be triggered based on current system state.
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

        for &idx in preemptable_indices.iter() {
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
        let block_size = crate::kv_cache::BLOCK_SIZE;
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

    /// Returns preemption statistics: (preempted_count, rejected_count).
    pub fn preemption_stats(&self) -> (u64, u64) {
        (
            self.preemption_manager.preempted_count(),
            self.preemption_manager.rejected_count(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Priority, Status};
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
            sampling_params: Default::default(),
            consecutive_decode_rounds: 0,
            priority: Priority::default(),
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
}
