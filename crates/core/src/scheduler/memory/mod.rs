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
    pub fn new(config: SchedulerConfig, num_blocks: usize) -> Self {
        Self {
            allocator: BlockAllocator::new(num_blocks),
            eviction_policy: EvictionPolicy::new(),
            preemption_manager: PreemptionManager::new(config),
        }
    }

    pub fn allocate(&mut self, num_blocks: usize) -> Option<Vec<BlockId>> {
        self.allocator.allocate(num_blocks)
    }

    pub fn free(&mut self, blocks: &[BlockId]) {
        self.allocator.free(blocks);
    }

    pub fn release_blocks(&mut self, blocks: &[BlockId]) {
        self.eviction_policy.release_blocks(blocks);
        self.allocator.free(blocks);
    }

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

    pub fn record_blocks(&mut self, blocks: &[BlockId]) {
        self.eviction_policy.record_blocks(blocks);
    }

    pub fn touch_blocks(&mut self, blocks: &[BlockId]) {
        self.eviction_policy.touch_blocks(blocks);
    }

    pub fn available_blocks(&self) -> usize {
        self.allocator.available()
    }

    pub fn total_blocks(&self) -> usize {
        self.allocator.total()
    }

    pub fn allocator_stats(&self) -> BlockAllocatorStats {
        self.allocator.stats()
    }

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

    pub fn execute_preemption(
        &mut self,
        running: &mut Vec<Sequence>,
        blocks_needed: usize,
    ) -> Vec<Sequence> {
        let mut preemptable: Vec<_> = running
            .iter()
            .filter(|s| s.status == Status::Decoding)
            .cloned()
            .collect();

        preemptable.sort_by(|a, b| {
            b.consecutive_decode_rounds
                .cmp(&a.consecutive_decode_rounds)
        });

        let mut blocks_freed = 0;
        let mut preempted = Vec::new();

        for seq in preemptable.iter() {
            if blocks_freed >= blocks_needed {
                break;
            }

            let block_count = seq.kv_blocks.len();
            self.free(seq.kv_blocks.as_ref());
            preempted.push(seq.clone());
            blocks_freed += block_count;
        }

        for seq in &preempted {
            running.retain(|s| s.id != seq.id);
        }

        preempted
    }

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
