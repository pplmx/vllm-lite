pub mod allocator;
pub mod eviction;

pub use allocator::{BlockAllocator, BlockAllocatorStats};
pub use eviction::{EvictionPolicy, EvictionPolicyStats};

use crate::types::{BlockId, Sequence};

pub struct MemoryManager {
    allocator: BlockAllocator,
    eviction_policy: EvictionPolicy,
}

impl Default for MemoryManager {
    fn default() -> Self {
        Self::new(1000)
    }
}

impl MemoryManager {
    pub fn new(num_blocks: usize) -> Self {
        Self {
            allocator: BlockAllocator::new(num_blocks),
            eviction_policy: EvictionPolicy::new(),
        }
    }

    pub fn allocate(&mut self, num_blocks: usize) -> Option<Vec<BlockId>> {
        self.allocator.allocate(num_blocks)
    }

    pub fn free(&mut self, blocks: &[BlockId]) {
        self.allocator.free(blocks);
        self.eviction_policy.release_blocks(blocks);
    }

    pub fn select_victims(
        &mut self,
        running_sequences: &[Sequence],
        num_blocks: usize,
    ) -> Vec<BlockId> {
        self.eviction_policy
            .select_victims(running_sequences, num_blocks)
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

    pub fn eviction_stats(&self) -> EvictionPolicyStats {
        self.eviction_policy.stats()
    }

    pub fn get_block_ref_count(&self, block: BlockId) -> usize {
        self.eviction_policy.get_block_ref_count(block)
    }

    pub fn invalidate_eviction_cache(&mut self) {
        self.eviction_policy.invalidate_cache();
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
        let mut manager = MemoryManager::new(10);

        let blocks = manager.allocate(3).unwrap();
        assert_eq!(blocks.len(), 3);
        assert_eq!(manager.available_blocks(), 7);

        manager.free(&blocks);
        assert_eq!(manager.available_blocks(), 10);
    }

    #[test]
    fn test_memory_manager_select_victims() {
        let mut manager = MemoryManager::new(10);

        let seq = make_sequence(1, vec![1, 2], Status::Decoding);
        let victims = manager.select_victims(&[seq], 1);
        assert!(victims.is_empty() || victims.len() == 1);
    }

    #[test]
    fn test_memory_manager_oom() {
        let mut manager = MemoryManager::new(2);
        manager.allocate(2).unwrap();
        assert!(manager.allocate(1).is_none());
    }
}
