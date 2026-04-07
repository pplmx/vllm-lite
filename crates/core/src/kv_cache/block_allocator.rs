use crate::types::BlockId;

const NULL_BLOCK: BlockId = BlockId::MAX;

#[derive(Clone, Default)]
pub struct BlockAllocatorStats {
    pub total_blocks: usize,
    pub available_blocks: usize,
    pub allocation_count: usize,
    pub free_count: usize,
}

pub struct BlockAllocator {
    num_blocks: usize,
    next_free: Vec<BlockId>,
    prev_free: Vec<BlockId>,
    first_free: BlockId,
    is_free: Vec<bool>,
    stats: BlockAllocatorStats,
}

impl BlockAllocator {
    pub fn new(num_blocks: usize) -> Self {
        let mut next_free = vec![0; num_blocks];
        let mut prev_free = vec![0; num_blocks];

        for i in 0..num_blocks {
            next_free[i] = if i + 1 < num_blocks {
                i + 1
            } else {
                NULL_BLOCK
            };
            prev_free[i] = if i > 0 { i - 1 } else { NULL_BLOCK };
        }

        Self {
            num_blocks,
            next_free,
            prev_free,
            first_free: 0,
            is_free: vec![true; num_blocks],
            stats: BlockAllocatorStats {
                total_blocks: num_blocks,
                available_blocks: num_blocks,
                allocation_count: 0,
                free_count: 0,
            },
        }
    }

    pub fn allocate(&mut self, num_blocks: usize) -> Option<Vec<BlockId>> {
        if self.stats.available_blocks < num_blocks {
            return None;
        }

        let mut blocks = Vec::with_capacity(num_blocks);
        for _ in 0..num_blocks {
            if self.first_free != NULL_BLOCK && self.is_free[self.first_free] {
                let block = self.first_free;
                blocks.push(block);
                self.remove_from_free_list(block);
                self.is_free[block] = false;
                self.stats.available_blocks -= 1;
            } else {
                return None;
            }
        }
        self.stats.allocation_count += 1;
        Some(blocks)
    }

    fn remove_from_free_list(&mut self, block: BlockId) {
        let next = self.next_free[block];
        let prev = self.prev_free[block];

        if prev != NULL_BLOCK {
            self.next_free[prev] = next;
        } else {
            self.first_free = next;
        }

        if next != NULL_BLOCK {
            self.prev_free[next] = prev;
        }
    }

    pub fn free(&mut self, blocks: &[BlockId]) {
        for &block in blocks {
            if block < self.num_blocks as BlockId {
                self.add_to_free_list(block);
                self.is_free[block] = true;
                self.stats.available_blocks += 1;
            }
        }
        self.stats.free_count += 1;
    }

    fn add_to_free_list(&mut self, block: BlockId) {
        let first = self.first_free;

        if first != NULL_BLOCK {
            self.prev_free[first] = block;
        }

        self.next_free[block] = first;
        self.prev_free[block] = NULL_BLOCK;
        self.first_free = block;
    }

    pub fn available(&self) -> usize {
        self.stats.available_blocks
    }

    pub fn total(&self) -> usize {
        self.num_blocks
    }

    pub fn stats(&self) -> BlockAllocatorStats {
        self.stats.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocate_and_free() {
        let mut alloc = BlockAllocator::new(10);

        let blocks = alloc.allocate(3).unwrap();
        assert_eq!(blocks.len(), 3);
        assert_eq!(alloc.available(), 7);

        alloc.free(&blocks);
        assert_eq!(alloc.available(), 10);
    }

    #[test]
    fn test_oom() {
        let mut alloc = BlockAllocator::new(2);
        alloc.allocate(2).unwrap();
        assert!(alloc.allocate(1).is_none());
    }

    #[test]
    fn test_free_order() {
        let mut alloc = BlockAllocator::new(5);

        let blocks1 = alloc.allocate(2).unwrap();
        let blocks2 = alloc.allocate(2).unwrap();

        alloc.free(&blocks2);
        alloc.free(&blocks1);

        assert_eq!(alloc.available(), 5);
    }

    #[test]
    fn test_block_allocation_exact_fit() {
        let mut alloc = BlockAllocator::new(3);
        let blocks = alloc.allocate(3).unwrap();
        assert_eq!(blocks.len(), 3);
        assert_eq!(alloc.available(), 0);
    }

    #[test]
    fn test_stats() {
        let mut alloc = BlockAllocator::new(10);

        assert_eq!(alloc.stats().allocation_count, 0);

        let blocks = alloc.allocate(3).unwrap();
        assert_eq!(alloc.stats().allocation_count, 1);

        alloc.free(&blocks);
        assert_eq!(alloc.stats().free_count, 1);
    }
}
