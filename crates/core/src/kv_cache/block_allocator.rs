use crate::types::BlockId;

pub struct BlockAllocator {
    num_blocks: usize,
    free_list: Vec<BlockId>,
}

impl BlockAllocator {
    pub fn new(num_blocks: usize) -> Self {
        let free_list: Vec<BlockId> = (0..num_blocks).collect();
        Self {
            num_blocks,
            free_list,
        }
    }

    pub fn allocate(&mut self, num_blocks: usize) -> Option<Vec<BlockId>> {
        if self.free_list.len() >= num_blocks {
            Some(
                (0..num_blocks)
                    .map(|_| self.free_list.pop().unwrap())
                    .collect(),
            )
        } else {
            None
        }
    }

    pub fn free(&mut self, blocks: &[BlockId]) {
        for &block in blocks {
            self.free_list.push(block);
        }
    }

    pub fn available(&self) -> usize {
        self.free_list.len()
    }

    pub fn total(&self) -> usize {
        self.num_blocks
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
}
