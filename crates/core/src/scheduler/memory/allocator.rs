use crate::types::BlockId;
use tracing::warn;

const NULL_BLOCK: BlockId = BlockId::MAX;

/// Bytes per KV block, used for VRAM budget tracking.
///
/// This is a default sizing — it does NOT track per-token KV cache size
/// or model-specific head dimensions. Callers needing exact VRAM accounting
/// should compute their own block size and use the helper APIs.
pub(crate) const BLOCK_BYTES: usize = 16 * 1024 * 1024;

/// `BlockAllocatorStats`: block allocator statistics.
#[derive(Clone, Default)]
pub struct BlockAllocatorStats {
    pub total_blocks: usize,
    pub available_blocks: usize,
    pub allocation_count: usize,
    pub free_count: usize,
}

/// `BlockAllocator`: block allocator.
pub struct BlockAllocator {
    num_blocks: usize,
    next_free: Vec<BlockId>,
    prev_free: Vec<BlockId>,
    first_free: BlockId,
    is_free: Vec<bool>,
    stats: BlockAllocatorStats,
}

impl BlockAllocator {
    #[must_use]
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
        tracing::debug!(
            requested = num_blocks,
            available = self.stats.available_blocks,
            "Block allocation requested"
        );
        if self.stats.available_blocks < num_blocks {
            return None;
        }

        debug_assert!(self.first_free != NULL_BLOCK || self.stats.available_blocks == 0);

        let mut blocks = Vec::with_capacity(num_blocks);
        for _ in 0..num_blocks {
            let block = self.first_free;
            if block == NULL_BLOCK {
                return None;
            }
            debug_assert!(self.is_free[block]);
            blocks.push(block);
            self.remove_from_free_list(block);
            self.is_free[block] = false;
            self.stats.available_blocks -= 1;
        }
        self.stats.allocation_count += 1;
        tracing::trace!(
            allocated = ?blocks,
            remaining_free = self.stats.available_blocks,
            "Blocks allocated"
        );
        Some(blocks)
    }

    fn remove_from_free_list(&mut self, block: BlockId) {
        let next = self.next_free[block];
        let prev = self.prev_free[block];

        if prev == NULL_BLOCK {
            self.first_free = next;
        } else {
            self.next_free[prev] = next;
        }

        if next != NULL_BLOCK {
            self.prev_free[next] = prev;
        }
    }

    pub fn free(&mut self, blocks: &[BlockId]) {
        tracing::trace!(
            blocks = ?blocks,
            freed_count = blocks.len(),
            remaining_free = self.stats.available_blocks,
            "Blocks freed"
        );
        for &block in blocks {
            if block < self.num_blocks as BlockId {
                if self.is_free[block] {
                    warn!(block = block, "freeing already-freed block");
                } else {
                    self.add_to_free_list(block);
                    self.is_free[block] = true;
                    self.stats.available_blocks += 1;
                }
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

    #[must_use]
    pub const fn available(&self) -> usize {
        self.stats.available_blocks
    }

    #[must_use]
    pub const fn total(&self) -> usize {
        self.num_blocks
    }

    #[must_use]
    pub fn stats(&self) -> BlockAllocatorStats {
        self.stats.clone()
    }

    /// Bytes per KV block. Static — same constant for all allocators in the
    /// process. Used for VRAM budget accounting.
    #[must_use]
    pub const fn bytes_per_block() -> usize {
        BLOCK_BYTES
    }

    /// Number of bytes currently allocated (in use) by this allocator.
    /// Computed as `(total_blocks - available_blocks) * BLOCK_BYTES`.
    #[must_use]
    pub const fn allocated_bytes(&self) -> usize {
        let allocated_blocks = self.num_blocks.saturating_sub(self.stats.available_blocks);
        allocated_blocks.saturating_mul(BLOCK_BYTES)
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

    #[test]
    fn test_bytes_per_block_constant() {
        assert_eq!(
            BlockAllocator::bytes_per_block(),
            16 * 1024 * 1024,
            "BLOCK_BYTES is the v18.0 VRAM accounting constant"
        );
    }

    #[test]
    fn test_allocated_bytes_scales_with_allocations() {
        let mut alloc = BlockAllocator::new(10);
        assert_eq!(alloc.allocated_bytes(), 0);

        let blocks = alloc.allocate(3).unwrap();
        assert_eq!(
            alloc.allocated_bytes(),
            3 * BlockAllocator::bytes_per_block()
        );

        let more = alloc.allocate(2).unwrap();
        assert_eq!(
            alloc.allocated_bytes(),
            5 * BlockAllocator::bytes_per_block()
        );

        alloc.free(&blocks);
        assert_eq!(
            alloc.allocated_bytes(),
            2 * BlockAllocator::bytes_per_block()
        );

        alloc.free(&more);
        assert_eq!(alloc.allocated_bytes(), 0);
    }
}
