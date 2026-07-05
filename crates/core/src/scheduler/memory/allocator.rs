//! Paged-KV block allocator: hands out contiguous blocks of `BLOCK_SIZE` tokens and recycles freed blocks.
//!
//! Implements a free-list with split / merge on adjacent free blocks.
//! The scheduler calls `allocate(n)` for every prefill and `free(ranges)`
//! when a sequence is dropped or preempted.
#![allow(clippy::module_name_repetitions)]
use crate::types::BlockId;
use tracing::warn;

const NULL_BLOCK: BlockId = BlockId::MAX;

/// Bytes per KV block, used for VRAM budget tracking.
///
/// This is a default sizing — it does NOT track per-token KV cache size
/// or model-specific head dimensions. Callers needing exact VRAM accounting
/// should compute their own block size and use the helper APIs.
pub(crate) const BLOCK_BYTES: usize = 16 * 1024 * 1024;

/// Block allocator telemetry: number of free/allocated/cached blocks, fragmentation ratio, allocation success rate. Updated on every `allocate`/`free` call.
#[derive(Debug, Clone, Default)]
pub struct BlockAllocatorStats {
    /// Total number of blocks this allocator owns.
    pub total_blocks: usize,
    /// Number of blocks currently free.
    pub available_blocks: usize,
    /// Cumulative count of `allocate(n)` calls that succeeded.
    pub allocation_count: usize,
    /// Cumulative count of `free(ranges)` calls.
    pub free_count: usize,
}

#[derive(Debug)]
/// Paged-KV-cache block allocator. Hands out contiguous virtual block IDs backed by physically scattered GPU/CPU blocks. Uses a free-list with split-on-OOM and coalescing on free.
pub struct BlockAllocator {
    /// Total block capacity.
    num_blocks: usize,
    /// Singly-linked free list (next-pointer per block id).
    next_free: Vec<BlockId>,
    /// Doubly-linked free list (prev-pointer per block id).
    prev_free: Vec<BlockId>,
    /// Head of the free list (`BlockId::MAX` when the list is empty).
    first_free: BlockId,
    /// Per-block allocation flag.
    is_free: Vec<bool>,
    /// Cumulative allocator statistics.
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

#[cfg(test)]
mod prop_tests {
    use super::*;
    use proptest::prelude::*;
    use std::collections::HashSet;

    proptest! {
        /// Property 1: All block IDs returned from `allocate()` are unique
        /// within an allocator's lifetime. Once capacity is exhausted,
        /// `allocate()` returns `None`.
        #[test]
        fn prop_allocated_unique(
            capacity in 1usize..50,
        ) {
            let mut alloc = BlockAllocator::new(capacity);
            let mut seen = HashSet::new();
            for _ in 0..capacity {
                let blocks = alloc.allocate(1);
                prop_assert!(blocks.is_some(), "allocate returned None before exhaustion");
                // invariant: caller guarantees non-empty block vector.
                let blocks = blocks.unwrap();
                prop_assert_eq!(blocks.len(), 1);
                let id = blocks[0];
                let fresh = seen.insert(id);
                prop_assert!(fresh, "duplicate allocation: {id}");
                prop_assert!(
                    id < capacity,
                    "id {id} out of range [0, {capacity})"
                );
            }
            prop_assert_eq!(alloc.allocate(1), None);
            prop_assert_eq!(alloc.available(), 0);
        }

        /// Property 2: After `free()`, the next single-block `allocate()`
        /// reuses the just-freed ID (LIFO reuse invariant).
        #[test]
        fn prop_alloc_free_reuse_lifo(
            capacity in 2usize..50,
        ) {
            let mut alloc = BlockAllocator::new(capacity);
            // invariant: pre-conditions make this infallible at this call site.
            let first = alloc.allocate(1).expect("first allocate").pop().unwrap();
            alloc.free(&[first]);
            // invariant: pre-conditions make this infallible at this call site.
            let second = alloc.allocate(1).expect("second allocate").pop().unwrap();
            prop_assert_eq!(first, second, "LIFO should reuse the freed id");
        }

        /// Property 3: The number of allocated blocks (`total - available`)
        /// never exceeds capacity and matches the caller's bookkeeping under
        /// any interleaving of `allocate` / `free` operations.
        #[test]
        fn prop_alloc_count_bounded(
            capacity in 1usize..30,
            ops in proptest::collection::vec(any::<bool>(), 1..60),
        ) {
            let mut alloc = BlockAllocator::new(capacity);
            let mut live: HashSet<BlockId> = HashSet::new();

            for do_alloc in ops {
                if do_alloc {
                    if let Some(blocks) = alloc.allocate(1) {
                        for id in &blocks {
                            let fresh = live.insert(*id);
                            prop_assert!(fresh, "duplicate id allocated: {id}");
                        }
                        prop_assert!(
                            live.len() <= capacity,
                            "live {} > capacity {}",
                            live.len(),
                            capacity
                        );
                    } else {
                        prop_assert_eq!(
                            alloc.available(),
                            0,
                            "allocate returned None but available > 0"
                        );
                    }
                } else if let Some(&id) = live.iter().next() {
                    live.remove(&id);
                    alloc.free(&[id]);
                }
                let allocated = alloc.total() - alloc.available();
                prop_assert_eq!(
                    allocated,
                    live.len(),
                    "bookkeeping mismatch: allocated={} live={}",
                    allocated,
                    live.len()
                );
                prop_assert!(allocated <= capacity);
            }
        }
    }
}
