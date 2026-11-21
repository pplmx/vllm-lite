//! Unit tests for `BlockAllocator`.
//!
//! Exercises the free-list invariants under hand-picked allocations:
//! exact-fit, OOM, repeated allocate/free cycles, and the
//! `allocated_bytes` / `BLOCK_BYTES` accounting helpers.
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

/// `free` must ignore out-of-range block ids and never double-count.
///
/// The guard is `if block < self.num_blocks as BlockId`, which protects the
/// `is_free[block]` index. `cargo mutants` relaxed it to `<=`, which indexes one
/// past the end. The neighbouring branch — freeing an already-free block —
/// must also leave the accounting untouched; a corrupted block id reaching
/// `free` is exactly what the guard exists for.
#[test]
fn test_free_ignores_out_of_range_and_duplicate_blocks() {
    let mut alloc = BlockAllocator::new(4);
    let blocks = alloc.allocate(2).expect("two blocks fit");
    assert_eq!(alloc.available(), 2);

    // A block id equal to `num_blocks` is out of range and must be ignored.
    alloc.free(&[4]);
    assert_eq!(
        alloc.available(),
        2,
        "an out-of-range id must not change the free count"
    );

    // A large bogus id must also be ignored, not panic.
    alloc.free(&[u32::MAX as BlockId]);
    assert_eq!(alloc.available(), 2);

    // Freeing the same block twice must count once.
    alloc.free(&blocks[0..1]);
    assert_eq!(alloc.available(), 3);
    alloc.free(&blocks[0..1]);
    assert_eq!(
        alloc.available(),
        3,
        "freeing an already-free block must not inflate the free count"
    );
}
