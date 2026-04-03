use vllm_core::kv_cache::BlockAllocator;

#[test]
fn test_allocate_exact_fit() {
    let mut allocator = BlockAllocator::new(10);
    let blocks = allocator.allocate(5).unwrap();
    assert_eq!(blocks.len(), 5);
    assert_eq!(allocator.available(), 5);
}

#[test]
fn test_allocate_partial() {
    let mut allocator = BlockAllocator::new(10);
    let blocks = allocator.allocate(3).unwrap();
    assert_eq!(blocks.len(), 3);
    assert_eq!(allocator.available(), 7);
}

#[test]
fn test_allocate_overflow_rejected() {
    let mut allocator = BlockAllocator::new(5);
    let result = allocator.allocate(10);
    assert!(result.is_none());
}

#[test]
fn test_allocate_zero() {
    let mut allocator = BlockAllocator::new(10);
    let blocks = allocator.allocate(0).unwrap();
    assert!(blocks.is_empty());
    assert_eq!(allocator.available(), 10);
}

#[test]
fn test_free_releases_blocks() {
    let mut allocator = BlockAllocator::new(10);
    let blocks = allocator.allocate(3).unwrap();
    allocator.free(&blocks);
    assert_eq!(allocator.available(), 10);
}
