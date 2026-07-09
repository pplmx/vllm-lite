//! Property-based tests for `BlockAllocator`.
//!
//! Three invariants hold across any sequence of `allocate` / `free`
//! operations:
//!
//! 1. Block IDs returned by `allocate` are unique within an allocator's
//!    lifetime and `allocate` returns `None` once capacity is exhausted.
//! 2. After `free` of a single block, the next single-block `allocate`
//!    reuses the just-freed ID (LIFO reuse).
//! 3. The number of allocated blocks (`total - available`) never exceeds
//!    capacity and matches the caller's bookkeeping under any
//!    interleaving of operations.
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
