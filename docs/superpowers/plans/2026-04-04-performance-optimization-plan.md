# Performance Optimization Implementation Plan

## Task 1: KV Cache Pool

Add memory pool for better block allocation:

- Add `KvCachePool` struct
- Implement `allocate()` / `deallocate()`
- Use in PagedKvCache

## Task 2: Prefix Hash

Add block hash tracking:

- Add `compute_block_hash()` method
- Add `block_hashes` field
- Add `find_matching_blocks()` method

## Task 3: Flash Attention Tile Sizes

Improve tile size selection:

- Support [64, 128, 256] tile sizes
- Auto-select based on sequence length

## Task 4: Fused Kernel

Add fused layer kernel:

- Create `fused_attention_layer()` function
- Test performance improvement

## Task 5: Final verification

- Run CI
- Compare benchmarks
