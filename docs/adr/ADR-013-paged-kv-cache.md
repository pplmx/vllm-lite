# ADR-013: Paged KV Cache over Contiguous Allocation

**Date:** 2026-06-27
**Status:** Accepted
**Context version:** v1.0

## Context

The KV cache stores, for each transformer layer, the key and value tensors of every past token in the sequence. For a 32-layer model with 32 heads, head_dim 128, and a 4K-token sequence, that's `32 * 2 * 32 * 128 * 4096 * 2 bytes ≈ 2 GB` per sequence — and per-sequence memory grows linearly with sequence length.

There are two allocation strategies:

- **Contiguous allocation**: each sequence gets a single contiguous tensor of size `max_seq_len * head_dim * num_heads * num_layers * 2 (K+V) * dtype_bytes`. Pre-allocated at admission, zero fragmentation, but reserves `max_seq_len` worth of memory even for short sequences.
- **Paged allocation**: memory is divided into fixed-size blocks (16 tokens each). Sequences own a list of block IDs; blocks are allocated on demand as the sequence grows. Inspired by OS virtual-memory paging.

Contiguous allocation has severe problems at scale:

- **Internal fragmentation**: a 100-token sequence in a 4096-token slot wastes 97.5% of its memory.
- **External fragmentation**: as sequences of varying lengths are admitted and freed, the contiguous allocator must coalesce free regions or accept unusable holes.
- **No prefix reuse**: if two sequences share a 500-token common prefix, contiguous allocation can't dedupe — each sequence stores its own full copy.
- **Worst-case admission**: a single 32K sequence must find a 32K contiguous hole, even if total free memory is 64K scattered in 2K pieces.

Paged allocation addresses all four problems:

- **No internal fragmentation** beyond `BLOCK_SIZE - 1` tokens (with `BLOCK_SIZE = 16`, max waste is 15 tokens per sequence).
- **No external fragmentation** in the practical sense — the block pool is a uniform free list.
- **Prefix reuse**: the prefix cache stores block hashes; a hit means "use these existing block IDs, allocate only the suffix".
- **Admission is easy**: any 16-token block is equivalent; the allocator never has to search for a contiguous region.

The cost is one indirection per attention access: the kernel must gather K/V tensors from a list of block pointers, not a single base pointer. This is well-known to be solvable (the PagedAttention paper from vLLM demonstrated this), and vllm-lite has implemented it in `crates/model/src/components/attention/`.

## Decision

The KV cache is paged into fixed-size blocks of **16 tokens** (`BLOCK_SIZE = 16` in `crates/traits/src/types.rs:6`):

```rust
// crates/traits/src/types.rs
/// BLOCK_SIZE: block size constant.
pub const BLOCK_SIZE: usize = 16;
/// BlockId: block id.
pub type BlockId = usize;
```

Each sequence is associated with a list of `BlockId` values (the block table). Blocks are allocated on demand as the sequence grows, and freed when the sequence completes or is preempted.

Implementation, split per ADR-005:

- **Logical layer** (`crates/core/src/scheduler/memory/allocator.rs`) — `BlockAllocator` tracks free blocks, hands them out on request, reclaims them on free. Owns the eviction and prefix-cache logic.
- **Physical layer** (`crates/model/src/paged_tensor/tensor_store.rs`) — `PagedKvCache` stores the actual K/V tensors indexed by block ID. The attention kernel reads from this via the block table.
- **Precision layer** (`crates/model/src/paged_tensor/quantization.rs` + `quant.rs`) — handles FP16/FP32/FP8 (ADR-004) storage inside each block.

The KV cache is exposed to attention as a block-table indirection; contiguous allocation is **not** supported and there is no contiguous fast path.

## Rationale

1. **Memory efficiency** — eliminates internal fragmentation; near-100% utilisation of allocated blocks (waste ≤ `BLOCK_SIZE - 1` tokens per sequence).
2. **Enables prefix caching** — the prefix cache (radix tree in `scheduler/radix_cache/`) hashes blocks; a hit means reusing the same block IDs across sequences with a common prefix. Contiguous allocation cannot do this.
3. **Predictable admission cost** — admitting a sequence requires only `ceil(prompt_len / BLOCK_SIZE)` blocks, findable in O(1) from the free list. No "find a 4K-hole" search.
4. **Proven design** — the PagedAttention paper (vLLM, Kwon et al., SOSP 2023) demonstrated this approach at scale; vllm-lite's implementation is directly inspired by it.
5. **Composes with continuous batching** (ADR-012) — freed blocks immediately re-enter the pool, no coalescing required.
6. **Composes with speculative decoding** (ADR-006) — the draft model can hold its own block table without affecting the target's.

Alternatives considered:

- **Contiguous allocation with slab allocator** — rejected; doesn't solve internal fragmentation or enable prefix caching.
- **Block size other than 16** — considered; values 8 and 32 were prototyped. 16 is the sweet spot — small enough to limit internal-fragmentation waste, large enough that the block-table indirection cost in attention is amortised.
- **Variable-size blocks** — rejected; uniform block size simplifies the free list, the block table, and the prefix cache.
- **Per-layer paged allocation with different block sizes per layer** — rejected; complicates the kernel for negligible gain.
- **No paged allocation (recompute KV each step)** — rejected; recompute cost dominates at any non-trivial sequence length.

## Consequences

**Positive:**

- Near-100% memory utilisation — wasted tokens ≤ `BLOCK_SIZE - 1` per sequence.
- Prefix cache reuses block IDs across sequences — common system prompts, chat templates, few-shot examples, etc. consume zero additional KV memory.
- Admission is O(1) — free blocks are uniform, no "best fit" search.
- Continuous batching (ADR-012) composes naturally — freed blocks re-enter the pool immediately.
- Memory pressure can be predicted exactly: `used_blocks = sum(ceil(seq_len / BLOCK_SIZE) for seq in running)`.

**Negative:**

- **Attention kernel complexity** — must gather K/V tensors from a list of block pointers instead of a single base pointer. The implementation is in `crates/model/src/components/attention/` and is more complex than the contiguous equivalent.
- **Block table size** — for a 32K-token sequence at `BLOCK_SIZE = 16`, the block table has 2048 entries. Negligible CPU memory but non-trivial GPU memory if cached there.
- **Prefix cache hash cost** — every block needs a hash for prefix-cache lookup. The radix tree in `scheduler/radix_cache/` makes this O(k) for k shared tokens, but k=1 lookups still pay the hash cost.
- **Sequences shorter than BLOCK_SIZE waste ~half a block** — 1-token sequences waste 15 tokens; for very short workloads this is worse than contiguous.
- **Debugging block ownership is harder** — "where does sequence X's K tensor for layer 5 live?" requires looking up the block table, finding the block ID, then indexing into the tensor store.

**Mitigations / migration paths:**

- The block-table indirection is implemented once in `paged_attention` (`components/attention/gqa.rs`); model-specific code doesn't need to know about it.
- Block-size tuning is a single constant (`BLOCK_SIZE`); changing it requires recompiling but no code changes.
- For very short workloads, batching multiple short sequences into one logical "packed" sequence is a future optimisation that composes with the existing block allocator.
- The prefix cache radix tree (`scheduler/radix_cache/`) is a self-contained module that can be swapped (e.g. for an LRU+hash approach) without touching the allocator or attention kernel.
