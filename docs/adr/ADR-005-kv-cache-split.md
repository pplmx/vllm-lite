# ADR-005: KV Cache Split Across Three Locations

**Date:** 2026-06-27
**Status:** Accepted
**Context version:** v1.0

## Context

The KV cache is the central data structure of the inference engine — it mediates every prefill, every decode step, every memory budget decision, every prefix-cache hit. Despite being one logical resource, the KV cache has at least three orthogonal concerns that evolve independently:

1. **Allocation policy** — which sequence gets which physical blocks, when to evict, when to share across sequences (prefix cache).
2. **Physical layout** — how tensor data is stored in memory, indexed by block ID, mapped into the attention kernel.
3. **Precision / quantization** — FP16 vs FP32 vs FP8, scale factors, dequant-on-read.

Bundling all three into a single `KvCache` struct was the obvious first design, and was in fact the original implementation. It quickly produced a god module: changes to eviction policy forced recompiles of the attention kernels; changes to FP8 encoding forced re-validation of allocation paths; tests for the radix-tree prefix cache had to construct full tensor stores just to verify lookup logic.

## Decision

Split KV-cache functionality across three locations, each owning one concern:

```text
crates/core/src/
├── kv_cache/
│   ├── mod.rs                    # Public re-exports
│   └── (block allocator lives in scheduler/memory/allocator.rs)
└── scheduler/
    └── memory/
        ├── allocator.rs          # Allocation policy (logical blocks, eviction, prefix cache)
        ├── eviction.rs           # Eviction policies (LRU, priority-based)
        └── mod.rs                # Module entry

crates/model/src/
└── paged_tensor/
    ├── mod.rs
    ├── tensor_store.rs           # Physical layout (PagedKvCache, block → tensor mapping)
    ├── quant.rs                  # Quantization config + per-channel scales
    └── quantization.rs           # Quantization primitives (quantize/dequantize ops)
```

The **logical layer** (`scheduler/memory/allocator.rs`) tracks which blocks belong to which sequence, when to allocate/free, and how prefix-cache hits are detected. It operates on opaque `BlockId` values and never sees tensor bytes.

The **physical layer** (`paged_tensor/tensor_store.rs`) implements `PagedKvCache` — the actual tensor storage indexed by block ID. It handles the layout of K and V tensors across blocks, and the interface that attention kernels consume.

The **quantization layer** (`paged_tensor/quantization.rs` + `quant.rs`) handles dtype selection and quantize/dequantize operations. It is independent of which blocks exist or which sequence owns them.

Each layer exposes a narrow API to the next. The scheduler talks to the physical layer via `PagedKvCache` (block IDs → tensor slices); the physical layer talks to the quantization layer via `quantize/dequantize` functions; nothing crosses layers in the other direction.

## Rationale

1. **Independent evolution** — Eviction policy can change (LRU → ARC) without touching tensor layouts. FP8 → FP4 can land without touching allocation code. Attention kernel rewrites don't require scheduler changes.
2. **Testability** — Allocation logic can be unit-tested with mock physical stores; quantization can be unit-tested without any block allocator; tensor layout can be tested without any scheduler.
3. **Layering matches the data flow** — The three concerns naturally form a DAG (policy → layout → precision), not a cycle.
4. **Compilation isolation** — Changing eviction policy recompiles ~10 files in `scheduler/memory/`; changing FP8 encoding recompiles ~5 files in `paged_tensor/`. The previous god-module design forced full-crate rebuilds for any change.
5. **Multiple implementations per layer** — Each layer can have alternative implementations (e.g. a `MockPagedKvCache` for tests, a `CudaPagedKvCache` for production) without rippling changes through the other layers.

Alternatives considered:

- **Single god module** — rejected; previously attempted, produced circular dependencies and 30+ minute compile times for any change.
- **Two layers (policy + combined layout/precision)** — rejected; layout and precision evolve at different cadences (precision changes with hardware; layout changes with kernel work).
- **Four+ layers** (separating prefix cache from allocator) — rejected; prefix cache is a *consumer* of the allocator, not a peer of it.

## Consequences

**Positive:**

- Each layer is small enough to fit in a developer's head (allocator ~600 LOC, tensor_store ~400 LOC, quantization ~300 LOC).
- Independent test suites per layer — fast, focused regression coverage.
- Pluggable implementations (e.g. swap LRU for ARC eviction, or FP8 for INT8) without touching unrelated code.
- Compile-time isolation: a change in one layer doesn't trigger full crate rebuilds.

**Negative:**

- Cross-layer changes (e.g. "FP8 quantization needs to know the block size for per-block scales") require touching two locations.
- New developers must learn three module trees to understand the cache.
- The boundary between layers is sometimes a judgment call (e.g. should "block-level scaling factor for FP8" live in the precision layer or the layout layer?).
- Documentation must cover all three layers, not one.

**Mitigations / migration paths:**

- The `crates/core/src/kv_cache/mod.rs` file re-exports the key types (`BlockAllocator`, `BLOCK_SIZE`) to give newcomers a single entry point.
- The architectural diagram in `crates/core/src/kv_cache/mod.rs` (`//! KV cache utilities: block allocation for the scheduler.`) explicitly names the three concerns.
- ADRs like this one capture the rationale so future contributors don't re-bundle.
