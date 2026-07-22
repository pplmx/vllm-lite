# P40 Design — `PagedKvCacheWrapper`: Production `BlockDataSource` Implementation

> **For:** P40 phase of the v31.0 "Perfection & Elegance" milestone.
> **Status:** Draft (awaiting sign-off)
> **Date:** 2026-07-22
> **ADR:** Extends [ADR-020](../../adr/ADR-020-multi-node-kv-block-transfer.md) — closes the **first half** of OPS-32a's deferred engine-integration gap.

## 1. Background & motivation

OPS-31d (P12, Phase 31-D) shipped the multi-node **protocol layer** end-to-end:

- `BlockDataSource` trait (`crates/dist/src/distributed_kv/block_data_source.rs`)
- `TransferKVBlock` gRPC RPC (`crates/dist/proto/node.proto`)
- `DistributedKVCache::fetch_block` with fan-out fallback
- 64 MiB symmetric message limit
- 7 integration tests using a `MockBlockDataSource`

ADR-020 §5 explicitly deferred the **engine integration** ("v32+ / OPS-32a"):

> The wiring that closes the loop end-to-end:
>
> ```
> PagedKvCache ─► PagedKvCacheWrapper: BlockDataSource
>        │
>        ▼
> GrpcState.block_data_source ─► transfer_kv_block handler
>        │
>        ▼
> PeerClient.fetch_block ─► DistributedKVCache::fetch_block
> ```
>
> is **not** shipped in v31.0. The `PagedKvCacheWrapper` requires
> plumbing `Arc<PagedKvCache>` from `crates/model/src/paged_tensor/`
> through `crates/core/src/scheduler/memory/MemoryManager` — a
> model-crate touch that the technical due diligence deferred. Without
> the wrapper, the gRPC server answers
> `Status::unavailable("TransferKVBlock called but no BlockDataSource
> wired in")` for every block transfer, so multi-node replication
> works for `(block_id, chain_hash)` *intent* but actual block bytes
> stay local-only in the default engine build. OPS-32a is the next
> phase for this work.

OPERATIONS.md §"Multi-Node (Experimental)" §"What is **not** yet
production-ready" (P12 follow-up, rewritten for honesty) names the same
gap:

> The `PagedKvCacheWrapper` (production `BlockDataSource` impl backed
> by `Arc<PagedKvCache>`) is **not yet implemented**. Until it ships,
> the gRPC server returns `Status::unavailable` for every
> `TransferKVBlock` request — meaning a node that detects via
> `lookup_prefix` that a peer has a prefix can replicate the
> `(block_id, chain_hash)` *intent* but cannot pull the actual KV
> tensor bytes back. Engine-level plumbing
> (`PagedKvCacheWrapper → MemoryManager`) lands in v32+ / OPS-32a.

## 2. Scope split (P40 vs. P41+)

The full OPS-32a work decomposes into **two bounded, independently
mergeable phases**. P40 ships **half**; the engine-level plumbing
deferral keeps the PR reviewable while delivering the load-bearing
implementation.

| Work | P40 (this design) | P41+ |
|------|-------------------|------|
| `PagedKvCacheWrapper` struct + `BlockDataSource` impl | ✅ ships | n/a |
| Read-path: K/V tensors → `Vec<u8>` (handles `quantized`) | ✅ ships | n/a |
| `has_block` impl via `block_hashes[0]` lookup | ✅ ships | n/a |
| `#[cfg(feature = "multi-node")]` gating (per ADR-008) | ✅ ships | n/a |
| `DistributedKVCache::with_paged_kv_cache(...)` ergonomic builder | ✅ ships | n/a |
| Integration test: real `PagedKvCache` round-trip via 2-node gRPC | ✅ ships | n/a |
| Docs: `OPERATIONS.md` §"What works" updated, "What is not" shrunk | ✅ ships | n/a |
| `Arc<PagedKvCache>` plumbed through `MemoryManager` | n/a (deferred) | ✅ P41 candidate |
| `EngineBuilder::with_paged_kv_cache(...)` builder method | n/a (deferred) | ✅ P41 candidate |
| `crates/server` main wiring (`ServerState → Engine → GrpcState`) | n/a (deferred) | ✅ P41 candidate |
| Hash-verification helper on wrapper side (receiver checks bytes) | n/a (deferred) | ✅ P41 candidate |
| `CacheConfig::peer_node_ids` for owner routing | n/a (deferred) | ✅ P32+ |

**Why split here:**
- The wrapper itself is a **concrete `BlockDataSource` impl** that
  can be unit-tested in isolation against a real `PagedKvCache` —
  the engine-level plumbing is what makes it **useful in production**,
  but the wrapper has standalone value (anyone who already holds an
  `Arc<PagedKvCache>` can wire it directly into a `DistributedKVCache`).
- Per the project's "Why P_N and not the alternatives" pattern (P21-P39),
  splitting OPS-32a at the natural seam keeps each PR reviewable.
- ADR-020 §5 names the **wrapper** and the **engine plumbing** as
  separate items; P40 closes the first one.

## 3. Goals

1. **G1 — Concrete `BlockDataSource` impl.** Add `PagedKvCacheWrapper`
   in `crates/model/src/paged_tensor/` (gated by
   `#[cfg(feature = "multi-node")]`) that holds an `Arc<PagedKvCache>`
   and implements `BlockDataSource::fetch_block` / `has_block`.
2. **G2 — Round-trip end-to-end.** A 2-node gRPC integration test
   proves: peer A writes real KV into a `PagedKvCache`, wraps it,
   serves via `TransferKVBlock`; peer B receives the bytes and
   deserializes them back into a `Tensor` shape compatible with
   `PagedKvCache::write_kv_batch` (caller decides whether to install).
3. **G3 — Quantization-correct.** The wrapper's `fetch_block`
   produces byte streams that respect the source's `quantized` flag
   — for `quantized=true`, dequantize before serializing (receiver
   gets f32 bytes, matches `PagedKvCache::write_kv` input contract).
4. **G4 — Bounded scope.** No model-crate lifecycle changes, no
   `MemoryManager` plumbing, no server main wiring. `vllm-dist`'s
   dependency tree is **unchanged** (wrapper lives in `vllm-model`,
   `vllm-model`'s `vllm-dist` optional dep already exists per
   ADR-008).
5. **G5 — Honest docs.** `OPERATIONS.md` §"Multi-Node (Experimental)"
   §"What works" grows from 3 bullets to 4 (add "Real PagedKvCache
   bytes can be served via TransferKVBlock"); §"What is not" shrinks
   by one item ("PagedKvCacheWrapper not implemented" → "PagedKvCache
   wrapper exists but is **not yet wired into the server"**).

## 4. Non-goals (P40 explicitly defers)

- Engine-level plumbing (`Arc<PagedKvCache>` → `MemoryManager`).
  Per ADR-020 §5 this is the **second** half of OPS-32a; deferred to
  P41+ to keep P40 reviewable.
- `EngineBuilder::with_paged_kv_cache(...)` builder method —
  requires the engine-plumbing change first.
- Server-side wiring in `crates/server/src/main.rs` (`ServerState`
  → Engine → GrpcState → `start_grpc_server_with_listener`).
- Hash verification on the wrapper side — the protocol layer
  (OPS-31d) already verifies `chain_hash` on the wire; re-hashing
  bytes inside the wrapper is redundant.
- `CacheConfig::peer_node_ids` (owner-based routing) — fan-out from
  OPS-31d is correct, just bandwidth-wasteful. Out of scope for P40.
- Receiver-side `write_kv_batch` integration (turning received bytes
  back into installed KV). This is part of the `MemoryManager`
  plumbing in P41+.

## 5. Architecture

### 5.1 New module: `crates/model/src/paged_tensor/paged_kv_cache_wrapper.rs`

```rust
//! Production [`BlockDataSource`] impl backed by [`PagedKvCache`].
//!
//! Closes the **first half** of OPS-32a (P40 / v31.0). Engine-level
//! plumbing (`Arc<PagedKvCache>` → `MemoryManager` → EngineBuilder)
//! lands in P41+ — this wrapper has standalone value: any caller that
//! holds an `Arc<PagedKvCache>` can wire it directly into a
//! [`DistributedKVCache`] or a [`GrpcState`] via
//! [`DistributedKVCache::with_block_data_source`] /
//! [`GrpcState::with_block_data_source`].
//!
//! ## Why this lives in `vllm-model`
//!
//! [`PagedKvCache`] lives in `vllm-model`; the wrapper needs access
//! to its private fields (`key_cache`, `value_cache`, `quantized`,
//! `block_hashes`). Per ADR-008, `vllm-dist` is feature-gated
//! (`--features multi-node`) and cannot pull in `vllm-model`, so
//! placing the wrapper in `vllm-model` and importing `vllm-dist` is
//! the only layering-correct location.
//!
//! ## Why gated by `multi-node`
//!
//! The wrapper compiles to a no-op when `multi-node` is off —
//! `vllm-model` then doesn't see `vllm-dist`, the `BlockDataSource`
//! trait doesn't exist, and the `paged_kv_cache_wrapper` module
//! becomes a `#[cfg(feature = "multi-node")] pub mod` that is
//! invisible in the default build. Mirrors how `crates/dist`'s
//! consumers in `vllm-core` already gate behind the same flag.

#[cfg(feature = "multi-node")]
use std::sync::Arc;

#[cfg(feature = "multi-node")]
use candle_core::Tensor;

#[cfg(feature = "multi-node")]
use vllm_dist::{BlockDataSource, FetchError};

#[cfg(feature = "multi-node")]
use super::tensor_store::PagedKvCache;

/// Wraps a [`PagedKvCache`] as a [`BlockDataSource`] for cross-node
/// transfer.
///
/// `Arc<PagedKvCache>` is the shared storage form so the wrapper can
/// be cheaply cloned into both a [`DistributedKVCache`] and a
/// [`GrpcState`]; both hold `Arc<dyn BlockDataSource>`.
#[cfg(feature = "multi-node")]
#[derive(Clone, Debug)]
pub struct PagedKvCacheWrapper {
    inner: Arc<PagedKvCache>,
}

#[cfg(feature = "multi-node")]
impl PagedKvCacheWrapper {
    /// Wrap an `Arc<PagedKvCache>` for use as a [`BlockDataSource`].
    #[must_use]
    pub const fn new(inner: Arc<PagedKvCache>) -> Self {
        Self { inner }
    }

    /// Borrow the wrapped cache (useful for tests that need to read
    /// back what was stored).
    #[must_use]
    pub fn inner(&self) -> &PagedKvCache {
        &self.inner
    }
}

#[cfg(feature = "multi-node")]
#[async_trait::async_trait]
impl BlockDataSource for PagedKvCacheWrapper {
    async fn fetch_block(&self, block_id: u64) -> Result<Vec<u8>, FetchError> {
        let block_id_us = usize::try_from(block_id)
            .map_err(|_| FetchError::NotFound(block_id))?;

        // The trait is async for future GPU-direct paths; today's
        // CPU-side read is sync. Spawning `spawn_blocking` would
        // pessimise the hot path (we're already off the GPU worker
        // thread by the time gRPC serves); a `block_in_place` is the
        // idiomatic alternative if this becomes a contention point.
        // For P40 the sync read is fine — see §6 performance notes.
        tokio::task::block_in_place(|| read_block_bytes(&self.inner, block_id_us))
    }

    async fn has_block(&self, block_id: u64) -> bool {
        // PagedKvCache tracks per-layer block hashes (one
        // HashMap per layer). We use layer 0 as the canonical
        // existence check — every block is written to every layer
        // or to none, so layer 0 is a sound witness.
        let block_id_us = match usize::try_from(block_id) {
            Ok(n) => n,
            Err(_) => return false,
        };
        self.inner
            .block_hashes()
            .first()
            .is_some_and(|layer_hashes| layer_hashes.values().any(|&bid| bid == block_id_us))
    }
}

#[cfg(feature = "multi-node")]
fn read_block_bytes(cache: &PagedKvCache, block_id: usize) -> Result<Vec<u8>, FetchError> {
    // Bounding check — block_id out of range is `NotFound`, not a
    // server-side 500. Matches OPS-31d's `MockBlockDataSource`
    // behaviour.
    if block_id >= cache.num_blocks() {
        return Err(FetchError::NotFound(block_id as u64));
    }

    // Concatenate per-layer K and V tensors for the given block_id
    // into a single flat Vec<u8>. The receiver uses
    // PagedKvCache::write_kv_batch to install the bytes back.
    //
    // Layout (matches PagedKvCache::write_kv's tensor shape):
    //   For each layer in [0, num_layers):
    //     K block:  (num_heads, block_size, head_dim) f32
    //     V block:  (num_heads, block_size, head_dim) f32
    //
    // We dequantize before serializing when `quantized=true` so the
    // receiver gets f32 bytes (matches `write_kv_batch`'s f32
    // contract).
    let mut bytes = Vec::with_capacity(estimate_bytes(cache));
    for layer_idx in 0..cache.num_layers() {
        let k_block = cache.read_kv_layer_k(layer_idx, block_id)
            .map_err(|_| FetchError::NotFound(block_id as u64))?;
        let v_block = cache.read_kv_layer_v(layer_idx, block_id)
            .map_err(|_| FetchError::NotFound(block_id as u64))?;
        let k_f32 = if cache.quantized {
            dequantize_to_f32(&k_block, cache.get_scale(layer_idx))
        } else {
            k_block
        };
        let v_f32 = if cache.quantized {
            dequantize_to_f32(&v_block, cache.get_scale(layer_idx))
        } else {
            v_block
        };
        bytes.extend_from_slice(&k_f32);
        bytes.extend_from_slice(&v_f32);
    }
    Ok(bytes)
}
```

**Helper signatures on `PagedKvCache`** (additions, all gated by
`#[cfg(feature = "multi-node")]` so they don't touch the default
build):

- `pub(crate) fn num_blocks(&self) -> usize` — already effectively
  present as `self.key_cache[0].dim(0)`; expose via a tiny getter.
- `pub(crate) fn num_layers(&self) -> usize` — already present
  (`self.num_layers`); expose.
- `pub(crate) fn block_hashes(&self) -> &[HashMap<u64, usize>]` —
  borrow the `block_hashes` Vec.
- `pub(crate) fn read_kv_layer_k(&self, layer: usize, block_id: usize)
  -> Result<Vec<f32>>` — narrow the K tensor for `[layer,
  block_id..block_id+1, ..]`, materialize to `Vec<f32>`.
- `pub(crate) fn read_kv_layer_v(&self, layer: usize, block_id: usize)
  -> Result<Vec<f32>>` — same for V.
- `pub(crate) fn get_scale(&self, layer: usize) -> f32` — already
  present (`self.scales[layer]`); expose.

### 5.2 Module declarations

`crates/model/src/paged_tensor/mod.rs`:

```rust
pub mod tensor_store;
#[cfg(feature = "multi-node")]
pub mod paged_kv_cache_wrapper;
```

`crates/model/src/paged_tensor/tensor_store/mod.rs`: add the four
new `pub(crate)` accessors above. Each is a thin wrapper around
existing fields/methods.

### 5.3 Re-exports

`crates/model/src/paged_tensor/mod.rs`:

```rust
#[cfg(feature = "multi-node")]
pub use paged_kv_cache_wrapper::PagedKvCacheWrapper;
```

`crates/dist/src/lib.rs`: **no changes** — the wrapper doesn't
extend `vllm-dist`'s public API; it's consumed *via*
`Arc<dyn BlockDataSource>` (trait object), which `vllm-dist` already
exports.

### 5.4 Integration test plan

Two new test files (both feature-gated behind
`#[cfg(feature = "multi-node")]` since they import `vllm-dist`):

**Unit tests** in `crates/model/src/paged_tensor/paged_kv_cache_wrapper.rs::tests`:

1. `wrapper_has_block_returns_true_for_written_block`
2. `wrapper_has_block_returns_false_for_unknown_block`
3. `wrapper_fetch_block_returns_bytes_for_written_block`
4. `wrapper_fetch_block_returns_not_found_for_oob_block`
5. `wrapper_fetch_block_handles_quantized_layer` (set
   `quantized=true`, write a block, fetch, verify the bytes are
   dequantized)
6. `wrapper_fetch_block_returns_empty_for_unwritten_layer_0` (write
   to layer 0 only — not realistic but pins the boundary)

**Integration tests** in
`crates/model/tests/paged_kv_cache_wrapper_e2e.rs` (new file):

Reuses the OPS-31d in-process gRPC pair pattern from
`crates/dist/tests/kv_block_transfer.rs`:

1. `peer_serves_real_paged_kv_bytes_via_wrapper` — 2-node pair;
   sender wraps a real `PagedKvCache`; receiver fetches a block via
   `DistributedKVCache::fetch_block`; deserialized byte length
   matches `num_layers * num_heads * BLOCK_SIZE * head_dim * 4 * 2`.
2. `wrapper_round_trip_with_quantization` — write quantized data,
   wrap, fetch, verify the receiver sees f32 bytes (not the
   quantized form).
3. `wrapper_fetch_block_chain_hash_verification_works` — combine
   the wrapper with `DistributedKVCache`'s `put` path: sender
   `put(block_id, hash)` then wrapper fetch; receiver verifies
   `chain_hash == expected_hash` before installing.

**Documented but not in P40:**

- `crates/dist/tests/kv_block_transfer.rs` does **not** need
  changes; the existing tests cover the protocol layer with
  `MockBlockDataSource`. The wrapper is additive; the integration
  tests in `crates/model/tests/` are the wrapper-specific
  coverage.

### 5.5 Performance notes

Today's wrapper does a **sync CPU read** of all layers of the
target block. For a typical Qwen3-7B block at F32:

- 32 layers × (num_heads × BLOCK_SIZE × head_dim × 4) bytes for K
  + same for V
- = 32 × (32 × 16 × 128 × 4) × 2 ≈ 16 MiB CPU read per fetch

This is acceptable for the protocol's existing 64 MiB message
limit (well under headroom). A future GPU-direct path can wrap
`tensor.to_device(&CpuDevice)` differently; P40 keeps the sync
read for the same reason OPS-31d's `MockBlockDataSource` is sync.

`tokio::task::block_in_place` is used (not `spawn_blocking`)
because:

- The gRPC handler is already on a `tonic`-managed thread pool;
  `spawn_blocking` would push the work to the blocking pool and
  back, doubling context-switch overhead.
- `block_in_place` keeps the work on the current thread but
  signals to tokio that this thread is otherwise idle, allowing
  the runtime to schedule other tasks onto it. This is the
  idiomatic pattern for "small CPU-bound section inside an async
  handler".

If profiling later shows this is a contention point, the fix is
a `spawn_blocking` swap — that's a one-line change in
`fetch_block`, deferred until measured.

## 6. Risks & mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| `PagedKvCache` private-field changes break the wrapper's read path | Low | Medium | The four new `pub(crate)` accessors are **thin pass-throughs** to existing fields/methods; `cargo check --all-features` + the new unit tests catch regressions |
| Wrapper silently mishandles the `quantized` case | Low | High | Explicit unit test `wrapper_fetch_block_handles_quantized_layer` + dedicated integration test `wrapper_round_trip_with_quantization` (sender writes quantized data, receiver sees f32 bytes) |
| Wrapper compiled even when `multi-node` is off → unused-code warnings or dead-code | Low | Low | Entire wrapper module gated by `#[cfg(feature = "multi-node")]`; `cargo build` (default) excludes it; `cargo build --workspace --all-features` includes it |
| `block_hashes` lookup misses a block because it's only written to non-zero layers | Low | High | Per `PagedKvCache::write_kv`, every write goes to all layers symmetrically. The wrapper's `has_block` uses layer 0 as the witness; a unit test pins the invariant |
| Byte-layout mismatch (sender format ≠ receiver format) | Low | High | The wrapper's layout is documented in its module-level docstring + `read_block_bytes`'s comment. Receiver-side `write_kv_batch` already expects f32 with shape `[1, num_heads, block_size, head_dim]` per layer; the wrapper produces exactly that, flattened |
| `usize::try_from(u64)` panic on 32-bit platforms | Negligible | Low | All `vllm-lite` deployments target 64-bit; explicit `try_from` + `FetchError::NotFound` mapping gives a clean error path anyway |
| Real-world block size exceeds `MAX_BLOCK_TRANSFER_BYTES` | Low | High | Wrapper produces per-layer-per-block bytes; OPS-31d's 64 MiB symmetric limit covers all realistic blocks (Qwen3-7B F32 ≈14 MiB/block; 4× headroom). Out-of-bound blocks return `FetchError::Transport` at the gRPC layer — already handled by OPS-31d |

## 7. Success criteria

- [ ] `cargo build -p vllm-model --features multi-node` is green
- [ ] `cargo build --workspace --all-features` is green
- [ ] `cargo build -p vllm-model` (default features) is green —
      the wrapper module is excluded
- [ ] `cargo clippy -p vllm-model --all-targets --features multi-node -- -D clippy::correctness -D clippy::suspicious -D clippy::perf` is green
- [ ] `cargo fmt --all --check` passes
- [ ] `cargo nextest run -p vllm-model --all-features --no-fail-fast`
      passes (existing tests + new wrapper tests + new integration
      tests)
- [ ] `cargo nextest run --workspace --all-features --no-fail-fast`
      passes (no regression in the 1,700+ existing tests)
- [ ] `bash .planning/phase-12e/check-public-api.sh` exits 0 (no
      public API delta for default features; multi-node API gains
      one new `pub` type `PagedKvCacheWrapper` + `pub` re-export)
- [ ] `OPERATIONS.md` §"Multi-Node (Experimental)" updated: "What
      works" adds the wrapper bullet; "What is not" shrinks by
      the wrapper item
- [ ] `CHANGELOG.md` has a P40 entry mirroring the P37/P38/P39
      style
- [ ] `.planning/v31.0-MASTER-PLAN.md` P40 row added with the
      deliverable + a forward-pointer to P41+ for engine plumbing
- [ ] Public-API delta: 1 new public type (`PagedKvCacheWrapper`)
      gated behind `multi-node` feature — zero default-features
      impact
- [ ] ADR-020 §"Status" line bumped from "Accepted" → "Accepted
      (P40 ships the wrapper; engine plumbing deferred to P41+)"

## 8. Out-of-scope follow-ups (P41+ candidates)

These remain **explicitly deferred** after P40 lands:

- **P41: Engine-level plumbing** — `Arc<PagedKvCache>` →
  `MemoryManager::block_data_source`, `EngineBuilder::with_paged_kv_cache`,
  `crates/server/src/main.rs` wiring. Closes the second half of
  OPS-32a. Real production benefit (multi-node now works
  end-to-end without manual `ServerState` plumbing).
- **OPS-32b: Streaming RPCs** — `StreamKVBlock` for blocks
  exceeding 64 MiB. Tonic's default `Stream` machinery; one proto
  addition + one client method.
- **OPS-32c: Wire compression** — fp16/int8 over the wire; the
  existing `num_tokens` reservation in
  `TransferKvBlockResponse` becomes a `wire_format` enum.
- **OPS-32d: Owner-based routing** — `CacheConfig::peer_node_ids`
  + `compute_owner_nodes`; replaces fan-out for clusters > 4
  nodes.
- **OPS-32e: Failure recovery** — `HashMismatch` retry with
  backoff + transient-error class. Today: hard error (matches
  OPS-31d stance).

After P40 + P41, the multi-node work has its full protocol +
production-shape wrapper + engine wiring. The remaining items
(32b-32e) are bandwidth/operability improvements, not
correctness blockers.

## 9. Decision log

| Decision | Rationale | Date |
|----------|-----------|------|
| Wrapper lives in `vllm-model`, not `vllm-dist` | `vllm-dist` cannot depend on `vllm-model` per ADR-008 layering; `PagedKvCache` is private to `vllm-model`; wrapper needs private-field access | 2026-07-22 |
| Entire wrapper module gated by `#[cfg(feature = "multi-node")]` | Mirrors OPS-31d's `vllm-dist` gating (ADR-008); default builds remain free of `BlockDataSource` machinery | 2026-07-22 |
| Sync CPU read via `block_in_place`, not `spawn_blocking` | Smaller context-switch overhead for the ≤16 MiB block sizes; matches OPS-31d's `MockBlockDataSource` pattern | 2026-07-22 |
| Dequantize before serializing when `quantized=true` | Receiver-side `write_kv_batch` expects f32; dequantizing at the source matches the contract | 2026-07-22 |
| `has_block` uses layer 0 as the canonical witness | Every `PagedKvCache::write_kv` writes to all layers symmetrically; layer 0 is a sound existence check | 2026-07-22 |
| Engine plumbing deferred to P41+ | Bounded scope; wrapper has standalone value; splits OPS-32a at its natural seam | 2026-07-22 |
| No `with_paged_kv_cache(...)` ergonomic on `DistributedKVCache` | `with_block_data_source(Arc::new(PagedKvCacheWrapper::new(cache)))` is already 1 line; a sugar method adds API surface for zero clarity gain. Wrapper is the unit of composition | 2026-07-22 |

## 10. See also

- OPS-31d phase plan: `.planning/phase-19/ops-31d-kv-block-transfer.md`
- ADR-020: `docs/adr/ADR-020-multi-node-kv-block-transfer.md`
- P39 spec (last completed): `docs/superpowers/specs/2026-07-21-p39-n-parallel-engine-wire-through-design.md`
- v31.0 master plan: `.planning/v31.0-MASTER-PLAN.md`
- OPS-31d §"Open: Engine integration (v32+)" — the deferred gap this design closes the first half of
- OPERATIONS.md §"Multi-Node (Experimental)" — the operator-facing doc updated by this work
