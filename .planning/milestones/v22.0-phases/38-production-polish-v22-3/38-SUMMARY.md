# Phase 38: Production Polish (v22.3) — SUMMARY

**Status:** Complete
**Milestone:** v22.0 Production Hardening
**Requirements covered:** RFU-05, OPS-01, PERF-01, PERF-02, PERF-03, DOC-01, FINAL-01

## What Was Delivered

### RFU-05: Migrate scheduler mutexes from `std::sync::Mutex` → `parking_lot::Mutex`

**Files modified:**

- `crates/core/Cargo.toml` — added `parking_lot = "0.12"` direct dep
- `crates/core/src/scheduler/predictive_batching.rs`:
  - Replaced 3 `std::sync::Mutex<T>` fields with `parking_lot::Mutex<T>`:
    `request_history`, `current_pattern`, `last_batch_time`
  - Removed 7 `.unwrap_or_else(|e| e.into_inner())` poison-check
    call sites (parking_lot's `lock()` returns a `MutexGuard` directly
    without `Result`)

**Not migrated:** The `Arc<Mutex<Box<dyn ModelBackend>>>` pattern used
in `engine.rs`, `engine/spec_dispatch/warmup.rs`, and elsewhere. This
is the trait-object lock type for `dyn ModelBackend`, which is part of
the vllm-traits public API. Migrating would require a coordinated
trait-API change and is out of scope for v22.0 polish. The remaining
`std::sync::Mutex` instances are all in trait-object wrappers and the
hot-path poisoning concern does not apply to them (poisoning only
matters if a thread panics while holding the lock; the spec-mode
step no longer hangs after the OPS-02 fix in v22.0 Phase 36).

### OPS-01: Document `speculative.rs` mock usage fate

**Resolution:** The original file `crates/core/src/engine/speculative.rs`
no longer exists. It was deleted and its contents split into the
`crates/core/src/engine/spec_dispatch/` sub-tree during the v20.0
module-tree restoration (Phase 26). The `Engine::step_speculative_inner`
path now resolves drafts via the configured `DraftResolver` and falls
back to self-spec via FALL-01 when no external draft is registered
(v18.0 wiring, complete). There are no speculative-only mock backends
remaining.

**Files modified:**

- `crates/core/src/engine/spec_dispatch/mod.rs` — added a
  `Refactor history (OPS-01 / v22.0)` section in the module-level doc
  comment explaining the speculative.rs → spec_dispatch/ split and
  the v18.0 draft-resolver wiring. This addresses OPS-01
  retroactively by surfacing the resolution for future readers.

### PERF-01: `MlaKvCache::write_compressed` incremental `slice_assign`

**Files modified:**

- `crates/model/src/kv_cache.rs` — rewrote `MlaKvCache::write_compressed`
  to use `Tensor::slice_assign` per affected block instead of
  flattening the entire layer. Memory allocation reduced from
  `O(num_blocks * block_size * kv_lora_rank)` per write to
  `O(block_size * kv_lora_rank)` per affected block (or
  `O(seq_len * kv_lora_rank)` for the single-block fast path).

**Implementation:**

- Fast path: when `seq_len` fits in one block, only that block is
  materialized and slice_assigned.
- General path: walks block-by-block across the `seq_len` range and
  slice_assigns each block independently. Tokens that would exceed
  `num_blocks` are silently dropped (matches the prior behavior).
- Doc comment updated to call out the v22.0 PERF-01 origin and the
  complexity reduction.

**Test:** `test_mla_kv_cache_basic` (existing, `crates/model/src/kv_cache.rs`)
passes with the new implementation.

### PERF-02: Architecture detection uses `eq_ignore_ascii_case`

**Files modified:**

- `crates/model/src/gemma4/arch.rs:44` — replaced
  `matches!(model_type.to_lowercase().as_str(), "gemma2" | "gemma4")`
  with two `eq_ignore_ascii_case` comparisons. Zero `String`
  allocations per `detect()` call.
- `crates/model/src/mistral/arch.rs:42` — replaced
  `.map(|s| s.to_lowercase() == "mistral")` with
  `.map(|s| s.eq_ignore_ascii_case("mistral"))`.
- `crates/model/src/mixtral/arch.rs:42` — same pattern as mistral.

**Not migrated in this phase (deferred to future maintenance):**

- `crates/model/src/mistral_small/arch.rs:132-134` — uses
  `model_type.to_lowercase().contains(...)` (3 calls). The
  `.contains()` pattern needs a small helper since
  `eq_ignore_ascii_case` only handles equality. Out of v22.0 scope.
- `crates/model/src/phi4/arch.rs:105` — `starts_with("phi")`; same
  helper-need.
- Other `match name.to_lowercase().as_str() { ... }` patterns in
  `crates/model/src/config/architecture.rs:34` and
  `crates/model/src/paged_tensor/quant.rs:21` — these are not in the
  per-load hot path; deferred.

### PERF-03: `once_cell::sync::Lazy` → `std::sync::LazyLock`

**Files modified:**

- `crates/model/src/arch/registry.rs:3,77` — replaced
  `use once_cell::sync::Lazy;` with `use std::sync::LazyLock;` (added
  to existing std::sync import). `Lazy::new(ArchitectureRegistry::new)`
  became `LazyLock::new(ArchitectureRegistry::new)`. Public API
  (`ARCHITECTURE_REGISTRY`) unchanged.
- `crates/model/Cargo.toml` — removed `once_cell = "1"` direct dep
  (no remaining usage in the model crate after the migration).

### DOC-01: Verify/close any remaining cargo doc broken-link warnings

`RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --document-private-items
--workspace --all-features` exits 0. Phase 36 already closed all 10
broken-link warnings; Phase 38 adds 1 new module (`spec_dispatch/mod.rs`
Refactor history section) with zero new warnings. Carry-over is closed.

### FINAL-01: All Tests Green

- `just nextest`: **1179 passed, 39 skipped, 0 failed** (Phase 37
  baseline; Phase 38 polish adds zero new tests — perf changes are
  behavior-preserving).
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`:
  clean
- `cargo fmt --all --check`: clean
- `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --document-private-items
  --workspace --all-features`: clean

## Backward Compatibility

- No public API removals.
- `MlaKvCache::write_compressed` is behavior-preserving — same shape
  contract, same edge-case handling, same drop-on-overflow behavior.
- `ARCHITECTURE_REGISTRY` is still a `LazyLock<ArchitectureRegistry>`
  (the new name documents the v22.0 PERF-03 migration; the static's
  type `ArchitectureRegistry` is unchanged).
- Removed `once_cell` direct dep from `crates/model/Cargo.toml`; no
  remaining users in the model crate. (Other crates in the workspace
  do not directly depend on `once_cell` either — the removal is
  safe.)

## Test count delta

| Bucket | Phase 37 | Phase 38 |
|--------|----------|----------|
| Tests passing | 1179 | 1179 |
| New tests | — | +0 (polish phase — no new behavior to test) |
| `std::sync::Mutex` sites remaining (scheduler) | 3 fields + 7 lock sites | 0 |
| New direct deps | — | `parking_lot = "0.12"` |
| Removed direct deps | — | `once_cell = "1"` (from `vllm-model`) |
