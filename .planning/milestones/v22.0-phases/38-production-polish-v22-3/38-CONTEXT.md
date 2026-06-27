# Phase 38: Production Polish (v22.3) - Context

**Gathered:** 2026-06-27
**Status:** Ready for planning
**Mode:** Smart discuss auto-accepted (infrastructure phase)

<domain>

## Phase Boundary

Eliminate production ergonomics smells and apply small perf wins:

1. **RFU-05**: Migrate scheduler/engine `std::sync::Mutex` → `parking_lot::Mutex`
   (24 sites per the original audit, actual count smaller after v20/v21
   cleanups — see Code Insights).
2. **OPS-01**: Decide and document `crates/core/src/engine/speculative.rs`
   mock usage fate. **Note**: as of v20/v21, `speculative.rs` has been
   refactored into `crates/core/src/engine/spec_dispatch/` — the
   original file no longer exists. OPS-01 is therefore satisfied by
   documenting this resolution.
3. **PERF-01**: `MlaKvCache::write_compressed` writes incrementally
   using `Tensor::slice_assign` or equivalent — no full-cache
   materialization.
4. **PERF-02**: Architecture detection uses `eq_ignore_ascii_case` instead
   of `model_type.to_lowercase()` — zero per-load `String` allocations in
   the arch detection path.
5. **PERF-03**: Lazy initialization in `crates/model/src/arch/registry.rs`
   uses `std::sync::LazyLock` (Rust 1.80+) — no new `once_cell::sync::Lazy`
   usage; existing `once_cell` usage migrated.
6. **DOC-01**: `cargo doc --workspace --no-deps` produces zero broken-link
   warnings — carry-over from OPS-03 confirmed closed.

FINAL-01: All 1179+ tests remain green post-polish.

</domain>

<decisions>

## Implementation Decisions

### the agent's Discretion

All implementation choices are at the agent's discretion — pure polish
phase. Specific guidance:

- **RFU-05**: Migrate `std::sync::Mutex` fields in scheduler code only.
  The `Arc<Mutex<Box<dyn ModelBackend>>>` pattern in engine.rs / spec_dispatch
  (used as the public lock type for `dyn ModelBackend` trait objects) is
  part of the vllm-traits contract and cannot change without breaking
  the public API. Touch only scheduler module mutexes where the lock
  type is internal.
- **OPS-01**: Document in the SUMMARY that `speculative.rs` no longer
  exists (refactored into `engine/spec_dispatch/` during v18.0/v20.0
  cleanup). Add a module-level doc comment in `engine/spec_dispatch/mod.rs`
  explaining the refactor history for future readers.
- **PERF-01**: Touch only the `write_compressed` implementation. The
  MLA KV cache is feature-gated; verify whether it is even compiled
  in default builds before investing significant scope.
- **PERF-02**: Apply to all `model_type.to_lowercase()` calls in arch
  detection (gemma4, mistral, mixtral, etc.). Cases where the result
  is then `.contains(...)` or `.starts_with(...)` need helper utilities.
- **PERF-03**: Replace `once_cell::sync::Lazy` with `std::sync::LazyLock`
  in `crates/model/src/arch/registry.rs`. Remove the unused `once_cell`
  dependency from `crates/model/Cargo.toml` if no other usage remains.

### Constraints (apply to all sub-plans)

- No new features. Strictly tech-debt execution.
- All 1179+ tests remain green post-polish.
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`
  remains clean after each sub-plan.
- Backward-compat: no public API changes; trait-object lock types
  remain `std::sync::Mutex<Box<dyn ModelBackend>>` to avoid breaking
  external callers.

</decisions>

<code_context>

## Existing Code Insights

### Reusable Assets

- **`PredictiveBatcher`** (`crates/core/src/scheduler/predictive_batching.rs:127`)
  — 3 `std::sync::Mutex` fields (`request_history`, `current_pattern`,
  `last_batch_time`). Each guarded by
  `.lock().unwrap_or_else(|e| e.into_inner())` — 7 such call sites.
- **`ArchitectureRegistry::ARCHITECTURE_REGISTRY`** —
  `once_cell::sync::Lazy<ArchitectureRegistry>` at
  `crates/model/src/arch/registry.rs:77`.
- **`MlaKvCache::write_compressed`** — location TBD; needs feature
  audit to confirm whether compiled in default build.
- **PERF-02 arch detection sites** (from grep):
  - `crates/model/src/gemma4/arch.rs:44`
  - `crates/model/src/mistral/arch.rs:42`
  - `crates/model/src/mixtral/arch.rs:42`
  - `crates/model/src/mistral_small/arch.rs:132-134` (3 sites, contains checks)
  - `crates/model/src/phi4/arch.rs:105` (starts_with)
  - others with `.to_lowercase().as_str()` match arms

### Established Patterns

- All public types use builder patterns; no new public API surface
  expected from this phase.
- Doc comments use `///` style with `PERF-XX:` and `RFU-XX:` tags so
  reviewers can grep audit-trail entries back to the original finding.

</code_context>

<specifics>

## Specific Ideas

- **parking_lot migration**: `use parking_lot::Mutex;` at top of
  `predictive_batching.rs`. Field types become `Mutex<T>`. Constructor
  uses `Mutex::new(...)`. All `.lock().unwrap_or_else(|e| e.into_inner())`
  calls become `.lock()` (parking_lot's `lock()` returns a `MutexGuard`
  directly without `Result`).
- **eq_ignore_ascii_case**: Replace `.to_lowercase() == "X"` with
  `.eq_ignore_ascii_case("X")`. For `matches!(x.to_lowercase().as_str(),
  "A" | "B")` use `x.eq_ignore_ascii_case("A") ||
  x.eq_ignore_ascii_case("B")` (or a small helper).
- **LazyLock**: `use std::sync::LazyLock;` instead of
  `use once_cell::sync::Lazy;`. `Lazy::new(...)` → `LazyLock::new(...)`.
  Public API of `ARCHITECTURE_REGISTRY` unchanged.

</specifics>

<deferred>

## Deferred Ideas

- **MlaKvCache full incremental rewrite** — scope depends on whether
  the cache is in default builds. If it is feature-gated and not
  commonly exercised, leaving it for a follow-up phase is acceptable.
- **Comprehensive audit-trail cleanup** — old `to_lowercase()` calls in
  `crates/model/src/phi4/arch.rs`, `mistral_small/arch.rs`, etc. — can
  be addressed incrementally in subsequent maintenance cycles.

</deferred>
