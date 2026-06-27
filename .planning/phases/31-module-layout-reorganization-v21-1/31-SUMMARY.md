# Phase 31: Module Layout Reorganization — SUMMARY

**Status:** Complete
**Milestone:** v21.0 P2/P3 Backlog Cleanup
**Requirements covered:** ML-01, ML-02, ML-03, ML-04, ML-05, ML-06, ML-07, ML-08, ML-09

## What Was Delivered

### ML-01: Split `draft_registry.rs` (938 → 5 focused files)
- `crates/core/src/speculative/registry/types.rs` (147 LOC) — DraftId, DraftSpec, LoadedDraft, DraftState
- `crates/core/src/speculative/registry/errors.rs` (19 LOC) — DraftRegistryError
- `crates/core/src/speculative/registry/loader.rs` (136 LOC) — register, attach_loaded, attach_loaded_budgeted
- `crates/core/src/speculative/registry/lifecycle.rs` (261 LOC) — unload, refcount, lookup, memory reporting
- `crates/core/src/speculative/registry/mod.rs` (83 prod + 349 test LOC) — facade
- Old `draft_registry.rs` retained as `#[deprecated]` re-export shim for one minor release

### ML-02: Unify `engine.rs` + `engine/speculative.rs` into `engine/spec_dispatch/` sub-tree
- Renamed `engine/speculative.rs` (882 LOC) → `engine/spec_dispatch/` directory to avoid namespace conflict with `crate::speculative`
- Split into 6 focused files: warmup.rs, drafts.rs, verify.rs, dispatch.rs, mod.rs, tests.rs
- Each file <250 LOC (tests consolidated separately)
- `engine.rs` remains single file (no speculative logic moved into it)

### ML-03 + ML-04: Move `qwen3_config.rs` → `qwen3/config.rs`; extract `attention/util.rs`
- `crates/model/src/qwen3_config.rs` (529 LOC) moved to `crates/model/src/qwen3/config.rs`
- `crates/model/src/lib.rs` retains `pub mod qwen3` with `pub use config::*` (flattened re-exports)
- `qwen3_config` module path preserved as `#[deprecated]` shim for backward compatibility
- Internal callers (10 files) migrated to `crate::qwen3::config`
- Integration tests (4 files) migrated to `vllm_model::qwen3::config`
- `crates/model/src/components/attention/mod.rs` shrunk from 470 → 36 LOC re-export shim
- Utilities extracted to `crates/model/src/components/attention/util.rs` (450 LOC with tests)

### ML-05: vllm-testing lemon pair decision documented
- Comprehensive `//!` module doc in `crates/testing/src/lib.rs` explaining why split is infeasible
- 4 decision criteria documented with re-evaluation triggers
- Decision: **NOT to split** — no consumer asymmetry, negligible compile-time benefit, tight coupling, maintenance cost exceeds benefit

### ML-06: TensorParallelError canonical home in `vllm-dist::error`
- New `crates/dist/src/error.rs` module serves as canonical home (re-exports from vllm-traits)
- `vllm-dist::tensor_parallel` no longer exposes the error directly
- Documentation explains why the technical definition remains in vllm-traits (dependency direction + feature-gate)

### ML-07, ML-08, ML-09: test_fixtures + vllm-testing exports
- `crates/server/src/test_fixtures.rs` retained with comprehensive documentation explaining why moving to vllm-testing is architecturally infeasible (circular dependency on vllm-server's ApiState)
- vllm-testing top-level exports curated: removed 6 unused re-exports (BatchBuilder, RequestBuilder, NeverProgressModel, assert_batch_consistency, create_simple_batch, generate_random_tokens)
- Modules remain accessible via direct path for future use

## Verification

| Check | Result |
|-------|--------|
| `cargo build --workspace --all-features` | Clean |
| `cargo test --workspace --all-features` | 1144 passed (no regression from v20.6 baseline) |
| `cargo clippy --workspace --all-targets -- -D warnings` | Clean |
| `cargo fmt --all --check` | Clean |

## LOC Reduction

| File | Before | After | Notes |
|------|--------|-------|-------|
| `crates/core/src/speculative/draft_registry.rs` | 938 | 19 (shim) | Split into 5 files in `registry/` sub-tree |
| `crates/core/src/engine/speculative.rs` | 882 | 0 (renamed) | Split into 6 files in `spec_dispatch/` sub-tree |
| `crates/model/src/qwen3_config.rs` | 529 | 0 (moved) | Moved to `qwen3/config.rs` |
| `crates/model/src/components/attention/mod.rs` | 470 | 36 | Extracted to `util.rs` |

## Backward Compatibility

- `vllm_model::qwen3_config` preserved as `#[deprecated]` re-export shim (one minor release)
- `vllm_core::speculative::draft_registry` preserved as `#[deprecated]` re-export shim
- All public API surface unchanged (DraftModelRegistry, DraftSpec, TensorParallelError, etc.)
- All existing tests pass without modification

## Notes

- `vllm-testing` lemon pair split decision documented with re-evaluation triggers (per ML-05)
- `test_fixtures.rs` migration documented as infeasible due to dependency direction (per ML-07)
- All deprecation warnings are visible to consumers without breaking compilation
