# Phase 31: Module Layout Reorganization — PLAN

**Phase:** 31
**Goal:** Reorganize oversized God modules into focused sub-trees so contributors can navigate and modify the codebase without cross-cutting concerns; error types live at their semantic boundaries.

**Requirements:** ML-01, ML-02, ML-03, ML-04, ML-05, ML-06, ML-07, ML-08, ML-09

## Success Criteria

1. `crates/core/src/speculative/draft_registry.rs` (938 LOC) decomposed into `registry/{loader,lifecycle,errors,model}.rs` + thin `mod.rs`; each leaf <300 LOC; all tests pass — **ML-01**
2. `engine.rs` (1059 LOC) and `engine/speculative.rs` (882 LOC) unified into `engine/speculative/` sub-tree; no duplicate re-exports — **ML-02**
3. `qwen3_config.rs` (529 LOC) moved to `qwen3/config.rs`; `components/attention/mod.rs` (470 LOC) utilities extracted to `components/attention/util.rs` — **ML-03, ML-04**
4. `TensorParallelError` lives in `vllm-dist::error` with re-export from `vllm-traits`; canonical path documented — **ML-06**
5. `vllm-testing` decision made (split or document infeasibility); `test_fixtures.rs` migrated into vllm-testing; unused exports verified or removed — **ML-05, ML-07, ML-08, ML-09**

## Plans

### 31-01: Split `draft_registry.rs` into `registry/` sub-tree — ML-01

**Tasks:**
1. Read `crates/core/src/speculative/draft_registry.rs` to identify concerns:
   - Type definitions (DraftId, DraftSpec, DraftState, LoadedDraft, DraftModelRegistry)
   - Loading logic (registry construction, model loading)
   - Lifecycle (refcount, auto-unload, registration tracking)
   - Error types (DraftRegistryError)
2. Create `crates/core/src/speculative/registry/` directory:
   - `mod.rs` — DraftModelRegistry facade + re-exports (thin shim)
   - `types.rs` — DraftId, DraftSpec, DraftState, LoadedDraft
   - `loader.rs` — Loading/registration logic
   - `lifecycle.rs` — Refcount + auto-unload
   - `errors.rs` — DraftRegistryError
3. Move code into appropriate files
4. Update `speculative/mod.rs` to re-export from `registry/`
5. Add `pub mod registry;` declaration
6. Verify: `cargo build -p vllm-core` compiles; `cargo test -p vllm-core` passes

**LOC target:** Each leaf file <300 LOC; `mod.rs` <50 LOC.

**Backward-compat:** Existing `use crate::speculative::draft_registry::...` paths continue to work via re-exports in `speculative/draft_registry.rs` (kept as compat shim with `#[deprecated]` after migration).

### 31-02: Collapse `engine.rs` + `engine/speculative.rs` into `engine/speculative/` sub-tree — ML-02

**Tasks:**
1. Analyze current structure:
   - `crates/core/src/engine.rs` (1059 LOC, top-level engine)
   - `crates/core/src/engine/speculative.rs` (882 LOC, single file inside engine subdir)
2. Move existing `engine/speculative.rs` content into `engine/speculative/` subdir:
   - `crates/core/src/engine/speculative/mod.rs` (current single-file content)
3. Audit `engine.rs` for speculative-related code; consider whether additional speculative helpers should move into `speculative/`
4. Verify no duplicate re-exports between engine.rs and engine/speculative/
5. Verify: `cargo build -p vllm-core` compiles; tests pass

**LOC target:** Engine stays under ~800 LOC after split; speculative/ sub-tree organized.

**Note:** This is largely a directory restructure (file → module). The actual split boundaries depend on what's in engine.rs that's speculative-specific.

### 31-03: Move `qwen3_config.rs` → `qwen3/config.rs`; extract `attention/mod.rs` utilities → `attention/util.rs` — ML-03, ML-04

**Tasks:**
1. `qwen3_config.rs` move:
   - Move file from `crates/model/src/qwen3_config.rs` to `crates/model/src/qwen3/config.rs`
   - Add `pub mod config;` to `qwen3/mod.rs`
   - Add `pub use config::*;` to `qwen3/mod.rs` for backward-compat
   - Update all `use crate::qwen3_config` references to `use crate::qwen3::config`
   - Verify: `cargo build -p vllm-model` compiles; tests pass
2. `attention/mod.rs` extraction:
   - Read `crates/model/src/components/attention/mod.rs` (470 LOC)
   - Identify utilities (helper functions, not re-exports)
   - Move utilities to `crates/model/src/components/attention/util.rs`
   - Keep `mod.rs` focused on re-exports + module declarations
   - Verify: `cargo build -p vllm-model` compiles; tests pass

**LOC target:** `attention/mod.rs` reduced to <200 LOC; utilities live in `util.rs`.

### 31-04: Decide vllm-testing lemon pair (split or document) — ML-05

**Tasks:**
1. Analyze `vllm-testing` structure:
   - `crates/testing/src/{lib,harness,request_factory,slow_model}.rs` + submodules
   - Tests consume: `vllm_testing::{TestHarness, RequestFactory, SlowModel, builders, fixtures, mocks, utils}`
2. Decision criteria for split (vllm-testkit + vllm-harness):
   - Whether the harness/slow-model infrastructure is conceptually distinct from test utilities
   - Whether splitting reduces compile times for unit-test-only consumers
   - Whether the split adds maintenance burden
3. **Pragmatic decision:** Document why split is infeasible (single 220 LOC harness + 157 LOC slow_model + 287 LOC request_factory don't justify a separate crate, would increase coupling, no compile-time benefit since all callers need the same test fixtures)
4. Update `crates/testing/src/lib.rs` module docs to include the rationale
5. Add `//! Lemon pair split decision: ...` section explaining the rationale

**Decision documentation:** Inline `//!` module doc explaining the architectural choice and criteria for re-evaluation.

### 31-05: Move `TensorParallelError` to `vllm-dist::error`; re-export from `vllm-traits` — ML-06

**Tasks:**
1. Create `crates/dist/src/error.rs`:
   - Re-export `pub use vllm_traits::TensorParallelError;`
   - Add module docs explaining: canonical home is vllm-dist semantically; technical definition lives in vllm-traits due to dependency direction (`vllm-dist → vllm-traits`); this module provides the canonical import path.
2. Update `crates/dist/src/tensor_parallel/mod.rs`:
   - Replace `pub use vllm_traits::TensorParallelError;` with `pub use crate::error::TensorParallelError;`
3. Add `pub mod error;` to `crates/dist/src/lib.rs`
4. Update `crates/dist/src/lib.rs` re-export to point at `error::TensorParallelError`
5. Verify: `cargo build -p vllm-dist --features multi-node` compiles; default build still excludes dist
6. Add documentation note in `crates/traits/src/types.rs` explaining why the technical definition lives there

**Canonical path:** `vllm_dist::error::TensorParallelError`
**Backward-compat:** `vllm_traits::TensorParallelError` continues to work (definition unchanged).

### 31-06: Move `crates/server/src/test_fixtures.rs` into `vllm-testing`; verify exports — ML-07, ML-08, ML-09

**Tasks:**
1. Analyze `crates/server/src/test_fixtures.rs` (64 LOC):
   - Contains: `api_state`, `api_state_with_mock_engine`, `spawn_mock_engine`
2. Move logic to `crates/testing/src/server_fixtures.rs` (new module):
   - Add `pub mod server_fixtures;` to `vllm-testing/src/lib.rs`
   - Add `pub use server_fixtures::*;` to prelude (or expose as needed)
3. Migration options:
   - **Option A:** Server tests directly use `vllm_testing::server_fixtures::*` (cleaner)
   - **Option B:** Keep `test_fixtures.rs` as a thin re-export shim
4. Pick Option A for cleanliness
5. Update server test files:
   - `crates/server/tests/models_handler_test.rs`
   - `crates/server/tests/chat_integration_test.rs`
   - Any other test files
6. Update internal server files using `crate::test_fixtures`:
   - `crates/server/src/openai/batch/handler.rs`
   - `crates/server/src/openai/embeddings.rs`
   - `crates/server/src/openai/completions.rs`
7. Delete `crates/server/src/test_fixtures.rs`
8. Verify unused exports in `vllm-testing`:
   - `SlowModel`, `TestHarness`, `BatchBuilder`, etc.
   - Document which are used by tests vs which are exports
   - Remove exports that are genuinely unused

**Verify:** `cargo test --workspace` passes; all test imports resolve.

## Execution Order

1. **31-04 first** (vllm-testing decision doc) — pure docs, no code changes
2. **31-05** (TensorParallelError) — small, contained, validates the multi-node feature-gate
3. **31-03** (qwen3_config + attention util) — moderate mechanical moves
4. **31-01** (draft_registry split) — larger refactor
5. **31-06** (test_fixtures migration) — moderate refactor across crates
6. **31-02** (engine speculative unify) — last (depends on all prior for stable signatures)

## Verification

After each plan completes:
- `cargo build --workspace --all-features` clean
- `cargo test --workspace` ≥1144 tests pass (no regression)
- `cargo clippy --workspace --all-targets -- -D warnings` clean
- `cargo fmt --all --check` clean
- Doc coverage does not regress (97.8% baseline)

## Risks

- **Circular import risk** in 31-05 if not careful — use `pub use` re-export, not direct reference
- **Re-export churn** in 31-01, 31-02, 31-03, 31-06 — every move triggers re-export shims
- **Test count regression** — refactors must preserve all 1144 tests
- **vllm-dist feature-gate breakage** — 31-05 must verify multi-node feature still compiles

## Rollback

Each plan is a discrete commit. Roll back any plan that breaks invariants via `git revert <sha>`.
