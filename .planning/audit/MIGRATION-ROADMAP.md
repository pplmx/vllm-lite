# v20.0+ Migration Roadmap — vllm-lite

**Generated:** 2026-06-27
**Source:** v19.0 audit synthesis (4 dimensions, 100 raw findings)
**Source documents:**
- `.planning/audit/SYNTHESIS.md` — cross-dimensional root causes
- `.planning/audit/BACKLOG.md` — prioritized findings table

This roadmap is **advisory**. It groups BACKLOG.md items into shippable phases for v20.0+. The user may merge, split, reorder, or drop phases based on priority.

---

## Proposed Phase Structure

### Phase 20.1 — P0 Critical Fixes (Layering + Error + Object-Safety)

**Goal:** Address all 5 P0 findings (architectural violations, error type safety, object safety).

**Estimated effort:** ~22.5 hours (~3 working days)

**Items:**
- ARCH-F-11 (P0) — break `vllm-model → vllm-dist` sibling-tier edge
- ARCH-F-12 (P0) — break `vllm-core → vllm-model` downward edge (cuda-graph)
- API-F-01 (P0) — convert `ModelError` struct → enum
- API-F-03 (P0) — convert `CudaGraphError` to `thiserror`
- API-F-02 (P0) — verify `Architecture`/`FlashAttention` `dyn` compile; introduce associated types where needed

**Approach:**
1. Move `TensorParallelConfig` and `TensorParallelError` to `vllm-traits` (or introduce a `vllm-tp-config` traits helper), then remove `vllm-model → vllm-dist` edge.
2. Extract `BatchCudaGraphExecutor` behind a trait in `vllm-traits`; have `vllm-model` provide the impl; engine invokes via trait. Move the feature gate.
3. Convert `ModelError` to enum with `ShapeMismatch`, `ForwardFailed`, `UnsupportedArchitecture`, etc.
4. Replace `CudaGraphError` manual impl with `#[derive(thiserror::Error)]`.
5. Audit each non-object-safe trait (`Architecture`, `FlashAttention`, `DraftLoader`, `PipelineStage`, `AllReduce`, `QkRotaryEmb`, `FormatLoader`, `Quantization`); introduce associated types or generic-erasure where the trait is used as `dyn`.

**Dependencies:** None (v19.0 baseline).

**Risk:** **High** — touches architectural boundaries. Needs:
- Full test re-run.
- Compile-time verification of all `dyn` usage sites (especially `Architecture` × 12, `FlashAttention` × 2).
- Backward-compat check for `ModelError` callers.

**Deliverable:** No P0 findings remain.

---

### Phase 20.2 — Module Tree Restoration + Dist Decision

**Goal:** Wire orphan modules into the source tree; decide fate of `vllm-dist`'s dead public API.

**Estimated effort:** ~12 hours (~1.5 working days)

**Items:**
- NAME-F-01 (P1) — wire `kv_cache_fp8.rs` (or delete if not needed)
- NAME-F-02 (P1) — wire `server/src/debug.rs` (or delete if not needed)
- NAME-F-04 (P1) — move/convert 3 test files in `model/src/`
- NAME-F-03 (P1) — rename `engine_v18_wiring.rs` → `engine_wiring.rs`
- ARCH-F-17 (P1) — decide fate of `distributed_kv`/`grpc`/`pipeline` modules (feature-gate or remove)

**Approach:**
1. Add `pub mod kv_cache_fp8;` to `components/mod.rs`. Verify it compiles. Document in `kv_cache_fp8.rs` (overlaps with DOCS-F-20).
2. Add `pub mod debug;` to `server/src/lib.rs`. Verify it compiles. Document.
3. Move `qwen3/model_tests.rs`, `qwen3_5/model_tests.rs`, `qwen3_5/speculative_tests.rs` to `crates/model/tests/` (or convert to `#[cfg(test)] mod tests {}`).
4. Rename file + update references.
5. For `vllm-dist`:
   - **Option A (recommended):** Feature-gate `distributed_kv`, `grpc`, `pipeline` behind `multi-node` feature (default off). Keep `TensorParallelConfig` public.
   - **Option B:** Remove the dead modules entirely; move `TensorParallelConfig` to `vllm-traits`.
   - Decision should be made in coordination with v20.5 ADRs.

**Dependencies:** None.

**Risk:** **Low** (mostly mechanical). The orphan module fixes may surface latent compile errors that the current dead-code state hides — this is a feature, not a bug.

**Deliverable:** No orphan modules. `vllm-dist` has clear scope.

---

### Phase 20.3 — Error Handling Standardization + God-Module Decomposition

**Goal:** Define project-wide error conventions; migrate existing errors; split `engine.rs` God module.

**Estimated effort:** ~64 hours (~8 working days)

**Items (P1):**
- API-F-04 — `From<PoisonError<T>>` for `EngineError`; replace 25+ mutex `.expect()`
- API-F-05 — introduce error context propagation (`anyhow::Context` or `#[track_caller]`)
- API-F-06 — add 4 variants to `EngineError` (`Timeout`, `Cancelled`, `ResourceExhausted`, `BackendUnavailable`)
- API-F-07 — replace 10 `Result<_, String>` sites with typed errors
- API-F-08 — use `#[source]` / `#[from]` for `EngineError::ModelError`
- API-F-09 — add `#[async_trait]` to `MetricsExporter`

**Items (P2):**
- API-F-14 — add `#[source]` to `DraftRegistryError::LoadFailed(String)`
- API-F-15 — replace 2 `Box<dyn Error>` in `model` lib
- API-F-16 — replace `Mutex::lock().unwrap()` in `predictive_batching.rs`
- API-F-18 — add `dyn Trait` compile-only tests per trait
- API-F-20 — split `FallbackStrategy` into sync + async
- API-F-21 — add `From<candle_core::Error>` for `EngineError`
- API-F-24 — carry `request_id`/`seq_id` in error context

**Items (P2 ARCH):**
- ARCH-F-04 — split `engine.rs` (1,038 LOC)
- ARCH-F-05 — split `draft_registry.rs` (929 LOC)
- ARCH-F-06 — collapse `engine.rs` + `engine/speculative.rs` into sub-tree
- ARCH-F-07 — move `qwen3_config.rs` to `qwen3/config.rs`
- ARCH-F-08 — split `attention/mod.rs` utilities
- ARCH-F-10 — split `vllm-testing` into `testkit` + `harness`
- ARCH-F-13 — move `TensorParallelError` to `vllm-dist`
- ARCH-F-14 — move `server/src/test_fixtures.rs` into `vllm-testing`
- ARCH-F-16 — migrate server tests to use `vllm-testing`
- ARCH-F-19 — verify or remove unused `vllm-testing` exports

**Approach:**
1. **Error conventions:** Document in `AGENTS.md`:
   - Each crate exports a `pub type Result<T> = std::result::Result<T, $ErrorEnum>`.
   - Use `thiserror::Error` derive.
   - Use `#[source]` for chain preservation.
   - Use `#[from]` for trivial conversions.
   - No `Result<_, String>` in production code.
2. **Migration:** Convert error types crate by crate (server → core → model → dist).
3. **Engine split:** Move `cuda_graph` glue and `speculative_resolver` glue into `engine/cuda_graph.rs` and `engine/speculative/`.

**Dependencies:** Phase 20.1 (ModelError must be enum first).

**Risk:** **Medium** — touches every crate; large refactor surface. Mitigate by:
- Migrate one crate at a time.
- Keep backward-compat shims (e.g., `From<OldError> for NewError`).
- Run full test suite per crate.

**Deliverable:** All errors typed; engine module < 800 LOC; `vllm-testing` cleanly split.

---

### Phase 20.4 — Documentation Coverage Push (Crates)

**Goal:** Raise doc coverage from 7.6% to ≥60% on `pub` items across all crates.

**Estimated effort:** ~46 hours (~6 working days)

**Items (P1):**
- DOCS-F-14 — `traits` crate (0% → 80%, 14 items)
- DOCS-F-15 — `dist` crate (2.7% → 80%, 36 items)
- DOCS-F-16 — `server` crate (4.9% → 80%, 65 items)
- DOCS-F-17 — `model` crate (8.5% → 80%, 170 items)
- DOCS-F-18 — `core` crate (9.0% → 80%, 99 items)
- DOCS-F-19 — `testing` crate (12.9% → 80%, 12 items)
- DOCS-F-20 — add `//!` module doc to 121 of 232 files
- DOCS-F-01 — workspace doc coverage push

**Approach:**
1. **Priority order** (per docs audit): `server/` → `model/loader/` → `traits/` → `core/{scheduler, error, types, sampling}.rs` → `model/` (architectures, attention, MLP).
2. **Templates:** Use `scheduler/mod.rs` (133-line architecture diagram), `arch/{mod, registry}.rs`, `testing/src/lib.rs` as templates.
3. **No examples** in `///` blocks for v20.4 (added examples risk going stale if APIs change). Phase 20.5+ can add `#[cfg(doc)]`-gated doctests if desired.

**Dependencies:** None.

**Risk:** **Low** (no semantic code change). Risk is that examples drift; mitigated by deferring examples.

**Deliverable:** Doc coverage ≥60% workspace; ≥80% on critical user-facing surfaces.

---

### Phase 20.5 — External Documentation Reconciliation + ADRs

**Goal:** Update README, AGENTS.md, REQUIREMENTS.md to reflect current state; write 6 ADRs for tribal knowledge.

**Estimated effort:** ~17 hours (~2 working days)

**Items (P1):**
- DOCS-F-02 — fix README broken code example
- DOCS-F-03 — update README/AGENTS.md architecture tables (10 architectures)
- DOCS-F-04 — update README/AGENTS.md to claim 6 crates
- DOCS-F-05 — add `/v1/models`, `/debug/*` endpoints to README
- DOCS-F-06 — update README test count claims
- DOCS-F-07 — write 6 ADRs (see below)
- DOCS-F-08 — ADR for KV cache split
- DOCS-F-09 — update README project structure tree
- DOCS-F-10 — expand README feature flags table
- DOCS-F-11 — validate AGENTS.md file:line references
- ARCH-F-03 — update docs to claim 6 crates

**Items (P2):**
- DOCS-F-21 — remove DeepSeek from `REQUIREMENTS.md:53` (or add directory back)
- DOCS-F-22 — ADR for vllm-dist investment/deprecation decision
- DOCS-F-24 — cross-link `.planning/PROJECT.md` "Key Decisions" to ADRs

**ADRs to write:**
1. **ADR-003** — Self-speculation 1/8 layer ratio
2. **ADR-004** — FP8 E4M3 format for KV cache compression
3. **ADR-005** — KV cache split across 3 locations
4. **ADR-006** — `vllm-dist` strategy (continue investing or deprecate)
5. **ADR-007** — Speculative decoding architecture (registry → resolver → budget → adaptive)
6. **ADR-008** — `cuda` vs `cuda-graph` feature split

**Approach:**
1. **README refresh:** Walk through every claim (per DOCS-04 audit, 18 specific items).
2. **AGENTS.md refresh:** Walk through every convention claim (per NAME audit, 26 items).
3. **ADR writing:** Use `ADR-001` / `ADR-002` as templates. Format: Context / Decision / Rationale / Consequences / Alternatives Considered.

**Dependencies:** Phase 20.4 (crates documented first).

**Risk:** **Low**.

**Deliverable:** README, AGENTS.md, REQUIREMENTS.md current; 6 ADRs published.

---

### Phase 20.6 — Naming + Polish

**Goal:** Final naming consistency pass; deprecation hygiene; comment cleanup.

**Estimated effort:** ~30 hours (~4 working days)

**Items (P1):**
- NAME-F-05 — rename `data` variable (31× in production)
- NAME-F-06 — rename `EmbeddingData` or document suffix
- NAME-F-07 — formalize verb policy in AGENTS.md
- NAME-F-04 — already addressed in v20.2
- API-F-10 — replace 4 unwraps in `gemma4/attention.rs`
- API-F-11 — replace 3 unwraps in `batch/handler.rs`
- DOCS-F-12 — resolve `quantize/gguf.rs:7` placeholder
- DOCS-F-13 — rephrase "Phase 18.3 will drive this" comments

**Items (P2):**
- NAME-F-08..F-20 — remaining naming (rename, AGENTS.md updates)
- API-F-12 — document builder vs struct-literal convention
- API-F-17 — introduce 22 builders (ergonomics)
- API-F-19 — public re-exports at crate roots
- API-F-22 — `Default` impl for object-safe traits
- DOCS-F-23 — reframe "Phase 5 Wave 4" comment
- API-F-27 — add `MIGRATING.md` (advisory)

**Approach:**
1. **Naming pass:** Mechanical rename using `cargo fix` + manual review.
2. **AGENTS.md consolidation:** All naming conventions from NAME audit (26 items) consolidated into a single "Naming" section.
3. **Comment sweep:** Resolve all stale comments (DOCS-F-12, DOCS-F-13, DOCS-F-23).

**Dependencies:** None (can run in parallel with v20.4/v20.5).

**Risk:** **Low** (mechanical).

**Deliverable:** All P1 findings resolved; remaining P2 items either fixed or deferred to v20.7+.

---

## Phase Dependencies (graph)

```text
v20.1 (P0 fixes)
   │
   ├─→ v20.3 (error handling + God-module split)
   │      └─ depends on v20.1 (ModelError must be enum first)
   │
   └─→ v20.2 (module tree + dist decision)
          └─ independent

v20.4 (doc coverage) ──→ v20.5 (external docs + ADRs)
                              └─ depends on v20.4

v20.6 (naming + polish) ── independent of v20.1-v20.5
```

**Parallel execution possible:** v20.1 → (v20.2 ‖ v20.4) → (v20.3 + v20.5) → v20.6.

---

## Total Estimated Effort

| Phase | Effort | Cumulative | Theme |
|-------|-------:|-----------:|-------|
| 20.1  | 22.5h  | 22.5h      | P0 critical fixes |
| 20.2  | 10h    | 32.5h      | Module tree + dist |
| 20.3  | 64h    | 96.5h      | Errors + God-module |
| 20.4  | 46h    | 142.5h     | Doc coverage (crates) |
| 20.5  | 17h    | 159.5h     | External docs + ADRs |
| 20.6  | 30h    | 189.5h     | Naming + polish |
| **Total** | **~190h** | — | **~5 working weeks (single engineer)** |

Note: P3 findings (13) and P2 items considered deferrable to v20.7+ are not included. See BACKLOG.md for full list.

---

## Risk Summary

| Phase | Risk | Mitigation |
|-------|------|------------|
| v20.1  | **High** — layering changes, error type breaking changes | Full test re-run; compile-only `dyn` verification; back-compat shims |
| v20.2  | **Low** — mechanical rewiring; orphan modules may have hidden compile errors | Treat as a feature (latent errors get fixed) |
| v20.3  | **Medium** — touches every crate | Migrate one crate at a time; per-crate test gates |
| v20.4  | **Low** — no semantic code change | Skip examples in `///` blocks; defer to v20.7+ if drift risk |
| v20.5  | **Low** — markdown + ADR only | Use templates; peer review each doc change |
| v20.6  | **Low** — mechanical rename + comment edits | Use `cargo fix`; manual review for variable names |

---

## Open Questions for User

1. **v20.1 layering fix:** Does `vllm-model → vllm-dist` removal break any planned v20+ multi-node work? (If yes, prefer feature-gating `dist` modules over removing the edge.)
2. **v20.3 error unification:** Single project-wide `Error` enum, or per-crate enums with a shared base trait? Per-crate is less invasive; project-wide is more uniform.
3. **v20.4 doc coverage target:** Accept 60% (achievable in ~46h) or push to 80% (would require ~60h)?
4. **v20.5 ADR-006 (`vllm-dist`):** Continue investing (wire up pipeline/distributed_kv) or deprecate (move `TensorParallelConfig` to `vllm-traits`)?
5. **v20.6 builder introduction (API-F-17):** Introduce 22 builders (mechanical, +8h), or leave `Default::default()` dominant and just document the convention?
6. **Phase ordering:** Keep serial (as proposed) or parallelize independent phases (v20.4 can start during v20.2)?

---

## Acceptance Criteria (per phase)

Each phase should ship with:

1. **All assigned findings resolved** (or explicitly deferred with rationale).
2. **Full test suite green:** `cargo test --workspace --all-features`.
3. **No new clippy warnings:** `cargo clippy --workspace --all-targets -- -D warnings`.
4. **No new TODOs/FIXMEs** introduced (preserve the codebase's excellent hygiene).
5. **Documentation updated:** `AGENTS.md` reflects any new conventions introduced.

---

*End of MIGRATION-ROADMAP.md*
