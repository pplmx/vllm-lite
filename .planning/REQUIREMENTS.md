# Requirements: vllm-lite

**Defined:** 2026-06-27 (v20.0); 2026-06-27 (v21.0 added)
**Milestone:** v21.0 P2/P3 Backlog Cleanup
**Core Value:** Fast, memory-efficient LLM inference with continuous batching, paged KV cache, and tensor parallelism — deployed with production-grade ops tooling and security.

## v21.0 Requirements (CURRENT)

Closure of remaining v19.0 audit findings (44 P2 + 13 P3, executed in v21.0). Each requirement maps to one sub-phase (v21.1-v21.5). Source backlog: `.planning/audit/BACKLOG.md`.

### Phase 31 (v21.1): Module Layout Reorganization (~37h)

- [ ] **ML-01**: `draft_registry.rs` (929 LOC) split into `draft_registry/` sub-tree with `loader.rs` + `lifecycle.rs` extracted
- [ ] **ML-02**: `engine.rs` + `engine/speculative.rs` collapsed into `engine/speculative/` sub-tree
- [ ] **ML-03**: `qwen3_config.rs` (487 LOC, top-level) moved into `qwen3/config.rs`
- [ ] **ML-04**: `attention/mod.rs` utilities (180+ LOC) extracted to `attention/util.rs`
- [ ] **ML-05**: `vllm-testing` split into `vllm-testkit` (factories, builders) + `vllm-harness` (SlowModel, TestHarness)
- [ ] **ML-06**: `TensorParallelError` moved to `vllm-dist::error`; re-exported from `vllm-traits`
- [ ] **ML-07**: `crates/server/src/test_fixtures.rs` moved into `vllm-testing`
- [ ] **ML-08**: Server tests migrated to use `vllm-testing` exports instead of local `test_fixtures`
- [ ] **ML-09**: Unused `vllm-testing` exports (`SlowModel`, `TestHarness`, builders) verified or removed

### Phase 32 (v21.2): API Consistency (~21h)

- [ ] **API-01**: Builder vs struct-literal convention documented in AGENTS.md
- [ ] **API-02**: `#[source]` attribute added to `DraftRegistryError::LoadFailed(String)`
- [ ] **API-03**: 2 `Box<dyn Error>` sites in `model` lib replaced with typed errors
- [ ] **API-04**: 8 `Mutex::lock().unwrap()` sites in `predictive_batching.rs` replaced with parking_lot or sync helper
- [ ] **API-05**: 22 builders introduced where only `Default` exists (ergonomics)
- [ ] **API-06**: `dyn Trait` compile-only tests added per trait (extending Phase 25 coverage)
- [ ] **API-07**: Public re-exports of common trait bounds added at crate roots
- [ ] **API-08**: `FallbackStrategy` split into sync + async traits
- [ ] **API-09**: Missing `From<candle_core::Error>` impl added for `EngineError`
- [ ] **API-10**: `Default` impl added for object-safe traits (`DraftVerifier`, `SchedulerObserver`)
- [ ] **API-11**: `request_id` / `seq_id` carried in error context as structured fields

### Phase 33 (v21.3): Naming Consistency (~5h)

- [ ] **NAM-01**: `flash_v3.rs` renamed to `flash_attention_v3.rs` (consistency with V2 naming)
- [ ] **NAM-02**: `*Manager` suffix convention documented in AGENTS.md
- [ ] **NAM-03**: `NodeInfo` evaluated for rename to `NodeSummary` / `NodeMetadata`; either renamed or documented why kept
- [ ] **NAM-04**: `create_*` vs `build_*` policy documented in AGENTS.md
- [ ] **NAM-05**: Async/sync split rationale documented in AGENTS.md
- [ ] **NAM-06**: Tensor-math single-letter variable exemption (`q`/`k`/`v`/`o`/`b`/`c`/`h`/`z`/`d`/`x`/`g`/`r`) documented in AGENTS.md
- [ ] **NAM-07**: Non-tensor single-letter variables in scheduler/sampling code renamed to descriptive names
- [ ] **NAM-08**: Test-file location convention documented in AGENTS.md (per NAME-F-04 finding)

### Phase 34 (v21.4): External Doc Fixes (~3.25h)

- [ ] **DOC-01**: DeepSeek reference removed from `REQUIREMENTS.md:53` OR `deepseek/` directory restored (whichever is correct)
- [ ] **DOC-02**: ADR created: "vllm-dist investment vs deprecation decision" (DOCS-F-22)
- [ ] **DOC-03**: "Phase 5 Wave 4" reference in `qwen3_5/speculative_tests.rs:1` reframed to current phase terminology
- [ ] **DOC-04**: `.planning/PROJECT.md` "Key Decisions" table cross-linked to ADRs

### Phase 35 (v21.5): P3 Actionable + Final Verification (~5h)

- [ ] **P3-01**: Dead `crates/traits/tests/mod.rs` cleaned up (ARCH-F-15)
- [ ] **P3-02**: `gemma4/attention.rs` `Tensor::zeros((1,1), …).unwrap()` non-test replaced with graceful error (API-F-25)
- [ ] **P3-03**: `MIGRATING.md` created with v15.0 → v21.0 versioned changelog (API-F-27)
- [ ] **P3-04**: `HalfOpenRejected(u32)` variant added to `CircuitBreakerError` (API-F-31)
- [ ] **P3-05**: `model` crate production `unwrap()` count re-verified post-v21.0 (API-F-32)
- [ ] **P3-06**: `CudaGraphError` `Clone` derive verified (kept or removed) (API-F-33)
- [ ] **FINAL-01**: All 1144+ tests remain green post-v21.0
- [ ] **FINAL-02**: `cargo clippy --workspace --all-targets --all-features -- -D warnings` clean
- [ ] **FINAL-03**: `cargo fmt --all --check` clean
- [ ] **FINAL-04**: `.planning/PROJECT.md` and `.planning/STATE.md` updated with v21.0 outcomes

## v20.0 Requirements (HISTORICAL — Shipped 2026-06-27)

All 48 requirements satisfied. See `v20.0-MILESTONE-AUDIT.md` for full verification.

### Phase 25 (v20.1): P0 Critical Fixes ✅

- [x] **P0-01**: `vllm-model` no longer depends on `vllm-dist`; `vllm-dist` feature-gated
- [x] **P0-02**: `vllm-core → vllm-model` edge made feature-gated (cuda-graph)
- [x] **P0-03**: `ModelError` is an enum; pattern-matchable; all `match ModelError` migrated
- [x] **P0-04**: 8 non-object-safe public traits made object-safe; `dyn Trait` verified
- [x] **P0-05**: `CudaGraphError` converted from hand-rolled Display/Error to thiserror

### Phase 26 (v20.2): Module Tree Restoration ✅

- [x] **MT-01**: `kv_cache_fp8.rs` (289 LOC) wired into `components/mod.rs`
- [x] **MT-02**: `debug.rs` (175 LOC) wired into `server/src/lib.rs`
- [x] **MT-03**: `engine_v18_wiring.rs` renamed to `engine_wiring.rs`
- [x] **MT-04**: `qwen3/model_tests.rs` moved from `src/` to `tests/`
- [x] **MT-05**: `qwen3_5/model_tests.rs` moved from `src/` to `tests/`
- [x] **MT-06**: `qwen3_5/speculative_tests.rs` moved from `src/` to `tests/`
- [x] **MT-07**: `vllm-dist` (~1,600 LOC unused modules) feature-gated

### Phase 27 (v20.3): Error Handling Standardization ✅

- [x] **ERR-01**: Zero `Result<_, String>` occurrences in production code
- [x] **ERR-02**: All error enums use `thiserror` derive
- [x] **ERR-03**: Mutex-poison `.expect()` calls replaced with `.lock().context()?-equivalent`
- [x] **ERR-04**: Cross-crate `From` impls added for all error types
- [x] **ERR-05**: `EngineError` has new variants: `Timeout`, `Cancelled`, `ResourceExhausted`, `BackendUnavailable`
- [x] **ERR-06**: Top-level server uses `anyhow::Result`
- [x] **ERR-07**: All error propagation paths preserve context

### Phase 28 (v20.4): Documentation Coverage Push ✅

- [x] **DOC-01**: Workspace doc coverage ≥60% on `pub` items (achieved 97.8%)
- [x] **DOC-02**: Undocumented `pub` items gain `///` doc comments (1,299 added)
- [x] **DOC-03**: Source files gain `//!` module-level docs (125 added)
- [x] **DOC-04**: README.md code example corrected to 3-arg form
- [x] **DOC-05**: README.md "Supported Architectures" list updated to 10 architectures
- [x] **DOC-06**: README.md crate count corrected (7 → 6)
- [x] **DOC-07**: AGENTS.md "Architecture" section reconciled

### Phase 29 (v20.5): External Documentation + ADRs ✅

- [x] **EXT-01**: README.md, AGENTS.md, Cargo.toml claims reconciled
- [x] **EXT-02..EXT-12**: 12 new ADRs created (self-spec, FP8, KV cache split, speculative overview, RTE-01..03, vllm-dist, FP8 orphan, CUDA graph, cross-crate errors, etc.)

### Phase 30 (v20.6): Naming + Final Polish ✅

- [x] **NAM-01/02**: 7 P1 + 19 P2 naming fixes applied
- [x] **DEP-01/02**: Deprecation markers + migration paths added
- [x] **CMT-01/02**: Stale comments + dead TODOs resolved
- [x] **FINAL-01..04**: All gates green (1144 tests, clippy clean, fmt clean, PROJECT/STATE updated)

## v2 Requirements (deferred)

Tracked but not in v21.0 scope:

### Refactor follow-ups

- **RFU-04**: Resolve 7 P3 negative findings (NAME-F-12..F-26) — informational only, no action
- **RFU-05**: Migrate from `Mutex::lock().expect()` to `parking_lot::Mutex` (eliminates poison check entirely; covers API-F-04 redux)
- **RFU-06**: Doc coverage push 97.8% → 99%+ (if v21.4 doesn't push higher incidentally)

### Architecture follow-ups

- **ARF-04**: Re-evaluate `vllm-dist` for resurrection if multi-node work resumes
- **ARF-05**: Replace `kv_cache_fp8` with unified FP8 module (if v21.1 integration reveals deeper design issues)
- **ARF-06**: Split `engine.rs` (1,038 LOC God module) into smaller modules (Phase 27 deferred item)
- **ARF-07**: Unify `engine/speculative` tree post-ML-02

### Operational

- **OPS-01**: Real-model benchmark (vs stub backend) — already deferred from v18.0
- **OPS-02**: Fix `Engine::step()` speculative-mode hang (pre-existing bug)
- **OPS-03**: Resolve 5 pre-existing cargo doc broken-link warnings (`engine.rs` + `components/mod.rs`)

## Out of Scope (v21.0)

Explicitly excluded from v21.0:

| Feature | Reason |
|---------|--------|
| Engine::step() speculative hang fix | Pre-existing bug, separate bug-fix milestone |
| Multi-node / vllm-dist resurrection | Feature-gated in v20.0; multi-node work future |
| Real-model benchmarks | Deferred from v18.0; out of scope for remediation milestones |
| New features | Milestone is purely backlog closure |
| Performance optimization | Orthogonal to audit findings; separate cycle if needed |
| Tree-based speculation | Too complex (carried from v18.0) |
| MIGRATING.md historical content | v21.0 creates skeleton only; backfill deferred |
| P3 negative findings (NAME-F-12..F-26) | Informational only; no action |

## Traceability

### v21.0 Requirements (CURRENT)

| Requirement | Phase | Status |
|-------------|-------|--------|
| ML-01       | Phase 31 | Pending |
| ML-02       | Phase 31 | Pending |
| ML-03       | Phase 31 | Pending |
| ML-04       | Phase 31 | Pending |
| ML-05       | Phase 31 | Pending |
| ML-06       | Phase 31 | Pending |
| ML-07       | Phase 31 | Pending |
| ML-08       | Phase 31 | Pending |
| ML-09       | Phase 31 | Pending |
| API-01      | Phase 32 | Pending |
| API-02      | Phase 32 | Pending |
| API-03      | Phase 32 | Pending |
| API-04      | Phase 32 | Pending |
| API-05      | Phase 32 | Pending |
| API-06      | Phase 32 | Pending |
| API-07      | Phase 32 | Pending |
| API-08      | Phase 32 | Pending |
| API-09      | Phase 32 | Pending |
| API-10      | Phase 32 | Pending |
| API-11      | Phase 32 | Pending |
| NAM-01      | Phase 33 | Pending |
| NAM-02      | Phase 33 | Pending |
| NAM-03      | Phase 33 | Pending |
| NAM-04      | Phase 33 | Pending |
| NAM-05      | Phase 33 | Pending |
| NAM-06      | Phase 33 | Pending |
| NAM-07      | Phase 33 | Pending |
| NAM-08      | Phase 33 | Pending |
| DOC-01      | Phase 34 | Pending |
| DOC-02      | Phase 34 | Pending |
| DOC-03      | Phase 34 | Pending |
| DOC-04      | Phase 34 | Pending |
| P3-01       | Phase 35 | Pending |
| P3-02       | Phase 35 | Pending |
| P3-03       | Phase 35 | Pending |
| P3-04       | Phase 35 | Pending |
| P3-05       | Phase 35 | Pending |
| P3-06       | Phase 35 | Pending |
| FINAL-01    | Phase 35 | Pending |
| FINAL-02    | Phase 35 | Pending |
| FINAL-03    | Phase 35 | Pending |
| FINAL-04    | Phase 35 | Pending |

### v20.0 Requirements (HISTORICAL — Shipped)

| Requirement Group | Phase | Status |
|-------------------|-------|--------|
| P0-01..05         | Phase 25 | Complete ✅ |
| MT-01..07         | Phase 26 | Complete ✅ |
| ERR-01..07        | Phase 27 | Complete ✅ |
| DOC-01..07        | Phase 28 | Complete ✅ |
| EXT-01..12        | Phase 29 | Complete ✅ |
| NAM/DEP/CMT/FINAL | Phase 30 | Complete ✅ |

**Coverage:**
- v21.0 requirements: 42 total (9 ML + 11 API + 8 NAM + 4 DOC + 6 P3 + 4 FINAL)
- Mapped to phases: 42
- Unmapped: 0 ✓
- v20.0 historical: 48/48 Complete ✓

---

*Requirements defined: 2026-06-27 (v20.0)*
*Last updated: 2026-06-27 after v21.0 P2/P3 Backlog Cleanup scope definition (42 requirements across Phase 31-35)*
