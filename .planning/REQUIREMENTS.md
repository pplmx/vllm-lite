# Requirements: vllm-lite

**Defined:** 2026-06-27
**Milestone:** v20.0 Codebase Remediation (BACKLOG.md-driven)
**Core Value:** Fast, memory-efficient LLM inference with continuous batching, paged KV cache, and tensor parallelism — deployed with production-grade ops tooling and security.

## v1 Requirements (v20.0)

Remediation of v19.0 audit findings (100 total). Each requirement maps to one sub-phase (v20.1-v20.6). Detailed backlog: `.planning/audit/BACKLOG.md`.

### Phase 25 (v20.1): P0 Critical Fixes

- [ ] **P0-01**: `vllm-model` no longer depends on `vllm-dist`; `vllm-dist` feature-gated and excluded from default build
- [ ] **P0-02**: `vllm-core → vllm-model` edge made feature-gated (cuda-graph)
- [ ] **P0-03**: `ModelError` is an enum (not struct); pattern-matchable; all existing `match ModelError` blocks migrated
- [ ] **P0-04**: 8 non-object-safe public traits either made object-safe (add `where Self: Sized`) or split into marker + functional traits; `dyn Trait` usage verified
- [ ] **P0-05**: `CudaGraphError` converted from hand-rolled Display/Error to thiserror enum

### Phase 26 (v20.2): Module Tree Restoration

- [ ] **MT-01**: `kv_cache_fp8.rs` (289 LOC) declared in `scheduler/memory/mod.rs` and reachable via `mod.rs`
- [ ] **MT-02**: `debug.rs` (175 LOC) declared in `server/src/lib.rs` and reachable
- [ ] **MT-03**: `engine_v18_wiring.rs` renamed to `engine_wiring.rs` (remove stage-info suffix)
- [ ] **MT-04**: `qwen3/model_tests.rs` moved from `src/` to `tests/` and registered in `qwen3/tests/`
- [ ] **MT-05**: `qwen3_5/model_tests.rs` moved from `src/` to `tests/`
- [ ] **MT-06**: `qwen3_5/speculative_tests.rs` moved from `src/` to `tests/`
- [ ] **MT-07**: `vllm-dist` (~1,600 LOC unused modules) feature-gated; not in default build

### Phase 27 (v20.3): Error Handling Standardization

- [ ] **ERR-01**: Zero `Result<_, String>` occurrences in production code (10 → 0)
- [ ] **ERR-02**: All error enums use `thiserror` derive; no hand-rolled Display/Error (except for trivial wrappers)
- [ ] **ERR-03**: Mutex-poison `.expect()` calls (25+) replaced with `.lock().context("acquiring mutex")?` or equivalent
- [ ] **ERR-04**: Cross-crate `From` impls added for all error types that cross crate boundaries
- [ ] **ERR-05**: `EngineError` has new variants: `Timeout`, `Cancelled`, `ResourceExhausted`, `BackendUnavailable`
- [ ] **ERR-06**: Top-level server uses `anyhow::Result` for boundary error reporting
- [ ] **ERR-07**: All error propagation paths preserve context via `.context()` or `.with_context()`

### Phase 28 (v20.4): Documentation Coverage Push

- [ ] **DOC-01**: Workspace doc coverage ≥60% on `pub` items (currently 7.6%)
- [ ] **DOC-02**: 776 undocumented `pub` items gain `///` doc comments
- [ ] **DOC-03**: 121 source files gain `//!` module-level docs
- [ ] **DOC-04**: README.md code example `SchedulerEngine::new(config, 1024)` corrected to 3-arg form
- [ ] **DOC-05**: README.md "Supported Architectures" list updated to all 10 registered architectures
- [ ] **DOC-06**: README.md crate count corrected (currently says 7; Cargo.toml has 6)
- [ ] **DOC-07**: AGENTS.md "Architecture" section reconciled with current crate structure

### Phase 29 (v20.5): External Documentation + ADRs

- [ ] **EXT-01**: README.md, AGENTS.md, Cargo.toml claims reconciled (cross-document consistency)
- [ ] **EXT-02**: ADR created: "Why self-spec uses 1/8 layer count" (v16.0 decision)
- [ ] **EXT-03**: ADR created: "FP8 E4M3 format choice for KV cache" (v15.0 decision)
- [ ] **EXT-04**: ADR created: "KV cache split strategy (prefix vs paged)" (v1.0 decision)
- [ ] **EXT-05**: ADR created: "Speculative decoding architecture overview" (v16.0 design)
- [ ] **EXT-06**: ADR created: "Per-request draft routing (RTE-01..03)" (v18.0 decision)
- [ ] **EXT-07**: ADR created: "Why `vllm-dist` is feature-gated" (v20.0 decision)
- [ ] **EXT-08**: ADR created: "FP8 quantizer orphan module decision" (v20.2 outcome)
- [ ] **EXT-09**: ADR created: "Cuda graph feature gating strategy" (v10.1 decision)
- [ ] **EXT-10**: ADR created: "Cross-crate error type boundaries" (v20.3 decision)
- [ ] **EXT-11**: 2+ additional ADRs covering v15.0-v18.0 tribal knowledge decisions
- [ ] **EXT-12**: `.planning/PROJECT.md` "What This Is" + "Core Value" updated if drifted

### Phase 30 (v20.6): Naming + Final Polish

- [ ] **NAM-01**: 7 P1 naming fixes applied (variable single-letter, redundant suffixes, etc.)
- [ ] **NAM-02**: 19 P2 naming consistency fixes applied
- [ ] **DEP-01**: All v20.0-removed public API items marked with `#[deprecated]`
- [ ] **DEP-02**: Migration paths documented for all newly-deprecated items
- [ ] **CMT-01**: Stale comments referencing old code removed or updated
- [ ] **CMT-02**: Dead TODOs / FIXMEs from v19.0 audit resolved or marked stale
- [ ] **FINAL-01**: All 287+ tests pass after v20.0 changes
- [ ] **FINAL-02**: `cargo clippy --workspace -- -D warnings` returns clean
- [ ] **FINAL-03**: `cargo fmt --all --check` returns clean
- [ ] **FINAL-04**: `.planning/PROJECT.md` and `.planning/STATE.md` updated with v20.0 outcomes

## v2 Requirements (deferred)

Tracked but not in current milestone scope:

### Refactor follow-ups

- **RFU-01**: Further doc coverage push from 60% → 80% (if 60% achieved in v20.4)
- **RFU-02**: Resolve 13 P3 informational findings from v19.0 audit
- **RFU-03**: Migrate from `Mutex::lock().expect()` to `parking_lot::Mutex` (eliminates poison check entirely)

### Architecture follow-ups

- **ARF-01**: Re-evaluate `vllm-dist` for resurrection if multi-node work resumes
- **ARF-02**: Replace `kv_cache_fp8` with unified FP8 module (if orphan integration reveals deeper design issues)
- **ARF-03**: Split `engine.rs` (1,038 LOC God module) into smaller modules

### Operational

- **OPS-01**: Real-model benchmark (vs stub backend) — already deferred from v18.0
- **OPS-02**: Fix `Engine::step()` speculative-mode hang (pre-existing bug)

## Out of Scope

Explicitly excluded from v20.0:

| Feature | Reason |
|---------|--------|
| New features | v20.0 is remediation only; no new capabilities |
| Performance optimization | Orthogonal to audit findings |
| `vllm-dist` removal | Feature-gated, not deleted (preserves code, allows future multi-node) |
| `vllm-dist` resurrection | Multi-node work is future; v20.0 only gates it |
| Tree-based speculation | Too complex (carried from v18.0) |
| Real-model benchmarks | Already deferred from v18.0 |
| New architecture (new crate) | Out of scope; v20.0 only removes/fixes |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| P0-01       | Phase 25 | Pending |
| P0-02       | Phase 25 | Pending |
| P0-03       | Phase 25 | Pending |
| P0-04       | Phase 25 | Pending |
| P0-05       | Phase 25 | Pending |
| MT-01       | Phase 26 | Pending |
| MT-02       | Phase 26 | Pending |
| MT-03       | Phase 26 | Pending |
| MT-04       | Phase 26 | Pending |
| MT-05       | Phase 26 | Pending |
| MT-06       | Phase 26 | Pending |
| MT-07       | Phase 26 | Pending |
| ERR-01      | Phase 27 | Pending |
| ERR-02      | Phase 27 | Pending |
| ERR-03      | Phase 27 | Pending |
| ERR-04      | Phase 27 | Pending |
| ERR-05      | Phase 27 | Pending |
| ERR-06      | Phase 27 | Pending |
| ERR-07      | Phase 27 | Pending |
| DOC-01      | Phase 28 | Pending |
| DOC-02      | Phase 28 | Pending |
| DOC-03      | Phase 28 | Pending |
| DOC-04      | Phase 28 | Pending |
| DOC-05      | Phase 28 | Pending |
| DOC-06      | Phase 28 | Pending |
| DOC-07      | Phase 28 | Pending |
| EXT-01      | Phase 29 | Pending |
| EXT-02      | Phase 29 | Pending |
| EXT-03      | Phase 29 | Pending |
| EXT-04      | Phase 29 | Pending |
| EXT-05      | Phase 29 | Pending |
| EXT-06      | Phase 29 | Pending |
| EXT-07      | Phase 29 | Pending |
| EXT-08      | Phase 29 | Pending |
| EXT-09      | Phase 29 | Pending |
| EXT-10      | Phase 29 | Pending |
| EXT-11      | Phase 29 | Pending |
| EXT-12      | Phase 29 | Pending |
| NAM-01      | Phase 30 | Pending |
| NAM-02      | Phase 30 | Pending |
| DEP-01      | Phase 30 | Pending |
| DEP-02      | Phase 30 | Pending |
| CMT-01      | Phase 30 | Pending |
| CMT-02      | Phase 30 | Pending |
| FINAL-01    | Phase 30 | Pending |
| FINAL-02    | Phase 30 | Pending |
| FINAL-03    | Phase 30 | Pending |
| FINAL-04    | Phase 30 | Pending |

**Coverage:**
- v20.0 requirements: 46 total
- Mapped to phases: 46
- Unmapped: 0 ✓

---

*Requirements defined: 2026-06-27*
*Last updated: 2026-06-27 after initial definition (v20.0)*
