# Requirements: vllm-lite

**Defined:** 2026-06-28
**Core Value:** Fast, memory-efficient LLM inference with continuous batching, paged KV cache, and tensor parallelism — deployed with production-grade ops tooling and security.

## v23.0 Requirements (Active)

Audit Remediation — 28 requirements across 4 categories. Each maps to one of 4 phases (Phase 40-43).

### Code Defects (CODE)

- [ ] **CODE-01**: `TensorParallelError` (traits/src/types.rs:87-112) is converted to `#[derive(thiserror::Error)]` matching the project's 19 other error enums
- [ ] **CODE-02**: `Engine::step()` error chain is preserved — replace `EngineError::ModelError(e.to_string())` at core/src/engine.rs:677 with `EngineError::from(e)` so logs retain the underlying `vllm_traits::ModelError` source
- [ ] **CODE-03**: `Box<dyn std::error::Error>` in `dist/src/grpc.rs:129` public API is replaced with a typed `GrpcError` enum (thiserror) per AGENTS.md convention
- [ ] **CODE-04**: Stub architecture policy is enforced — either implement real forward passes for `gemma3`/`llama4`/`phi4`/`mistral_small`, or reject `loader.load()` with stub capability at runtime (in non-test builds)
- [ ] **CODE-05**: `SchedulerEngine::prefix_cache_hit_rate()` placeholder (core/src/scheduler/engine.rs:555-559) is either implemented against metrics counters or removed from the public API

### Stale Documentation (DOC)

- [ ] **DOC-02**: CLAUDE.md is rewritten to reflect v23.0 — crate count (6, not 4), Rust version (1.85, not 1.75), Engine signature (non-generic `Box<dyn ModelBackend>`), correct module references
- [ ] **DOC-03**: README.md:459-473 Scheduling policy example compiles — imports changed to `vllm_core::scheduler::policy::{FcfsPolicy, SjfPolicy, PriorityPolicy}`
- [ ] **DOC-04**: CHANGELOG.md backfilled with v19.0, v20.0, v21.0, v22.0 entries (synthesized from `.planning/milestones/v{19,20,21,22}.0-*.md`)
- [ ] **DOC-05**: MIGRATING.md v22.0 entry added — covers security middleware wiring, parking_lot migration, LazyLock upgrade
- [ ] **DOC-06**: `docs/architecture.md` created — unified v23.0 architecture overview covering engine orchestration, scheduler split, paged_tensor split, KV cache layer, model registry pattern, multi-model spec flow
- [ ] **DOC-07**: README.md test count badge updated (1100+ → 1179) and version pin note added
- [ ] **DOC-08**: `docs/optimization_guide.md:50` outdated `Engine::with_config` API example fixed to match current `Option<M>` signature
- [ ] **DOC-09**: `docs/optimization_guide.md` performance numbers tagged with date and matched against v22.0 bench results

### Comment Cleanup (CMT)

- [ ] **CMT-01**: ~85 module-level `//! <mod>: <mod>.` placeholder docs deleted across `core/`, `model/components/`, `model/paged_tensor/`, `model/kernels/`
- [ ] **CMT-02**: ~530 function/struct/method `/// <name>: <name>.` placeholder docs deleted (auto-generated noise polluting public API)
- [ ] **CMT-03**: 13 copies of `/// builder: construct via builder for documented field ergonomics.` replaced with type-specific documentation
- [ ] **CMT-04**: Phase/audit IDs (v18.0, Plan 17.x, SEC-06, PERF-01, etc.) stripped from user-visible rustdoc in 70 files — consolidated into one internal reference doc per module
- [ ] **CMT-05**: 4 actively wrong comments fixed — `core/src/lib.rs:7` "in progress" claim, `types.rs:264/273` double-name corruption, `server/src/{lib,health}.rs` triple-header pattern
- [ ] **CMT-06**: `qwen3_config` deprecation shim (crates/model/src/lib.rs:44-52) updated or deleted — `since = "0.21.0"` references nonexistent version

### Architecture Cleanup (ARCH)

- [ ] **ARCH-01**: `scheduler/batch_planner.rs` (369 LOC) deleted — zero production callers (deletion test passes: no callers concentrate)
- [ ] **ARCH-02**: `scheduler/predictive_batching.rs` (498 LOC) deleted — zero production callers
- [ ] **ARCH-03**: `core/src/kv_cache/mod.rs` (7 LOC) deleted — pass-through shim for BLOCK_SIZE re-export
- [ ] **ARCH-04**: Unused internal modules consolidated — `core/src/sync.rs` (12), `routing/HashRouter` (191), `ha/{FailoverManager,LeaderElection}` (328), `circuit_breaker/*` (556) — no engine seam
- [ ] **ARCH-05**: 4 stub architectures (`gemma3`/`llama4`/`phi4`/`mistral_small`, ~1100 LOC) collapsed into one parameterized `StubArchitecture` struct
- [ ] **ARCH-06**: `core → model` upward dependency via `cuda-graph` feature fixed — CUDA graph types moved to `vllm-traits` or new `vllm-kernels` crate below both
- [ ] **ARCH-07**: Unused `reqwest` dependency removed from `crates/server/Cargo.toml`
- [ ] **ARCH-08**: `rayon` moved from `[dependencies]` to `[dev-dependencies]` in `crates/model/Cargo.toml` (only used in tests)
- [ ] **ARCH-09**: 3 `greedy_sample`/`argmax` implementations unified — `core/sampling.rs`, `model/causal_lm/mod.rs`, `engine/spec_dispatch/drafts.rs` collapse to one canonical impl
- [ ] **ARCH-10**: 2 `Architecture` types unified — `arch::Architecture` (trait) and `config::Architecture` (enum) collapse to single concept

### Final Verification (FINAL — per-phase + end-of-milestone)

- [ ] **FINAL-01**: All 1179 tests pass after each phase (Phase 40-43)
- [ ] **FINAL-02**: `cargo clippy --workspace --all-targets --all-features -- -D warnings` clean (Phase 43)
- [ ] **FINAL-03**: `cargo fmt --all --check` clean (Phase 43)
- [ ] **FINAL-04**: Test count ≥ 1179 post-v23.0 (no regression, remediation scope)
- [ ] **FINAL-05**: `.planning/PROJECT.md` + `.planning/STATE.md` updated with v23.0 outcomes (Phase 43)

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| New model capabilities (long context >32K, multimodal/vision) | v24+ candidate; v23.0 is remediation only |
| Engine.rs full single-responsibility split (1057 LOC → sub-modules) | ARF-06 still partial; ARF-* in v23.0 doesn't include God module split |
| Multi-node / vllm-dist resurrection | OPS-05 still deferred; feature-gated only |
| Real-model benchmarks | OPS-04 still deferred (no GPU env) |
| Doc coverage 97.8% → 99%+ | RFU-06; not in remediation scope |
| Stub architecture full implementation | CODE-04 only handles policy (implement OR reject), not full forward impl |
| New architectures (Falcon, DeepSeek, etc.) | Out of remediation scope |
| Online fine-tuning, WebAssembly | Long-term vision |
| Tree-based speculation | Too complex (carried from v18.0) |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| CODE-01 | Phase 40 | Pending |
| CODE-02 | Phase 40 | Pending |
| CODE-03 | Phase 40 | Pending |
| CODE-04 | Phase 40 | Pending |
| CODE-05 | Phase 40 | Pending |
| DOC-02 | Phase 41 | Pending |
| DOC-03 | Phase 41 | Pending |
| DOC-04 | Phase 41 | Pending |
| DOC-05 | Phase 41 | Pending |
| DOC-06 | Phase 41 | Pending |
| DOC-07 | Phase 41 | Pending |
| DOC-08 | Phase 41 | Pending |
| DOC-09 | Phase 41 | Pending |
| CMT-01 | Phase 42 | Pending |
| CMT-02 | Phase 42 | Pending |
| CMT-03 | Phase 42 | Pending |
| CMT-04 | Phase 42 | Pending |
| CMT-05 | Phase 42 | Pending |
| CMT-06 | Phase 42 | Pending |
| ARCH-01 | Phase 43 | Pending |
| ARCH-02 | Phase 43 | Pending |
| ARCH-03 | Phase 43 | Pending |
| ARCH-04 | Phase 43 | Pending |
| ARCH-05 | Phase 43 | Pending |
| ARCH-06 | Phase 43 | Pending |
| ARCH-07 | Phase 43 | Pending |
| ARCH-08 | Phase 43 | Pending |
| ARCH-09 | Phase 43 | Pending |
| ARCH-10 | Phase 43 | Pending |
| FINAL-01 | Phase 40-43 | Pending |
| FINAL-02 | Phase 43 | Pending |
| FINAL-03 | Phase 43 | Pending |
| FINAL-04 | Phase 43 | Pending |
| FINAL-05 | Phase 43 | Pending |

**Coverage:**
- v23.0 requirements: 28 functional + 5 FINAL = 33 total
- Mapped to phases: 33
- Unmapped: 0 ✓

---
*Requirements defined: 2026-06-28*
*Last updated: 2026-06-28 after v22.0 milestone audit*
