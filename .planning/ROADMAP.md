# Roadmap: vllm-lite

## Milestones

- ✅ **v16.0 Speculative Decoding** — Phases 16.1-16.4 (shipped 2026-04-28)
- ✅ **v17.0 Production Speculative Decoding** — Phases 17.1-17.4 (shipped 2026-05-13)
- ✅ **v18.0 Multi-Model Speculative Decoding** — Phases 18.1-18.4 + Phase 19 gap closure (shipped 2026-06-27)
- ✅ **v19.0 Codebase Health Audit** — Phases 20-24 (shipped 2026-06-27; analysis-only; see `.planning/audit/`)
- ✅ **v20.0 Codebase Remediation** — Phases 25-30 (shipped 2026-06-27; 48/48 reqs; 1144 tests; clippy/fmt clean; doc 97.8%)
- ✅ **v21.0 P2/P3 Backlog Cleanup** — Phases 31-35 (shipped 2026-06-27; 38 of 42 actively addressed; 1146 tests; 100% backlog closure)
- ✅ **v22.0 Production Hardening** — Phases 36-39 (shipped 2026-06-27; 21/21 reqs; 1179 tests; clippy/fmt/doc clean; P0 bug fixed, security wired, polish applied)
- ◆ **v23.0 Audit Remediation** — Phases 40-43 (active 2026-06-28; 29 functional + 5 FINAL reqs; ~50h estimated; remediates 22 findings from v22.0 post-ship audit across P0 code / stale docs / placeholder docs / dead code)

## Current Position

**Active milestone:** v23.0 Audit Remediation ◆ ACTIVE (roadmap created 2026-06-28; 4 phases planned; Phase 40 ready for `/gsd-discuss-phase 40`)

**Previous milestone:** v22.0 Production Hardening ✅ SHIPPED 2026-06-27 (Phases 36-39; 21/21 reqs; 1179 tests; FINAL-01..05 green)

## Phases

### v22.0 Production Hardening (shipped)

- [x] **Phase 36: Critical Bug Fixes** (v22.1) — Fix `Engine::step()` speculative-mode hang; resolve cargo doc warnings; resolve gguf parser TODO ✅ shipped 2026-06-27
- [x] **Phase 37: Security Hardening** (v22.2) — JWT signature verification; wire RbacMiddleware; request size limits; audit log test; Grafana credentials to .env; TLS hardening ✅ shipped 2026-06-27
- [x] **Phase 38: Production Polish** (v22.3) — parking_lot::Mutex migration; speculative mock fate decision; MlaKvCache write_compressed perf; arch detection perf; LazyLock migration ✅ shipped 2026-06-27
- [x] **Phase 39: Engine Refactor + Final Verification** (v22.4) — Split engine.rs God module; unify engine/spec_dispatch tree; FINAL gates (clippy/fmt/test/docs) ✅ shipped 2026-06-27

### v23.0 Audit Remediation (active)

- [ ] **Phase 40: Critical Code Fixes** (v23.1) — `TensorParallelError` → thiserror; `Engine::step()` error chain preserved; `Box<dyn Error>` → typed `GrpcError`; stub architecture policy; `prefix_cache_hit_rate()` implemented or removed (~6h)
- [ ] **Phase 41: Stale Documentation** (v23.2) — CLAUDE.md rewrite; README fix; CHANGELOG backfill v19-v22; MIGRATING v22; new `docs/architecture.md`; README badge update; optimization_guide API fix; perf numbers date tag (~16h)
- [ ] **Phase 42: Placeholder Doc Cleanup** (v23.3) — Delete ~85 module placeholder docs; delete ~530 function placeholder docs; replace 13 builder copy-paste docs; strip phase IDs from 70 files; fix 4 wrong comments; `qwen3_config` deprecation cleanup (~6h)
- [ ] **Phase 43: Architecture Cleanup + Final Verification** (v23.4) — Delete 7 dead modules (~2000 LOC); consolidate 4 stub architectures; fix `core → model` upward dep; remove unused `reqwest`; move `rayon` to dev-deps; unify 3 `greedy_sample` impls + 2 `Architecture` types; FINAL gates (~20h)

## Phase Details

### 🚧 v22.0 Production Hardening (Phases 36-39) — PLANNING

**Milestone Goal:** Push vLLM-lite to production-ready state — resolve v21.0-deferred P0/P1 tech debt, wire security middleware (currently no-op pass-throughs), polish production ergonomics, refactor the `engine.rs` God module.

**Source:** `.planning/PROJECT.md` (Current Milestone section), `.planning/REQUIREMENTS.md` (v22.0 section), v18.0/v20.0/v21.0 carry-over tech-debt items

**Effort estimate:** ~75h (~1 working month)

**Dependency graph:**

```text
36 (Critical Bug Fixes)
   │
   ├─→ 37 (Security Hardening)
   │      └─ depends on 36 (test stability from hang fix enables reliable security integration tests)
   │
   └─→ 38 (Production Polish)
          ├─ depends on 36 (test stability required for refactor verification)
          └─ depends on 37 (security wiring must not be perturbed by polish)
                  │
                  └─→ 39 (Engine Refactor + FINAL gates)
                          └─ depends on 36, 37, 38 (all prior work must be in place before FINAL gates run)
```

**Phase dependency chain (recommended):** 36 → 37 → 38 → 39 (linear; P0 bug fix first to unblock reliable integration testing; security wiring before polish so JWT/RBAC code paths are stable; engine refactor last so it can absorb security + polish changes without churn)

**Constraints (apply to ALL v22.0 phases):**

- All 1146+ existing tests must remain green throughout (FINAL-01 invariant per phase)
- Public API changes require `#[deprecated(since, note)]` markers + migration path (DEP-01/02 from v20.6)
- `vllm-dist` remains feature-gated (ADR-008 outcome from v20.1)
- `cargo clippy --workspace --all-targets -- -D warnings` must remain clean after each phase
- No new features, no architectural redesign — strictly tech-debt execution (production hardening scope)
- Backward-compat: security wiring may add new config keys but must not break existing config

---

#### Phase 36: Critical Bug Fixes (v22.1)

**Goal**: Eliminate pre-existing P0/P1 bugs blocking production deployment — `Engine::step()` speculative-mode hang, cargo doc broken links, gguf parser placeholder

**Depends on**: v21.0 baseline (Phase 35 shipped — 1146 tests green, clippy/fmt clean, doc coverage 97.8%)

**Requirements**: OPS-02, OPS-03, GGUF-01, FINAL-01

**Success Criteria** (what must be TRUE):

  1. `Engine::step()` in speculative mode completes deterministically without hanging — 2 previously `#[ignore]`'d Phase 19 e2e tests in `crates/core/tests/engine_v18_wiring.rs` and `crates/core/tests/draft_resolver_integration.rs` now pass (their `#[ignore]` markers removed)
  2. `cargo doc --workspace --no-deps` produces zero broken-link warnings (5 pre-existing warnings in `engine.rs` + `components/mod.rs` resolved — typically via `[link]` attributes, intra-doc URLs, or removing the broken references)
  3. `crates/model/src/quantize/gguf.rs` contains no actionable TODO comments related to the parser placeholder (resolved by either implementing the placeholder or removing it with a documented rationale)
  4. **FINAL-01**: All 1146+ tests remain green post-fix — `cargo test --workspace --all-features` returns 0 failures; new tests added for hang regression and doc-warning coverage must not reduce the test count

**Plans**: TBD

Plans:

- [ ] 36-01: Diagnose and fix `Engine::step()` speculative-mode hang (root cause + regression test)
- [ ] 36-02: Resolve 5 cargo doc broken-link warnings in `engine.rs` + `components/mod.rs`
- [ ] 36-03: Resolve gguf parser TODO in `crates/model/src/quantize/gguf.rs`
- [ ] 36-04: FINAL-01 verification — `cargo test --workspace --all-features` ≥1146 pass

---

#### Phase 37: Security Hardening (v22.2)

**Goal**: Make security middleware actually enforce its policies — currently JWT parses but doesn't verify, RBAC is a no-op pass-through, TLS has `unwrap()` panics, Grafana credentials are hardcoded

**Depends on**: Phase 36 (test stability from hang fix enables reliable security integration tests; cargo doc warnings clean so security doc references resolve)

**Requirements**: SEC-01, SEC-02, SEC-03, SEC-04, SEC-05, SEC-06, FINAL-01

**Success Criteria** (what must be TRUE):

  1. JWT tokens with invalid, expired, or tampered signatures are rejected by the server with HTTP 401 — implementation uses HMAC-SHA256 for `secret`-based JWTs and RSA/ECDSA for `public_key_pem`-based JWTs; verified by integration tests covering each rejection case (validates SEC-01)
  2. `RbacMiddleware` actually denies requests from users lacking required permissions (currently a no-op pass-through at `rbac.rs:82-84`); denied requests return HTTP 403 with structured error; integration tests verify permission-required paths (validates SEC-02)
  3. HTTP request bodies exceeding the configured size limit are rejected by `tower_http::limit::RequestBodyLimitLayer` with HTTP 413 Payload Too Large; default limit configurable via server config (validates SEC-03)
  4. Audit log (`crates/server/src/security/audit.rs`) emits an event for every authenticated request; integration test asserts the audit log contains entries for each auth scenario (success, JWT failure, RBAC denial) (validates SEC-04)
  5. `docker-compose.yml` contains no hardcoded Grafana credentials — `GRAFANA_ADMIN_USER` and `GRAFANA_ADMIN_PASSWORD` are read from `.env` (already gitignored) via Compose variable substitution (validates SEC-05)
  6. TLS handshake failures in `security/tls.rs:63` return a structured error instead of `unwrap()` panic — server fails gracefully on malformed certificates, key material, or handshake negotiation; test verifies the error path (validates SEC-06)
  7. **FINAL-01**: All auth/middleware integration tests pass; no new security regressions — `cargo test -p vllm-server` returns 0 failures; coverage of JWT/RBAC/size-limit/audit paths is exercised end-to-end

**Plans**: TBD

Plans:

- [ ] 37-01: Implement JWT signature verification (HMAC-SHA256 + RSA/ECDSA) + tests — SEC-01
- [ ] 37-02: Wire `RbacMiddleware` into request pipeline + permission tests — SEC-02
- [ ] 37-03: Add `RequestBodyLimitLayer` with configurable limit + 413 tests — SEC-03
- [ ] 37-04: Audit log integration test (assert events emitted per auth scenario) — SEC-04
- [ ] 37-05: Move Grafana credentials from `docker-compose.yml` to `.env` — SEC-05
- [ ] 37-06: Replace `unwrap()` in `security/tls.rs:63` with structured error + test — SEC-06
- [ ] 37-07: FINAL-01 verification — auth/middleware integration tests green

---

#### Phase 38: Production Polish (v22.3)

**Goal**: Eliminate production ergonomics smells and apply small perf wins — remove `std::sync::Mutex` poison checks, decide speculative mock fate, fix `MlaKvCache::write_compressed` full-cache materialization, optimize arch detection, migrate to `std::sync::LazyLock`

**Depends on**: Phase 36 (test stability required for refactor verification), Phase 37 (security wiring must be in place before polish so JWT/RBAC code paths remain stable)

**Requirements**: RFU-05, OPS-01, PERF-01, PERF-02, PERF-03, DOC-01, FINAL-01

**Success Criteria** (what must be TRUE):

  1. Scheduler and engine paths use `parking_lot::Mutex` exclusively — zero `std::sync::Mutex` instances remain in `crates/core/src/scheduler/` and `crates/core/src/engine.rs`; zero `.lock().unwrap()` poison-check calls (24 sites from CONCERNS.md eliminated) (validates RFU-05)
  2. `crates/core/src/engine/speculative.rs` mock usage has a documented fate — either wired to real draft loading via `ServerDraftLoader` or annotated as mock-only with a comment block explaining production behavior (validates OPS-01)
  3. `MlaKvCache::write_compressed` writes incrementally using `Tensor::slice_assign` or equivalent — no full-cache materialization (memory allocation proportional to the slice written, not the full cache); benchmark or memory-profile test shows reduced allocation (validates PERF-01)
  4. Architecture detection uses `eq_ignore_ascii_case` instead of `model_type.to_lowercase()` — zero per-load `String` allocations in the arch detection path (validates PERF-02)
  5. Lazy initialization in `crates/model/src/arch/registry.rs` uses `std::sync::LazyLock` (Rust 1.80+) — no new `once_cell::sync::Lazy` usage; existing `once_cell` usage migrated or explicitly retained with rationale (validates PERF-03)
  6. `cargo doc --workspace --no-deps` produces zero broken-link warnings — DOC-01 carry-over from OPS-03 confirmed closed; if Phase 36 left any warnings unresolved, this phase closes them (validates DOC-01)
  7. **FINAL-01**: All 1146+ tests remain green post-polish — `cargo test --workspace --all-features` returns 0 failures; mutex migration + perf changes do not regress any test

**Plans**: TBD

Plans:

- [ ] 38-01: Migrate scheduler/engine mutexes from `std::sync::Mutex` → `parking_lot::Mutex` (24 sites) — RFU-05
- [ ] 38-02: Decide and document `speculative.rs` mock usage fate — OPS-01
- [ ] 38-03: Refactor `MlaKvCache::write_compressed` to incremental `slice_assign` — PERF-01
- [ ] 38-04: Replace `model_type.to_lowercase()` with `eq_ignore_ascii_case` — PERF-02
- [ ] 38-05: Migrate `once_cell::sync::Lazy` → `std::sync::LazyLock` in `arch/registry.rs` — PERF-03
- [ ] 38-06: Verify/close any remaining cargo doc broken-link warnings — DOC-01
- [ ] 38-07: FINAL-01 verification — `cargo test --workspace --all-features` ≥1146 pass

---

#### Phase 39: Engine Refactor + Final Verification (v22.4)

**Goal**: Decompose the `engine.rs` God module into focused sub-modules, unify the `engine/spec_dispatch` tree, and verify all FINAL gates (clippy/fmt/test/docs) are green for v22.0 ship

**Depends on**: Phase 36, Phase 37, Phase 38 (all prior phases must be complete before FINAL gates run — engine refactor must not perturb prior fixes)

**Requirements**: ARF-06, ARF-07, FINAL-01, FINAL-02, FINAL-03, FINAL-04, FINAL-05

**Success Criteria** (what must be TRUE):

  1. `engine.rs` is decomposed into focused sub-modules — the original 1,038 LOC file is split such that no single engine source file exceeds 300 LOC; sub-modules have single-responsibility naming (`engine/step.rs`, `engine/sequence.rs`, `engine/batch.rs`, etc.); behavior is unchanged (validates ARF-06)
  2. `engine/spec_dispatch` tree is unified post-Phase-31 (ML-02) — duplicate abstractions between `engine.rs` and `engine/speculative/` are collapsed into the canonical sub-tree; no `pub use` re-export shims remain where the underlying module is directly accessible (validates ARF-07)
  3. **FINAL-01**: All 1146+ tests remain green post-refactor — `cargo test --workspace --all-features` returns 0 failures; refactor preserves all existing test coverage (validates FINAL-01)
  4. **FINAL-02**: `cargo clippy --workspace --all-targets --all-features -- -D warnings` exits 0 — zero warnings, zero errors across all targets and features (validates FINAL-02)
  5. **FINAL-03**: `cargo fmt --all --check` exits 0 — workspace formatting matches rustfmt output (validates FINAL-03)
  6. **FINAL-04**: `cargo test --workspace --all-features` reports ≥ 1146 tests passing (no regression from v21.0 baseline; hardening scope, no expected growth) (validates FINAL-04)
  7. **FINAL-05**: `.planning/PROJECT.md` (Current Milestone + Validated sections) and `.planning/STATE.md` (Current Position + Performance Metrics + v22.0 outcomes) are updated; v22.0 production-ready status declared; v23.0 candidates surfaced (long context, multimodal, multi-node resurrection) (validates FINAL-05)

**Plans**: TBD

Plans:

- [ ] 39-01: Split `engine.rs` (1,038 LOC) into focused sub-modules — ARF-06
- [ ] 39-02: Unify `engine/spec_dispatch` tree (collapse duplicate abstractions) — ARF-07
- [ ] 39-03: FINAL-01 verification — `cargo test --workspace --all-features` ≥1146 pass post-refactor
- [ ] 39-04: FINAL-02 verification — `cargo clippy --workspace --all-targets --all-features -- -D warnings` clean
- [ ] 39-05: FINAL-03 verification — `cargo fmt --all --check` clean
- [ ] 39-06: FINAL-04 verification — `cargo test --workspace --all-features` ≥1146 tests total
- [ ] 39-07: FINAL-05 — Update `.planning/PROJECT.md` + `.planning/STATE.md` with v22.0 outcomes

---

### ◆ v23.0 Audit Remediation (Phases 40-43) — ACTIVE

**Milestone Goal:** Remediate 22 findings from the v22.0 post-ship audit across 4 categories (P0 code defects, stale documentation, placeholder docs, dead code) — strict no-new-features scope; deliver an audit-remediated codebase ready for v24+ new capabilities

**Source:** `.planning/PROJECT.md` (Current Milestone section), `.planning/REQUIREMENTS.md` (v23.0 section), `.planning/v22.0-MILESTONE-AUDIT.md` (22 audit findings)

**Effort estimate:** ~50h (~6h Phase 40 + ~16h Phase 41 + ~6h Phase 42 + ~20h Phase 43)

**Dependency graph:**

```text
40 (Critical Code Fixes)
   │
   ├─→ 41 (Stale Documentation)
   │      └─ depends on 40 (test stability from P0 fix enables reliable doc verification)
   │
   └─→ 42 (Placeholder Doc Cleanup)
          ├─ depends on 40 (test stability from P0 fix)
          └─ depends on 41 (docs must be current before stripping phase IDs / fixing wrong comments to avoid churn)
                  │
                  └─→ 43 (Architecture Cleanup + FINAL gates)
                          └─ depends on 40, 41, 42 (all prior work must be in place before FINAL gates run; ARCH-* touches many files and must absorb prior changes without churn)
```

**Phase dependency chain (recommended):** 40 → 41 → 42 → 43 (linear; P0 code first to unblock reliable testing; docs before comment cleanup to avoid freshly-added phase IDs being stripped; ARCH last to absorb accumulated changes without breaking subsequent work)

**Constraints (apply to ALL v23.0 phases):**

- All 1179+ existing tests must remain green throughout (FINAL-01 invariant per phase)
- Public API changes require `#[deprecated(since, note)]` markers + migration path (DEP-01/02 from v20.6)
- `vllm-dist` remains feature-gated (ADR-008 outcome from v20.1)
- `cargo clippy --workspace --all-targets -- -D warnings` must remain clean after each phase
- No new features, no architectural redesign — strictly audit remediation (4 categories only: P0 code / stale docs / placeholder docs / dead code)
- Backward-compat: dead code removal is internal refactor; no breaking API changes
- Test count goal: ≥ 1179 post-v23.0 (remediation scope, no growth expected)

---

#### Phase 40: Critical Code Fixes (v23.1)

**Goal**: Resolve P0 code defects from v22.0 audit — error types are structured enums with preserved source chains; stub architectures have explicit policy

**Depends on**: v22.0 baseline (Phase 39 shipped — 1179 tests green, clippy/fmt clean, doc coverage 97.8%)

**Requirements**: CODE-01, CODE-02, CODE-03, CODE-04, CODE-05, FINAL-01

**Success Criteria** (what must be TRUE):

  1. `TensorParallelError` (traits/src/types.rs:87-112) is converted to `#[derive(thiserror::Error)]` matching the project's 19 other error enums — existing call sites compile unchanged; the enum gains structured variants + `#[error("...")]` messages + `#[source]` chains (validates CODE-01)
  2. `Engine::step()` preserves error source chain — `EngineError::ModelError(e.to_string())` at core/src/engine.rs:677 is replaced with `EngineError::from(e)` (or equivalent `#[source]` wiring) so logs retain the underlying `vllm_traits::ModelError`; `Display` output shows the nested cause chain end-to-end (validates CODE-02)
  3. `Box<dyn std::error::Error>` in `dist/src/grpc.rs:129` return type is replaced with a typed `GrpcError` enum (thiserror) per AGENTS.md convention — downstream callers match on variants instead of stringifying; no `Box<dyn Error>` remains in `crates/dist/src/` public API (validates CODE-03)
  4. Stub architecture policy is enforced — `gemma3`/`llama4`/`phi4`/`mistral_small` cannot be loaded via `ModelLoader::load()` in non-test builds; the `StubArchitecture` (introduced by Phase 43 ARCH-05) returns a structured `LoadError::StubNotAllowed` error when `allow_stub` capability is not present in non-test profiles (validates CODE-04)
  5. `SchedulerEngine::prefix_cache_hit_rate()` placeholder (core/src/scheduler/engine.rs:555-559) is either implemented against metrics counters (returns the real hit rate from `prefix_cache_hits` / `prefix_cache_queries`) or removed from the public API surface (with all in-tree callers updated) (validates CODE-05)
  6. **FINAL-01**: All 1179 tests remain green post-fix — `cargo test --workspace --all-features` returns 0 failures; thiserror conversions + stub policy + placeholder resolution introduce no regressions

**Plans**: TBD

Plans:

- [ ] 40-01: Convert `TensorParallelError` to `#[derive(thiserror::Error)]` + call-site verification — CODE-01
- [ ] 40-02: Replace `EngineError::ModelError(e.to_string())` with `EngineError::from(e)` to preserve source chain — CODE-02
- [ ] 40-03: Replace `Box<dyn Error>` in `dist/src/grpc.rs:129` with typed `GrpcError` enum — CODE-03
- [ ] 40-04: Enforce stub architecture policy — reject `loader.load()` for stubs in non-test builds — CODE-04
- [ ] 40-05: Implement or remove `SchedulerEngine::prefix_cache_hit_rate()` placeholder — CODE-05
- [ ] 40-06: FINAL-01 verification — `cargo test --workspace --all-features` ≥1179 pass

---

#### Phase 41: Stale Documentation (v23.2)

**Goal**: All user-facing documentation reflects v23.0 reality — no broken examples, no stale crate/version references, no missing migration notes

**Depends on**: Phase 40 (test stability from P0 fix enables reliable doc verification; `StubArchitecture` rename impacts docs)

**Requirements**: DOC-02, DOC-03, DOC-04, DOC-05, DOC-06, DOC-07, DOC-08, DOC-09, FINAL-01

**Success Criteria** (what must be TRUE):

  1. `CLAUDE.md` is rewritten to reflect v23.0 — crate count corrected to 6 (not 4), Rust version corrected to 1.85 (not 1.75), Engine signature corrected to non-generic `Box<dyn ModelBackend>`, broken `qwen3/attention.rs` reference removed or replaced with valid path (validates DOC-02)
  2. `README.md` Scheduling policy example compiles — imports changed to `vllm_core::scheduler::policy::{FcfsPolicy, SjfPolicy, PriorityPolicy}`; lines 459-473 no longer reference the old (non-existent) `vllm_core::scheduler::FcfsPolicy` path; copy-paste snippet pastes into a test file without error (validates DOC-03)
  3. `CHANGELOG.md` contains entries for v19.0, v20.0, v21.0, v22.0 milestones — synthesized from `.planning/milestones/v{19,20,21,22}.0-*.md` with key accomplishments + stats + tech debt roll-forward per milestone (validates DOC-04)
  4. `MIGRATING.md` contains a v22.0 entry covering security middleware wiring (JWT verify, RBAC, RequestBodyLimitLayer), parking_lot migration, and LazyLock upgrade — with example diffs showing how to update consumers of changed APIs (validates DOC-05)
  5. `docs/architecture.md` exists with a unified v23.0 architecture overview — engine orchestration, scheduler split (queue/preemption/eviction/batch), paged_tensor split (logical vs physical KV cache), registry pattern, multi-model spec flow; cross-references relevant ADRs (validates DOC-06)
  6. `README.md` test count badge updated from `1100+` to `1179+` with a version pin note; `docs/optimization_guide.md` `Engine::with_config` API example (line 50) updated to match the current `Option<M>` signature; performance numbers in `optimization_guide.md` tagged with date and reconciled with v22.0 bench results (validates DOC-07, DOC-08, DOC-09)
  7. **FINAL-01**: All 1179 tests remain green — doc updates touch markdown + Rust comments but introduce no logic changes

**Plans**: TBD

Plans:

- [ ] 41-01: Rewrite `CLAUDE.md` for v23.0 reality (crates, Rust version, Engine signature, broken refs) — DOC-02
- [ ] 41-02: Fix `README.md:459-473` Scheduling policy example imports — DOC-03
- [ ] 41-03: Backfill `CHANGELOG.md` with v19-v22 milestone entries — DOC-04
- [ ] 41-04: Add `MIGRATING.md` v22.0 entry (security middleware, parking_lot, LazyLock) — DOC-05
- [ ] 41-05: Create `docs/architecture.md` unified v23.0 overview — DOC-06
- [ ] 41-06: Update README badge + fix optimization_guide API example + tag perf numbers — DOC-07, DOC-08, DOC-09
- [ ] 41-07: FINAL-01 verification — `cargo test --workspace --all-features` ≥1179 pass

---

#### Phase 42: Placeholder Doc Cleanup (v23.3)

**Goal**: rustdoc surface is honest and informative — no auto-generated noise, no phase/audit ID leakage, no incorrect claims

**Depends on**: Phase 40 (test stability from P0 fix), Phase 41 (docs must be current before stripping phase IDs / fixing wrong comments to avoid churn on freshly-added content)

**Requirements**: CMT-01, CMT-02, CMT-03, CMT-04, CMT-05, CMT-06, FINAL-01

**Success Criteria** (what must be TRUE):

  1. ~85 module-level `//! <mod>: <mod>.` placeholder docs are deleted across `crates/core/src/`, `crates/model/src/components/`, `crates/model/src/paged_tensor/`, `crates/model/src/kernels/` — remaining module docs are substantive (explain purpose, key types, invariants) or absent (validates CMT-01)
  2. ~530 function/struct/method `/// <name>: <name>.` placeholder docs are deleted from public API surfaces — remaining rustdoc on public items explains behavior, parameters, returns, errors, or is explicitly `/// # Implementation details` / `/// See <type>` cross-references (validates CMT-02)
  3. 13 copies of `/// builder: construct via builder for documented field ergonomics.` are replaced with type-specific documentation — each builder gets a concrete example of construction + a list of field semantics (validates CMT-03)
  4. Phase/audit IDs (v18.0, Plan 17.x, SEC-06, PERF-01, ARF-07, etc.) are stripped from user-visible rustdoc in 70 files — internal reference docs consolidated into one module-internal `docs/references.md` per affected module; rustdoc no longer exposes milestone/commit IDs to library consumers (validates CMT-04)
  5. Four actively wrong comments are fixed: `core/src/lib.rs:7` "in progress" claim removed (work is shipped); `types.rs:264/273` double-name corruption (e.g. `/// Foo: Foo.`) corrected to single-name or substantive doc; `server/src/{lib,health}.rs` triple-header pattern collapsed to one canonical header (validates CMT-05)
  6. `qwen3_config` deprecation shim (crates/model/src/lib.rs:44-52) is updated or deleted — `since = "0.21.0"` references a nonexistent version; either point `since` at the real version (e.g. `0.20.0`) or remove the shim entirely if no consumers remain (validates CMT-06)
  7. **FINAL-01**: All 1179 tests remain green post-cleanup — comment deletions don't change code semantics; `cargo doc --workspace --no-deps` produces zero warnings

**Plans**: TBD

Plans:

- [ ] 42-01: Delete ~85 module-level `//! <mod>: <mod>.` placeholder docs — CMT-01
- [ ] 42-02: Delete ~530 `/// <name>: <name>.` function/struct/method placeholder docs — CMT-02
- [ ] 42-03: Replace 13 `/// builder: construct via builder...` copies with type-specific docs — CMT-03
- [ ] 42-04: Strip phase/audit IDs from 70 files of user-visible rustdoc — CMT-04
- [ ] 42-05: Fix 4 actively wrong comments (lib.rs:7, types.rs:264/273, server/lib.rs + health.rs) — CMT-05
- [ ] 42-06: Update or delete `qwen3_config` deprecation shim — CMT-06
- [ ] 42-07: FINAL-01 verification — `cargo test --workspace --all-features` ≥1179 pass; `cargo doc` 0 warnings

---

#### Phase 43: Architecture Cleanup + Final Verification (v23.4)

**Goal**: ~2000 LOC of dead code removed; architecture consistency restored (one Architecture concept, one greedy_sample); final verification gates green for v23.0 ship

**Depends on**: Phase 40, Phase 41, Phase 42 (all prior phases must be complete before FINAL gates run — ARCH-* touches many files and must absorb prior changes without churn)

**Requirements**: ARCH-01, ARCH-02, ARCH-03, ARCH-04, ARCH-05, ARCH-06, ARCH-07, ARCH-08, ARCH-09, ARCH-10, FINAL-01, FINAL-02, FINAL-03, FINAL-04, FINAL-05

**Success Criteria** (what must be TRUE):

  1. Dead scheduler + KV cache modules are deleted — `scheduler/batch_planner.rs` (369 LOC), `scheduler/predictive_batching.rs` (498 LOC), `core/src/kv_cache/mod.rs` (7 LOC) removed; `cargo build --workspace` and `cargo test --workspace` both confirm zero production callers concentrated (ARCH-04 backward-compat shims already removed by v20.0/v21.0) (validates ARCH-01, ARCH-02, ARCH-03)
  2. Unused internal modules are consolidated — `core/src/sync.rs` (12 LOC), `routing/HashRouter` (191 LOC), `ha/{FailoverManager,LeaderElection}` (328 LOC), `circuit_breaker/*` (556 LOC) are either deleted entirely or scoped to `pub(crate)` so they cannot leak into the public API; total LOC reduction ≥ 1000 lines (validates ARCH-04)
  3. Four stub architectures are collapsed — `gemma3`, `llama4`, `phi4`, `mistral_small` (combined ~1100 LOC) collapse into one parameterized `StubArchitecture` struct that accepts an architecture name string and produces the same shape of model metadata; no behavior change for test-suite consumers (validates ARCH-05)
  4. `core → model` upward dependency via `cuda-graph` feature is fixed — CUDA graph types (`CudaGraph`, `CudaGraphError`, capture/replay traits) are extracted from `vllm-model` into `vllm-traits` (preferred) or a new `vllm-kernels` crate below both `vllm-core` and `vllm-model`; `vllm-core` no longer requires the `cuda-graph` feature to depend on `vllm-model` for non-graph functionality; `cargo tree` shows clean layering (validates ARCH-06)
  5. Unused dependencies are cleaned — `reqwest` removed from `crates/server/Cargo.toml` (verify zero references in `crates/server/src/` first); `rayon` moved from `[dependencies]` to `[dev-dependencies]` in `crates/model/Cargo.toml` (only used in tests/benches); `cargo build` + `cargo test` both succeed with reduced dep graph (validates ARCH-07, ARCH-08)
  6. Duplicated implementations are unified — 3 `greedy_sample`/`argmax` implementations (`core/sampling.rs`, `model/causal_lm/mod.rs`, `engine/spec_dispatch/drafts.rs`) collapse to one canonical impl in `core/sampling.rs`; 2 `Architecture` types (`arch::Architecture` trait + `config::Architecture` enum) collapse to a single concept (single trait + associated enum, or single enum that implements a trait) (validates ARCH-09, ARCH-10)
  7. **FINAL-01**: All 1179 tests remain green post-cleanup — `cargo test --workspace --all-features` returns 0 failures; ARCH deletions + dependency moves + impl unification introduce no regressions
  8. **FINAL-02**: `cargo clippy --workspace --all-targets --all-features -- -D warnings` exits 0 — zero warnings, zero errors across all targets and features (validates FINAL-02)
  9. **FINAL-03**: `cargo fmt --all --check` exits 0 — workspace formatting matches rustfmt output (validates FINAL-03)
  10. **FINAL-04**: `cargo test --workspace --all-features` reports ≥ 1179 tests passing — no regression from v22.0 baseline; remediation scope, no expected growth (validates FINAL-04)
  11. **FINAL-05**: `.planning/PROJECT.md` (Current Milestone + Validated sections) and `.planning/STATE.md` (Current Position + Performance Metrics + v23.0 outcomes) are updated; v23.0 audit-remediated status declared; v24.0 candidates surfaced (long context, multimodal, multi-node resurrection, engine.rs full split, doc coverage push) (validates FINAL-05)

**Plans**: TBD

Plans:

- [ ] 43-01: Delete `scheduler/batch_planner.rs` (369 LOC) — ARCH-01
- [ ] 43-02: Delete `scheduler/predictive_batching.rs` (498 LOC) — ARCH-02
- [ ] 43-03: Delete `core/src/kv_cache/mod.rs` (7 LOC) pass-through shim — ARCH-03
- [ ] 43-04: Consolidate unused internal modules (`sync.rs`, `HashRouter`, `ha/`, `circuit_breaker/`) — ARCH-04
- [ ] 43-05: Collapse 4 stub architectures into parameterized `StubArchitecture` — ARCH-05
- [ ] 43-06: Fix `core → model` upward dependency via CUDA graph types → `vllm-traits` or new crate — ARCH-06
- [ ] 43-07: Remove unused `reqwest` from `crates/server/Cargo.toml` — ARCH-07
- [ ] 43-08: Move `rayon` to `[dev-dependencies]` in `crates/model/Cargo.toml` — ARCH-08
- [ ] 43-09: Unify 3 `greedy_sample`/`argmax` implementations to one canonical impl — ARCH-09
- [ ] 43-10: Unify 2 `Architecture` types (`arch::Architecture` trait + `config::Architecture` enum) — ARCH-10
- [ ] 43-11: FINAL-01 verification — `cargo test --workspace --all-features` ≥1179 pass post-cleanup
- [ ] 43-12: FINAL-02 verification — `cargo clippy --workspace --all-targets --all-features -- -D warnings` clean
- [ ] 43-13: FINAL-03 verification — `cargo fmt --all --check` clean
- [ ] 43-14: FINAL-04 verification — `cargo test --workspace --all-features` ≥1179 tests total
- [ ] 43-15: FINAL-05 — Update `.planning/PROJECT.md` + `.planning/STATE.md` with v23.0 outcomes

---

## Progress

**Execution Order:**
v22.0: 36 → 37 → 38 → 39 (linear chain; P0 bug fix → security wiring → polish → engine refactor + FINAL gates)
v23.0: 40 → 41 → 42 → 43 (linear chain; P0 code → stale docs → placeholder doc cleanup → architecture cleanup + FINAL gates)

| Phase                                                   | Milestone | Plans Complete | Status        | Completed    |
| ------------------------------------------------------- | --------- | -------------- | ------------- | ------------ |
| 36 Critical Bug Fixes (v22.1)                           | v22.0     | 4/4            | ✅ Complete   | 2026-06-27   |
| 37 Security Hardening (v22.2)                           | v22.0     | 7/7            | ✅ Complete   | 2026-06-27   |
| 38 Production Polish (v22.3)                            | v22.0     | 7/7            | ✅ Complete   | 2026-06-27   |
| 39 Engine Refactor + Final Verification (v22.4)         | v22.0     | 7/7            | ✅ Complete   | 2026-06-27   |
| 40 Critical Code Fixes (v23.1)                          | v23.0     | 0/TBD          | 🔵 Not started | —          |
| 41 Stale Documentation (v23.2)                          | v23.0     | 0/TBD          | 🔵 Not started | —          |
| 42 Placeholder Doc Cleanup (v23.3)                      | v23.0     | 0/TBD          | 🔵 Not started | —          |
| 43 Architecture Cleanup + Final Verification (v23.4)    | v23.0     | 0/TBD          | 🔵 Not started | —          |

**Historical phases:** See archived roadmaps at `.planning/milestones/v{16,17,18,19,20,21}.0-ROADMAP.md` and `.planning/milestones/v{20,21,22}.0-phases/` for full Phase Details of shipped milestones.

## Archived Milestones

For historical phase details, see:
- `.planning/milestones/v22.0-{ROADMAP,REQUIREMENTS,PROJECT}.md`
- `.planning/milestones/v22.0-phases/` (4 phase directories with CONTEXT/PLAN/SUMMARY per phase)
- `.planning/v22.0-MILESTONE-AUDIT.md`
- `.planning/milestones/v21.0-{ROADMAP,REQUIREMENTS,PROJECT}.md`
- `.planning/milestones/v21.0-phases/` (5 phase directories with CONTEXT/PLAN/SUMMARY per phase)
- `.planning/v21.0-MILESTONE-AUDIT.md`
- `.planning/milestones/v20.0-{ROADMAP,REQUIREMENTS,PROJECT}.md`
- `.planning/milestones/v20.0-phases/`
- `.planning/milestones/v19.0-{ROADMAP,REQUIREMENTS}.md`
- etc.

## Next Steps

To execute the v23.0 milestone:
1. `/gsd-discuss-phase 40` — gather context for the critical code fix phase
2. `/gsd-plan-phase 40` — produce PLAN.md for Phase 40
3. `/gsd-execute-phase 40` — run all plans in Phase 40
4. Repeat for Phase 41, 42, 43
5. `/gsd-complete-milestone` after Phase 43 FINAL gates green — archive v23.0 and prepare v24.0 candidates

---

*Roadmap updated: 2026-06-28 — v23.0 Audit Remediation milestone planning complete (Phases 40-43; 34 requirements mapped across 4 sub-phases; ~50h estimated; linear dependency chain 40→41→42→43; preserves v22.0 invariants: 1179+ tests green, clippy/fmt/doc clean, doc coverage 97.8%, #[deprecated] for public API changes, vllm-dist feature-gated; remediates 22 findings from v22.0 post-ship audit across 4 categories: P0 code defects, stale documentation, placeholder docs, dead code)*
