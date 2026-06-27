# Roadmap: vllm-lite

## Milestones

- ✅ **v16.0 Speculative Decoding** — Phases 16.1-16.4 (shipped 2026-04-28)
- ✅ **v17.0 Production Speculative Decoding** — Phases 17.1-17.4 (shipped 2026-05-13)
- ✅ **v18.0 Multi-Model Speculative Decoding** — Phases 18.1-18.4 + Phase 19 gap closure (shipped 2026-06-27)
- ✅ **v19.0 Codebase Health Audit** — Phases 20-24 (shipped 2026-06-27; analysis-only; see `.planning/audit/`)
- ✅ **v20.0 Codebase Remediation** — Phases 25-30 (shipped 2026-06-27; 48/48 reqs; 1144 tests; clippy/fmt clean; doc 97.8%)
- ✅ **v21.0 P2/P3 Backlog Cleanup** — Phases 31-35 (shipped 2026-06-27; 38 of 42 actively addressed; 1146 tests; 100% backlog closure)
- ✅ **v22.0 Production Hardening** — Phases 36-39 (shipped 2026-06-27; 21/21 reqs; 1179 tests; clippy/fmt/doc clean; P0 bug fixed, security wired, polish applied)

## Current Position

**Active milestone:** v22.0 Production Hardening ✅ SHIPPED (all 4 phases complete; awaiting `/gsd-audit-milestone` for milestone audit)

## Phases

- [x] **Phase 36: Critical Bug Fixes** (v22.1) — Fix `Engine::step()` speculative-mode hang; resolve cargo doc warnings; resolve gguf parser TODO ✅ shipped 2026-06-27
- [x] **Phase 37: Security Hardening** (v22.2) — JWT signature verification; wire RbacMiddleware; request size limits; audit log test; Grafana credentials to .env; TLS hardening ✅ shipped 2026-06-27
- [x] **Phase 38: Production Polish** (v22.3) — parking_lot::Mutex migration; speculative mock fate decision; MlaKvCache write_compressed perf; arch detection perf; LazyLock migration ✅ shipped 2026-06-27
- [x] **Phase 39: Engine Refactor + Final Verification** (v22.4) — Split engine.rs God module; unify engine/spec_dispatch tree; FINAL gates (clippy/fmt/test/docs) ✅ shipped 2026-06-27

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

## Progress

**Execution Order:**
v22.0: 36 → 37 → 38 → 39 (linear chain; P0 bug fix → security wiring → polish → engine refactor + FINAL gates)

| Phase                                                | Milestone | Plans Complete | Status      | Completed    |
| ---------------------------------------------------- | --------- | -------------- | ----------- | ------------ |
| 36 Critical Bug Fixes (v22.1)                        | v22.0     | 4/4            | ✅ Complete | 2026-06-27   |
| 37 Security Hardening (v22.2)                        | v22.0     | 7/7            | ✅ Complete | 2026-06-27   |
| 38 Production Polish (v22.3)                         | v22.0     | 7/7            | ✅ Complete | 2026-06-27   |
| 39 Engine Refactor + Final Verification (v22.4)      | v22.0     | 7/7            | ✅ Complete | 2026-06-27   |

**Historical phases:** See archived roadmaps at `.planning/milestones/v{16,17,18,19,20,21}.0-ROADMAP.md` and `.planning/milestones/v{20,21}.0-phases/` for full Phase Details of shipped milestones.

## Archived Milestones

For historical phase details, see:
- `.planning/milestones/v21.0-{ROADMAP,REQUIREMENTS,PROJECT}.md`
- `.planning/milestones/v21.0-phases/` (5 phase directories with CONTEXT/PLAN/SUMMARY per phase)
- `.planning/v21.0-MILESTONE-AUDIT.md`
- `.planning/milestones/v20.0-{ROADMAP,REQUIREMENTS,PROJECT}.md`
- `.planning/milestones/v20.0-phases/`
- `.planning/milestones/v19.0-{ROADMAP,REQUIREMENTS}.md`
- etc.

## Next Steps

To execute the v22.0 milestone:
1. `/gsd-discuss-phase 36` — gather context for the critical bug fix phase
2. `/gsd-plan-phase 36` — produce PLAN.md for Phase 36
3. `/gsd-execute-phase 36` — run all plans in Phase 36
4. Repeat for Phase 37, 38, 39
5. `/gsd-complete-milestone` after Phase 39 FINAL gates green — archive v22.0 and prepare v23.0 candidates

---

*Roadmap updated: 2026-06-27 — v22.0 Production Hardening milestone planning complete (Phases 36-39; 21 requirements mapped across 4 sub-phases; ~75h estimated; linear dependency chain 36→37→38→39; preserves v21.0 invariants: 1146+ tests green, clippy/fmt clean, doc coverage 97.8%, #[deprecated] for public API changes, vllm-dist feature-gated)*
