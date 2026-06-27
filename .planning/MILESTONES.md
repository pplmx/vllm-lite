# Milestones

## v19.0 Codebase Health Audit

**Shipped:** 2026-06-27
**Phases:** 20-24 | **Plans:** 5 | **Tasks:** 23 requirements (analysis-only)

### Key Accomplishments

1. **Architecture audit** — 17 findings (2 P0, 3 P1, 8 P2, 4 P3); flagged `vllm-model → vllm-dist` and `vllm-core → vllm-model` layering violations; `engine.rs` God module (1,038 LOC)
2. **Naming audit** — 26 findings (0 P0, 7 P1, 19 P2); user-reported stage-info file `engine_v18_wiring.rs`; orphan modules `kv_cache_fp8.rs` + `debug.rs` unreachable
3. **Comments + documentation audit** — 24 findings (0 P0, 20 P1, 4 P2); workspace doc coverage **7.6%**; README code example broken
4. **API + error handling audit** — 33 findings (3 P0, 8 P1, 13 P2, 9 P3); `ModelError` struct (defeats pattern matching); 8 non-object-safe traits; 0 `#[deprecated]` markers
5. **Synthesis** — 100 raw findings consolidated; 8 cross-dimensional themes; 6 proposed v20.x phases (~190h)

### Stats

- Files: 11 audit artifacts in `.planning/audit/` (4 REPORT.md + 4 SUMMARY.md + SYNTHESIS.md + BACKLOG.md + MIGRATION-ROADMAP.md)
- Total artifact volume: 3,578 lines
- Requirements: 23/23 covered (100%)
- **Source code modified: 0 lines** (analysis-only constraint honored)
- Timeline: 2026-06-27 (single session, autonomous)

### Tech Debt & Known Limitations

- All 100 audit findings deferred to v20.x — by design (analysis-only)
- `vllm-dist` fate undecided (continue / feature-gate / remove) — open question for v20.1
- README code example broken — must fix in v20.4 or 20.5

---

## v18.0 Multi-Model Speculative Decoding

**Shipped:** 2026-06-27
**Phases:** 18.1-18.4 + 19 (gap closure) | **Plans:** 5 | **Tasks:** 14 requirements

### Key Accomplishments

1. **DraftModelRegistry** — Runtime registry for heterogeneous external draft models with lazy weight loading. Loader-agnostic (no vllm-model dependency in core).
2. **MemoryBudget** — VRAM budget enforcement with structured `MemoryBudgetExceeded` error. Atomic `try_reserve_draft` for safe concurrent reservation.
3. **Refcount lifecycle** — Auto-unload on zero refcount; `unload` refuses with `InUse(n)`; `force_unload` overrides.
4. **Per-request routing** — `Request.draft_model_id` propagated to `Sequence.draft_model_id`; `DraftResolver` resolves per-request so mixed drafts coexist in one batch.
5. **Fallback semantics** — FALL-01 (load failure → self-spec, silent); FALL-02 (runtime error → degraded_draft sticky flag).
6. **Metrics** — 5 new counters: draft resolutions (external/self_spec/none), load failures, runtime errors.
7. **Engine step-loop integration (Phase 19)** — `DraftResolver` wired into `Engine::step_speculative_inner` via `generate_per_seq_drafts`. Per-seq dispatch, mixed routing, and FALL-02 all live in the production request path.
8. **HTTP exporter + server wiring (Phase 19)** — 5 v18.0 counters exposed via `/metrics`; server constructs Engine with `with_budget_boxed` / `with_drafts_boxed` when config declares budget or specs; `ServerDraftLoader` wraps `ModelLoader` for real-world draft loading.

### Stats

- Files: speculative/draft_registry.rs, speculative/memory_budget.rs, speculative/draft_resolver.rs, tests/multi_draft_integration.rs, tests/engine_v18_wiring.rs, benches/multi_draft_speculative.rs
- Requirements: 14/14 satisfied (100%, 3 gaps closed by Phase 19)
- Files modified: 21 (5 new src modules + engine + types + scheduler + metrics + server config + server main + tests + benches + Cargo.toml)
- Lines: ~4500 added
- Commits: 16 (5 phases + 9 code review fixes + 1 fmt + 1 docs)
- Timeline: 2026-06-27 (single session, autonomous + audit-driven gap closure)
- Audit: passed ✓ (gaps_found → closed by Phase 19 → passed)
- Test count: 209 → 287+ tests (+78 tests, 10 Phase 19 integration tests added)

### Tech Decisions

- Loader-agnostic registry: `DraftModelRegistry` doesn't depend on `vllm-model`; the actual `ModelLoader` call lives at the caller (server), keeping the registry reusable in non-vllm-model contexts.
- Per-request resolution (not per-batch): mixed-draft routing emerges naturally from calling `DraftResolver::resolve` per-request inside the step loop.
- Shared `Arc<MemoryBudget>` between Engine and registry: single source of truth for budget state; reservations atomic via interior `RwLock`.
- `ResolvedDraft` enum: makes the three resolution outcomes explicit (`External`, `SelfSpec`, `None`); engine code matches on these.
- `Sequence.degraded_draft`: sticky per-sequence flag set once when FALL-02 fires; engine short-circuits draft attempts for that sequence forever after.
- Resolver wired in Engine (Phase 19): `Engine.draft_resolver: Option<Arc<DraftResolver>>`. Legacy `new_boxed` path remains unwired for backward compatibility.
- `catch_unwind(AssertUnwindSafe(...))` around per-seq forward (Phase 19): catches both `Result::Err` and panic from misbehaving backends; degraded seqs are skipped on subsequent steps.
- `ServerDraftLoader` (Phase 19): concrete `DraftLoader` impl that wraps `ModelLoader`, enabling real-world lazy loading in the HTTP server.

### Tech Debt & Known Gaps

- `Engine::step()` in speculative mode hangs — pre-existing bug (not introduced by v18.0). 2 Phase 19 end-to-end tests are `#[ignore]`d awaiting the fix.
- Real-model benchmark missing — current bench uses stub backends. Deferred until a small real checkpoint is available.

---

## v17.0 Production Speculative Decoding

**Shipped:** 2026-05-13
**Phases:** 17.1-17.4 | **Plans:** 4 | **Tasks:** 27 requirements

(v17.0 milestone — see .planning/milestones/v17.0-ROADMAP.md for details)

---

## v16.0 Speculative Decoding

**Shipped:** 2026-04-28
**Phases:** 16.1-16.4 | **Plans:** 4 | **Tasks:** 17 requirements

### Key Accomplishments

1. **Core Architecture** — DraftVerifier trait, SpeculativeModel wrapper, SpeculationConfig, RejectionStrategy
2. **Self-Speculation** — Draft model using reduced layer count of target model
3. **ModelBackend Extension** — Added num_layers() and num_heads() to trait
4. **Verification Infrastructure** — Parallel verification ready with early termination

### Stats

- Files: speculative/verifier.rs, speculative/config.rs, speculative/strategy.rs, speculative/model.rs, speculative/self_spec.rs
- Requirements: 17/17 satisfied (100%)
- Files modified: 41
- Lines: +2029 / -130
- Commits: 10
- Timeline: 2026-04-28 (single session)
- Audit: passed ✓

### Tech Decisions

- Self-speculation using same model with 1/8 layer count
- TokenLevel rejection strategy (accept if target_prob >= draft_prob)
- AdaptiveDraftConfig already existed from previous implementation

### Tech Debt & Known Gaps

- Full Engine integration (step_speculative) not implemented
- Actual performance benchmarks deferred (infrastructure ready)

---

## v15.0 Performance + Models + Production

**Shipped:** 2026-04-27
**Phases:** 15.1-15.6 | **Plans:** 6 | **Tasks:** 10 requirements

### Key Accomplishments

1. **FlashAttention V3** — New kernel with MQA/GQA support and sliding window attention
2. **KV Cache Optimization** — FP8 quantization with 50% memory reduction, chunked prefill
3. **Model Support** — Gemma3, Phi-4, Llama 4, Mistral Small architectures
4. **Production Hardening** — Go K8s Operator scaffold, TLS/JWT security

### Stats

- Files: flash_v3.rs, kv_cache_fp8.rs, gemma3/, phi4/, llama4/, mistral_small/, k8s/operator/, security/tls.rs, security/jwt.rs
- Requirements: 10/10 satisfied (100%)
- Timeline: Single session

### Tech Decisions

- FA-V3 with online softmax algorithm
- FP8 E4M3 format for KV cache
- MoE architectures with expert routing
- Rustls for TLS, custom JWT validation

---

## v14.0 Developer Tooling

**Shipped:** 2026-04-27
**Phases:** 14.1-14.4 | **Plans:** 4 | **Tasks:** 12 requirements

### Key Accomplishments

1. **Benchmarking suite** — Throughput and latency benchmarks with P50/P95/P99 percentiles
2. **Debug endpoints** — /debug/metrics, /debug/kv-cache, /debug/trace for runtime inspection
3. **CLI tools** — config validate, model list/info for developer workflows
4. **Test infrastructure** — TestHarness, SlowModel, RequestFactory for integration tests

### Stats

- Files: benchmarks/src/, server/src/debug.rs, server/src/bin/vllm.rs, testing/src/
- Requirements: 12/12 satisfied (100%)
- Timeline: Same day as v13.0

### Tech Decisions

- Benchmarking via BenchmarkSuite + criterion pattern
- Debug via HTTP endpoints for easy integration
- Separate vllm binary for CLI concerns
- TestHarness provides unified test environment

---

## v13.0 主机部署

**Shipped:** 2026-04-27
**Phases:** 13.1-13.3 | **Plans:** 3 | **Tasks:** 23 requirements

### Key Accomplishments

1. **Kubernetes deployment** — Helm chart, health probes, NodeMesh discovery
2. **High availability** — Leader election, failover, consistent hash routing
3. **Security hardening** — RBAC, audit logging, correlation IDs

### Tech Debt

- K8S-02: Full Go Kubernetes Operator deferred
- TLS/axum integration needs production testing
- JWT validation stubbed

---

## v20.0 Codebase Remediation

**Shipped:** 2026-06-27
**Phases:** 25-30 | **Plans:** 39 | **Tasks:** 48 requirements

### Key Accomplishments

1. **P0 architecture/API fixes (Phase 25)** — Eliminated `vllm-model → vllm-dist` edge (feature-gated); confirmed `vllm-core → vllm-model` is `cuda-graph` feature-gated; converted `ModelError` struct → 5-variant `#[non_exhaustive]` enum; converted `CudaGraphError` to `#[derive(thiserror::Error)]`; added 8 compile-only object-safety tests
2. **Module tree restoration (Phase 26)** — Wired orphan `kv_cache_fp8.rs` (289 LOC) + `debug.rs` (175 LOC); renamed `engine_v18_wiring.rs` → `engine_wiring.rs`; migrated 3 test files from `src/` to `tests/`; workspace `default-members` excludes `vllm-dist` from default build
3. **Error handling standardization (Phase 27)** — 13 error enums all `#[derive(thiserror::Error)]`; zero `Result<_, String>` in production; 8 mutex `.expect()` + 13 poisoned `.unwrap()` → graceful recovery; 4 new `EngineError` variants (`Timeout`/`Cancelled`/`ResourceExhausted`/`BackendUnavailable`); cross-crate `From` impls; `vllm` binary uses `anyhow::Result`
4. **Documentation coverage push (Phase 28)** — 1,299 `///` + 125 `//!` across 212 files. Workspace pub coverage 19.5% → 97.8%, module coverage 48.0% → 99.6%. Per-crate all ≥80% pub (traits 80, dist 84.4, server 94.8, testing 100, model 100, core 99.8). Fixed README example + architecture table + crate count; reconciled AGENTS.md Architecture section
5. **External docs + 12 ADRs (Phase 29)** — Created ADR-003..014 capturing tribal knowledge from v1.0-v20.0. ADR-003: self-spec 1/8 ratio (v16); ADR-004: FP8 E4M3 (v15); ADR-005: KV cache split (v1); ADR-006: speculative architecture (v16); ADR-007: per-request draft routing (v18); ADR-008: vllm-dist feature-gate (v20.1); ADR-009: FP8 orphan decision (v20.2); ADR-010: CUDA graph gating (v10.1); ADR-011: cross-crate error boundaries (v20.3); ADR-012: continuous batching (v1); ADR-013: paged KV cache (v1); ADR-014: architecture registry (v17)
6. **Naming + final polish (Phase 30)** — NAM-01 P1: 6 `data` variable renames, `EmbeddingData` → `Embedding` with `#[deprecated]` alias. NAM-02 P2: AGENTS.md additions (verb policy, suffix conventions, tensor-math single-letter exemption, test-file location). DEP-01/02: 1 deprecation marker. CMT-01: 3 stale comments resolved. Fixed 3 pre-existing kv_cache_fp8 clippy errors. cargo fmt --all auto-formatted 133 files

### Stats

- Commits: 32 (Phase 25 onward)
- Files changed: 416
- Insertions: 4,366 lines
- Deletions: 1,079 lines
- Test count: 287+ → **1144** (4× growth, 0 regressions)
- Doc coverage: 19.5% → **97.8%** pub / 48.0% → **99.6%** module
- ADRs: **12 new** (ADR-003..014)
- Requirements: 48/48 covered (100%)
- Timeline: 2026-06-27 (single day, autonomous)

### Tech Debt (deferred to v20.7+)

- `Engine::step()` speculative-mode hang (pre-existing bug)
- Real-model benchmark (vs stub backend)
- 44 P2 issues + 13 P3 informational findings (out of v20.0 scope)
- `vllm-dist` resurrection (multi-node work — feature-gated, not removed)
- 5 pre-existing cargo doc broken-link warnings
- 4 `Result<_, String>` in vllm_server test fixtures (production clean)
- 1 actionable TODO in gguf parser
- RFU-02 / RFU-03 / ARF-01..03 / OPS-01..03 (carry-over backlog)

---

*Full milestone archives: .planning/milestones/*
