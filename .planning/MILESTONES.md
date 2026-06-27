# Milestones

## v18.0 Multi-Model Speculative Decoding

**Shipped:** 2026-06-27
**Phases:** 18.1-18.4 | **Plans:** 4 | **Tasks:** 14 requirements

### Key Accomplishments

1. **DraftModelRegistry** — Runtime registry for heterogeneous external draft models with lazy weight loading. Loader-agnostic (no vllm-model dependency in core).
2. **MemoryBudget** — VRAM budget enforcement with structured `MemoryBudgetExceeded` error. Atomic `try_reserve_draft` for safe concurrent reservation.
3. **Refcount lifecycle** — Auto-unload on zero refcount; `unload` refuses with `InUse(n)`; `force_unload` overrides.
4. **Per-request routing** — `Request.draft_model_id` propagated to `Sequence.draft_model_id`; `DraftResolver` resolves per-request so mixed drafts coexist in one batch.
5. **Fallback semantics** — FALL-01 (load failure → self-spec, silent); FALL-02 (runtime error → degraded_draft sticky flag).
6. **Metrics** — 5 new counters: draft resolutions (external/self_spec/none), load failures, runtime errors.

### Stats

- Files: speculative/draft_registry.rs, speculative/memory_budget.rs, speculative/draft_resolver.rs, tests/multi_draft_integration.rs, benches/multi_draft_speculative.rs
- Requirements: 14/14 satisfied (100%)
- Files modified: 17 (3 new src modules + engine + types + scheduler + metrics + tests + benches + Cargo.toml)
- Lines: ~3500 added
- Commits: 4 (one per phase)
- Timeline: 2026-06-27 (single session, autonomous)
- Audit: passed ✓
- Test count: 209 → 277 (+68 tests)

### Tech Decisions

- Loader-agnostic registry: `DraftModelRegistry` doesn't depend on `vllm-model`; the actual `ModelLoader` call lives at the caller (server), keeping the registry reusable in non-vllm-model contexts.
- Per-request resolution (not per-batch): mixed-draft routing emerges naturally from calling `DraftResolver::resolve` per-request inside the step loop.
- Shared `Arc<MemoryBudget>` between Engine and registry: single source of truth for budget state; reservations atomic via interior `RwLock`.
- `ResolvedDraft` enum: makes the three resolution outcomes explicit (`External`, `SelfSpec`, `None`); engine code matches on these.
- `Sequence.degraded_draft`: sticky per-sequence flag set once when FALL-02 fires; engine short-circuits draft attempts for that sequence forever after.

### Tech Debt & Known Gaps

- Engine `step_speculative` still uses the single `draft_model` field from v17. The DraftResolver exists and is unit-tested but isn't yet called from the step loop. Deferred to v19.0.
- HTTP exporter doesn't yet expose v18.0 metrics (counters exist in `EnhancedMetricsCollector`). Deferred to v19.0.
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

*Full milestone archives: .planning/milestones/*
