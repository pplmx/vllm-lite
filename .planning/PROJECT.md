# vllm-lite

## What This Is

A production-ready LLM inference engine in Rust, optimized for single and multi-node GPU deployment with Kubernetes integration, high availability, and enterprise security.

## Core Value

Fast, memory-efficient LLM inference with continuous batching, paged KV cache, and tensor parallelism — deployed with production-grade ops tooling and security.

## Current Milestone: v20.0 Codebase Remediation (BACKLOG.md-driven)

**Goal:** 执行 v19.0 审计产出的修复 backlog,按 MIGRATION-ROADMAP.md 提议顺序执行 6 个子 phase(v20.1→v20.6),系统化恢复 codebase 健康度;**这是 v19.0 审计清单的修复 milestone**,所有改动基于 `.planning/audit/BACKLOG.md` 的 100 个 finding。

**Target features (按 phase 分组):**

- **Phase 25 (v20.1): P0 关键修复** — 消除 `vllm-model → vllm-dist` 依赖 (feature-gate),`ModelError` struct → enum,8 个 non-object-safe trait 同步修
- **Phase 26 (v20.2): 模块树恢复** — orphan 模块挂回,stage-info 文件改名,3 个未注册 test 文件迁出
- **Phase 27 (v20.3): 错误处理标准化** — 13 个错误类型统一,`Result<_, String>` 消除,context propagation
- **Phase 28 (v20.4): 文档覆盖率提升** — workspace 7.6% → ≥60%,`///` 补 776 个 pub item,121 个文件补 `//!`,README 修复
- **Phase 29 (v20.5): 外部文档调和** — README/AGENTS.md 准确性,补 12+ 缺失 ADR
- **Phase 30 (v20.6): 命名 + 收尾** — 7 P1 + 19 P2 命名,弃用卫生,注释清理

## Current State

**Current Milestone:** v20.0 Codebase Remediation (planning)
**Latest Shipped:** v19.0 Codebase Health Audit (2026-06-27, 23/23 requirements, audit passed, 0 source code modified)
**Status:** v19.0 收官;v20.0 修复 milestone 启动

### Phase 17 Achievements (v17.0 shipped)

- ✅ Engine integration (`step_speculative_inner`, commit `52f77ce`)
- ✅ Seamless fallback parity tests (`qwen3_5/speculative_tests.rs`)
- ✅ Real hardware benchmark suite (`latency_percentiles`, `speculative_vs_baseline`, `bench_throughput`)
- ✅ Adaptive draft depth (`AdaptiveSpeculativeDecoder` + EWMA + deadband + cooldown)
- ✅ Speculative warmup (`warmup_draft_kv` after prefill)
- ✅ Acceptance rate monitoring + Prometheus + `/debug/metrics`
- ⏸ MULTI-01/02 deferred to v18.0

### Phase 16 Achievements

- ✅ Speculative Decoding architecture (DraftVerifier, SpeculativeModel, Config)
- ✅ Self-speculation with reduced layer count and weight sharing
- ✅ Parallel verification with token-level rejection
- ✅ Draft accuracy tracking and metrics infrastructure
- ✅ ModelBackend trait extended with num_layers()/num_heads()

## Requirements

### Validated

<!-- Shipped from Phase 1-12 -->
- ✓ 核心推理引擎 — Continuous Batching, Paged KV Cache, Prefix Caching (Phase 1)
- ✓ 多模型支持 — Llama, Mistral, Qwen, DeepSeek, Gemma4, Mixtral (Phase 6)
- ✓ OpenAI 兼容 API — Chat, Completions, Embeddings (Phase 7)
- ✓ 生产就绪 — 监控、日志、可靠性 (Phase 5)
- ✓ FlashAttention V2 实现 (Phase 10.1)
- ✓ CUDA Graph 优化完善 (Phase 10.1)
- ✓ PD 分离完善 (Phase 10.2)
- ✓ Chunked Prefill 优化 (Phase 10.2)
- ✓ 性能基准测试 (Phase 10.3)
- ✓ Pipeline Parallelism (Phase 11.1)
- ✓ Distributed KV Cache (Phase 11.2)
- ✓ AWQ/GPTQ quantization support (Phase 12.1)
- ✓ Backpressure handling for streaming (Phase 12.2)
- ✓ Predictive batching (Phase 12.3)

<!-- Shipped from Phase 13 -->
- ✓ Multi-node cluster support — v13.0 (NodeMesh, gRPC, consistent hash)
- ✓ Kubernetes integration — v13.0 (Helm chart, health probes, ConfigMap)
- ✓ High availability — v13.0 (leader election, failover, HPA metrics)
- ✓ Security hardening — v13.0 (TLS, mTLS, RBAC, audit logging)
- ✓ Structured logging with correlation IDs — v13.0

<!-- Shipped from Phase 14 -->
- ✓ Benchmarking suite — v14.0 (Throughput, latency, P50/P95/P99)
- ✓ Debug endpoints — v14.0 (/debug/metrics, /debug/kv-cache, /debug/trace)
- ✓ CLI tools — v14.0 (config validate, model list/info)
- ✓ Test infrastructure — v14.0 (TestHarness, SlowModel, RequestFactory)

<!-- Shipped from Phase 16 -->
- ✓ Speculative Decoding architecture — v16.0 (DraftVerifier, SpeculativeModel, Config)
- ✓ Self-speculation with layer sharing — v16.0 (1/8 layer count, weight reuse)
- ✓ Parallel verification infrastructure — v16.0 (token acceptance, early termination)
- ✓ Draft accuracy metrics — v16.0 (DraftAccuracyTracker, acceptance rate)

<!-- Shipped from Phase 17 (v17.0) -->
- ✓ Engine integration — v17.0 (`step_speculative_inner`, commit `52f77ce`)
- ✓ Seamless fallback parity tests — v17.0 (`qwen3_5/speculative_tests.rs`)
- ✓ Real hardware benchmark suite — v17.0 (`latency_percentiles`, `speculative_vs_baseline`)
- ✓ Baseline comparison benchmarks — v17.0 (`bench_throughput`)
- ✓ Adaptive draft depth — v17.0 (`AdaptiveSpeculativeDecoder` + EWMA + deadband)
- ✓ Acceptance rate monitoring — v17.0 (Prometheus `speculative_adjustments_total`)
- ✓ Speculative warmup — v17.0 (`Engine::warmup_draft_kv` after prefill)

<!-- Shipped from Phase 18 (v18.0) -->
- ✓ DraftModelRegistry — v18.0 (runtime registry, lazy weight loading, loader-agnostic)
- ✓ MemoryBudget — v18.0 (VRAM budget enforcement, atomic `try_reserve_draft`)
- ✓ Refcount lifecycle — v18.0 (auto-unload on zero refcount; `unload` vs `force_unload`)
- ✓ Per-request routing — v18.0 (`Request.draft_model_id`, `DraftResolver` for mixed routing)
- ✓ Fallback semantics — v18.0 (FALL-01 silent fallback, FALL-02 sticky `degraded_draft`)
- ✓ Phase 19 gap closure — v18.0 (resolver wired into `step_speculative_inner`, HTTP exporter, server config, `ServerDraftLoader`)

<!-- Shipped from v19.0 (Codebase Health Audit) -->
- ✓ Architecture audit — v19.0 (17 findings; 2 P0 layering violations, 1 P1 God module)
- ✓ Naming audit — v19.0 (26 findings; 7 P1; orphan modules `kv_cache_fp8.rs`/`debug.rs`; stage-info file `engine_v18_wiring.rs`)
- ✓ Comments + documentation audit — v19.0 (24 findings; 20 P1; doc coverage 7.6%; README broken example)
- ✓ API + error handling audit — v19.0 (33 findings; 3 P0; `ModelError` struct; 8 non-object-safe traits)
- ✓ Synthesis + remediation backlog — v19.0 (100 findings consolidated; 8 themes; 6 proposed v20.x phases; ~190h)

<!-- Shipped from Phase 15 -->
- ✓ FlashAttention V3 — v15.0 (MQA/GQA, sliding window)
- ✓ KV cache FP8 quantization — v15.0 (50% memory reduction)
- ✓ Chunked prefill — v15.0 (large prompt handling)
- ✓ Gemma3 architecture — v15.0 (sliding window attention)
- ✓ Phi-4 architecture — v15.0 (rotary embedding)
- ✓ Llama 4 architecture — v15.0 (MoE support)
- ✓ Mistral Small architecture — v15.0 (expert routing)
- ✓ Go K8s Operator scaffold — v15.0 (controller-runtime)
- ✓ TLS termination — v15.0 (rustls)
- ✓ JWT validation — v15.0 (middleware)

### Active

**v20.0: Codebase Remediation (BACKLOG.md-driven)**

#### Phase 25 (v20.1): P0 关键修复

- [ ] **P0-01**: Eliminate `vllm-model → vllm-dist` dependency (feature-gate `vllm-dist`)
- [ ] **P0-02**: Make `vllm-core → vllm-model` feature-gated
- [ ] **P0-03**: Convert `ModelError` struct → enum (pattern-matchable)
- [ ] **P0-04**: Make 8 non-object-safe traits object-safe (add `where Self: Sized` or split traits)
- [ ] **P0-05**: Refactor `CudaGraphError` to use thiserror (currently hand-rolled Display/Error)

#### Phase 26 (v20.2): 模块树恢复

- [ ] **MT-01**: Wire `kv_cache_fp8.rs` into module tree (currently orphan, 289 LOC unreachable)
- [ ] **MT-02**: Wire `debug.rs` into module tree (currently orphan, 175 LOC unreachable)
- [ ] **MT-03**: Rename `engine_v18_wiring.rs` → `engine_wiring.rs` (stage-info file)
- [ ] **MT-04**: Migrate `qwen3/model_tests.rs` from `src/` → `tests/`
- [ ] **MT-05**: Migrate `qwen3_5/model_tests.rs` from `src/` → `tests/`
- [ ] **MT-06**: Migrate `qwen3_5/speculative_tests.rs` from `src/` → `tests/`
- [ ] **MT-07**: Decide and resolve fate of `vllm-dist` (~1,600 LOC unused) — feature-gate per v20.1

#### Phase 27 (v20.3): 错误处理标准化

- [ ] **ERR-01**: Eliminate 10 `Result<_, String>` anti-patterns (use proper error types)
- [ ] **ERR-02**: Convert `CudaGraphError` to thiserror enum (P0-05 redux for clarity)
- [ ] **ERR-03**: Refactor 25+ mutex-poison `.expect()` calls (use `.context()` or convert to Result)
- [ ] **ERR-04**: Add missing `From` impls for cross-crate error conversion
- [ ] **ERR-05**: Add `EngineError::Timeout`, `Cancelled`, `ResourceExhausted`, `BackendUnavailable` variants
- [ ] **ERR-06**: Adopt `anyhow` for top-level server error reporting
- [ ] **ERR-07**: Add `.context()` to error propagation paths losing information

#### Phase 28 (v20.4): 文档覆盖率提升

- [ ] **DOC-01**: Raise workspace doc coverage 7.6% → ≥60%
- [ ] **DOC-02**: Add `///` doc comments to 776 undocumented `pub` items
- [ ] **DOC-03**: Add `//!` module-level docs to 121 files lacking them
- [ ] **DOC-04**: Fix README.md broken code example (`SchedulerEngine::new(config, 1024)` → 3-arg form)
- [ ] **DOC-05**: Update README.md "Supported Architectures" list (currently lists 5-6 vs 10 actually registered)
- [ ] **DOC-06**: Reconcile crate count in README.md (claims 7, Cargo.toml has 6)
- [ ] **DOC-07**: Update AGENTS.md "Architecture" section to match current state

#### Phase 29 (v20.5): 外部文档调和 + ADRs

- [ ] **EXT-01**: Reconcile README.md vs AGENTS.md vs Cargo.toml claims
- [ ] **EXT-02**: Create ADR: "Why self-spec uses 1/8 layer count" (tribal knowledge)
- [ ] **EXT-03**: Create ADR: "FP8 E4M3 format choice for KV cache"
- [ ] **EXT-04**: Create ADR: "KV cache split strategy (prefix vs paged)"
- [ ] **EXT-05**: Create ADR: "Speculative decoding architecture overview"
- [ ] **EXT-06**: Create ADR: "Per-request draft routing (RTE-01..03)"
- [ ] **EXT-07**: Create ADR: "Why `vllm-dist` is feature-gated (v20.1 decision)"
- [ ] **EXT-08**: Create ADR: "FP8 quantizer orphan module decision (v20.2 outcome)"
- [ ] **EXT-09**: Create ADR: "Cuda graph feature gating strategy"
- [ ] **EXT-10**: Create ADR: "Cross-crate error type boundaries (v20.3 decision)"
- [ ] **EXT-11**: Document 2+ additional ADRs from v15.0-v18.0 tribal knowledge
- [ ] **EXT-12**: Update `.planning/PROJECT.md` Core Value section if drifted

#### Phase 30 (v20.6): 命名 + 收尾

- [ ] **NAM-01**: Apply 7 P1 naming fixes (variable single-letter, redundant suffixes, etc.)
- [ ] **NAM-02**: Apply 19 P2 naming consistency fixes
- [ ] **DEP-01**: Add `#[deprecated]` markers to public API items removed or replaced in v20.0
- [ ] **DEP-02**: Provide migration paths for all newly-deprecated items
- [ ] **CMT-01**: Clean up stale comments referencing old code (per DOCS-03 audit)
- [ ] **CMT-02**: Remove or update dead TODOs / FIXMEs discovered during v19.0 audit
- [ ] **FINAL-01**: Verify all 287+ tests pass post-remediation
- [ ] **FINAL-02**: Verify `cargo clippy --workspace -- -D warnings` clean
- [ ] **FINAL-03**: Verify `cargo fmt --all --check` clean
- [ ] **FINAL-04**: Update `.planning/PROJECT.md` and `.planning/STATE.md` with v20.0 outcomes

### Out of Scope (v20.0)

- New features — milestone is purely remediation; no new capabilities
- Performance optimization — orthogonal to audit findings; separate cycle if needed
- Real-model benchmarks — already deferred from v18.0
- New architecture (e.g., adding a new crate) — out of scope; v20.0 only removes/fixes
- Tree-based speculation — still too complex (carried from v18.0)
- vllm-dist resurrection — feature-gated but not deleted; multi-node work is future

### Out of Scope (carried from earlier milestones)

- Tree-based speculation (draft tree) — sigmoidally complex
- Medusa-style multiple heads — incompatible with off-the-shelf models
- Speculative decoding for prefill — compute-bound, only decode
- Dynamic model switching mid-request — complex state management
- Draft model retraining/fine-tuning — out of engine scope

## v20.0 Replaces (Deferred Then Promoted) — now active

The v19.0 audit identified 100 findings (5 P0 + 38 P1 + 44 P2 + 13 P3). v20.0 executes the remediation backlog in 6 sub-phases per `.planning/audit/MIGRATION-ROADMAP.md`:

| Sub-phase | Focus | Estimate |
|-----------|-------|---------:|
| 25 (v20.1) | P0 critical fixes + object-safety + vllm-dist feature-gate | ~22h |
| 26 (v20.2) | Module tree restoration + orphan files | ~12h |
| 27 (v20.3) | Error handling standardization | ~30h |
| 28 (v20.4) | Doc coverage 7.6% → ≥60% | ~40h |
| 29 (v20.5) | External docs reconciliation + ADRs | ~16h |
| 30 (v20.6) | Naming + deprecation hygiene + final verification | ~20h |
| **Total** | | **~190h** |

Key v20.0 decisions (from user input):
- **vllm-dist fate**: feature-gate (not remove, not retain)
- **Object-safety + ModelError**: fixed together in v20.1 (both P0)
- **Scope**: all 6 sub-phases within single v20.0 milestone

## v18.0 Replaces (Deferred Then Promoted) — now historical

The following v17 deferred items shipped in v18.0:

- ✓ **SPEC-MULTI-01 → MMLT-01..03** (external draft model support)
- ✓ **SPEC-MULTI-02 → LIFE-01..03** (lifecycle management)
- ✓ **SPEC-MULTI-03 → MEM-01..03** (GPU memory budgeting)
- ✓ **NEW: RTE-01..03** (request-level routing) — emerged from "请求间动态选择" design decision
- ✓ **NEW: FALL-01..02** (fallback semantics) — required for safety in production

### Out of Scope

- WebAssembly support — 长期愿景
- Multi-tenant isolation — Enterprise feature
- Online fine-tuning — 长期愿景
- Real-time fine-tuning — 长期愿景
- Vision end-to-end — deferred from v14.0 (architecture only)

## Context

v19.0 audit shipped 23/23 requirements, 0 source code modified (analysis-only). 100 findings consolidated:

- **5 P0** (must fix): layering violations, `ModelError` struct, non-object-safe traits
- **38 P1** (should fix): orphan modules, stage-info files, doc coverage 7.6%, error ergonomics
- **44 P2** + **13 P3** (optional)

Full backlog: `.planning/audit/BACKLOG.md` — proposed phasing: `.planning/audit/MIGRATION-ROADMAP.md`

v20.0 build-on:

- Direct execution of prioritized audit findings, organized into 6 sub-phases
- Backward-compat: existing public API must remain stable (with `#[deprecated]` for removed items)
- All 287+ existing tests must remain green
- v18.0 speculative decoding functionality preserved

Tech stack: Rust + Candle, multi-GPU CUDA support, Kubernetes, gRPC.
Codebase state (v19.0 end): 7 crates; speculative decoding complete (v18.0); draft registry + memory budget live in core; HTTP server with OpenAI-compatible API; 100 remediation findings ready.

## Constraints

- **Tech**: Rust + Candle, multi-GPU CUDA support
- **Compatibility**: Must maintain single-GPU API compatibility
- **Performance**: Scale-up with linear throughput improvement
- **Cluster**: Multi-node coordination via gRPC

## Key Decisions

| Decision                  | Rationale                                          | Outcome                     |
| ------------------------- | -------------------------------------------------- | --------------------------- |
| Multi-node architecture   | Horizontal scaling beyond single host              | Implemented — v13.0         |
| K8s Operator vs Helm-only | Operator for declarative management                | Scaffolded — v15.0          |
| Consensus protocol        | Raft vs etcd for HA leader election                | Using K8s Lease API — v13.0 |
| TLS approach              | mTLS for cluster internal, simple TLS for external | Implemented — v15.0         |
| FA-V3 kernel approach     | FlashAttention V3 integration                      | Implemented — v15.0         |
| KV cache compression      | FP8 E4M3 format                                    | Implemented — v15.0         |
| Speculative strategy      | Self-speculation with 1/8 layer count              | Implemented — v16.0         |
| Token rejection           | TokenLevel (accept if target_p >= draft_p)         | Implemented — v16.0         |
| Draft weight sharing      | Zero-copy weight references, no extra GPU memory   | Implemented — v16.0         |
| Multi-draft routing       | Per-request `draft_model_id` for heterogeneous A/B | Implemented — v18.0         |
| External draft lifecycle  | Runtime registry + refcount + unload frees KV      | Implemented — v18.0         |
| VRAM budget strategy      | Load-time estimate + runtime check; refuse on over | Implemented — v18.0         |
| Audit-before-refactor     | Analyze codebase health before non-functional work | Implemented — v19.0         |
| Analysis-only milestone   | Produce audit reports without code changes; backlog consumed by v20.0+ | Implemented — v19.0 |
| vllm-dist feature-gate    | Keep code, exclude from default build; enable for multi-node | Planned — v20.0              |
| Object-safety co-fix      | Fix `ModelError` + non-object-safe traits together in v20.1 | Planned — v20.0              |
| Single big v20.0 milestone | All 6 sub-phases in one milestone (vs splitting v20.1-v20.6) | Planned — v20.0              |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---

*Last updated: 2026-06-27 — v20.0 remediation milestone started; v19.0 archived (100 findings consolidated, 6 proposed sub-phases)*
