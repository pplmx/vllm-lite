# vllm-lite

## What This Is

A production-ready LLM inference engine in Rust, optimized for single and multi-node GPU deployment with Kubernetes integration, high availability, and enterprise security.

## Core Value

Fast, memory-efficient LLM inference with continuous batching, paged KV cache, and tensor parallelism — deployed with production-grade ops tooling and security.

## Current Milestone: v19.0 Codebase Health Audit (Analysis-Only)

**Goal:** 对 vllm-lite 整个 codebase 做多维度深度审计(架构/命名/注释文档/API+错误处理),产出优先级清单与具体修复建议;**本 milestone 不执行任何修改**,清单用于指导后续 milestone(v20.0+)的重构工作。

**Target features (审计产出物):**

- 架构审计报告 — crate 依赖图、模块边界、循环依赖、分层一致性
- 命名规范审计 — 文件名/类名/方法名/变量名的一致性 + 语义清晰度
- 注释 + 文档审计 — /// 文档覆盖度、模块级 README、过期注释、TODO 清理
- API + 错误处理审计 — 公开 API 一致性、错误类型冗余/缺失、信息完整度
- 综合优先级清单 — P0/P1/P2 排序,每项标注影响范围、修复成本、建议阶段

## Current State

**Current Milestone:** v19.0 Codebase Health Audit (planning)
**Latest Shipped:** v18.0 Multi-Model Speculative Decoding (2026-06-27, 14/14 requirements + Phase 19 gap closure)
**Status:** v18.0 收官;开始 v19.0 审计规划

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

**v19.0: Codebase Health Audit (analysis-only, no code changes)**

#### Architecture audit

- [ ] **ARCH-01**: Crate dependency graph audited (verify directional flow: traits ← core ← model/server/dist; no upward deps)
- [ ] **ARCH-02**: Module boundary audit complete (each module has single, clear responsibility; no God modules)
- [ ] **ARCH-03**: Circular dependency scan complete (cargo metadata + manual review)
- [ ] **ARCH-04**: Layering consistency audit (e.g., scheduler doesn't import from server; model doesn't import from core)
- [ ] **ARCH-05**: Test architecture audit (unit/integration/bench separation; shared testing crate hygiene)

#### Naming audit

- [ ] **NAME-01**: File naming audit complete (identify casually-named files like `17_*.rs` stage-info-named files; document each finding)
- [ ] **NAME-02**: Type/struct/enum naming consistency audit (PascalCase, descriptive, no redundant suffixes)
- [ ] **NAME-03**: Function/method naming audit (snake_case, action verbs, consistent prefix patterns)
- [ ] **NAME-04**: Variable naming audit (descriptive, no single-letter except indices; consistent naming for similar concepts)
- [ ] **NAME-05**: Module name audit (matches file name; consistent depth)

#### Comments + documentation audit

- [ ] **DOCS-01**: Doc-comment coverage measured for public API (target: ≥80% on `pub` items)
- [ ] **DOCS-02**: Module-level documentation audit (each module has `//!` or top-of-file context)
- [ ] **DOCS-03**: Stale comment / TODO audit (identify comments referencing old code, dead TODOs, misleading docstrings)
- [ ] **DOCS-04**: External documentation audit (root README, AGENTS.md, .planning docs accuracy against current codebase)
- [ ] **DOCS-05**: Architecture-decision records (ADRs) — identify documented rationale vs. tribal knowledge

#### API + error handling audit

- [ ] **API-01**: Public API surface consistency (function signatures, return types, builder patterns)
- [ ] **API-02**: Error type audit (thiserror usage, error variants coverage, error message quality)
- [ ] **API-03**: Error ergonomics audit (Result types, error context propagation, `From` conversions)
- [ ] **API-04**: Trait design audit (object safety, async/sync consistency, default method usage)
- [ ] **API-05**: Deprecation hygiene (deprecated items properly marked, migration paths documented)

#### Synthesis

- [ ] **SYNTH-01**: Cross-dimensional synthesis report complete (correlates findings across ARCH/NAME/DOCS/API)
- [ ] **SYNTH-02**: Prioritized remediation backlog produced (P0/P1/P2 with impact, cost, suggested phase)
- [ ] **SYNTH-03**: Suggested v20.0+ migration roadmap (which findings group into which future phase)

### Out of Scope (v19.0 — analysis only, no code changes)

- Any code modification (renaming, refactoring, re-architecture) — deferred to v20.0+
- New features — not the goal of an audit milestone
- Performance optimization — separate audit/optimization cycle
- Test re-runs or CI changes — no code touched, so existing CI still validates
- Migration execution — even renaming is deferred to a future milestone

### Out of Scope (carried from earlier milestones)

- Tree-based speculation (draft tree) — sigmoidally complex
- Medusa-style multiple heads — incompatible with off-the-shelf models
- Speculative decoding for prefill — compute-bound, only decode
- Dynamic model switching mid-request — complex state management
- Draft model retraining/fine-tuning — out of engine scope

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

v18.0 shipped 14/14 requirements satisfied (5 phases, 2026-06-27):

- DraftModelRegistry (loader-agnostic), MemoryBudget, refcount lifecycle
- Per-request routing, fallback semantics (silent FALL-01, sticky FALL-02)
- Phase 19 gap closure: resolver wired into step_speculative_inner, HTTP exporter, server config, ServerDraftLoader
- 287+ tests, 16 commits, audit passed

v19.0 build-on:

- 7 crates have grown over 18 milestones; accumulated technical debt likely exists in:
  - File naming (user-reported: some files named with stage info like `17_*.rs`)
  - Module boundaries (speculative/, scheduler/, kv_cache/, paged_tensor/ all grew organically)
  - Documentation drift (some doc-comments may reference pre-v15 APIs)
  - Error handling consistency (each subsystem may have its own error type)
- Audit is non-destructive: findings become input to v20.0+ refactoring phases

Tech stack: Rust + Candle, multi-GPU CUDA support, Kubernetes, gRPC.
Codebase state (v18.0 end): 7 crates; speculative decoding complete; draft registry + memory budget live in core; HTTP server with OpenAI-compatible API + v18.0 metrics exposed.

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
| Audit-before-refactor     | Analyze codebase health before non-functional work | Planned — v19.0             |

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

*Last updated: 2026-06-27 — v19.0 audit milestone started; v18.0 archived (14/14 requirements shipped, 5 phases)*
