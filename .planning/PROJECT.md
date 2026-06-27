# vllm-lite

## What This Is

A production-ready LLM inference engine in Rust, optimized for single and multi-node GPU deployment with Kubernetes integration, high availability, and enterprise security.

## Core Value

Fast, memory-efficient LLM inference with continuous batching, paged KV cache, and tensor parallelism — deployed with production-grade ops tooling and security.

## Current Milestone: v22.0 Production Hardening

**Status:** 🚧 Planning (2026-06-27)

**Goal:** 把 vLLM-lite 推到 production-ready 状态 — 解决 v21.0 延后的 P0/P1 tech debt、安全接线、生产打磨、引擎重构。

**Target themes:**

- 🔴 **关键 bug 修复**: `Engine::step()` speculative-mode hang (pre-existing bug, 实际可用性风险)
- 🔐 **安全加固**: JWT 签名验证 (目前 broken) + RBAC 接线 (目前 no-op pass-through)
- 🔧 **生产打磨**: `parking_lot::Mutex` 迁移 (消除 poison 检查) + cargo doc 警告修复 + gguf TODO
- 🏗️ **引擎重构**: 拆分 `engine.rs` God module (从 v20.0 延后) + 统一 `engine/speculative` 树

**Phase 编号延续:** v21.0 收尾于 Phase 35 → v22.0 从 Phase 36 开始 → v22.0 收尾于 Phase 39 (4 phases)

**估算:** ~75h (1 working month)

**Scope 决策 (来自用户输入):**

- **方向**: Production Hardening (而非新特性如 long context / multimodal / multi-node)
- **Research**: Skip (基于现有 PROJECT.md + CODEBASE 地图直接进入 requirements)
- **优先级**: P0 bug 修复 + 安全 > 生产打磨 > 引擎重构

**Out of Scope (v22.0):**

- New model capabilities (long context, multimodal) — 下一里程碑考虑
- Multi-node / vllm-dist resurrection — feature-gated, multi-node work future
- Real-model benchmarks — requires GPU env (still deferred from v18.0)
- Performance optimization — orthogonal to hardening scope
- Tree-based speculation — too complex (carried from v18.0)

## Current State

**Current Milestone:** v22.0 Production Hardening (planning)
**Latest Shipped:** v21.0 P2/P3 Backlog Cleanup (2026-06-27, 5/5 phases complete, all FINAL gates green)
**Status:** v22.0 启动中;1146 tests pass,clippy clean,fmt clean,15 ADRs,doc coverage 97.8%

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
- ✓ 多模型支持 — Llama, Mistral, Qwen, Qwen2/3, Qwen3.5, Gemma4, Mixtral, Llama4, Mistral Small, Phi-4 (Phase 6+)
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

<!-- Shipped from v20.0 (Codebase Remediation) -->
- ✓ **P0 critical fixes — v20.1**: vllm-dist feature-gated; ModelError struct→enum; 8 non-object-safe traits made object-safe; CudaGraphError thiserror-converted
- ✓ **Module tree restoration — v20.2**: kv_cache_fp8 + debug orphan modules wired; engine_v18_wiring.rs renamed; 3 unregistered test files migrated; vllm-dist feature-gated
- ✓ **Error handling standardization — v20.3**: 13 error enums thiserror-converted; Result<_,String> eliminated; mutex-poison .expect() fixed; EngineError +4 new variants; anyhow adopted at server boundary
- ✓ **Doc coverage push — v20.4**: workspace doc coverage 19.5% → 97.8% (target ≥60% exceeded); 776+ pub items documented; 121+ files with module docs; README fixed
- ✓ **External docs + ADRs — v20.5**: README/AGENTS.md reconciled; 12 new ADRs created (self-spec, FP8, KV cache split, speculative overview, RTE-01..03 routing, vllm-dist feature-gate, FP8 orphan decision, CUDA graph gating, cross-crate errors, etc.)
- ✓ **Naming + final polish — v20.6**: 7 P1 + 19 P2 naming fixes; `EmbeddingData` → `Embedding` rename with #[deprecated] alias; 3 stale comments resolved; 3 kv_cache_fp8 clippy errors fixed; cargo fmt --all clean across 133 files; 1144 tests pass; clippy clean

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

**v22.0: Production Hardening (CURRENT)**

#### Phase 36 (v22.1): 关键 Bug 修复 (Critical Bug Fixes) — ~25h

- [ ] **OPS-02**: Fix `Engine::step()` speculative-mode hang (pre-existing bug from pre-v18.0; blocks 2 Phase 19 e2e tests)
- [ ] **OPS-03**: Resolve 5 pre-existing cargo doc broken-link warnings (`engine.rs` + `components/mod.rs`)
- [ ] **GGUF-01**: Resolve actionable TODO in `crates/model/src/quantize/gguf.rs` (gguf parser post-v20.7)
- [ ] **FINAL-01**: All 1146+ tests remain green post-fix

#### Phase 37 (v22.2): 安全加固 (Security Hardening) — ~25h

- [ ] **SEC-01**: Implement JWT cryptographic signature verification (HMAC-SHA256 for `secret`; RSA/ECDSA for `public_key_pem`) — currently parses but never verifies
- [ ] **SEC-02**: Wire `RbacMiddleware` into request pipeline (currently no-op pass-through at `rbac.rs:82-84`)
- [ ] **SEC-03**: Add request size limits (`tower_http::limit::RequestBodyLimitLayer`)
- [ ] **SEC-04**: Audit log integration test (verify `security/audit.rs` emits events for authenticated requests)
- [ ] **SEC-05**: Move hardcoded Grafana credentials from `docker-compose.yml` to `.env` file
- [ ] **SEC-06**: TLS hardening — replace `unwrap()` with proper error in `security/tls.rs:63`
- [ ] **FINAL-01**: All auth/middleware integration tests pass

#### Phase 38 (v22.3): 生产打磨 (Production Polish) — ~15h

- [ ] **RFU-05**: Migrate from `std::sync::Mutex` → `parking_lot::Mutex` (eliminate poison check entirely) in scheduler/engine paths
- [ ] **OPS-01**: Decide fate of `speculative.rs` mock usage in production paths (real draft loading or document mock-only)
- [ ] **PERF-01**: `MlaKvCache::write_compressed` — replace full-cache materialization with `slice_assign` if available
- [ ] **PERF-02**: Replace `model_type.to_lowercase()` in arch detection with `eq_ignore_ascii_case`
- [ ] **PERF-03**: Replace `once_cell::sync::Lazy` with `std::sync::LazyLock` (Rust 1.80+)
- [ ] **DOC-01**: Resolve 5 cargo doc broken-link warnings (if not closed in OPS-03)
- [ ] **FINAL-01**: All 1146+ tests remain green

#### Phase 39 (v22.4): 引擎重构 (Engine Refactor) + Final Verification — ~10h

- [ ] **ARF-06**: Split `engine.rs` God module (1,038 LOC) into focused sub-modules (Phase 27 deferred)
- [ ] **ARF-07**: Unify `engine/spec_dispatch` tree post-ML-02 (collapse duplicate abstractions)
- [ ] **FINAL-01**: All 1146+ tests remain green post-refactor
- [ ] **FINAL-02**: `cargo clippy --workspace --all-targets --all-features -- -D warnings` clean
- [ ] **FINAL-03**: `cargo fmt --all --check` clean
- [ ] **FINAL-04**: `cargo test --workspace --all-features` ≥ 1146 tests pass
- [ ] **FINAL-05**: `.planning/PROJECT.md` and `.planning/STATE.md` updated with v22.0 outcomes

### Out of Scope (v22.0)

- New model capabilities (long context >32K, multimodal/vision) — 下一 milestone 候选
- Multi-node / vllm-dist resurrection — feature-gated; multi-node work future
- Real-model benchmarks — requires GPU environment (still deferred from v18.0)
- Performance optimization beyond audit findings — orthogonal to hardening scope
- Tree-based speculation — too complex (carried from v18.0)
- New architectures (e.g., Falcon, DeepSeek) — out of scope for hardening milestone
- Online fine-tuning, WebAssembly — 长期愿景

### Out of Scope (v21.0 — historical)

- Engine::step() speculative-mode hang fix — **moved to v22.0 OPS-02**
- Real-model benchmarks — still deferred (no GPU env)
- Multi-node / vllm-dist resurrection — feature-gated
- v20.0 carry-over (5 cargo doc warnings, 1 gguf TODO) — **moved to v22.0 OPS-03 + GGUF-01**

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

## v21.0 Replaces (Deferred Then Promoted) — now active

The v19.0 audit produced 100 findings. v20.0 executed 5 P0 + 38 P1. v21.0 closes the remaining 44 P2 + 13 P3 backlog per `.planning/audit/BACKLOG.md`:

| Sub-phase | Focus | Items | Effort |
|-----------|-------|------:|-------:|
| 31 (v21.1) | 模块布局重组 (ARCH-F-05..F-19) | 9 | ~37h |
| 32 (v21.2) | API 一致性 (API-F-12..F-24) | 11 | ~21h |
| 33 (v21.3) | 命名一致性 (NAME-F-08..F-19) | 8 | ~5h |
| 34 (v21.4) | 外部 doc 修复 (DOCS-F-21..F-24) | 4 | ~3.25h |
| 35 (v21.5) | P3 actionable + Final verification | 6 + 4 FINAL | ~5h |
| **Total** | | **42** | **~71h** |

Note: 2 of the 44 P2 items (NAME-F-13, NAME-F-20) are zero-effort "no action" findings per BACKLOG.md, so effective P2 work = 42.

Key v21.0 decisions (from user input):
- **Scope**: All 44 P2 + selected actionable P3 (not just P2 high-impact subset)
- **Phase numbering**: Continue from Phase 30 → Phase 31
- **Carry-over from v20.0 audit**: Engine::step() hang + cargo doc warnings + gguf TODO — out of scope for v21.0 (separate bug-fix milestone if desired)

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

v20.0 build-on (SHIPPED 2026-06-27):

- Direct execution of prioritized audit findings, organized into 6 sub-phases
- Backward-compat: existing public API must remain stable (with `#[deprecated]` for removed items)
- All 1100+ existing tests must remain green
- v18.0 speculative decoding functionality preserved
- **Outcome**: 5 P0 + 38 P1 = 43 findings fixed; 1144 tests pass; clippy/fmt clean; doc coverage 97.8%

v21.0 build-on:

- Direct execution of remaining 44 P2 + selected actionable P3 (13 total), organized into 5 sub-phases
- Same backward-compat constraints (no breaking changes without `#[deprecated]`)
- Same test invariants (1100+ tests must remain green; aim for ≥ 1144 post-v21.0)
- v20.0 architectural improvements preserved
- **Goal**: 100% backlog closure (all 100 v19 findings addressed)

v22.0 build-on (CURRENT):

- Production hardening: P0 bug fix (Engine::step hang) + security wiring (JWT/RBAC) + production polish (parking_lot, cargo doc, gguf TODO) + engine refactor (God module split)
- 4 sub-phases (Phase 36-39), organized by severity (P0 first, P3 last)
- Backward-compat preserved (no breaking API changes)
- Test invariants: 1146 tests must remain green; aim for ≥ 1146 post-v22.0 (no growth expected — hardening scope)
- v21.0 doc coverage + audit findings preserved
- **Goal**: Production-ready vLLM-lite — no P0/P1 bugs, security wired, no cargo doc warnings, engine.rs manageable

Tech stack: Rust + Candle, multi-GPU CUDA support, Kubernetes, gRPC.
Codebase state (v22.0 start): 6 crates; speculative decoding complete (v18.0); draft registry + memory budget live in core; HTTP server with OpenAI-compatible API; 100 v19 audit findings → 100 fixed (v20+v21); v22.0 hardening: 4 remaining tech-debt themes (P0 bug, security wiring, production polish, engine refactor).

## Constraints

- **Tech**: Rust + Candle, multi-GPU CUDA support
- **Compatibility**: Must maintain single-GPU API compatibility
- **Performance**: Scale-up with linear throughput improvement
- **Cluster**: Multi-node coordination via gRPC

## Key Decisions

| Decision                  | Rationale                                          | Outcome                     | ADR                                       |
| ------------------------- | -------------------------------------------------- | --------------------------- | ----------------------------------------- |
| Multi-node architecture   | Horizontal scaling beyond single host              | Implemented — v13.0         | —                                         |
| K8s Operator vs Helm-only | Operator for declarative management                | Scaffolded — v15.0          | —                                         |
| Consensus protocol        | Raft vs etcd for HA leader election                | Using K8s Lease API — v13.0 | —                                         |
| TLS approach              | mTLS for cluster internal, simple TLS for external | Implemented — v15.0         | —                                         |
| Component sharing         | Traits crate as boundary between core and model    | Implemented — v1.0          | [ADR-001](../docs/adr/ADR-001-component-sharing-strategy.md) |
| Feature flag design       | Single binary, compile-time + runtime feature gating | Implemented — v1.0          | [ADR-002](../docs/adr/ADR-002-feature-flag-design.md) |
| FA-V3 kernel approach     | FlashAttention V3 integration                      | Implemented — v15.0         | —                                         |
| KV cache compression      | FP8 E4M3 format                                    | Implemented — v15.0         | [ADR-004](../docs/adr/ADR-004-fp8-e4m3-kv-cache.md) |
| KV cache split across 3 locations | Logical vs Physical vs Prefix for memory tiering | Implemented — v1.0          | [ADR-005](../docs/adr/ADR-005-kv-cache-split.md) |
| Continuous batching       | Token-level scheduling instead of request-level    | Implemented — v1.0          | [ADR-012](../docs/adr/ADR-012-continuous-batching.md) |
| Paged KV cache            | Block-based allocation, copy-on-write semantics    | Implemented — v1.0          | [ADR-013](../docs/adr/ADR-013-paged-kv-cache.md) |
| Architecture registry     | Dynamic registration enables extensibility         | Implemented — v6.0          | [ADR-014](../docs/adr/ADR-014-architecture-registry.md) |
| Speculative architecture  | Self-spec + external draft + per-request routing   | Implemented — v16.0-v18.0   | [ADR-006](../docs/adr/ADR-006-speculative-decoding-architecture.md) |
| Self-speculation 1/8 ratio | Reduced layer count with weight sharing            | Implemented — v16.0         | [ADR-003](../docs/adr/ADR-003-self-speculation-1-8-layer-ratio.md) |
| Per-request draft routing | Heterogeneous draft selection per request           | Implemented — v18.0         | [ADR-007](../docs/adr/ADR-007-per-request-draft-routing.md) |
| Multi-draft routing       | Per-request `draft_model_id` for heterogeneous A/B | Implemented — v18.0         | [ADR-007](../docs/adr/ADR-007-per-request-draft-routing.md) |
| External draft lifecycle  | Runtime registry + refcount + unload frees KV      | Implemented — v18.0         | [ADR-007](../docs/adr/ADR-007-per-request-draft-routing.md) |
| VRAM budget strategy      | Load-time estimate + runtime check; refuse on over | Implemented — v18.0         | [ADR-007](../docs/adr/ADR-007-per-request-draft-routing.md) |
| CUDA Graph feature-gating | Trait abstraction so core doesn't depend on model  | Implemented — v10.1         | [ADR-010](../docs/adr/ADR-010-cuda-graph-feature-gating.md) |
| Audit-before-refactor     | Analyze codebase health before non-functional work | Implemented — v19.0         | —                                         |
| Analysis-only milestone   | Produce audit reports without code changes; backlog consumed by v20.0+ | Implemented — v19.0 | —                                         |
| vllm-dist feature-gate    | Keep code, exclude from default build; enable for multi-node | Planned — v20.0              | [ADR-008](../docs/adr/ADR-008-vllm-dist-feature-gated.md), [ADR-015](../docs/adr/ADR-015-vllm-dist-investment-decision.md) |
| FP8 quantizer orphan      | Move into `components/` module tree                | Shipped — v20.2             | [ADR-009](../docs/adr/ADR-009-fp8-quantizer-orphan-decision.md) |
| Cross-crate error boundaries | Typed enums per crate; `From<E>` impls at boundaries | Implemented — v20.3       | [ADR-011](../docs/adr/ADR-011-cross-crate-error-boundaries.md) |
| Object-safety co-fix      | Fix `ModelError` + non-object-safe traits together in v20.1 | Planned — v20.0              | —                                         |
| Single big v20.0 milestone | All 6 sub-phases in one milestone (vs splitting v20.1-v20.6) | Shipped — v20.0              | —                                         |
| EmbeddingData rename | Rename + #[deprecated] alias (vs breaking change) | Shipped — v20.0              | —                                         |
| Verb policy formalization | Document `get_/load_/read_/create_/build_` semantics in AGENTS.md rather than enforce mechanically | Shipped — v20.0              | —                                         |
| P2/P3 scope for v21.0 | All 44 P2 + selected actionable P3 (vs only high-impact subset) | Planned — v21.0              | —                                         |
| v21.0 phase structure | 5 phases (Phase 31-35) mirroring v20.0's 6-phase pattern; each phase ~one theme | Planned — v21.0              | —                                         |
| P3 items mostly informational | Only ~6 of 13 P3 items need action; rest are "no action" or "vacuous positive" findings | Planned — v21.0              | —                                         |

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

*Last updated: 2026-06-27 — **v22.0 Production Hardening milestone STARTED** (4 phases planned: Phase 36 Critical Bug Fixes → Phase 37 Security Hardening → Phase 38 Production Polish → Phase 39 Engine Refactor + FINAL gates); v21.0 complete (1146 tests pass, clippy/fmt clean, 15 ADRs, 100% backlog closure); v22.0 focus: P0 bug fix (Engine::step hang) + security wiring (JWT/RBAC) + production polish + engine refactor*
