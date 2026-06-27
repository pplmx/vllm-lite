# vllm-lite

## What This Is

A production-ready LLM inference engine in Rust, optimized for single and multi-node GPU deployment with Kubernetes integration, high availability, and enterprise security.

## Core Value

Fast, memory-efficient LLM inference with continuous batching, paged KV cache, and tensor parallelism — deployed with production-grade ops tooling and security.

## Current Milestone: v23.0 审计修复 (Audit Remediation)

**Status:** ◆ Active (planning)

**Goal:** 修复 v22.0 审计发现的 22 项 P0/P1 问题, 覆盖代码缺陷、文档陈旧、占位 doc、dead code 4 类, 不引入新特性

**Target themes (按类别切分 4 phases):**

- 🔴 **A. P0 代码缺陷**: TensorParallelError thiserror 化、错误链保留、`Box<dyn Error>` 消除、stub 架构策略
- 📚 **B. 文档陈旧**: CLAUDE.md / README / CHANGELOG / MIGRATING 同步, 新建 `docs/architecture.md` 总览
- 🧹 **C. 占位 doc 清理**: 删除 ~85 个模块占位 doc + ~530 个函数占位 doc + 13 处 builder 复制粘贴
- 🗑️ **D. 架构 dead code**: 删除 ~7 个零调用模块 (~2000 行)、合并 4 stub 架构、修复 core→model 上行依赖、清理未使用依赖

**Phase 编号延续:** v22.0 收尾于 Phase 39 → v23.0 从 Phase 40 开始 → v23.0 收尾于 Phase 43 (4 phases)

**估算:** ~50h (按审计报告分类累计)

**Scope 决策 (来自用户输入):**

- **方向**: 全面修复审计发现 (A+B+C+D 全量, 而非 P0-only)
- **Research**: Skip (基于现有 v22.0 审计报告, 已知问题清单, 无新领域需研究)
- **Phase 粒度**: 标准 ~4 phases (与 v20.0/v21.0/v22.0 一致, 按问题类别切分)
- **版本号**: v23.0 (跨多个主题, 不属于 v22.x 补丁)
- **优先级**: P0 代码 > 文档陈旧 > 占位 doc 清理 > dead code 清理

**Out of Scope (v23.0):**

- 新特性 (long context / multimodal / tool calling) — v24+ 候选
- Engine.rs 完全拆分到 sub-modules (1057 LOC → 多文件) — ARF-06 仍 partial
- Multi-node / vllm-dist resurrection — OPS-05 仍 deferred
- Real-model benchmark — OPS-04 仍 deferred (无 GPU env)
- Doc coverage 97.8% → 99%+ (RFU-06) — 不在本期范围
- Stub 架构完整实现 — 仅做策略统一 (实现 / 拒绝加载), 不实现 forward

## Previous Milestone: v22.0 Production Hardening ✅ SHIPPED

**Status:** ✅ Complete (2026-06-27)

## Current State

**Current Milestone:** v23.0 审计修复 (Audit Remediation) — ◆ Planning (2026-06-28)
**Previous Milestone:** v22.0 Production Hardening ✅ SHIPPED (2026-06-27)
**Latest Shipped:** v22.0 (4/4 phases complete, all FINAL gates green)
**Status:** v22.0 完成 (1179 tests pass, clippy/fmt/cargo-doc clean, 15 ADRs, doc coverage 97.8%); v23.0 planning — 4 phases (40-43) addressing 22 audit findings

### v22.0 Achievements

**Phase 36 (v22.1) — Critical Bug Fixes** ✅
- ✅ **OPS-02**: Fixed `Engine::step()` speculative-mode hang. Root cause: DashMap shard re-entry in `record_per_request_acceptance` (entry guard held across `len()` call). Fix: scope the entry to release the shard lock before calling `len()`. Two `#[ignore]`'d e2e tests in `engine_wiring.rs` now pass (`test_fall02_engine_step_catches_runtime_error`, `test_engine_step_routes_to_correct_draft_backend`); 7 `#[ignore]`'d unit tests in `engine/spec_dispatch/tests.rs` also unblocked.
- ✅ **OPS-03**: Resolved 10 cargo doc broken-link warnings (`crates/testing/src/lib.rs`, `crates/core/src/engine.rs`, `crates/core/src/speculative/registry/{mod,lifecycle}.rs`, `crates/model/src/components/{attention/mod,block,decoder_block/mod}.rs`).
- ✅ **GGUF-01**: Documented gguf parser placeholder (`crates/model/src/quantize/gguf.rs`) — stub loader with explicit ADR-009 cross-reference.

**Phase 37 (v22.2) — Security Hardening** ✅
- ✅ **SEC-01**: JWT signature verification via `jsonwebtoken = "9"`. Algorithm allowlist: HS256 (HMAC), RS256/RS384/RS512 (RSA), ES256/ES384 (ECDSA). `alg: none` rejected. 12 JWT tests cover wrong-secret, tampered, expired, invalid-iss, invalid-aud, none-alg, RSA/ECDSA error paths.
- ✅ **SEC-02**: `RbacMiddleware` wired (was a no-op pass-through). Path → action mapping: `/v1/models → read`, `/v1/chat/completions → execute`, `/admin/* → manage_users`. HTTP 403 + structured JSON on denial. 5 RBAC integration tests.
- ✅ **SEC-03**: `RequestBodyLimitLayer` (1 MiB default, configurable). HTTP 413 on overflow. 4 size-limit tests.
- ✅ **SEC-04**: Audit log integration test (4 tests) — auth-success, auth-failure, no-events-on-health, RBAC-denial-still-emits-auth-event.
- ✅ **SEC-05**: Grafana credentials moved from hardcoded `docker-compose.yml` to `${VAR:-default}` substitution. `.env.example` documents required keys.
- ✅ **SEC-06**: TLS `ca_cert_path.unwrap()` replaced with structured `TlsError::InvalidConfig`. Regression test asserts no panic.

**Phase 38 (v22.3) — Production Polish** ✅
- ✅ **RFU-05**: Migrated 3 scheduler `std::sync::Mutex` fields + 7 poison-check call sites in `predictive_batching.rs` to `parking_lot::Mutex` (new direct dep `parking_lot = "0.12"`). Trait-object lock types in `engine.rs` left unchanged (public API stability).
- ✅ **OPS-01**: `engine/speculative.rs` mock fate documented in `engine/spec_dispatch/mod.rs` — file was deleted during v20.0 module-tree restoration; speculative path now resolves drafts via `DraftResolver` with FALL-01 self-spec fallback (v18.0 wiring).
- ✅ **PERF-01**: `MlaKvCache::write_compressed` rewritten with per-block `Tensor::slice_assign`. Allocation reduced from `O(num_blocks * block_size * kv_lora_rank)` to `O(block_size * kv_lora_rank)` per affected block.
- ✅ **PERF-02**: `model_type.to_lowercase()` → `eq_ignore_ascii_case` in gemma4, mistral, mixtral arch detection. Zero per-load `String` allocations.
- ✅ **PERF-03**: `once_cell::sync::Lazy` → `std::sync::LazyLock` in `crates/model/src/arch/registry.rs`. `once_cell` direct dep removed from `vllm-model`.
- ✅ **DOC-01**: 0 cargo doc broken-link warnings (carried-over from OPS-03 closed).

**Phase 39 (v22.4) — Engine Refactor + Final Verification** ✅
- ✅ **ARF-06**: Partial completion. `engine.rs` (1057 LOC) was already split into `engine/spec_dispatch/` sub-tree during v20.0 (dispatch.rs, drafts.rs, verify.rs, warmup.rs, tests.rs). Remaining `engine.rs` body consolidates the Engine struct + scheduler/engine/batch.rs step functions. Full single-responsibility sub-module split deferred to v23.0+ to avoid end-of-milestone regression risk.
- ✅ **ARF-07**: `engine/spec_dispatch/` is the canonical speculative sub-tree (post-Phase-31 ML-02 / v17.0). No duplicate abstractions remain between `engine.rs` and `engine/spec_dispatch/`.
- ✅ **FINAL-01**: 1179 tests pass, 39 skipped (slow checkpoint tests).
- ✅ **FINAL-02**: `cargo clippy --workspace --all-targets --all-features -- -D warnings` clean.
- ✅ **FINAL-03**: `cargo fmt --all --check` clean.
- ✅ **FINAL-04**: 1179 tests total (≥ 1146 v21.0 baseline; +33 net).
- ✅ **FINAL-05**: PROJECT.md + STATE.md updated with v22.0 outcomes.

**v23.0 deferred items (rolled forward to v24.0+ candidates):**
- Engine.rs 完全拆分到 sub-modules (ARF-06 still partial; current 1057 LOC file)
- Long context (>32K) — NMC-01
- Multimodal/Vision — NMC-02
- Tool calling — NMC-03
- Doc coverage push to 99%+ — RFU-06
- Multi-node / vllm-dist resurrection — OPS-05
- Real-model benchmark — OPS-04

**v22.0 deferred items (now historical — partially addressed in v23.0):**
- Full engine.rs single-responsibility split — still partial; v23.0 ARCH-* doesn't include God module split
- Long context (>32K) — NMC-01 (v24+)
- Multimodal/Vision — NMC-02 (v24+)
- Tool calling — NMC-03 (v24+)
- Doc coverage push to 99%+ — RFU-06 (deferred further)
- Multi-node / vllm-dist resurrection — OPS-05 (v24+)
- Real-model benchmark — OPS-04 (deferred further)

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

<!-- Shipped from v22.0 (Production Hardening) -->
- ✓ **Engine::step() speculative-mode hang fix — v22.0 (OPS-02)**: DashMap shard re-entry fix; 2 e2e + 7 unit tests unblocked
- ✓ **Cargo doc broken-link warnings (10) — v22.0 (OPS-03, DOC-01)**: 0 broken links across workspace
- ✓ **GGUF parser stub documentation — v22.0 (GGUF-01)**: ADR-009 cross-reference added
- ✓ **JWT signature verification — v22.0 (SEC-01)**: jsonwebtoken v9, 6 algorithms, 12 tests
- ✓ **RbacMiddleware wired — v22.0 (SEC-02)**: HTTP 403 + structured JSON, 5 RBAC tests
- ✓ **Request body size limit — v22.0 (SEC-03)**: RequestBodyLimitLayer, 1 MiB default, HTTP 413
- ✓ **Audit log integration test — v22.0 (SEC-04)**: 4 tests (auth-success, auth-failure, no-events, RBAC-denial)
- ✓ **Grafana credentials to .env — v22.0 (SEC-05)**: docker-compose substitution + .env.example
- ✓ **TLS ca_cert_path.unwrap() replaced — v22.0 (SEC-06)**: structured TlsError::InvalidConfig
- ✓ **parking_lot migration — v22.0 (RFU-05)**: 3 scheduler Mutex + 7 poison checks migrated
- ✓ **speculative.rs mock fate documented — v22.0 (OPS-01)**: DraftResolver resolves drafts; FALL-01 fallback
- ✓ **MlaKvCache::write_compressed perf — v22.0 (PERF-01)**: O(num_blocks * block_size * kv_lora_rank) → O(block_size * kv_lora_rank)
- ✓ **eq_ignore_ascii_case — v22.0 (PERF-02)**: zero per-load String allocations
- ✓ **once_cell → LazyLock — v22.0 (PERF-03)**: Rust 1.80+ stdlib
- ✓ **engine/spec_dispatch unification — v22.0 (ARF-07)**: canonical sub-tree post-Phase-31
- ✓ **All FINAL gates green — v22.0 (FINAL-01..05)**: 1179 tests, clippy/fmt/cargo-doc clean, 15 ADRs, 97.8% doc coverage

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

**v23.0: 审计修复 (Audit Remediation)** — 4 phases (Phase 40-43), 22 requirements

#### Phase 40 (v23.1): P0 代码缺陷修复 (Critical Code Fixes) — ~6h

- [ ] **CODE-01**: Convert `TensorParallelError` (traits/src/types.rs:87-112) to `#[derive(thiserror::Error)]`
- [ ] **CODE-02**: Replace `EngineError::ModelError(e.to_string())` at `core/src/engine.rs:677` with `EngineError::from(e)` to preserve source chain
- [ ] **CODE-03**: Replace `Box<dyn std::error::Error>` in `dist/src/grpc.rs:129` return type with typed `GrpcError` enum
- [ ] **CODE-04**: Decide stub architecture policy — implement forward passes OR reject `allow_stub` at runtime in non-test builds
- [ ] **CODE-05**: Implement or remove `SchedulerEngine::prefix_cache_hit_rate()` placeholder (`core/src/scheduler/engine.rs:555-559`)
- [ ] **FINAL-01**: All 1179 tests pass

#### Phase 41 (v23.2): 文档陈旧修复 (Stale Documentation) — ~16h

- [ ] **DOC-02**: Rewrite CLAUDE.md — fix crate count (6 not 4), Rust version (1.85 not 1.75), Engine signature (non-generic), and remove broken `qwen3/attention.rs` reference
- [ ] **DOC-03**: Fix README.md:459-473 — correct imports to `vllm_core::scheduler::policy::{FcfsPolicy, SjfPolicy, PriorityPolicy}`
- [ ] **DOC-04**: Backfill CHANGELOG.md with v19.0, v20.0, v21.0, v22.0 entries synthesized from `.planning/milestones/v{19,20,21,22}.0-*.md`
- [ ] **DOC-05**: Add MIGRATING.md v22.0 entry covering security middleware, Mutex migration, LazyLock upgrade
- [ ] **DOC-06**: Create `docs/architecture.md` — unified v23.0 architecture overview (engine orchestration, scheduler split, paged_tensor, KV cache, registry, multi-model spec flow)
- [ ] **DOC-07**: Update README.md test count badge (1100+ → 1179) and add version pin note
- [ ] **DOC-08**: Fix `docs/optimization_guide.md:50` outdated `Engine::with_config` API example
- [ ] **FINAL-01**: All 1179 tests pass

#### Phase 42 (v23.3): 占位 doc 清理 (Placeholder Doc Cleanup) — ~6h

- [ ] **CMT-01**: Delete ~85 module-level `<mod>: <mod>.` placeholder docs across `core/`, `model/components/`, `model/paged_tensor/`, `model/kernels/`, etc.
- [ ] **CMT-02**: Delete ~530 function/struct/method `<name>: <name>.` placeholder docs (auto-generated noise)
- [ ] **CMT-03**: Replace 13 copies of `/// builder: construct via builder for documented field ergonomics.` with type-specific docs
- [ ] **CMT-04**: Strip phase/audit IDs (v18.0, Plan 17.x, SEC-06, PERF-01, etc.) from user-visible rustdoc in 70 files — keep only one internal reference doc
- [ ] **CMT-05**: Fix 4 actively wrong comments: `core/src/lib.rs:7` "in progress", `types.rs:264/273` double-name corruption, `server/src/{lib,health}.rs` triple-header
- [ ] **CMT-06**: Update `qwen3_config` deprecation shim wording or delete it (`crates/model/src/lib.rs:44-52`)
- [ ] **FINAL-01**: All 1179 tests pass

#### Phase 43 (v23.4): 架构 dead code 清理 (Architecture Cleanup) — ~20h

- [ ] **ARCH-01**: Delete `scheduler/batch_planner.rs` (369 LOC) — zero production callers
- [ ] **ARCH-02**: Delete `scheduler/predictive_batching.rs` (498 LOC) — zero production callers
- [ ] **ARCH-03**: Delete `core/src/kv_cache/mod.rs` (7 LOC) — pass-through shim for BLOCK_SIZE
- [ ] **ARCH-04**: Delete or `pub(crate)` `core/src/sync.rs` (12 LOC), `routing/HashRouter` (191 LOC), `ha/{FailoverManager,LeaderElection}` (328 LOC), `circuit_breaker/*` (556 LOC) — no engine seam
- [ ] **ARCH-05**: Collapse 4 stub architectures (`gemma3/llama4/phi4/mistral_small`, ~1100 LOC) into one parameterized `StubArchitecture`
- [ ] **ARCH-06**: Fix `core → model` upward dependency — extract CUDA graph types into `vllm-traits` or new `vllm-kernels` crate
- [ ] **ARCH-07**: Remove unused `reqwest` from `crates/server/Cargo.toml`
- [ ] **ARCH-08**: Move `rayon` from `[dependencies]` to `[dev-dependencies]` in `crates/model/Cargo.toml`
- [ ] **ARCH-09**: Unify the 3 `greedy_sample`/`argmax` implementations (core/sampling.rs, model/causal_lm/mod.rs, engine/spec_dispatch/drafts.rs)
- [ ] **ARCH-10**: Unify the 2 `Architecture` types (`arch::Architecture` trait vs `config::Architecture` enum)
- [ ] **FINAL-01**: All 1179 tests pass
- [ ] **FINAL-02**: `cargo clippy --workspace --all-targets --all-features -- -D warnings` clean
- [ ] **FINAL-03**: `cargo fmt --all --check` clean
- [ ] **FINAL-04**: Test count ≥ 1179 (no regression)
- [ ] **FINAL-05**: `.planning/PROJECT.md` + `.planning/STATE.md` updated with v23.0 outcomes

### Out of Scope (v23.0)

- New model capabilities (long context >32K, multimodal/vision) — v24+ 候选
- Multi-node / vllm-dist resurrection — OPS-05 仍 deferred
- Real-model benchmarks — OPS-04 仍 deferred (无 GPU env)
- Engine.rs 完全拆分到 sub-modules (1057 LOC → 多文件) — ARF-06 仍 partial
- Doc coverage 97.8% → 99%+ (RFU-06) — 不在本期范围
- Stub 架构完整实现 — 仅做策略统一 (实现 / 拒绝加载), 不实现 forward
- New architectures (e.g., Falcon, DeepSeek) — out of scope for remediation milestone
- Online fine-tuning, WebAssembly — 长期愿景

### Out of Scope (v22.0 — historical)

- Engine::step() speculative-mode hang fix — **shipped in v22.0 (OPS-02)**
- Real-model benchmarks — still deferred (no GPU env)
- Multi-node / vllm-dist resurrection — feature-gated
- v20.0 carry-over (5 cargo doc warnings, 1 gguf TODO) — **shipped in v22.0 (OPS-03 + GGUF-01)**

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

v22.0 build-on (SHIPPED 2026-06-27):

- Production hardening: P0 bug fix (Engine::step hang) + security wiring (JWT/RBAC) + production polish (parking_lot, cargo doc, gguf TODO) + engine refactor (God module split)
- 4 sub-phases (Phase 36-39), organized by severity (P0 first, P3 last)
- Backward-compat preserved (no breaking API changes)
- Test invariants: 1146 tests must remain green; aim for ≥ 1146 post-v22.0 (no growth expected — hardening scope)
- v21.0 doc coverage + audit findings preserved
- **Outcome**: 21/21 requirements covered, 1179 tests pass (+33), 0 cargo doc warnings, 0 clippy, 0 fmt, 15 ADRs

v23.0 build-on (CURRENT — planning 2026-06-28):

- Audit remediation: 22 findings from v22.0 post-ship audit across 4 categories (P0 code / stale docs / placeholder docs / dead code)
- 4 sub-phases (Phase 40-43), organized by category (CODE → DOC → CMT → ARCH)
- Backward-compat preserved (no breaking API changes; dead code removal is internal refactor)
- Test invariants: 1179 tests must remain green; aim for ≥ 1179 post-v23.0 (no growth expected — remediation scope)
- v22.0 production hardening + doc coverage + FINAL gates preserved
- **Goal**: Clean v23.0 — audit-remediated codebase, honest rustdoc, no dead code, no P0 violations, structured top-level docs

Tech stack: Rust + Candle, multi-GPU CUDA support, Kubernetes, gRPC.
Codebase state (v23.0 start): 6 crates; speculative decoding complete (v18.0); draft registry + memory budget live in core; HTTP server with OpenAI-compatible API; 100 v19 audit findings → 100 fixed (v20+v21); 21 v22.0 hardening findings → 21 fixed; v22.0 post-ship audit found 22 new items across 4 categories (P0 code defects, stale docs, placeholder docs, dead code).

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

*Last updated: 2026-06-28 — **v23.0 审计修复 milestone STARTED** (4 phases planned: Phase 40 P0 代码 → Phase 41 文档陈旧 → Phase 42 占位 doc → Phase 43 dead code + FINAL gates); v22.0 complete (1179 tests pass, 21 hardening findings closed, 0 warnings); v23.0 focus: 22 audit findings across 4 categories (P0 code / stale docs / placeholder docs / dead code)*
