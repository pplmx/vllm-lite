# vllm-lite

## What This Is

A production-ready LLM inference engine in Rust, optimized for single and multi-node GPU deployment with Kubernetes integration, high availability, and enterprise security.

## Core Value

Fast, memory-efficient LLM inference with continuous batching, paged KV cache, and tensor parallelism — deployed with production-grade ops tooling and security.

## Current Milestone: v18.0 Multi-Model Speculative Decoding

**Goal:** 兑现 v17 延期的 MULTI-01/02/03 — 引入外部 draft model（不同架构/尺寸），实现请求级 draft 路由，并发 target + draft 显存预算。

**Target features:**

- 外部 drafter 模型加载 — Engine 接受独立 draft model（不同架构/size），独立 KV cache
- Draft model 生命周期管理 — 运行时注册/卸载，KV cache 回收，registry 跟踪活跃 drafts
- GPU 显存预算 — 加载前估算 + 运行时并发检查，超额拒绝
- 请求级 draft 路由 — Scheduler 按 request 选择 draft，多 draft 共存 batch
- Fallback 兼容 — 若外部 draft 加载失败，回退到 self-spec（v17 已有的能力）

## Current State

**Current Milestone:** v18.0 Multi-Model Speculative Decoding (planning)
**Latest Shipped:** v17.0 Production Speculative Decoding (2026-06-26, 21/21 SPECs)
**Status:** v17.0 收官；开始 v18.0 规划

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

**v18.0: Multi-Model Speculative Decoding**

- [ ] **MMLT-01**: Engine loads separate draft model instance (different architecture/size from target)
- [ ] **MMLT-02**: Draft model uses own ModelBackend instance with independent KV cache
- [ ] **MMLT-03**: Draft weights loaded lazily (deferred until first request using it)
- [ ] **LIFE-01**: DraftModelRegistry registers/loads/unloads draft models at runtime
- [ ] **LIFE-02**: Unloading a draft model frees its KV cache blocks via MemoryManager
- [ ] **LIFE-03**: Registry tracks active drafts + reference counts
- [ ] **MEM-01**: Engine enforces total VRAM budget for target + N concurrent drafts
- [ ] **MEM-02**: Load-time weight-size estimation; runtime KV-cache growth tracking
- [ ] **MEM-03**: Engine refuses to load draft if budget would exceed
- [ ] **RTE-01**: Request can specify `draft_model_id` in SamplingParams or Request
- [ ] **RTE-02**: Scheduler routes request to correct draft model instance
- [ ] **RTE-03**: Multiple drafts coexist in same batch (mixed draft routing)
- [ ] **FALL-01**: External draft load failure → fallback to self-spec (v17 path)
- [ ] **FALL-02**: Runtime draft error → graceful degradation to non-speculative decode

### Out of Scope (carried from v17)

- Tree-based speculation (draft tree) — sigmoidally complex
- Medusa-style multiple heads — incompatible with off-the-shelf models
- Speculative decoding for prefill — compute-bound, only decode
- Dynamic model switching mid-request — complex state management
- Draft model retraining/fine-tuning — out of engine scope

## v18.0 Replaces (Deferred Then Promoted)

The following v17 deferred items are now active in v18.0:

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

v17.0 shipped 21/21 SPECs satisfied (Wave 5 benchmark suite closure, 2026-06-26):

- Engine integration, fallback parity, real hardware bench suite
- Adaptive depth with EWMA + deadband, warmup after prefill
- Metrics: acceptance rate, Prometheus counters, `/debug/metrics`
- MULTI-01/02 deferred to v18.0 per original design

v18.0 build-on:

- Self-speculation path remains baseline fallback (zero extra VRAM, weight sharing)
- External draft model is opt-in extension; lazy load keeps cold-start fast
- Request-level draft routing enables heterogeneous deployment (multi-tenant, A/B testing)

Tech stack: Rust + Candle, multi-GPU CUDA support, Kubernetes, gRPC.
Codebase: Speculative decoding module (verifier, model, config, strategy, self_spec); draft registry to be added in v18.0.

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
| Multi-draft routing       | Per-request `draft_model_id` for heterogeneous A/B | Planned — v18.0             |
| External draft lifecycle  | Runtime registry + refcount + unload frees KV      | Planned — v18.0             |
| VRAM budget strategy      | Load-time estimate + runtime check; refuse on over | Planned — v18.0             |

## Evolution

This document evolves at phase transitions and milestone boundaries.

---

*Last updated: 2026-06-27 — v18.0 milestone started; v17.0 archived (21/21 SPECs shipped)*
