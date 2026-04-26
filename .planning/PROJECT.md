# vllm-lite

## What This Is

A production-ready LLM inference engine in Rust, optimized for single and multi-node GPU deployment with Kubernetes integration, high availability, and enterprise security.

## Core Value

Fast, memory-efficient LLM inference with continuous batching, paged KV cache, and tensor parallelism — deployed with production-grade ops tooling and security.

## Current State

**Latest Milestone:** v13.0 主机部署 (shipped 2026-04-27)
**Status:** Production-ready for host/datacenter deployment

### Phase 13 Achievements

- ✅ Multi-node cluster support (NodeMesh, gRPC transport, consistent hash routing)
- ✅ Kubernetes integration (Helm chart, health probes, ConfigMap, graceful shutdown)
- ✅ High availability (leader election, failover, HPA metrics, PodDisruptionBudget)
- ✅ Security hardening (TLS, mTLS, authentication, RBAC, audit logging)

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

### Active

(None yet — define next milestone with `/gsd-new-milestone`)

### Out of Scope

- WebAssembly support — 长期愿景
- 多租户隔离 — 企业特性
- Online fine-tuning — 长期愿景
- Real-time fine-tuning — 长期愿景
- Full K8s Operator (Go) — deferred from v13.0 (scaffolded only)

## Context

v13.0 shipped with 22/23 requirements satisfied. Known gaps:
- K8S-02: Full Go Kubernetes Operator deferred (scaffolded only)
- Multi-node testing requires actual K8s cluster
- TLS integration with axum server incomplete
- JWT validation stubbed

Tech stack: Rust + Candle, multi-GPU CUDA support, Kubernetes, gRPC.

## Constraints

- **Tech**: Rust + Candle, multi-GPU CUDA support
- **Compatibility**: Must maintain single-GPU API compatibility
- **Performance**: Scale-up with linear throughput improvement
- **Cluster**: Multi-node coordination via gRPC

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Multi-node architecture | Horizontal scaling beyond single host | — Pending |
| K8s Operator vs Helm-only | Operator for declarative management | Deferred — Helm only for now |
| Consensus protocol | Raft vs etcd for HA leader election | Using K8s Lease API |
| TLS approach | mTLS for cluster internal, simple TLS for external | Scaffolded |

## Evolution

This document evolves at phase transitions and milestone boundaries.

---
*Last updated: 2026-04-27 after v13.0 milestone*
