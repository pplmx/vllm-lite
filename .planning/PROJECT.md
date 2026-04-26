# vllm-lite Phase 13: 主机部署

## What This Is

Phase 13 milestone focused on production host deployment — multi-node clusters, Kubernetes integration, high availability, operations tooling, and security hardening.

## Core Value

Make vllm-lite production-ready for host/datacenter deployment with cluster support, Kubernetes Operator, HA/DR capabilities, ops tools, and enterprise security.

## Current Milestone: v13.0 主机部署

**Goal:** Production-ready host deployment with cluster, K8s, HA, ops, and security

**Target features:**
- Multi-node/cluster support — Distributed inference across servers
- Kubernetes integration — Operator, Helm Chart, auto-scaling
- High availability/fault recovery — Hot standby, multi-replica, failover
- Operations tools — Deployment scripts, health checks, config management
- Security hardening — TLS, authentication, RBAC, audit logs

## Requirements

### Validated

<!-- Shipped from Phase 1-11 -->

- ✓ 核心推理引擎 — Continuous Batching, Paged KV Cache, Prefix Caching (Phase 1)
- ✓ 多模型支持 — Llama, Mistral, Qwen, DeepSeek, Gemma4, Mixtral (Phase 6)
- ✓ OpenAI 兼容 API — Chat, Completions, Embeddings (Phase 7)
- ✓ 生产就绪 — 监控、日志、可靠性 (Phase 5)
- ✓ FlashAttention V2 实现 — Phase 10.1
- ✓ CUDA Graph 优化完善 — Phase 10.1
- ✓ PD 分离完善 — Phase 10.2
- ✓ Chunked Prefill 优化 — Phase 10.2
- ✓ 性能基准测试 — Phase 10.3
- ✓ Pipeline Parallelism — Phase 11.1
- ✓ Distributed KV Cache — Phase 11.2
- ✓ AWQ/GPTQ quantization support — Phase 12.1
- ✓ Backpressure handling for streaming — Phase 12.2
- ✓ Predictive batching — Phase 12.3

### Active

(None yet — to be defined)

### Out of Scope

- WebAssembly 支持 — 长期愿景
- 多租户隔离 — 企业特性
- Online fine-tuning — 长期愿景
- Real-time fine-tuning — 长期愿景

## Context

Phase 12 completed with AWQ/GPTQ quantization, streaming improvements, and predictive batching. The engine is production-ready for single-node multi-GPU.

v13.0 focuses on multi-node cluster deployment:
- Existing: `crates/dist/` — Tensor Parallel support
- Needed: Multi-node coordination, Kubernetes integration, HA/DR

## Constraints

- **Tech**: Rust + Candle, multi-GPU CUDA support
- **Compatibility**: Must maintain single-GPU API compatibility
- **Performance**: Scale-up with linear throughput improvement
- **Cluster**: Initial focus on single-node multi-GPU

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Multi-node architecture | Horizontal scaling beyond single host | — Pending |
| K8s Operator vs Helm-only | Operator for declarative management | — Pending |
| Consensus protocol | Raft vs etcd for HA leader election | — Pending |
| TLS approach | mTLS for cluster internal, simple TLS for external | — Pending |

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
*Last updated: 2026-04-27 after Phase 13 initialization*
