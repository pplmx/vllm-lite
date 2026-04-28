# vllm-lite

## What This Is

A production-ready LLM inference engine in Rust, optimized for single and multi-node GPU deployment with Kubernetes integration, high availability, and enterprise security.

## Core Value

Fast, memory-efficient LLM inference with continuous batching, paged KV cache, and tensor parallelism — deployed with production-grade ops tooling and security.

## Current State

**Latest Milestone:** v16.0 Speculative Decoding (planning)
**Status:** Production-ready with FA-V3, FP8 KV cache, 12 model architectures

### Phase 16 Goals

- Speculative Decoding architecture (draft model + verification)
- KV cache reuse across draft verification
- Adaptive speculation depth based on content patterns
- Benchmarks showing 2-3x speedup on repetitive tasks

### Phase 15 Achievements

- ✅ FlashAttention V3 with MQA/GQA support
- ✅ FP8 KV cache quantization (50% memory reduction)
- ✅ Gemma3, Phi-4, Llama 4, Mistral Small architectures
- ✅ Go K8s Operator scaffold (controller-runtime)
- ✅ TLS + JWT security hardening

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

**v16.0: Speculative Decoding** — Draft-then-verify token generation for 2-3x speedup

### Out of Scope

- WebAssembly support — 长期愿景
- Multi-tenant isolation — Enterprise feature
- Online fine-tuning — 长期愿景
- Real-time fine-tuning — 长期愿景
- Vision end-to-end — Architecture ready, no model integration yet

### Deferred from v15.0

- JWTSigner cryptographic verification — simplified validation for scaffold
- Go Operator cluster testing — scaffolded, needs K8s cluster

### Out of Scope

- WebAssembly support — 长期愿景
- Multi-tenant isolation — Enterprise feature
- Online fine-tuning — 长期愿景
- Real-time fine-tuning — 长期愿景
- Vision end-to-end — deferred from v14.0 (architecture only)

## Context

v14.0 shipped with 12/12 requirements satisfied. v15.0 focus areas:
- Performance: FA-V3 needs kernel implementation, KV cache compression research
- Models: Architecture detection for Gemma3, Phi-4, Llama 4, Mistral Small
- Production: Go Operator full implementation, TLS/JWT completion

Tech stack: Rust + Candle, multi-GPU CUDA support, Kubernetes, gRPC.

## Constraints

- **Tech**: Rust + Candle, multi-GPU CUDA support
- **Compatibility**: Must maintain single-GPU API compatibility
- **Performance**: Scale-up with linear throughput improvement
- **Cluster**: Multi-node coordination via gRPC

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Multi-node architecture | Horizontal scaling beyond single host | Implemented — v13.0 |
| K8s Operator vs Helm-only | Operator for declarative management | Scaffolded — v15.0 |
| Consensus protocol | Raft vs etcd for HA leader election | Using K8s Lease API — v13.0 |
| TLS approach | mTLS for cluster internal, simple TLS for external | Implemented — v15.0 |
| FA-V3 kernel approach | FlashAttention V3 integration | Implemented — v15.0 |
| KV cache compression | FP8 E4M3 format | Implemented — v15.0 |

## Evolution

This document evolves at phase transitions and milestone boundaries.

---
*Last updated: 2026-04-28 — v16.0 Speculative Decoding started*
