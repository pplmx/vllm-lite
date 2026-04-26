# vllm-lite Phase 11: 分布式支持

## What This Is

Phase 11 milestone focused on distributed computing support — enabling multi-GPU inference through Pipeline Parallelism and cluster-wide KV Cache sharing.

## Core Value

Enable vllm-lite to scale across multiple GPUs and nodes, supporting larger models and higher throughput in distributed deployments.

## Current Milestone: v11.0 分布式支持

**Goal:** Multi-GPU support with Pipeline Parallelism and Distributed KV Cache

**Target features:**
- Pipeline Parallelism — Multi-GPU tensor pipeline
- Distributed KV Cache — Cluster-wide KV cache sharing

## Requirements

### Validated

<!-- Shipped from Phase 1-10 -->

- ✓ 核心推理引擎 — Continuous Batching, Paged KV Cache, Prefix Caching (Phase 1)
- ✓ 多模型支持 — Llama, Mistral, Qwen, DeepSeek, Gemma4, Mixtral (Phase 6)
- ✓ OpenAI 兼容 API — Chat, Completions, Embeddings (Phase 7)
- ✓ 生产就绪 — 监控、日志、可靠性 (Phase 5)
- ✓ FlashAttention V2 实现 — Phase 10.1
- ✓ CUDA Graph 优化完善 — Phase 10.1
- ✓ PD 分离完善 — Phase 10.2
- ✓ Chunked Prefill 优化 — Phase 10.2
- ✓ 性能基准测试 — Phase 10.3

### Active

(None yet — to be defined)

### Out of Scope

- WebAssembly 支持 — 长期愿景
- 多租户隔离 — 企业特性
- Online fine-tuning — 长期愿景

## Context

Phase 10 completed with FlashAttention V2, CUDA Graph pooling, and Chunked Prefill. The engine is now ready for horizontal scaling.

Existing distributed infrastructure needs:
- `crates/dist/` — Tensor Parallel support (already exists, needs expansion)
- Communication layer for multi-node coordination
- KV Cache invalidation/ coherency protocol

## Constraints

- **Tech**: Rust + Candle, multi-GPU CUDA support
- **Compatibility**: Must maintain single-GPU API compatibility
- **Performance**: Scale-up with linear throughput improvement
- **Cluster**: Initial focus on single-node multi-GPU

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Pipeline Parallelism | Split model layers across GPUs | — Pending |
| Distributed KV Cache | Share KV across nodes | — Pending |
| Communication | gRPC for inter-node | — Pending |

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
*Last updated: 2026-04-26 after Phase 11 initialization*
