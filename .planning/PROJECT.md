# vllm-lite Phase 10: 性能优化

## What This Is

vllm-lite Phase 10 milestone focused on performance optimizations — faster attention computation, reduced kernel launch overhead, and improved scheduling strategies for better throughput and latency.

## Core Value

Deliver production-grade performance optimizations that make vllm-lite competitive with full vLLM on standard benchmarks while maintaining the lightweight Rust implementation.

## Requirements

### Validated

<!-- Shipped and confirmed valuable from Phase 1-9 -->

- ✓ 核心推理引擎 — Continuous Batching, Paged KV Cache, Prefix Caching (Phase 1)
- ✓ 多模型支持 — Llama, Mistral, Qwen, DeepSeek, Gemma4, Mixtral (Phase 6)
- ✓ OpenAI 兼容 API — Chat, Completions, Embeddings (Phase 7)
- ✓ 生产就绪 — 监控、日志、可靠性 (Phase 5)

### Active

- [ ] FlashAttention V2 实现
- [ ] CUDA Graph 优化完善
- [ ] PD 分离完善
- [ ] Chunked Prefill 优化

### Out of Scope

- Pipeline Parallelism — Phase 11
- Distributed KV Cache — Phase 11
- WebAssembly 支持 — 长期愿景
- 多租户隔离 — 企业特性

## Context

从 Phase 9 架构重构后，代码结构更加模块化。现在是实现性能优化的最佳时机，因为：
- 共享组件层已建立 (attention, mlp, norm, positional)
- Feature flags 已支持 cuda/gguf/real_weights 可选
- 849 tests passing，基础扎实

现有性能相关代码：
- `crates/model/src/components/attention/gqa.rs` — 当前 GQA 实现
- `crates/model/src/kernels/cuda_graph.rs` — CUDA Graph 框架
- `crates/core/src/scheduler/` — 调度器模块

## Constraints

- **Tech**: Rust + Candle，必须保持 CPU fallback
- **Compatibility**: 必须保持现有 API 兼容性
- **Performance**: 目标：prefill 吞吐量提升 20%+，decode 延迟降低 10%+

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| FlashAttention V2 | 相比 V1 有显著性能提升 | — Pending |
| CUDA Graph 优化 | 减少 kernel 启动开销 | — Pending |
| PD 分离完善 | 充分利用 prefill/decode 计算特性差异 | — Pending |
| Chunked Prefill 改进 | 支持更长上下文，减少 OOM | — Pending |

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
*Last updated: 2026-04-26 after Phase 10 initialization*
