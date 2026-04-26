# Phase 10 Roadmap: 性能优化

## Overview

**Milestone:** Phase 10 — 性能优化
**Core Value:** 交付生产级性能优化，使 vllm-lite 在标准基准测试中具有竞争力
**Phases:** 3 | **Requirements:** 5 | **Started:** 2026-04-26

---

## Phase Summary

| # | Phase | Goal | Requirements | Status |
|---|-------|------|--------------|--------|
| 10.1 | Kernel 优化 | FlashAttention V2 + CUDA Graph | PERF-01, PERF-02 | ✅ Complete |
| 10.2 | 调度优化 | PD 分离 + Chunked Prefill | PERF-03, PERF-04 | ✅ Complete |
| 10.3 | 基准测试 | 性能验证 + 文档 | QUAL-01 | ✅ Complete |

---

## Phase 10.1: Kernel 优化

**Goal:** 实现 FlashAttention V2 和 CUDA Graph 优化

**Requirements:**
- PERF-01: FlashAttention V2 实现
- PERF-02: CUDA Graph 优化完善

**Success Criteria:**
1. FlashAttention V2 实现并通过精度验证 (误差 < 1e-3)
2. CUDA Graph 覆盖范围扩大，kernel 启动开销减少 30%+
3. 单元测试覆盖核心路径

**Implementation Notes:**
- 参考 `crates/model/src/kernels/flash_attention.rs`
- 参考 `crates/model/src/kernels/cuda_graph.rs`
- 需要与 RoPE 实现协调

---

## Phase 10.2: 调度优化

**Goal:** 完善 PD 分离和 Chunked Prefill

**Requirements:**
- PERF-03: PD 分离完善
- PERF-04: Chunked Prefill 优化

**Success Criteria:**
1. PD 分离调度正确工作，prefill 吞吐量提升 20%+
2. Chunked Prefill 支持 32k+ 上下文无 OOM
3. 调度器指标正确收集

**Implementation Notes:**
- 参考 `crates/core/src/scheduler/`
- PD 分离需要 PhaseScheduler 修改
- Chunked Prefill 需要内存管理协调

---

## Phase 10.3: 基准测试

**Goal:** 性能验证和文档

**Requirements:**
- QUAL-01: 性能基准测试

**Success Criteria:**
1. 基准测试套件可运行
2. 性能对比数据记录
3. 优化文档更新

**Implementation Notes:**
- 创建 `benches/` 目录下的基准测试
- 使用真实模型和输入进行测试
- 记录优化前后对比

---

## Phase Transition Triggers

After each phase, run verification and update ROADMAP.md progress.

---

## Long-term Vision

Phase 11: 分布式支持 (Pipeline Parallelism, Distributed KV Cache)

---
*Roadmap created: 2026-04-26*
*Last updated: 2026-04-26 after initial creation*
