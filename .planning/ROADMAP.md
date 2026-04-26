# Phase 12 Roadmap: 高级功能

## Overview

**Milestone:** Phase 12 — 高级功能
**Core Value:** Expand vllm-lite with more quantization, streaming, and batching features
**Phases:** 3 | **Requirements:** 3 | **Started:** 2026-04-26

---

## Phase Summary

| # | Phase | Goal | Requirements | Status |
|---|-------|------|--------------|--------|
| 12.1 | 量化扩展 | AWQ/GPTQ 支持 | QUANT-01 | Not Started |
| 12.2 | 流式改进 | 背压处理优化 | STREAM-01 | Not Started |
| 12.3 | 智能批处理 | 预测性批处理 | BATCH-01 | Not Started |

---

## Phase 12.1: 量化扩展

**Goal:** Add AWQ and GPTQ quantization support beyond current GGUF Q4_K_M

**Requirements:**
- QUANT-01: AWQ/GPTQ support

**Success Criteria:**
1. AWQ weight loading and dequantization
2. GPTQ weight loading and dequantization
3. Runtime compatibility with existing attention kernels
4. Memory savings vs FP16 baseline

**Implementation Notes:**
- Reference `crates/model/src/paged_tensor/quantization.rs`
- Implement dequantization kernels
- Test with real quantized weights

---

## Phase 12.2: 流式改进

**Goal:** Improve streaming with backpressure handling and flow control

**Requirements:**
- STREAM-01: Streaming improvements

**Success Criteria:**
1. Backpressure handling for slow clients
2. Buffer management improvements
3. Graceful degradation under load
4. Connection lifecycle management

**Implementation Notes:**
- Reference `crates/server/` for streaming endpoints
- Implement flow control mechanism
- Add buffer size limits

---

## Phase 12.3: 智能批处理

**Goal:** Implement predictive batching for better throughput

**Requirements:**
- BATCH-01: Predictive batching

**Success Criteria:**
1. Request pattern detection
2. Proactive batching decisions
3. Latency/throughput balance tuning
4. Metrics for batching effectiveness

**Implementation Notes:**
- Reference `crates/core/src/scheduler/`
- Implement prediction heuristics
- Add batching strategy configuration

---

## Phase Transition Triggers

After each phase, run verification and update ROADMAP.md progress.

---

## Long-term Vision

Phase 13: Mobile/Edge optimization

---
*Roadmap created: 2026-04-26*
*Last updated: 2026-04-26 after initial creation*
