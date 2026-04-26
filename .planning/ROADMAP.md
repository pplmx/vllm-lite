# Phase 11 Roadmap: 分布式支持

## Overview

**Milestone:** Phase 11 — 分布式支持
**Core Value:** Enable vllm-lite to scale across multiple GPUs and nodes
**Phases:** 2 | **Requirements:** 2 | **Started:** 2026-04-26

---

## Phase Summary

| # | Phase | Goal | Requirements | Status |
|---|-------|------|--------------|--------|
| 11.1 | Pipeline Parallelism | Multi-GPU layer pipeline | PP-01 | Not Started |
| 11.2 | Distributed KV Cache | Cluster KV sharing | KV-01 | Not Started |

---

## Phase 11.1: Pipeline Parallelism

**Goal:** Implement multi-GPU pipeline parallelism for layer distribution

**Requirements:**
- PP-01: Pipeline Parallelism implementation

**Success Criteria:**
1. Model layers split across available GPUs
2. Forward pass correctly pipelines through stages
3. Backward pass coordination works
4. Linear speedup with GPU count (target: 1.8x per GPU)

**Implementation Notes:**
- Reference `crates/dist/` for existing TP support
- Implement stage partitioning
- Add pipeline scheduler

---

## Phase 11.2: Distributed KV Cache

**Goal:** Enable KV cache sharing across cluster nodes

**Requirements:**
- KV-01: Distributed KV Cache

**Success Criteria:**
1. KV cache invalidation protocol
2. Cross-node cache coherence
3. Memory reduction vs replication
4. Latency overhead < 10%

**Implementation Notes:**
- Design cache coherence protocol
- Implement inter-node communication
- Handle cache misses gracefully

---

## Phase Transition Triggers

After each phase, run verification and update ROADMAP.md progress.

---

## Long-term Vision

Phase 12: Enterprise features (Multi-tenancy, TLS, Audit logging)

---
*Roadmap created: 2026-04-26*
*Last updated: 2026-04-26 after initial creation*
