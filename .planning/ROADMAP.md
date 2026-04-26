# Roadmap: vllm-lite

## Milestones

- ✅ **v13.0 主机部署** — Phases 13.1-13.3 (shipped 2026-04-27)
- 🚧 **v14.0 Developer Tooling** — Phases 14.1-14.4 (in progress)

## Phase History

<details>
<summary>✅ v13.0 主机部署 (Phases 13.1-13.3) — SHIPPED 2026-04-27</summary>

- [x] Phase 13.1: K8s 基础 (10/10 req, K8S-02 partial) — completed 2026-04-27
- [x] Phase 13.2: 高可用 (7/7 req) — completed 2026-04-27
- [x] Phase 13.3: 安全加固 (6/6 req) — completed 2026-04-27

Full details: [.planning/milestones/v13.0-ROADMAP.md](.planning/milestones/v13.0-ROADMAP.md)

</details>

## Phases

- [ ] **Phase 14.1: Benchmarking** — Throughput/latency benchmarks with warmup handling
- [ ] **Phase 14.2: Debug Utilities** — Request tracing, KV cache dump, metrics snapshot
- [ ] **Phase 14.3: CLI Tools** — Config validation, model listing, model info
- [ ] **Phase 14.4: Test Infrastructure** — Test harness, mock models, request factory

---

## Phase Details

### Phase 14.1: Benchmarking

**Goal:** Developer can run standardized benchmarks measuring throughput and latency
**Depends on:** Phase 13 (completed)
**Requirements:** BENCH-01, BENCH-02, BENCH-03
**Success Criteria** (what must be TRUE):
  1. Developer can run throughput benchmark that reports tokens/sec under concurrent load
  2. Developer can run latency benchmark that reports TTFT, P50, P95, P99 percentiles
  3. Benchmark warmup discards initial iterations, producing stable benchmark results
**Plans:** TBD
**UI hint:** no

### Phase 14.2: Debug Utilities

**Goal:** Developer can inspect and trace engine internals during execution
**Depends on:** Phase 14.1
**Requirements:** DEBUG-01, DEBUG-02, DEBUG-03
**Success Criteria** (what must be TRUE):
  1. Request tracing via tracing spans shows hierarchical execution in logs
  2. KV cache dump endpoint returns current cache state in readable format
  3. Metrics snapshot endpoint returns current metrics as JSON
**Plans:** TBD
**UI hint:** no

### Phase 14.3: CLI Tools

**Goal:** Developer can manage models and validate configuration via CLI
**Depends on:** Phase 14.2
**Requirements:** CLI-01, CLI-02, CLI-03
**Success Criteria** (what must be TRUE):
  1. CLI validates config file syntax and schema, reports validation errors
  2. CLI lists available models in model directory with names and sizes
  3. CLI displays model metadata (architecture, parameters, config options)
**Plans:** TBD
**UI hint:** no

### Phase 14.4: Test Infrastructure

**Goal:** Test infrastructure provides reusable components for integration tests
**Depends on:** Phase 14.3
**Requirements:** TEST-01, TEST-02, TEST-03
**Success Criteria** (what must be TRUE):
  1. TestHarness::new() initializes test environment with common utilities
  2. NeverProgressModel blocks indefinitely, SlowModel delays for deterministic testing
  3. Request factory generates test requests with configurable tokens, settings
**Plans:** TBD
**UI hint:** no

---

## Progress

| Phase | Milestone | Requirements | Status |
|-------|-----------|--------------|--------|
| 1-11 | Prior milestones | Various | Complete |
| 12 | Advanced features | AWQ/GPTQ, backpressure, predictive batching | Complete |
| 13.1 | K8s 基础 | 9/10 (K8S-02 partial) | Complete |
| 13.2 | 高可用 | 7/7 | Complete |
| 13.3 | 安全加固 | 6/6 | Complete |
| 14.1 | Benchmarking | BENCH-01, BENCH-02, BENCH-03 | Not started |
| 14.2 | Debug Utilities | DEBUG-01, DEBUG-02, DEBUG-03 | Not started |
| 14.3 | CLI Tools | CLI-01, CLI-02, CLI-03 | Not started |
| 14.4 | Test Infrastructure | TEST-01, TEST-02, TEST-03 | Not started |

---

## Long-term Vision

- Phase 15: Cross-region replication
- WebAssembly support (out of scope for now)
- Multi-tenant KV cache isolation (requires coherence v2)

---

*Roadmap updated: 2026-04-27*
*Full milestone archives: .planning/milestones/*
