# Requirements: vllm-lite

**Defined:** 2026-04-27
**Core Value:** Fast, memory-efficient LLM inference with continuous batching, paged KV cache, and tensor parallelism

## v1 Requirements

### Benchmarking

- [ ] **BENCH-01**: Developer can run throughput benchmark measuring tokens/sec under concurrent load
- [ ] **BENCH-02**: Developer can run latency benchmark reporting TTFT, P50, P95, P99 percentiles
- [ ] **BENCH-03**: Benchmark runner handles warmup by discarding initial iterations

### Debug Utilities

- [ ] **DEBUG-01**: Developer can enable request tracing via tracing spans to debug request execution
- [ ] **DEBUG-02**: Developer can dump KV cache state to inspect cached prompts
- [ ] **DEBUG-03**: Developer can snapshot current metrics via HTTP endpoint

### CLI Tools

- [ ] **CLI-01**: Developer can validate config file syntax and schema at startup
- [ ] **CLI-02**: Developer can list available models in model directory
- [ ] **CLI-03**: Developer can view model metadata (architecture, params, config)

### Test Infrastructure

- [ ] **TEST-01**: Test harness provides common utilities (TestHarness::new()) for integration tests
- [ ] **TEST-02**: Mock model variants available (NeverProgressModel, SlowModel) for deterministic testing
- [ ] **TEST-03**: Request factory generates test requests with configurable properties

## v2 Requirements

### Benchmarking

- **BENCH-04**: Throughput sweep at multiple concurrency levels with throughput curve output
- **BENCH-05**: Memory profiling tracking peak GPU memory during benchmark
- **BENCH-06**: CUDA graph impact comparison (with/without graphs)

### Debug Utilities

- **DEBUG-04**: Cache hit analysis explaining prefix match/miss reasons
- **DEBUG-05**: Memory timeline exportable for visualization (perfetto format)
- **DEBUG-06**: Batch composition visualization (ASCII art)

### CLI Tools

- **CLI-04**: Model download from HuggingFace Hub
- **CLI-05**: Config generation scaffolding with defaults
- **CLI-06**: Interactive benchmark runner with live progress

### Test Infrastructure

- **TEST-04**: Property-based tests with proptest (1000+ auto-generated cases)
- **TEST-05**: Fuzzing corpus for edge case inputs
- **TEST-06**: CI performance regression check (fail PR if >5% regression)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Full GUI debugger | Binary bloat, complexity - CLI REPL sufficient |
| Cloud-based profiling | Privacy concerns, latency - local perfetto export only |
| Real-time dashboard | Complexity - use existing Prometheus + Grafana |
| Multi-user debug session | Auth complexity - single-user CLI tool |
| Web UI for benchmarking | Over-engineering - CLI + JSON output |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| BENCH-01 | Phase 14.1 | Pending |
| BENCH-02 | Phase 14.1 | Pending |
| BENCH-03 | Phase 14.1 | Pending |
| DEBUG-01 | Phase 14.2 | Pending |
| DEBUG-02 | Phase 14.2 | Pending |
| DEBUG-03 | Phase 14.2 | Pending |
| CLI-01 | Phase 14.3 | Pending |
| CLI-02 | Phase 14.3 | Pending |
| CLI-03 | Phase 14.3 | Pending |
| TEST-01 | Phase 14.4 | Pending |
| TEST-02 | Phase 14.4 | Pending |
| TEST-03 | Phase 14.4 | Pending |

**Coverage:** 12/12 v1 requirements mapped (0 unmapped)

---

## Phase Details

### Phase 14.1: Benchmarking
- **Goal:** Developer can run standardized benchmarks measuring throughput and latency
- **Requirements:** BENCH-01, BENCH-02, BENCH-03
- **Success Criteria:**
  1. Throughput benchmark reports tokens/sec under concurrent load
  2. Latency benchmark reports TTFT, P50, P95, P99 percentiles
  3. Warmup discards initial iterations, producing stable results

### Phase 14.2: Debug Utilities
- **Goal:** Developer can inspect and trace engine internals during execution
- **Requirements:** DEBUG-01, DEBUG-02, DEBUG-03
- **Success Criteria:**
  1. Request tracing shows hierarchical spans in logs
  2. KV cache dump returns readable cache state
  3. Metrics snapshot returns JSON endpoint response

### Phase 14.3: CLI Tools
- **Goal:** Developer can manage models and validate configuration via CLI
- **Requirements:** CLI-01, CLI-02, CLI-03
- **Success Criteria:**
  1. Config validation reports syntax/schema errors
  2. Model listing shows available models with names/sizes
  3. Model info displays architecture, parameters, config

### Phase 14.4: Test Infrastructure
- **Goal:** Test infrastructure provides reusable components for integration tests
- **Requirements:** TEST-01, TEST-02, TEST-03
- **Success Criteria:**
  1. TestHarness::new() initializes test environment
  2. NeverProgressModel blocks, SlowModel delays deterministically
  3. Request factory generates configurable test requests

---
*Requirements defined: 2026-04-27*
*Last updated: 2026-04-27 — roadmap created with phase details*
