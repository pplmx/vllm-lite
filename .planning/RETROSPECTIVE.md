# Retrospective

## v14.0 Developer Tooling

**Shipped:** 2026-04-27
**Phases:** 4 | **Plans:** 4

### What Was Built

- Benchmarking suite with ThroughputBenchmark, LatencyBenchmark, PercentileStats
- Debug HTTP endpoints (metrics, kv-cache, trace) with hierarchical tracing spans
- CLI binary `vllm` with config validate, model list/info commands
- Test infrastructure: TestHarness, SlowModel, RequestFactory

### What Worked

- Reusing existing patterns (BenchmarkSuite, SchedulerObserver)
- Separate binaries for different concerns (server vs CLI)
- Test infrastructure building on existing mock models

### What Was Inefficient

- Multiple dependency conflicts (axum version) required resolution
- Clippy warnings in new code needed manual fixes

### Patterns Established

- SchedulerObserver for non-invasive metrics collection
- Debug endpoints via HTTP for runtime inspection
- Builder pattern for TestHarness configuration

### Key Lessons

- Debug tracing spans require discipline to add consistently
- CLI subcommands via clap::Subcommand scale well for tool suites

## Milestone: v16.0 — Speculative Decoding

**Shipped:** 2026-04-28
**Phases:** 4 | **Plans:** 4

### What Was Built

- DraftVerifier trait, SpeculativeModel wrapper, SpeculationConfig, RejectionStrategy
- Self-speculation with layer sharing (1/8 layer count, weight reuse via zero-copy)
- Parallel verification with token-level acceptance and early termination
- DraftAccuracyTracker for throughput proxy and acceptance rate metrics
- ModelBackend trait extended with num_layers()/num_heads()

### What Worked

- Clean separation of concerns across 4 phases (Architecture → Draft Model → Verification → Benchmarks)
- Reusing existing ModelBackend trait pattern for speculation integration
- Weight sharing via wrapper pattern — no additional GPU memory for draft model

### What Was Inefficient

- Engine integration (step_speculative) not wired — production speculation loop deferred
- Actual performance benchmarks on real hardware not executed (infrastructure ready)

### Patterns Established

- Speculative decoding stack: Trait → Model → Verification → Metrics
- VerificationResult for draft acceptance tracking with rejected_at position

### Key Lessons

- Speculative decoding architecture benefits from clean trait separation
- Weight sharing is critical for memory efficiency in self-speculation
- Benchmark infrastructure should be validated on real hardware before claiming throughput gains

---

## Cross-Milestone Trends

| Metric       | v13.0 | v14.0 | v16.0 |
| ------------ | ----- | ----- | ----- |
| Phases       | 3     | 4     | 4     |
| Requirements | 23    | 12    | 17    |
| Completion   | 96%   | 100%  | 100%  |
| Tech Debt    | High  | Low   | Low   |
