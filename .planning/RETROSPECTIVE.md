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

---

## Cross-Milestone Trends

| Metric | v13.0 | v14.0 |
|--------|-------|-------|
| Phases | 3 | 4 |
| Requirements | 23 | 12 |
| Completion | 96% | 100% |
| Tech Debt | High | Low |
