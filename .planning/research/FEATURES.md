# Feature Landscape: vllm-lite v14.0 Developer Tooling

**Domain:** LLM Inference Engine Developer Tooling
**Researched:** 2026-04-27

## Table Stakes

Features users expect. Missing = product feels incomplete.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Throughput benchmark | Measure tokens/sec under load | Med | Use criterion + custom runner |
| Latency benchmarks | Measure TTFT, inter-token latency | Med | P50, P95, P99 percentiles |
| Request tracing | Debug specific request issues | Med | Use tracing spans |
| Metrics endpoint | `/v1/metrics` already exists | Low | Extend with tooling-specific metrics |
| Config validation | Catch config errors at startup | Low | Use serde schema validation |
| Model listing | Know what models are available | Low | CLI `list-models` command |

## Differentiators

Features that set product apart. Not expected, but valued.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Interactive debug REPL | Step through request execution | High | NanoClaw-style REPL |
| KV cache visualizer | See cache state graphically | High | Web UI or CLI tree |
| Prefix cache analysis | Understand prompt reuse rates | Med | CLI `analyze-cache` command |
| Speculative decoding profiler | Tune draft parameters | Med | Track acceptance rates |
| GPU memory timeline | Visualize memory allocation | Med | Use perfetto |
| Fuzzing harness | Catch edge cases automatically | Med | cargo-fuzz integration |

## Anti-Features

Features to explicitly NOT build.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Full GUI debugger | Binary bloat, complexity | CLI REPL is sufficient |
| Cloud-based profiling | Privacy concerns, latency | Local perfetto export |
| Real-time dashboard | Complexity, maintenance | Prometheus + Grafana (existing) |
| Multi-user debug session | Auth complexity | Single-user CLI tool |
| Web UI for benchmarking | Over-engineering | CLI + JSON output |

## Feature Dependencies

```
Feature A → Feature B (B requires A)

Benchmark Suite
├── Basic throughput → Latency percentiles
├── Single request → Concurrent requests
└── Local metrics → Prometheus export

Debug Utilities
├── Request tracing → KV cache inspection
├── Span recording → Trace playback
└── Cache stats → Prefix analysis

CLI Tools
├── Model listing → Model info
├── Config validation → Config generation
└── Basic commands → Interactive REPL

Test Infrastructure  
├── Mock models → Integration tests
├── Property tests → Fuzzing harness
└── Unit tests → E2E benchmarks
```

## MVP Recommendation

Prioritize:

1. **Throughput benchmark** - Core metric, easy to implement
2. **Latency benchmarks** - P50/P95/P99, important for SLA
3. **Request tracing** - Uses existing tracing infrastructure
4. **Metrics extension** - Adds tooling metrics to existing endpoint
5. **Config validation** - Prevents startup errors

Defer:
- **Interactive REPL** (High complexity, nice-to-have)
- **Fuzzing harness** (Good for long-term quality, not MVP)
- **GPU memory timeline** (Requires CUDA-specific tooling)

## Benchmarking Feature List

### Required (MVP)

| Feature | Description | Acceptance Criteria |
|---------|-------------|---------------------|
| Throughput test | Sustained load test | Reports tokens/sec at N concurrent requests |
| Latency test | Timing per request | Reports TTFT, P50, P95, P99 latency |
| Warmup handling | Skip cold-start data | Discards first N iterations as warmup |

### Nice-to-Have

| Feature | Description | Acceptance Criteria |
|---------|-------------|---------------------|
| Throughput sweep | Test multiple concurrency levels | Generate throughput curve |
| Memory profiling | Track memory during test | Report peak memory usage |
| CUDA graph impact | Compare with/without CUDA graphs | Report speedup percentage |

## Debug Feature List

### Required (MVP)

| Feature | Description | Acceptance Criteria |
|---------|-------------|---------------------|
| Request trace | Log all operations for a request | JSON output with timestamps |
| KV cache dump | Inspect cache state | List all cached prompts |
| Metrics snapshot | Point-in-time metrics | Export current metric values |

### Nice-to-Have

| Feature | Description | Acceptance Criteria |
|---------|-------------|---------------------|
| Cache hit analysis | Explain why prefix hit/miss | Show matching and divergence |
| Memory timeline | GPU memory over time | Export for visualization |
| Batch visualization | See batch composition | ASCII art of batch |

## CLI Feature List

### Required (MVP)

| Feature | Description | Command |
|---------|-------------|---------|
| Serve server | Run inference server | `vllm-server -m model` (default) |
| Validate config | Check config file | `vllm-tool validate config.yaml` |
| List models | Show available models | `vllm-tool list-models ./models` |
| Get model info | Show model metadata | `vllm-tool model-info -m llama` |

### Nice-to-Have

| Feature | Description | Command |
|---------|-------------|---------|
| Model download | Download from HuggingFace | `vllm-tool download Qwen/Qwen2-0.5B` |
| Config generate | Scaffold config | `vllm-tool init-config` |
| Benchmark | Run benchmark suite | `vllm-tool benchmark --duration 60` |

## Test Infrastructure Feature List

### Required (MVP)

| Feature | Description | Acceptance Criteria |
|---------|-------------|---------------------|
| Integration test helpers | Common test utilities | `TestHarness::new()` |
| Mock model variants | Different behavior mocks | `NeverProgressModel`, `SlowModel` |
| Request factory | Generate test requests | `TestRequest::random()` |

### Nice-to-Have

| Feature | Description | Acceptance Criteria |
|---------|-------------|---------------------|
| Property-based tests | Generative testing | 1000+ test cases auto-generated |
| Fuzzing corpus | Edge case inputs | Model behavior under mutation |
| Benchmark CI check | Reject regressions | Fail PR if regression > 5% |

---

## Sources

- [vLLM Performance Guide](https://docs.vllm.ai/en/latest/dev PERFORMANCE.html)
- [Criterion Examples](https://github.com/bheisner/criterion.rs)
- [Rust Fuzzing Book](https://rust-fuzz.github.io/book/)
- [Proptest Tutorial](https://proptest-rs.github.io/proptest-book/)
