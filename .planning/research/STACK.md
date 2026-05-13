# Technology Stack: Production Speculative Decoding

**Project:** vLLM-lite — Speculative Decoding (v17.0 milestone)
**Researched:** 2026-05-13

## Recommended Stack

The project already has a well-defined Rust + Candle stack. This document focuses on the speculative decoding *additions* — what new technologies, if any, are needed.

### No New Dependencies Required

All features in the v17.0 milestone can be implemented using the existing technology stack:

| Technology                                | Already In Use         | For Spec Decode                      |
| ----------------------------------------- | ---------------------- | ------------------------------------ |
| Rust + Candle                             | Core inference engine  | Model forward for draft + target     |
| `criterion` (bench)                       | benchmarks/ Cargo.toml | Benchmark suite for spec vs non-spec |
| `metrics` / `metrics-exporter-prometheus` | MetricsCollector       | Spec decode acceptance rate counters |
| `serde` / `serde_json`                    | Config serialization   | Spec decode config serialization     |
| `tracing`                                 | Logging system         | Spec decode step tracing             |
| `tokio`                                   | Async runtime (server) | Async benchmark runners              |

### Additional Library Considerations (Optional, Not Required)

| Library                     | Purpose                                       | Why Use / Why Not                                                                                                                                                                |
| --------------------------- | --------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ndarray` / `ndarray-stats` | Percentile computation on GPU latency samples | Not needed. Current `PercentileStats` in `benches/src/percentile.rs` works on CPU-side f64 vectors. `ndarray-stats` would only help if we processed thousands of samples on GPU. |
| `histogram` crate           | Latency histogram bucketing for Prometheus    | Potentially useful for production metrics. Prometheus prefers pre-bucketed histograms. Could replace the current raw sample collection approach. NOT required for v17.0.         |
| `csv` crate                 | Export benchmark results to CSV               | Useful for benchmark comparison across runs. Currently results are in-memory `HashMap<String, BenchmarkResult>`. Add for v17.1.                                                  |

## Alternatives Considered

| Category         | Recommended                                   | Alternative                 | Why Not                                                                                                                                                                                                                                                        |
| ---------------- | --------------------------------------------- | --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Draft management | Self-speculation (1/8 layers, weight sharing) | Separate draft model        | Self-spec: zero additional GPU memory, works immediately with existing ArchitectureRegistry. Separate model: doubles memory, requires lifecycle management, but can give higher acceptance rates (specialized drafter). Self-spec is the right starting point. |
| Benchmark runner | Existing `BenchmarkSuite`                     | `iai` / `criterion` benches | Criterion-style microbenchmarks measure single-op latency (e.g., `flash_attention::forward`). For spec vs non-spec comparison, we need end-to-end request-level benchmarks. The existing `ThroughputBenchmark` / `LatencyBenchmark` are designed for this.     |
| Metrics format   | Prometheus counters                           | OpenTelemetry metrics       | Prometheus is already integrated (`metrics-exporter-prometheus`). Three spec decode counters is trivially addable. OpenTelemetry adds deployment complexity.                                                                                                   |
| Adaptive control | Simple threshold + EWMA                       | PID controller              | A full PID controller is overkill for a single-variable system with noisy measurements. EWMA smooth + deadband ±5% + cooldown 10 steps has fewer tuning parameters and is easier to debug.                                                                     |

## Existing Infrastructure (Already Built)

All of this is already in place and can be extended:

- **`AdaptiveSpeculativeDecoder`** in `crates/core/src/speculative/adaptive.rs` — sliding window tracking, threshold adjustment
- **`DraftAccuracyTracker`** in same file — per-token acceptance tracking
- **`BenchmarkSuite`** in `benches/src/lib.rs` — named benchmarks with warmup, iteration config
- **`ThroughputBenchmark`** / **`LatencyBenchmark`** in `benches/src/e2e.rs` — request-level benchmark runners
- **`PercentileStats`** in `benches/src/percentile.rs` — P50/P95/P99 computation
- **`MetricsCollector`** in `crates/core/src/metrics/` — counter + gauge tracking with Prometheus export
- **`DraftVerifier`** in `crates/core/src/speculative/verifier.rs` — TokenLevel rejection strategy
- **`SelfSpeculativeModel`** in `crates/core/src/speculative/self_spec.rs` — 1/8 layer, weight-shared draft

## Sources

- Current vLLM-lite codebase: `Cargo.toml` (workspace), `benches/Cargo.toml`, `crates/core/src/metrics/` — **HIGH confidence**

---

*Stack research for: Production Speculative Decoding in LLM Inference Engine*
*Researched: 2026-05-13*
