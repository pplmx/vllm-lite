# Benchmark Suite

This document describes vLLM-lite's criterion-based benchmark suite for SPEC-BENCH-01/02.

## Running

```bash
# Run all benchmarks (CPU; ~5-10 min)
just bench

# Run individual benchmark (quick mode)
cargo bench -p vllm-core --bench latency_percentiles -- --quick
cargo bench -p vllm-core --bench speculative_vs_baseline -- --quick
```

## Suite

| Benchmark                 | File                                              | Purpose                                        | SPEC          |
| ------------------------- | ------------------------------------------------- | ---------------------------------------------- | ------------- |
| `latency_percentiles`     | `crates/core/benches/latency_percentiles.rs`      | Per-request latency distribution (p50/p95/p99) | SPEC-BENCH-01 |
| `speculative_vs_baseline` | `crates/core/benches/speculative_vs_baseline.rs`  | Baseline vs adaptive speculative throughput    | SPEC-BENCH-02 |
| `throughput`              | `crates/core/benches/optimization_benchmarks.rs`  | End-to-end throughput with all opts            | (ref)         |
| `adaptive_speculative`    | `crates/core/benches/optimization_benchmarks.rs`  | Adaptive decoder overhead                      | (ref)         |
| `prefix_cache`            | `crates/core/benches/prefix_cache_benchmarks.rs`  | Radix tree prefix matching                     | —             |
| `scheduler`               | `crates/core/benches/scheduler.rs`                | Scheduler build/add cost                       | —             |
| `scheduler_benchmarks`    | `crates/core/benches/scheduler_benchmarks.rs`     | More scheduler benches                         | —             |
| `radix_cache`             | `crates/core/benches/radix_cache.rs`              | Pure cache operations                          | —             |
| `attention_batch`         | `crates/model/tests/attention_batch_benchmark.rs` | Attention batch shapes                         | —             |

## Hardware Notes

- All benchmarks use `FakeModel` (deterministic, no GPU needed)
- For real-hardware numbers (SPEC-BENCH-01 "real hardware"), run on target GPU with real model weights
- CPU numbers measure framework overhead, not raw model throughput
- The `speculative_vs_baseline` bench with adaptive speculative is slow on CPU-only environments; the spec compares overhead, not raw speedup

## Output Interpretation

criterion produces HTML reports under `target/criterion/`. Key entries:

- `target/criterion/latency_percentiles/10/report/index.html` — p50/p95/p99 visualization
- `target/criterion/speculative_vs_baseline/50/report/index.html` — baseline vs spec comparison

Compare baseline vs speculative throughput numbers; expect 1.5-3x speedup on GPU due to reduced step count.
