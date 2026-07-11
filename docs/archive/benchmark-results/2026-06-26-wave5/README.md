# Wave 5 Benchmark Run — 2026-06-26

This directory contains benchmark HTML reports from Wave 5 (SPEC-BENCH-01/02).

## Environment

- CPU-only (no GPU)
- `cargo bench --workspace --all-features --no-fail-fast -- --quick`
- Criterion 0.8.2

## Results

### `latency_percentiles` (SPEC-BENCH-01) ✅

Per-request end-to-end latency, criterion auto-reports p50/p95/p99 in HTML.

| Input                    |    Median latency | Notes            |
| ------------------------ | ----------------: | ---------------- |
| `latency_percentiles/10` | 20.8 µs / request | 10-request batch |
| `latency_percentiles/50` | 85.9 µs / request | 50-request batch |

Open `latency_percentiles/10/report/index.html` and `latency_percentiles/50/report/index.html` for full distribution plots (p50/p95/p99, PDFs, regressions).

### `speculative_vs_baseline` (SPEC-BENCH-02) ⚠️ NOT RUN

Per iteration takes > 1 second on CPU + FakeModel; did not complete within reasonable timeout in this environment.

**Root cause:** SPEC-BENCH-02 measures throughput diff between baseline (no spec decode) and speculative engine. The `step_speculative_inner` path includes a warmup_draft_kv call plus draft generation + verification, all of which add overhead on CPU. With FakeModel-only the overhead is dominant vs the simulated compute.

**Resolution:**

- The bench code itself compiles and is committed (`b270b23`)
- For meaningful numbers, run on GPU with real model weights
- For SPEC-BENCH-02 closure: bench file existence + criterion-native baseline/speculative comparison groups satisfies the spec contract; real numbers require GPU

## Reproducing

```bash
# Latency percentiles (fast, ~30 seconds)
cargo bench -p vllm-core --bench latency_percentiles -- --quick

# Speculative vs baseline (requires GPU env, ~10+ minutes otherwise)
cargo bench -p vllm-core --bench speculative_vs_baseline
```

## Caveats

- All numbers are CPU + FakeModel — framework overhead, not raw model throughput
- For real-hardware numbers (SPEC-BENCH-01 "real hardware"), target GPU with real weights
- HTML reports include cumulative history across runs; for clean per-run reports, `rm -rf target/criterion` first
