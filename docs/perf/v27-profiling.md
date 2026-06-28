# v27.0 Profiling Guide

## Overview

Profiling identifies hotspots — functions consuming the most CPU time — that
optimization efforts should target. v27.0 uses two complementary tools:

1. **`cargo-flamegraph`** — generates SVG flamegraphs from sampling data; works
   with any criterion version (recommended for our criterion 0.8 benches)
2. **`pprof` crate (0.15)** — embedded sampling profiler; useful for
   custom/manual integration in benches and tests

`pprof` is added as a workspace dev-dependency; `cargo-flamegraph` is installed
separately as a cargo subcommand.

## Version note: pprof vs criterion

`pprof` 0.15's `criterion` feature pins `criterion 0.5`, which conflicts with
our workspace's `criterion 0.8`. For this reason:

- The `pprof::criterion::PProfProfiler` integration **does not compile** against
  our criterion 0.8 benches. **Do not enable** the `criterion` feature on pprof.
- Use `cargo-flamegraph` (wraps pprof at the binary level, criterion-agnostic).
- For manual control, use `pprof::ProfilerGuard` directly inside bench code.

The `flamegraph` feature is enabled; the `criterion` feature is not.

## Prerequisites

```bash
# Install cargo-flamegraph
cargo install flamegraph

# Linux: may need perf for sampling
sudo sysctl kernel.perf_event_paranoid=-1   # or run as root

# Verify
which cargo-flamegraph
cargo flamegraph --help | head -5
```

## CPU sampling (cargo-flamegraph)

```bash
cd /workspace/vllm-lite/crates/model

# Generate flamegraph for a specific bench target
# Set seq_len small to keep sampling under a minute
cargo flamegraph --bench gqa_forward -- --bench gqa_forward_smoke/cpu_smoke

# Output: flamegraph.svg in current directory (or target/ subdir)
```

Open `flamegraph.svg` in a browser. The widest functions are the hottest.

For `vllm-core` benches:

```bash
cd /workspace/vllm-lite/crates/core
cargo flamegraph --bench radix_cache -- --bench radix_cache_smoke
```

## Manual pprof integration (ProfilerGuard)

When you need sampling within a specific bench function (not whole-binary),
use `pprof::ProfilerGuard` directly. This works with criterion 0.8.

```rust
use pprof::ProfilerGuardBuilder;
use std::fs::File;
use std::io::Write;

fn bench_with_profile(c: &mut criterion::Criterion) {
    let mut group = c.benchmark_group("profile_demo");
    group.bench_function("hot_loop", |b| {
        b.iter(|| {
            // Start sampling for this iteration only
            let guard = ProfilerGuardBuilder::default()
                .frequency(1000)
                .blocklist(&["libc", "libsystem", "pthread"])
                .build()
                .unwrap();

            // ... hot code under measurement ...

            // Stop and dump (here on every iter; in practice gate behind a flag)
            if let Ok(report) = guard.report().build() {
                let file = File::create("flamegraph.svg").unwrap();
                report.flamegraph(file).unwrap();
            }
        });
    });
    group.finish();
}
```

Note: dumping on every iteration is expensive. In practice, gate profiling
behind a feature flag or env var (e.g., `VLLM_PROFILE=1`) and run a single
representative iteration.

## Interpreting flamegraphs

- **Width = CPU time** (wider = more time consumed)
- **Color**: warm colors (red, orange) typically indicate hot paths
- **Call stack**: functions higher in the stack call functions below
- **Self-time vs total-time**: hover over a function to see both

For vllm-lite, look for:

- Tensor ops (`candle::softmax`, `candle::matmul`, `candle::embedding`)
- Allocation hot spots (`Vec::push`, `HashMap::insert`, `to_string`)
- Cache-unfriendly patterns (random access, large struct copies)
- Lock contention (`parking_lot::Mutex::lock`)
- Recurrent `clone()` of large tensors (`Tensor::copy`)

## Limitations

- **CPU sampling requires kernel support** (`perf_event_paranoid` setting)
- **Flamegraph overhead**: sampling can 2-5x bench runtime; do not include in CI
- **macOS**: uses `dtrace` instead of `perf`; different setup
- **GPU profiling**: requires NVIDIA tools (`nsys`, `ncu`); not covered here

## Sub-phase order (H-8 to H-10)

| Phase | Target                          | Output                                                    |
|-------|---------------------------------|-----------------------------------------------------------|
| H-8   | GQA forward                     | docs/perf/v27-profile-gqa.md                              |
| H-9   | MLA + FlashAttn                 | docs/perf/v27-profile-mla.md, docs/perf/v27-profile-flash.md |
| H-10  | PagedKV + BatchComposer         | docs/perf/v27-profile-pkv.md, docs/perf/v27-profile-batch.md |

Each sub-phase should:

1. Identify candidate hotspots via flamegraph or guided reasoning
2. Capture a baseline benchmark (TPS, latency p50/p99)
3. Apply one targeted optimization
4. Re-benchmark and document speedup

## References

- pprof crate: <https://docs.rs/pprof/>
- cargo-flamegraph: <https://github.com/flamegraph-rs/flamegraph>
- Brendan Gregg's flamegraph guide: <http://www.brendangregg.com/flamegraphs.html>
- Criterion profiling docs: <https://github.com/bheisler/criterion.rs/blob/master/book/src/user_guide/profiling.md>
