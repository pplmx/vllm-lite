# Domain Pitfalls: vllm-lite v14.0 Developer Tooling

**Domain:** LLM Inference Engine Developer Tooling
**Researched:** 2026-04-27

## Critical Pitfalls

Mistakes that cause rewrites or major issues.

### Pitfall 1: Profiling Overhead in Hot Path

**What goes wrong:** Tooling code adds latency to `Engine::step()`, increasing inference latency.

**Why it happens:** Adding tracing spans, metrics recording, or serialization in the critical path.

**Consequences:**
- Performance regression in production
- Benchmarks show incorrect numbers (tooling overhead included)
- SLA violations

**Prevention:**
```rust
// BAD: Always serialize on hot path
fn step(&mut self) -> Result<Vec<(SeqId, TokenId)>> {
    let result = self.execute_regular(&batch)?;
    
    // BAD: Serialization in hot path
    if self.trace_enabled {
        let trace = serde_json::to_string(&result).unwrap();
        self.trace_buffer.push(trace);
    }
    
    result
}

// GOOD: Buffer and flush async
fn step(&mut self) -> Result<Vec<(SeqId, TokenId)>> {
    let result = self.execute_regular(&batch)?;
    
    // GOOD: Record raw data, serialize off critical path
    if self.trace_enabled {
        self.trace_buffer.push(TraceEvent {
            timestamp: Instant::now(),
            result: (&result.seq_ids, result.next_tokens.len()),
        });
    }
    
    result
}
```

**Detection:** Benchmark with/without `--features profiling`, expect < 1% overhead.

---

### Pitfall 2: Memory Bloat in Metrics Collection

**What goes wrong:** Metrics buffer grows unbounded, consuming memory.

**Why it happens:** Accumulating events without bounds or TTLs.

**Consequences:**
- OOM on long-running servers
- Memory metrics show tooling as source of leak
- False memory regression in benchmarks

**Prevention:**
```rust
// BAD: Unbounded accumulation
pub struct ToolingMetrics {
    latency_samples: Vec<u64>,  // Grows forever!
}

// GOOD: Bounded buffer with circular behavior
pub struct ToolingMetrics {
    latency_samples: VecDeque<u64>,
    max_samples: usize,
}

impl ToolingMetrics {
    pub fn record_latency(&mut self, ns: u64) {
        if self.latency_samples.len() >= self.max_samples {
            self.latency_samples.pop_front();
        }
        self.latency_samples.push_back(ns);
    }
}
```

**Detection:** Monitor memory after 24h run, set upper bound on buffers.

---

### Pitfall 3: Blocking in Async Context

**What goes wrong:** Benchmark or debug code uses blocking I/O in async runtime.

**Why it happens:** Using `std::fs` instead of `tokio::fs`, `thread::sleep` instead of `tokio::time::sleep`.

**Consequences:**
- Server hangs during benchmarks
- Starvation of other async tasks
- "Too many open files" errors

**Prevention:**
```rust
// BAD: Blocking I/O
async fn run_benchmark(&self) {
    let data = std::fs::read("config.json").unwrap();  // Blocks!
    tokio::time::sleep(Duration::from_secs(1)).await;  // Fine
}

// GOOD: Async I/O
async fn run_benchmark(&self) {
    let data = tokio::fs::read("config.json").await.unwrap();  // Non-blocking!
    tokio::time::sleep(Duration::from_secs(1)).await;
}
```

**Detection:** Use `tokio-console` to detect blocked tasks.

---

## Moderate Pitfalls

### Pitfall 4: Benchmark Non-Determinism

**What goes wrong:** Benchmark results vary wildly between runs.

**Why it happens:** Not accounting for warmup, not isolating requests, using wall-clock time without synchronization.

**Consequences:**
- Can't detect regressions
- Conflicting benchmark reports
- CI/CD failures on flaky benchmarks

**Prevention:**
```rust
// GOOD: Proper benchmark setup
fn run_throughput_benchmark() -> BenchmarkResult {
    // 1. Warmup phase (discard results)
    for _ in 0..WARMUP_ITERATIONS {
        engine.add_request(test_request());
        engine.step();
    }
    
    // 2. Synchronize before measurement
    engine.sync();  // Ensure GPU is idle
    
    // 3. Timed phase
    let start = Instant::now();
    for _ in 0..MEASURE_ITERATIONS {
        engine.add_request(test_request());
        engine.step();
    }
    let elapsed = start.elapsed();
    
    // 4. Statistical analysis
    calculate_statistics(elapsed)
}
```

---

### Pitfall 5: Config Validation After Engine Start

**What goes wrong:** Invalid config causes panic mid-initialization.

**Why it happens:** Validation happens too late, after partial state setup.

**Consequences:**
- Crash during startup
- Half-initialized state
- Hard to debug

**Prevention:**
```rust
// GOOD: Validate early
fn AppConfig::load(path: PathBuf) -> Self {
    let raw = std::fs::read_to_string(&path).unwrap();
    let config: AppConfig = serde_yaml::from_str(&raw).unwrap();
    
    // Validate BEFORE any state
    config.validate().expect("Invalid config");
    
    config
}

fn AppConfig::validate(&self) -> Result<(), ConfigError> {
    if self.engine.max_batch_size > 8192 {
        return Err(ConfigError::BatchSizeTooLarge);
    }
    if self.engine.num_kv_blocks == 0 {
        return Err(ConfigError::KvBlocksRequired);
    }
    Ok(())
}
```

---

### Pitfall 6: Tool Binary Bloating Server

**What goes wrong:** Adding CLI subcommands bloats the server binary.

**Why it happens:** Putting tooling code in `vllm-server` instead of `vllm-tool`.

**Consequences:**
- Slower compile times
- Larger binary
- Users pay for tooling they don't use

**Prevention:**
```rust
// GOOD: Separate binaries
[workspace]
members = [
    "crates/server",      # vllm-server only
    "crates/tooling",     # vllm-tool only
]

// Server: minimal
[package]
name = "vllm-server"

# Tooling: rich
[package]
name = "vllm-tool"
```

---

## Minor Pitfalls

### Pitfall 7: Ignoring GPU State in Benchmarks

**What goes wrong:** Benchmarks don't reset GPU between runs.

**Why it happens:** Not calling `cudaDeviceReset()` or not waiting for GPU to finish.

**Consequences:**
- First benchmark run is always slower (cold start)
- Memory fragmentation affects later runs

**Prevention:**
```rust
#[cfg(feature = "cuda")]
fn reset_gpu_state() {
    unsafe { cudaDeviceReset() };
}

fn run_benchmark_suite() {
    reset_gpu_state();
    
    for benchmark in benchmarks {
        reset_gpu_state();  // Isolate each benchmark
        benchmark.run();
    }
}
```

---

### Pitfall 8: Not Versioning Benchmark Format

**What goes wrong:** Changing output format breaks scripts.

**Why it happens:** No version in output format, no changelog.

**Prevention:**
```rust
#[derive(Serialize)]
struct BenchmarkReport {
    version: &'static str,  // "v1.0"
    timestamp: DateTime<Utc>,
    results: Vec<BenchmarkResult>,
}

impl BenchmarkReport {
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap()
    }
}
```

---

## Phase-Specific Warnings

| Phase | Pitfall | Mitigation |
|-------|---------|------------|
| Infrastructure | Observer trait changes break observers | Add new events via enum extension, not trait changes |
| Benchmarking | Non-deterministic results | Require warmup, use statistical validation |
| Debugging | Trace buffer unbounded | Implement circular buffer with max size |
| CLI | Config schema drift | Keep schema in sync with code, validate on load |
| Testing | Test flakiness from GPU state | Reset GPU in test setup/teardown |

## Sources

- [Criterion.rs Best Practices](https://bheisner.github.io/criterion.rs/book/user_guide/)
- [Tokio Anti-Patterns](https://tokio.rs/tokio/topics/anti-patterns)
- [Rust Fuzzing Pitfalls](https://rust-fuzz.github.io/book/introduction.html)
