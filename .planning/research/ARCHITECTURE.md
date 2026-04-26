# Architecture: vllm-lite v14.0 Developer Tooling

**Domain:** LLM Inference Engine Developer Tooling Integration
**Researched:** 2026-04-27

## Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        vllm-lite Architecture                           │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     Tooling Layer (NEW)                          │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │   │
│  │  │ Benchmark   │  │  Debug      │  │  CLI        │             │   │
│  │  │ Suite       │  │  Utilities  │  │  Tools      │             │   │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │   │
│  │         │                │                │                     │   │
│  │  ┌──────┴────────────────┴────────────────┴──────┐             │   │
│  │  │            ToolingBridge (shared crate)        │             │   │
│  │  │  - Metrics export                               │             │   │
│  │  │  - Trace context                               │             │   │
│  │  │  - Config schema                               │             │   │
│  │  └──────┬────────────────┬────────────────┬──────┘             │   │
│  └─────────┼────────────────┼────────────────┼────────────────────┘   │
│            │                │                │                         │
│  ┌─────────┴────────────────┴────────────────┴────────────────────┐   │
│  │                     Core Engine (EXISTING)                      │   │
│  │                                                                  │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │   │
│  │  │   Engine    │  │ Scheduler   │  │   KV Cache  │             │   │
│  │  │             │──│  Engine     │──│  (Prefix)   │             │   │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │   │
│  │         │                │                │                     │   │
│  │  ┌──────┴────────────────┴────────────────┴──────┐             │   │
│  │  │         SchedulerObservers (extensible)        │             │   │
│  │  │  - on_request_start                           │             │   │
│  │  │  - on_batch_built                             │             │   │
│  │  │  - on_forward_complete                        │             │   │
│  │  │  - on_request_end                             │             │   │
│  │  └───────────────────────────────────────────────┘             │   │
│  │                                                                  │   │
│  │  ┌───────────────────────────────────────────────┐             │   │
│  │  │     EnhancedMetricsCollector (extensible)      │             │   │
│  │  │  - throughput counters                         │             │   │
│  │  │  - latency histograms                          │             │   │
│  │  │  - resource gauges                             │             │   │
│  │  └───────────────────────────────────────────────┘             │   │
│  │                                                                  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Integration Points

### 1. SchedulerObservers (PRIMARY)

**Location:** `crates/core/src/scheduler/observer.rs`

**Existing:** `SchedulerObservers` with basic event types.

**Extension needed:** Add new event types for tooling:

```rust
// NEW: Extend ObserverEvent enum
pub enum ObserverEvent {
    // ... existing events ...
    
    // NEW: Benchmarking events
    RequestReceived { request_id: u64, prompt_len: usize },
    BatchScheduled { batch_size: usize, prefill_tokens: usize, decode_tokens: usize },
    ForwardComplete { duration_ns: u64, tokens_processed: usize },
    PrefixCacheHit { request_id: u64, hit_ratio: f32 },
    
    // NEW: Debug events
    KvCacheAllocation { block_count: usize, seq_id: u64 },
    EvictionTriggered { blocks_evicted: usize, reason: EvictionReason },
    PreemptionOccurred { seq_id: u64, reason: PreemptionReason },
    
    // NEW: Tracing events
    SpanStart { span_id: u64, operation: OperationType },
    SpanEnd { span_id: u64, status: ResultStatus },
}
```

**Integration pattern:**
```rust
// In SchedulerEngine::add_request()
pub fn add_request(&mut self, mut req: Request) -> SeqId {
    self.metrics.record_request();
    
    // NEW: Emit observer event for tooling
    self.observers.emit(ObserverEvent::RequestReceived {
        request_id: req.id,
        prompt_len: req.prompt.len(),
    });
    
    // ... rest of implementation
}
```

### 2. EnhancedMetricsCollector (METRICS PIPELINE)

**Location:** `crates/core/src/metrics/collector.rs`

**Existing:** Atomic counters and gauges for internal metrics.

**Extension needed:** Add tooling-specific metrics:

```rust
// NEW: Add to EnhancedMetricsCollector
pub struct EnhancedMetricsCollector {
    // ... existing fields ...
    
    // NEW: Benchmark metrics
    benchmark_iterations: AtomicU64,
    benchmark_total_duration_ns: AtomicU64,
    
    // NEW: Debug metrics
    trace_events_total: AtomicU64,
    kv_cache_inspections: AtomicU64,
    
    // NEW: Resource metrics
    peak_memory_bytes: AtomicU64,
    cuda_graph_capture_time_ns: AtomicU64,
}
```

**Export endpoint:** `GET /v1/metrics` already exists - tooling can extend format.

### 3. Engine Step Hot Path (PROFILING)

**Location:** `crates/core/src/engine.rs`

**Integration point:** Wrap model forward with profiling spans:

```rust
// NEW: In Engine::execute_regular()
fn execute_regular(&mut self, batch: &Batch) -> Result<BatchOutput> {
    let start = Instant::now();
    
    // NEW: Optional profiling span
    #[cfg(feature = "profiling")]
    let _span = tracing::info_span!("model_forward", 
        batch_size = batch.seq_ids.len(),
        total_tokens = batch.total_tokens
    ).entered();
    
    let result = self.target_model.lock().unwrap().forward(...);
    
    // NEW: Record timing if enabled
    #[cfg(feature = "profiling")]
    if result.is_ok() {
        self.metrics.record_inference_latency(start.elapsed().as_nanos() as u64);
    }
    
    result
}
```

### 4. Prefix Cache Inspection (DEBUG)

**Location:** `crates/core/src/kv_cache/prefix_cache.rs`

**New API for tooling:**

```rust
// NEW: Add to PrefixCache
impl PrefixCache {
    /// Get cache statistics for debugging
    pub fn get_stats(&self) -> PrefixCacheStats {
        PrefixCacheStats {
            total_entries: self.entries.len(),
            total_blocks: self.compute_total_blocks(),
            hit_rate: self.compute_hit_rate(),
            memory_usage_bytes: self.estimate_memory_usage(),
        }
    }
    
    /// Dump cache entries for debugging
    pub fn dump_entries(&self) -> Vec<CacheEntryDebug> {
        // Returns serializable debug info
    }
}
```

### 5. CLI Extension Points (CLI TOOLING)

**Location:** `crates/server/src/cli.rs`

**Current:** `CliArgs` with clap derive.

**Extension pattern:**

```rust
// NEW: Add tooling subcommand
#[derive(Parser, Debug)]
enum Command {
    /// Run the inference server (default)
    Serve(RunArgs),
    
    /// Tooling commands
    #[command(subcommand)]
    Tool(ToolCommand),
}

#[derive(Subcommand, Debug)]
enum ToolCommand {
    /// Benchmark the engine
    Benchmark(BenchmarkArgs),
    
    /// Validate configuration
    Validate(ConfigPath),
    
    /// Inspect KV cache state
    DebugCache(DebugCacheArgs),
    
    /// List available models
    ListModels(PathBuf),
}
```

## New Components

### Crate Structure

```
crates/
├── tooling/                    # NEW: Tooling integration crate
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── metrics/            # Metrics aggregation
│       ├── trace/              # Distributed tracing
│       ├── bench/              # Benchmark utilities
│       └── debug/              # Debug utilities
```

### ToolingBridge - Shared Types

```rust
// crates/tooling/src/lib.rs
pub mod metrics;
pub mod trace;
pub mod bench;
pub mod debug;

pub use metrics::ToolingMetrics;
pub use trace::TraceContext;
pub use bench::{BenchmarkConfig, BenchmarkResult};
pub use debug::{CacheSnapshot, RequestTrace};
```

## Data Flow

### Benchmarking Flow

```
User CLI: `vllm-tool benchmark --concurrency 16 --duration 60s`
    │
    ▼
ToolCommand::Benchmark → BenchmarkRunner
    │
    ├──▶ Load model config
    │
    ├──▶ Create benchmark requests
    │
    ├──▶ Execute via Engine API
    │       │
    │       ├──▶ SchedulerObservers records timing
    │       │
    │       └──▶ EnhancedMetricsCollector aggregates
    │
    └──▶ Output results
        - Tokens/second
        - Latency percentiles
        - Memory usage
        - Prefix cache hit rate
```

### Debug Flow

```
User CLI: `vllm-tool debug request --id 12345`
    │
    ▼
ToolCommand::DebugRequest → DebugService
    │
    ├──▶ Query SchedulerObservers trace log
    │
    ├──▶ Query EnhancedMetricsCollector for metrics
    │
    ├──▶ Query PrefixCache for cache state
    │
    └──▶ Output formatted trace
        - Timeline of operations
        - Token generation steps
        - KV cache usage
        - Any errors
```

## Build Order

### Phase 1: Infrastructure (Week 1)

| Order | Component | Files Modified | Files Added |
|-------|-----------|----------------|-------------|
| 1.1 | Create `crates/tooling` crate | `Cargo.toml` (workspace) | `crates/tooling/Cargo.toml`, `crates/tooling/src/lib.rs` |
| 1.2 | Extend `SchedulerObservers` | `crates/core/src/scheduler/observer.rs` | - |
| 1.3 | Extend `EnhancedMetricsCollector` | `crates/core/src/metrics/collector.rs` | - |
| 1.4 | Add feature flag `profiling` | `crates/core/Cargo.toml`, `crates/core/src/engine.rs` | - |

### Phase 2: Benchmarking (Week 2)

| Order | Component | Files Modified | Files Added |
|-------|-----------|----------------|-------------|
| 2.1 | Benchmark runner framework | - | `crates/tooling/src/bench/mod.rs` |
| 2.2 | Throughput benchmark | - | `crates/tooling/src/bench/throughput.rs` |
| 2.3 | Latency benchmark | - | `crates/tooling/src/bench/latency.rs` |
| 2.4 | Integrate with existing benches | `benches/integration.rs` | - |

### Phase 3: Debug Utilities (Week 3)

| Order | Component | Files Modified | Files Added |
|-------|-----------|----------------|-------------|
| 3.1 | Tracing spans in engine | `crates/core/src/engine.rs` | - |
| 3.2 | KV cache inspection API | `crates/core/src/kv_cache/prefix_cache.rs` | - |
| 3.3 | Debug HTTP endpoints | `crates/server/src/openai/mod.rs` | `crates/server/src/openai/debug.rs` |
| 3.4 | Request trace formatter | - | `crates/tooling/src/debug/trace.rs` |

### Phase 4: CLI Improvements (Week 4)

| Order | Component | Files Modified | Files Added |
|-------|-----------|----------------|-------------|
| 4.1 | Add subcommand enum | `crates/server/src/cli.rs` | - |
| 4.2 | Model management commands | - | `crates/tooling/src/cli/models.rs` |
| 4.3 | Config validation | - | `crates/tooling/src/cli/validate.rs` |
| 4.4 | Debug cache command | - | `crates/tooling/src/cli/debug.rs` |

### Phase 5: Test Infrastructure (Week 5)

| Order | Component | Files Modified | Files Added |
|-------|-----------|----------------|-------------|
| 5.1 | Integration test helpers | `crates/testing/src/lib.rs` | `crates/testing/src/integration/*.rs` |
| 5.2 | Property-based tests | - | `crates/testing/src/property/*.rs` |
| 5.3 | Fuzzing harness | - | `fuzz/Cargo.toml`, `fuzz/src/*.rs` |

## Anti-Patterns to Avoid

### Anti-Pattern 1: Blocking Tooling in Hot Path

**What:** Performing expensive operations (serialization, logging) in `Engine::step()`.

**Why bad:** Increases latency, breaks SLA.

**Instead:** Always use async/non-blocking patterns, buffer events.

### Anti-Pattern 2: Direct Engine State Modification

**What:** Tooling modifying scheduler state or KV cache directly.

**Why bad:** Race conditions, undefined behavior.

**Instead:** Use read-only APIs via observers, never mutate from tooling.

### Anti-Pattern 3: Feature-Creep CLI

**What:** Adding too many commands to main binary.

**Why bad:** Binary bloat, slower compile times.

**Instead:** Use separate `vllm-tool` binary, keep server minimal.

---

## Sources

- [Rust Benchmarking Guide](https://doc.rust-lang.org/nightly/unstable-book/library-features/test.html)
- [Criterion Documentation](https://bheisner.github.io/criterion.rs/)
- [Tokio Tracing](https://tokio.rs/tokio/tracing)
- [ Clap Subcommands](https://docs.rs/clap/latest/clap/_derive/_tutorial/index.html)
