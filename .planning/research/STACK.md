# Technology Stack: vllm-lite v14.0 Developer Tooling

**Project:** vllm-lite Developer Tooling
**Researched:** 2026-04-27

## Recommended Stack

### Core Framework
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| Rust stable | 1.85+ | Language | Existing codebase uses 1.85, no need to upgrade |
| criterion | 0.5 | Micro-benchmarking | Mature, async-friendly, statistical rigor |
| tracing | 0.1 | Distributed tracing | Already used in engine (tracing crate) |
| clap | 4.x | CLI argument parsing | Already used (v4.x in deps) |

### Benchmarking
| Technology | Version | Purpose | When to Use |
|-----------|---------|---------|-------------|
| criterion | 0.5 | Statistical micro-benchmarks | Unit-level performance testing |
| tokio test | 1.x | Async benchmarking | Server endpoint benchmarks |
| perfetto | - | GPU profiling | CUDA kernel profiling |

### Debugging
| Technology | Version | Purpose | When to Use |
|-----------|---------|---------|-------------|
| tracing | 0.1 | Structured spans | Request tracing |
| tracing-subscriber | 0.3 | Output formatting | Development debugging |
| opentelemetry | 0.21 | Trace export | Production observability |

### CLI
| Technology | Version | Purpose | When to Use |
|-----------|---------|---------|-------------|
| clap | 4.x | Argument parsing | Already a dep |
| serde | 1.x | Config serialization | Config validation |
| prettytable-rs | 0.10 | Table formatting | CLI output |

### Testing
| Technology | Version | Purpose | When to Use |
|-----------|---------|---------|-------------|
| cargo-fuzz | - | Coverage-guided fuzzing | Input fuzzing |
| proptest | 1.x | Property-based testing | Generative testing |
| tokio test | 1.x | Async integration tests | Already in use |

### Existing Dependencies (Already in Workspace)

From `Cargo.toml` workspace dependencies:
```toml
tokio = { version = "1", features = ["sync", "rt", "macros"] }
metrics = "0.22"
tracing = "0.1"  # Add: tracing-subscriber
serde = { version = "1", features = ["derive"] }
serde_json = "1"
thiserror = "2"
```

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| Benchmarking | criterion | i Benchmark | criterion is more battle-tested with better statistical output |
| Tracing | tracing (existing) | log + tracing compat | Already in use, opentelemetry integration exists |
| CLI | clap (existing) | picocli | clap is already a dependency |
| Fuzzing | cargo-fuzz | libfuzzer | cargo-fuzz integrates better with Rust, no external deps |
| Property tests | proptest | quickcheck | proptest has better shrinking, more flexible strategies |

## New Dependencies to Add

### crates/tooling/Cargo.toml
```toml
[package]
name = "vllm-tooling"
version = "0.1.0"
edition = "2024"

[dependencies]
# Core
tokio = { workspace = true }
serde = { workspace = true }
serde_json = { workspace = true }
thiserror = { workspace = true }

# Benchmarking
criterion = { version = "0.5", features = ["html_reports"] }
tokio-bench = "0.1"  # Not real - use tokio::spawn + Instant

# Tracing
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter", "json"] }
tracing-opentelemetry = { workspace = true }
opentelemetry_sdk = { workspace = true }

# CLI (extends existing clap in server)
clap = { workspace = true }
prettytable-rs = "0.10"

# Testing
proptest = "1"

[dev-dependencies]
cargo-fuzz = "0.11"
```

### crates/core/Cargo.toml additions
```toml
[features]
default = ["cuda", "gguf"]
cuda = ["vllm-model/cuda"]
gguf = ["vllm-model/gguf"]
# NEW
profiling = []  # Enable profiling spans in hot path
```

### crates/testing/Cargo.toml additions
```toml
[dependencies]
proptest = "1"
quickcheck = "1"
```

## Installation

```bash
# Add to workspace
cargo add -p vllm-tooling criterion tracing tracing-subscriber proptest prettytable-rs

# Enable profiling feature
cargo build -p vllm-core --features profiling

# Run benchmarks
cargo bench -p benches

# Fuzzing (requires nightly)
cargo +nightly fuzz run model_forward
```

## Sources

- [Criterion Rust Benchmarking](https://bheisner.github.io/criterion.rs/)
- [Tokio Tracing Tutorial](https://tokio.rs/tokio/tutorial/tracing)
- [Proptest Book](https://proptest-rs.github.io/proptest-book/)
- [Clap CLI Tutorial](https://docs.rs/clap/latest/clap/_tutorial/)
