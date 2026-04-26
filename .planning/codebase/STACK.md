# Technology Stack

**Analysis Date:** 2026-04-26

## Languages

**Primary:**
- Rust 1.85+ (workspace-level requirement in `Cargo.toml:12`)

**Secondary:**
- Markdown (documentation)
- YAML (configuration files)

## Runtime

**Environment:**
- Native binary execution
- tokio async runtime for server operations

**Package Manager:**
- Cargo (Rust's native package manager)
- Lockfile: `Cargo.lock` (present, tracked in git)

## Frameworks

**Core ML:**
- **Candle** 0.10.2 - ML framework powering model inference
  - `crates/model/Cargo.toml:11-12`
  - Supports CPU and CUDA backends via feature flags

**HTTP Server:**
- **Axum** 0.8.8 - Web framework for API endpoints
  - `crates/server/Cargo.toml:21`
  - Provides routing, middleware, and async HTTP handling

**Async Runtime:**
- **Tokio** 1.x - Async runtime
  - `Cargo.toml:21`
  - Features: sync, rt, macros (workspace)
  - Server: full feature set including multi-threaded runtime

## Key Dependencies

### ML & Model Loading
| Package | Version | Purpose | File |
|---------|---------|---------|------|
| `candle-core` | 0.10.2 | Tensor operations, ML primitives | `crates/model/Cargo.toml:11` |
| `candle-nn` | 0.10.2 | Neural network layers | `crates/model/Cargo.toml:12` |
| `safetensors` | 0.7.0 | Safe model weight loading | `crates/model/Cargo.toml:14` |
| `gguf` | 0.1 | GGUF quantization format (optional) | `crates/model/Cargo.toml:22` |
| `half` | 2 | FP16/BF16 type support | `crates/model/Cargo.toml:17` |

### Tokenization
| Package | Version | Purpose | File |
|---------|---------|---------|------|
| `tokenizers` | 0.22 | HuggingFace-compatible tokenizer | `crates/model/Cargo.toml:21` |
| `tiktoken` | 3 | OpenAI-compatible BPE tokenizer | `crates/model/Cargo.toml:20` |

### Server & API
| Package | Version | Purpose | File |
|---------|---------|---------|------|
| `axum` | 0.8 | HTTP API framework | `crates/server/Cargo.toml:21` |
| `tower` | 0.5 | Middleware utilities | `crates/server/Cargo.toml:22` |
| `clap` | 4 | CLI argument parsing | `crates/server/Cargo.toml:31` |
| `serde` | 1 | Serialization/deserialization | `Cargo.toml:23` |
| `serde_json` | 1 | JSON handling | `Cargo.toml:24` |
| `serde_yaml` | 0.9 | YAML configuration | `crates/server/Cargo.toml:26` |

### Observability
| Package | Version | Purpose | File |
|---------|---------|---------|------|
| `tracing` | 0.1 | Structured logging | `crates/core/Cargo.toml:15` |
| `tracing-subscriber` | 0.3 | Log format/filtering | `crates/server/Cargo.toml:28` |
| `tracing-appender` | 0.2 | File logging | `crates/server/Cargo.toml:29` |
| `metrics` | 0.22 | Metrics collection | `Cargo.toml:25` |
| `metrics-exporter-prometheus` | 0.13 | Prometheus export (optional) | `Cargo.toml:26` |
| `opentelemetry` | 0.21 | Distributed tracing (optional) | `Cargo.toml:28-31` |

### Utilities
| Package | Version | Purpose | File |
|---------|---------|---------|------|
| `thiserror` | 2 | Error handling | `Cargo.toml:20` |
| `rand` | 0.10 | Random number generation | `Cargo.toml:22` |
| `uuid` | 1 | Request ID generation | `crates/server/Cargo.toml:30` |
| `dashmap` | 5.5 | Concurrent hash maps | `Cargo.toml:32` |
| `crossbeam` | 0.8 | Parallel data structures | `crates/core/Cargo.toml:14` |
| `rayon` | 1.10 | Data parallelism | `crates/model/Cargo.toml:18` |
| `memmap2` | 0.9 | Memory-mapped file I/O | `crates/model/Cargo.toml:19` |
| `reqwest` | 0.12 | HTTP client (model downloads) | `crates/server/Cargo.toml:32` |

### Testing
| Package | Version | Purpose | File |
|---------|---------|---------|------|
| `criterion` | 0.8 | Benchmarking | `crates/core/Cargo.toml:32` |
| `proptest` | 1.5 | Property-based testing | `crates/testing/Cargo.toml:15` |
| `tempfile` | 3 | Temporary test files | `crates/model/Cargo.toml:27` |

## Build System

**Tool:** Cargo (Rust's package manager)

**Build Commands:**
```bash
cargo build --release              # Release build
cargo build --workspace            # All crates
just build                         # Alias for release build
```

**Release Profile Optimizations:**
```toml
[profile.release]
opt-level = 3
lto = "fat"           # Fat LTO for maximum optimization
codegen-units = 1     # Single codegen unit
panic = "abort"       # Abort on panic (smaller binary)
strip = true          # Strip debug symbols
```

## Feature Flags

| Feature | Description | Enables |
|---------|-------------|---------|
| `cuda` | GPU acceleration | `candle-core/cuda`, `candle-nn/cuda` |
| `gguf` | GGUF model format | `gguf` crate |
| `full` | All features | `cuda` + `gguf` |
| `prometheus` | Metrics export | `metrics-exporter-prometheus` |
| `opentelemetry` | Distributed tracing | `opentelemetry` stack |
| `cuda-graph` | CUDA graph optimization | `vllm-model` integration |

**Default features:** `prometheus` (for vllm-core)

## Development Tools

**Pre-commit Hooks:**
- `commitizen` v4.13.9 - Conventional commit enforcement
- `rumdl` - Markdown linting
- Built-in hooks: file fixer, TOML/YAML validation

**Code Quality:**
```bash
just clippy    # Linting (required before commit)
just fmt-check # Format validation
just doc-check # Documentation validation
```

**Testing:**
```bash
just nextest       # Fast tests (skips #[ignore])
just nextest-all   # All tests including slow
just ci            # Full CI pipeline
```

## Platform Requirements

**Development:**
- Rust 1.85+
- Standard build tools (gcc/clang, cmake)

**Production:**
- Linux (primary target)
- x86_64 architecture
- CUDA-capable GPU (optional, for `cuda` feature)

---

*Stack analysis: 2026-04-26*
