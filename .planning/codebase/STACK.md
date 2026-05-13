# Technology Stack

**Analysis Date:** 2026-05-13

## Languages

**Primary:**

- Rust (Edition 2024, MSRV 1.85) - All inference engine, server, and core logic across all 7 workspace crates

**Secondary:**

- Go 1.22 - Kubernetes operator (`k8s/operator/go.mod`)
- YAML - Configuration, Kubernetes manifests, CI workflows

## Runtime

**Environment:**

- Rust toolchain 1.85+ (stable/beta matrices in CI)
- Go 1.22 (k8s operator only)

**Package Manager:**

- Cargo (workspace with `resolver = "2"`)
- Lockfile: `Cargo.lock` present (committed)
- Go modules: `k8s/operator/go.mod`

## Frameworks

**Core:**

- [Tokio](https://tokio.rs/) 1.x - Async runtime (multi-threaded, sync primitives, I/O, networking, signals)
- [Axum](https://docs.rs/axum/) 0.7 - HTTP web framework (routing, middleware, SSE streaming, state extraction)
- [Candle](https://github.com/huggingface/candle) 0.10.2 - ML tensor library (GPU via CUDA, CPU fallback)
- [Tonic](https://docs.rs/tonic/) 0.12 - gRPC framework for distributed inference

**CLI:**

- [Clap](https://docs.rs/clap/) 4.x - CLI argument parsing with derive macros and env var support

**Testing:**

- cargo test (built-in) - Unit tests embedded in `#[cfg(test)]` modules
- cargo-nextest - Test runner (used via `just nextest` for better performance)
- [Criterion](https://docs.rs/criterion/) 0.8 - Benchmark harness
- [Proptest](https://docs.rs/proptest/) 1.11 - Property-based testing (vllm-testing dev dependency)

**Build/Dev:**

- `justfile` - Build automation
- `rustfmt` - Code formatting (4-space indent, 100-char soft limit)
- `clippy` - Linting (`-D warnings`)
- `cargo fix` - Auto-fix clippy warnings
- `cargo-audit` - Security vulnerability scanning
- `rumdl` - Markdown linter
- `prek` - Pre-commit hook manager (commitizen, rumdl, whitespace checks)

## Key Dependencies

**Critical (core logic):**

- `candle-core` 0.10.2 / `candle-nn` 0.10.2 - ML framework for tensor operations and neural network layers
- `thiserror` 2.x - Error type derivation
- `tokio` 1.x - Async runtime (sync, rt, rt-multi-thread, macros, time, io-util, fs, net, signal)
- `serde` 1.x / `serde_json` 1.x - Serialization/deserialization
- `tracing` 0.1 - Structured logging
- `rand` 0.10 - Random number generation

**Infrastructure (server):**

- `axum` 0.7 - HTTP framework with `tokio` feature
- `tower` 0.5 / `tower-http` 0.5 - Middleware (CORS, trace)
- `clap` 4.x - CLI with `derive` and `env` features
- `reqwest` 0.12 - HTTP client with blocking support
- `uuid` 1.x - Request ID generation (v4)

**Monitoring:**

- `metrics` 0.22 - Metrics instrumentation
- `metrics-exporter-prometheus` 0.13 - Prometheus export format (optional, default-on via `prometheus` feature)
- `opentelemetry` 0.21 / `opentelemetry-otlp` 0.14 - OTLP export (optional)
- `opentelemetry_sdk` 0.21 - OpenTelemetry SDK
- `tracing-opentelemetry` 0.22 - Tracing-to-OpenTelemetry bridge (optional)
- `tracing-subscriber` 0.3 - Log subscriber with JSON and env-filter features
- `tracing-appender` 0.2 - File-based log appender with rotation

**Distributed:**

- `tonic` 0.12 / `tonic-build` 0.12 - gRPC server/client and codegen
- `prost` 0.13 / `prost-build` 0.13 - Protobuf codec
- `tokio-stream` 0.1 - Stream utilities for gRPC
- `dashmap` 5.5 - Concurrent hashmap
- `crossbeam` 0.8 - Concurrent data structures

**Model Loading:**

- `safetensors` 0.7.0 - SafeTensor checkpoint loading
- `gguf` 0.1 - GGUF format loading (optional, `gguf` feature)
- `memmap2` 0.9 - Memory-mapped file I/O
- `half` 2.x - FP16 type support
- `rayon` 1.12 - Parallel iteration
- `tiktoken` 3.x / `tokenizers` 0.22 - Tokenization (always enabled)

**Security:**

- `tokio-rustls` 0.26 / `rustls-pemfile` 2 - TLS/mTLS transport
- `base64` 0.22 - Base64 encoding/decoding for JWT
- `chrono` 0.4 - Timestamp handling

**Kubernetes Operator (Go):**

- `k8s.io/api` v0.30.0 / `k8s.io/apimachinery` v0.30.0 / `k8s.io/client-go` v0.30.0 - K8s API and client
- `sigs.k8s.io/controller-runtime` v0.18.0 - Controller framework
- `sigs.k8s.io/controller-tools` v0.14.0 - CRD generation

## Feature Flags

**Workspace-level (`Cargo.toml`):**

| Crate        | Feature                | Description                                                |
| ------------ | ---------------------- | ---------------------------------------------------------- |
| vllm-core    | `prometheus` (default) | Prometheus metrics export (`metrics-exporter-prometheus`)  |
| vllm-core    | `opentelemetry`        | OTLP telemetry export (`dep:opentelemetry`)                |
| vllm-core    | `cuda-graph` (default) | CUDA Graph support (`dep:vllm-model`)                      |
| vllm-model   | `cuda`                 | Candle CUDA backend (`candle-core/cuda`, `candle-nn/cuda`) |
| vllm-model   | `gguf`                 | GGUF checkpoint loading (`dep:gguf`)                       |
| vllm-model   | `full`                 | All features (`cuda` + `gguf`)                             |
| vllm-traits  | `candle`               | Candle tensor types support (`candle-core`)                |
| vllm-traits  | `kernels`              | GPU kernel support                                         |
| vllm-testing | `cuda`                 | Candle CUDA backend                                        |

## Configuration

**Environment Variables:**

Core configuration loaded via env vars (all with `VLLM_` prefix):

- `VLLM_HOST` / `VLLM_PORT` - Server bind address (default: `0.0.0.0:8000`)
- `VLLM_MODEL` - Path to model directory (required)
- `VLLM_TENSOR_PARALLEL_SIZE` - Tensor parallelism degree (default: `1`, max: 64)
- `VLLM_KV_BLOCKS` - Number of KV cache blocks (default: `1024`, max: 65536)
- `VLLM_KV_QUANTIZATION` - Enable KV cache quantization (default: `false`)
- `VLLM_MAX_BATCH_SIZE` - Maximum batch size (default: `256`)
- `VLLM_MAX_WAITING_BATCHES` - Max waiting batches (default: `10`)
- `VLLM_MAX_DRAFT_TOKENS` - Max draft tokens for speculative decoding (default: `8`, range: 0-64)
- `VLLM_ADAPTIVE_SPECULATIVE` - Enable adaptive speculative decoding (default: `false`)
- `VLLM_API_KEY` - API key(s) for auth (repeatable)
- `VLLM_API_KEYS_FILE` - File path with API keys (one per line)
- `VLLM_LOG_LEVEL` - Log verbosity: trace/debug/info/warn/error (default: `info`)
- `VLLM_LOG_DIR` - Directory for JSON log file output
- `VLLM_CONFIG_PATH` - Path to YAML configuration file
- `RUST_LOG` - tracing-subscriber log filter (env-filter)
- `RUST_BACKTRACE` - Backtrace on panic

**Build:**

- `Cargo.toml` (workspace root + individual crates at `crates/*/Cargo.toml`)
- `Cargo.lock` (committed)
- `rust-toolchain` - Managed via CI `dtolnay/rust-toolchain` action
- No `.env` file detected in repository — environment configuration is CLI/env-var driven

**Dev:**

- `.editorconfig` - UTF-8, LF, 4-space indent for `.rs` files
- `.pre-commit-config.yaml` - prek-managed hooks
- `.rumdl.toml` - Markdown lint configuration

## Platform Requirements

**Development:**

- Rust 1.85+ (stable)
- Go 1.22+ (only for k8s operator development)
- CUDA toolkit (optional, for GPU inference and CUDA Graph compilation)
- `build-essential`, `cmake`, `clang`, `llvm` (for C/C++ deps in build)

**Production:**

- Container: `Dockerfile` (multi-stage: `rust:1.82-bookworm` builder → `debian:bookworm-slim` runtime)
- Orchestration: Kubernetes (Helm chart at `k8s/charts/vllm-lite/`, CRD operator at `k8s/operator/`)
- Ports: 8000 (HTTP API), 9090 (Prometheus metrics), 50051 (gRPC for distributed)
- Non-root user `vllm` (UID 1000)
- Health checks: Kubernetes-style liveness (`/health/live`) and readiness (`/health/ready`) probes
- Resource requirements: 4-8 CPU cores, 8-16 GB memory (GPU recommended)

---

*Stack analysis: 2026-05-13*
