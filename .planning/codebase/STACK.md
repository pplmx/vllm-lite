# Technology Stack

**Last updated:** 2026-05-09
**Focus:** Tech

## Language & Runtime

- **Rust** (edition 2024) — all crates
- Minimum Rust version: 1.85
- Build system: Cargo workspace with 7 crates
- Package manager: Cargo + `just` (task runner)

## Core Framework

| Component | Library | Version |
|-----------|---------|---------|
| Deep learning framework | `candle-core`, `candle-nn` | 0.10.2 |
| Async runtime | `tokio` | 1.x (full features) |
| Web framework | `axum` | 0.7 |
| Serialization | `serde`, `serde_json` | 1.x |
| Tokenization | `tiktoken`, `tokenizers` | 3.x, 0.22 |
| Tensor storage format | `safetensors` | 0.7.0 |
| Model format | `gguf` (optional) | 0.1 |
| gRPC | `tonic`, `prost` | 0.12, 0.13 |
| CLI | `clap` | 4.x |
| Checkpoint weights | `half` (FP16), `memmap2` | 2.x, 0.9 |

## Observability

| Component | Library | Version |
|-----------|---------|---------|
| Structured logging | `tracing` | 0.1 |
| JSON log output | `tracing-subscriber` (json feature) | 0.3 |
| Log file rotation | `tracing-appender` | 0.2 |
| Performance metrics | `metrics` | 0.22 |
| Prometheus export | `metrics-exporter-prometheus` | 0.13 |
| OpenTelemetry | `opentelemetry`, `opentelemetry-otlp`, `opentelemetry_sdk` | 0.21 |
| Tracing integration | `tracing-opentelemetry` | 0.22 |

## Concurrency & Data Structures

| Component | Library | Version |
|-----------|---------|---------|
| Concurrent operations | `crossbeam` | 0.8 |
| Concurrent hashmap | `dashmap` | 5.5 |
| Async networking | `tower`, `tower-http` | 0.4, 0.5 |
| Parallel computation | `rayon` | 1.10 |

## Feature Flags

| Feature | Crate | Description |
|---------|-------|-------------|
| `cuda` | vllm-model | Candle CUDA GPU acceleration |
| `gguf` | vllm-model | GGUF model format loading |
| `full` | vllm-model | All features (cuda + gguf) |
| `cuda-graph` | vllm-core | CUDA Graph optimization (depends on vllm-model) |
| `prometheus` | vllm-core | Prometheus metrics export |
| `opentelemetry` | vllm-core | OpenTelemetry tracing export |

## Workspace Crates (7)

| Crate | Path | Dependencies |
|-------|------|-------------|
| `vllm-traits` | `crates/traits/` | serde, thiserror, candle-core (opt) |
| `vllm-core` | `crates/core/` | vllm-traits, tokio, crossbeam, dashmap |
| `vllm-model` | `crates/model/` | vllm-traits, vllm-dist, candle, safetensors, tiktoken |
| `vllm-server` | `crates/server/` | vllm-core, vllm-model, axum, tokio |
| `vllm-dist` | `crates/dist/` | vllm-traits, candle, tonic, prost |
| `vllm-testing` | `crates/testing/` | vllm-traits, vllm-core, candle |
| `vllm-lite-benchmarks` | `benches/` | vllm-core, vllm-model, vllm-traits |

## Development Tools

| Tool | Usage |
|------|-------|
| `cargo nextest` | Test runner (preferred) |
| `cargo clippy` | Linting |
| `cargo fmt` | Formatting |
| `cargo doc` | Documentation generation |
| `cargo tarpaulin` | Code coverage |
| `cargo criterion` | Benchmarks |
| `proptest` | Property-based testing |
| `just` | Task runner (see `justfile`) |

## Build Configuration

```toml
[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
panic = "abort"
strip = true
```
