//! OpenTelemetry (OTLP) push-based exporter — gated by the `opentelemetry`
//! feature on `vllm-core`. Streams engine metrics + tracing spans to any
//! OTel-compatible collector (Jaeger / Tempo / Datadog / Honeycomb / etc.).

#![cfg(feature = "opentelemetry")]
// Implementation lands in Tasks 2-4. This skeleton exists so the feature flag
// compiles both with and without the OTLP deps enabled.
