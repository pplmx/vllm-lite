# ADR-021: OTLP Metrics + Traces Exporter (P43)

- **Status:** Accepted (2026-07-22)
- **Deciders:** vLLM-lite maintainers
- **Tags:** observability, opentelemetry, metrics, tracing

## Context

Engine telemetry today is Prometheus-scrape only (text format at `/metrics`).
New SRE shops overwhelmingly standardise on OpenTelemetry — Jaeger, Tempo,
Datadog, Honeycomb all ingest OTLP natively. With multi-node KV block transfer
(P40–P42) now shipped, distributed tracing across nodes becomes more valuable:
operators can trace a single request from HTTP ingress → scheduler → speculative
decode → KV fetch on a remote peer.

The v30.0 cleanup removed the old `opentelemetry` Cargo feature — but that was
a **no-op**: no `#[cfg(feature = "...")]` gates existed in the code. P43
re-adds the feature with a real push-based exporter.

## Decision

Add an opt-in `opentelemetry` feature on `vllm-core` (and propagate through
`vllm-server`). When enabled:

1. **`OtlpExporter`** — a background tokio task polls
   `EnhancedMetricsCollector` every `metrics_export_interval_secs` (default 30s)
   and ships via `opentelemetry-otlp` `grpc-tonic`. A 21-entry schema map
   (`SCHEMA_MAP`) translates Prometheus metric names to OTel instrument names.

2. **`tracing-opentelemetry` bridge** — `init_tracing_with_otlp()` wraps the
   existing `tracing_subscriber` registry with a `tracing-opentelemetry::layer`,
   bridging every `tracing::info_span!` to an OTel span. The `OtlpGuard`
   (Drop-flush) shuts down the `SdkTracerProvider` on graceful exit.

3. **Server bootstrap** — `ObservabilityConfig` in the YAML config +
   `--otlp-endpoint` CLI override. `spawn_otlp_exporter()` takes the engine's
   `Arc<EnhancedMetricsCollector>` and returns an `OtlpHandle` (Drop-flush).

4. **Default off** — `opentelemetry` is not in the default feature set.
   `cargo build` / `cargo test` without `--features opentelemetry` is unaffected.

### Config YAML

```yaml
observability:
  otlp:
    enabled: true
    endpoint: "http://localhost:4317"
    metrics_export_interval_secs: 30
    trace_sampling_ratio: 1.0
    service_name: "vllm-lite"
    service_version: "0.1.0"
    protocol: "Grpc"
```

## Consequences

**Positive:**
- Operators can stream telemetry to any OTLP-compatible collector.
- Distributed tracing works across multi-node peers.
- `otlp` module is fully `#[cfg]`-gated — zero impact on single-node builds.
- 5 new unit tests + 5 integration tests + stub collector for CI.

**Negative / trade-offs:**
- 5 additional dependencies (`opentelemetry`, `opentelemetry_sdk`,
  `opentelemetry-otlp`, `opentelemetry-semantic-conventions`,
  `tracing-opentelemetry`, `opentelemetry-proto` in tests).
- Bounded public API delta: 8 new types re-exported from `vllm-core`.
- File-logging support (`log_dir`) is unavailable when OTLP tracing init
  replaces `logging::init_logging` — tracked as a v32 follow-up.

## Alternatives

- **HTTP/protobuf transport:** deferred — gRPC/tonic is the OTel default and
  matches the existing `vllm-dist` gRPC usage.
- **Full replacement of `PrometheusExporter`:** rejected — keep both. Prometheus
  is zero-cost for local scrape; OTLP is opt-in for distributed tracing.
- **OTLP Logs export:** deferred — `tracing-subscriber` JSON format is already
  log-aggregator friendly (Loki, Datadog logs).

## Related

- Spec: `docs/superpowers/specs/2026-07-22-p43-otlp-exporter-design.md`
- Plan: `docs/superpowers/plans/2026-07-22-p43-otlp-exporter.md`
- Production-readiness doc: `ops/technical-due-diligence/production-readiness.md`
  (was listed as one of 5 v32+ code follow-ups; P43 closes it)
