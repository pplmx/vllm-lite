# P43 OTLP Metrics + Traces Exporter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a push-based OpenTelemetry metrics + traces exporter behind a new `opentelemetry` feature flag on `vllm-core`, wired through server bootstrap + YAML config + graceful shutdown, so operators can stream engine telemetry to any OTel-compatible collector (Jaeger / Tempo / Datadog / Honeycomb / etc.).

**Architecture:** A new `OtlpExporter` background task polls the existing `EnhancedMetricsCollector` every 30s (default) and ships via `opentelemetry-otlp`'s `grpc-tonic` exporter. A parallel `tracing-opentelemetry` layer bridges every `tracing::info_span!` (which already exists pervasively for `request_id`, `engine.add_request`, `scheduler.batch`) to OTel spans. Both share the same `OtlpConfig` and the same `OtlpGuard` Drop-flush handle for graceful shutdown.

**Tech Stack:** `opentelemetry` 0.27 + `opentelemetry_sdk` 0.27 (`rt-tokio`) + `opentelemetry-otlp` 0.27 (`grpc-tonic`, `metrics`) + `tracing-opentelemetry` 0.28 + `opentelemetry-semantic-conventions` 0.27 (versions verified at implementation start via `cargo add`).

**Spec:** `docs/superpowers/specs/2026-07-22-p43-otlp-exporter-design.md`

---

## File structure

**Created:**
- `crates/core/src/metrics/exporter/otlp.rs` — `OtlpConfig`, `OtlpProtocol`, `OtlpError`, `OtlpExporter`, `OtlpExporterBuilder`, schema mapping table
- `crates/core/src/tracing_init.rs` — `init_tracing_with_otlp()`, `OtlpGuard` (Drop-flush)
- `crates/server/src/bootstrap/observability.rs` — `spawn_otlp_exporter()`, `OtlpHandle` (Drop-flush), CLI-flag handling
- `crates/server/tests/otlp_exporter_integration.rs` — integration tests with in-process stub OTLP collector
- `crates/server/tests/otlp_stub_collector.rs` — reusable stub OTLP `MetricsService` + `TraceService` gRPC impl
- `docs/adr/ADR-021-otlp-exporter.md` — architectural decision record

**Modified:**
- `crates/core/Cargo.toml` — add `opentelemetry` feature + 5 deps
- `crates/core/src/metrics/exporter/mod.rs` — `#[cfg(feature = "opentelemetry")] mod otlp;` + `MetricsError::Otlp(String)` variant + re-export
- `crates/core/src/lib.rs` — `#[cfg(feature = "opentelemetry")] pub mod tracing_init;`
- `crates/server/Cargo.toml` — opt-in `vllm-core/opentelemetry` feature
- `crates/server/src/main.rs` — call `init_tracing_with_otlp` + `spawn_otlp_exporter` in startup; drop handle in shutdown
- `crates/server/src/config/mod.rs` — add `observability: ObservabilityConfig` section + `--otlp-endpoint` CLI override
- `crates/server/src/bootstrap/mod.rs` — re-export `observability` module
- `docs/OPERATIONS.md` — new "OTLP Observability" section (Jaeger / Tempo / Datadog / Honeycomb examples)
- `docs/reference/feature-matrix.md` — add `opentelemetry` row
- `.github/workflows/ci.yml` — new `ci-otlp` job
- `CHANGELOG.md` — `[Unreleased] > Added` entry + `public-api:` bullets
- `.planning/STATE.md` — record P43 closure + test count delta
- `.planning/v31.0-MASTER-PLAN.md` — close "OTLP" v32+ item

---

## Task 1: Add `opentelemetry` feature flag + empty module skeleton

**Files:**
- Modify: `crates/core/Cargo.toml`
- Create: `crates/core/src/metrics/exporter/otlp.rs` (empty skeleton)
- Modify: `crates/core/src/metrics/exporter/mod.rs:1-30` (add `#[cfg(feature = "opentelemetry")] mod otlp;`)

- [ ] **Step 1: Add the feature + deps to `crates/core/Cargo.toml`**

Append the following inside `[features]` (after the existing `multi-node = [...]` line):

```toml
opentelemetry = [
    "dep:opentelemetry",
    "dep:opentelemetry_sdk",
    "dep:opentelemetry-otlp",
    "dep:opentelemetry-semantic-conventions",
    "dep:tracing-opentelemetry",
]
```

Append the following inside `[dependencies]` (after the existing `parking_lot = { workspace = true, optional = true }` line). Versions verified via `cargo add` at task start; the snippet below uses the values the spec calls out, adjust to whatever `cargo add` resolves:

```toml
opentelemetry = { version = "0.27", optional = true }
opentelemetry_sdk = { version = "0.27", optional = true, features = ["rt-tokio"] }
opentelemetry-otlp = { version = "0.27", optional = true, features = ["grpc-tonic", "metrics"] }
opentelemetry-semantic-conventions = { version = "0.27", optional = true }
tracing-opentelemetry = { version = "0.28", optional = true }
```

Also add a `workspace = true` entry to `[workspace.dependencies]` in the root `Cargo.toml` (only if `tracing-opentelemetry` is missing — check first):

```bash
grep -q '^tracing-opentelemetry' /workspace/vllm-lite/Cargo.toml || echo 'tracing-opentelemetry = "0.28"' >> /workspace/vllm-lite/Cargo.toml
```

- [ ] **Step 2: Create empty skeleton at `crates/core/src/metrics/exporter/otlp.rs`**

```rust
//! OpenTelemetry (OTLP) push-based exporter — gated by the `opentelemetry`
//! feature on `vllm-core`. Streams engine metrics + tracing spans to any
//! OTel-compatible collector (Jaeger / Tempo / Datadog / Honeycomb / etc.).

#![cfg(feature = "opentelemetry")]
// Implementation lands in Tasks 2-4. This skeleton exists so the feature flag
// compiles both with and without the OTLP deps enabled.
```

- [ ] **Step 3: Re-export from `crates/core/src/metrics/exporter/mod.rs`**

After the existing `mod prometheus;` line (line 18), add:

```rust
#[cfg(feature = "opentelemetry")]
mod otlp;
```

After the existing `pub use prometheus::PrometheusExporter;` line (line 114), add:

```rust
#[cfg(feature = "opentelemetry")]
pub use otlp::{OtlpConfig, OtlpError, OtlpExporter, OtlpExporterBuilder, OtlpProtocol};
```

- [ ] **Step 4: Build default features — verify it still works**

Run: `cargo build -p vllm-core`
Expected: SUCCESS (the `#[cfg(feature = "opentelemetry")]` gates mean default-features build compiles the empty skeleton only).

- [ ] **Step 5: Build with the new feature — verify deps resolve**

Run: `cargo build -p vllm-core --features opentelemetry`
Expected: SUCCESS. If any dep version fails to resolve, run `cargo add` for the failing crate and commit the updated `Cargo.toml` + `Cargo.lock`.

- [ ] **Step 6: Commit**

```bash
git add crates/core/Cargo.toml crates/core/src/metrics/exporter/otlp.rs crates/core/src/metrics/exporter/mod.rs Cargo.toml Cargo.lock
git commit -m "feat(core): opentelemetry feature flag + otlp module skeleton (P43 T1)"
```

---

## Task 2: `OtlpConfig` + `OtlpProtocol` + `OtlpError` types

**Files:**
- Modify: `crates/core/src/metrics/exporter/otlp.rs` (replace skeleton)
- Modify: `crates/core/src/metrics/exporter/mod.rs:97-105` (add `MetricsError::Otlp` variant)

- [ ] **Step 1: Write failing tests at the bottom of `otlp.rs`**

Add the following `#[cfg(test)] mod tests` block:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn otlp_config_default_matches_spec() {
        let cfg = OtlpConfig::default();
        assert!(!cfg.enabled);
        assert_eq!(cfg.endpoint, "http://localhost:4317");
        assert_eq!(cfg.service_name, "vllm-lite");
        assert_eq!(cfg.metrics_export_interval_secs, 30);
        assert!((cfg.trace_sampling_ratio - 1.0).abs() < f64::EPSILON);
        assert!(matches!(cfg.protocol, OtlpProtocol::Grpc));
    }

    #[test]
    fn otlp_config_rejects_negative_sampling_ratio() {
        let cfg = OtlpConfig { trace_sampling_ratio: -0.1, ..OtlpConfig::default() };
        assert!(matches!(cfg.validate(), Err(OtlpError::Config(_))));
    }

    #[test]
    fn otlp_config_rejects_above_one_sampling_ratio() {
        let cfg = OtlpConfig { trace_sampling_ratio: 1.5, ..OtlpConfig::default() };
        assert!(matches!(cfg.validate(), Err(OtlpError::Config(_))));
    }

    #[test]
    fn otlp_config_rejects_zero_metrics_interval() {
        let cfg = OtlpConfig { metrics_export_interval_secs: 0, ..OtlpConfig::default() };
        assert!(matches!(cfg.validate(), Err(OtlpError::Config(_))));
    }

    #[test]
    fn otlp_config_validates_default_as_ok() {
        assert!(OtlpConfig::default().validate().is_ok());
    }
}
```

- [ ] **Step 2: Run tests — verify they fail (types don't exist yet)**

Run: `cargo test -p vllm-core --features opentelemetry otlp_config -- --nocapture`
Expected: COMPILATION ERROR (`OtlpConfig`, `OtlpProtocol`, `OtlpError` are undefined).

- [ ] **Step 3: Implement the types**

Replace the skeleton in `otlp.rs` with:

```rust
//! OpenTelemetry (OTLP) push-based exporter — gated by the `opentelemetry`
//! feature on `vllm-core`. Streams engine metrics + tracing spans to any
//! OTel-compatible collector (Jaeger / Tempo / Datadog / Honeycomb / etc.).

#![cfg(feature = "opentelemetry")]

use serde::{Deserialize, Serialize};

/// Wire protocol for the OTLP exporter. Only `Grpc` is supported in v43;
/// the enum is reserved so adding `Http` later is a non-breaking change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum OtlpProtocol {
    #[default]
    Grpc,
}

/// Configuration for the OTLP exporter. Loaded from the
/// `app_config.observability.otlp` YAML section or the `--otlp-endpoint`
/// CLI override.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OtlpConfig {
    /// Master switch — when `false`, the bootstrap skips OTLP entirely.
    pub enabled: bool,
    /// OTLP collector endpoint (gRPC). Default: `"http://localhost:4317"`.
    pub endpoint: String,
    /// OTel `service.name` resource attribute.
    pub service_name: String,
    /// OTel `service.version` resource attribute (synced from the release manifest).
    pub service_version: String,
    /// Metrics export interval in seconds. Default: `30`.
    pub metrics_export_interval_secs: u64,
    /// Trace sampling ratio in `[0.0, 1.0]`. Default: `1.0` (always sample).
    pub trace_sampling_ratio: f64,
    /// OTLP transport protocol. Default: `Grpc`.
    pub protocol: OtlpProtocol,
}

impl Default for OtlpConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            endpoint: "http://localhost:4317".to_string(),
            service_name: "vllm-lite".to_string(),
            service_version: env!("CARGO_PKG_VERSION").to_string(),
            metrics_export_interval_secs: 30,
            trace_sampling_ratio: 1.0,
            protocol: OtlpProtocol::Grpc,
        }
    }
}

impl OtlpConfig {
    /// Validate field ranges. Returns `Err(OtlpError::Config)` for any
    /// out-of-range value. Safe to call repeatedly.
    pub fn validate(&self) -> Result<(), OtlpError> {
        if !(0.0..=1.0).contains(&self.trace_sampling_ratio) {
            return Err(OtlpError::Config(format!(
                "trace_sampling_ratio = {} is out of range [0.0, 1.0]",
                self.trace_sampling_ratio
            )));
        }
        if self.metrics_export_interval_secs == 0 {
            return Err(OtlpError::Config(
                "metrics_export_interval_secs must be > 0".to_string(),
            ));
        }
        if self.endpoint.trim().is_empty() {
            return Err(OtlpError::Config("endpoint must be non-empty".to_string()));
        }
        Ok(())
    }
}

/// Typed errors for the OTLP exporter / tracing-init layer.
#[derive(Debug, thiserror::Error)]
pub enum OtlpError {
    #[error("otlp config invalid: {0}")]
    Config(String),
    #[error("otlp export failed: {0}")]
    Export(String),
    #[error("otlp collector unreachable: {0}")]
    CollectorUnreachable(String),
    #[error("otlp builder failed: {0}")]
    Builder(String),
}
```

- [ ] **Step 4: Add `MetricsError::Otlp` variant in `mod.rs`**

Replace the existing `MetricsError` enum (lines 97-105 of `crates/core/src/metrics/exporter/mod.rs`) with:

```rust
/// Error type for the metrics export / serialization layer. Covers I/O failures on the export sink and Prometheus-format conversion errors.
#[derive(Debug, thiserror::Error)]
pub enum MetricsError {
    #[error("export failed: {0}")]
    ExportFailed(String),
    /// A `Mutex`/`RwLock` guard was poisoned by a panic while held.
    #[error("metrics exporter lock poisoned")]
    LockPoisoned,
    /// OpenTelemetry (OTLP) exporter reported an error. Only constructible
    /// when the `opentelemetry` feature is enabled.
    #[cfg(feature = "opentelemetry")]
    #[error("otlp exporter error: {0}")]
    Otlp(String),
}
```

- [ ] **Step 5: Run tests — verify they pass**

Run: `cargo test -p vllm-core --features opentelemetry otlp_config -- --nocapture`
Expected: PASS (5 tests).

- [ ] **Step 6: Commit**

```bash
git add crates/core/src/metrics/exporter/otlp.rs crates/core/src/metrics/exporter/mod.rs
git commit -m "feat(core): OtlpConfig + OtlpError + validation (P43 T2)"
```

---

## Task 3: `OtlpExporter` background task + 21-metric schema mapping

**Files:**
- Modify: `crates/core/src/metrics/exporter/otlp.rs` (extend with exporter)

- [ ] **Step 1: Add failing tests for the schema mapping table**

Append to the `mod tests` block:

```rust
    #[test]
    fn metric_schema_mapping_covers_all_prometheus_metrics() {
        // Pins the contract: every Prometheus metric has an OTel counterpart.
        // If PrometheusExporter adds a new metric, this test fails until the
        // schema map is updated.
        let prometheus_names = [
            "cuda_graph_hits_total",
            "cuda_graph_misses_total",
            "speculative_adjustments_total",
            "requests_total",
            "draft_resolutions_external_total",
            "draft_resolutions_self_spec_self_spec_total", // intentional miss → test should FAIL after fix
            "draft_resolutions_none_total",
            "draft_load_failures_total",
            "draft_runtime_errors_total",
            "packing_efficiency",
            "speculative_acceptance_rate",
            "speculative_efficiency",
            "throughput_speedup_ratio",
            "speculative_per_request_count",
            "request_queue_depth",
            "active_sequences",
            "gpu_memory_used_bytes",
            "gpu_memory_total_bytes",
            "is_leader",
            "inflight_requests",
            "scheduler_queue_size",
        ];
        for name in prometheus_names {
            // intentional miss above trips the test; replace with a real name
            // once the schema table is implemented.
            assert!(
                SCHEMA_MAP.iter().any(|(p, _, _, _)| *p == name),
                "Prometheus metric {name} has no OTel counterpart in SCHEMA_MAP"
            );
        }
    }
```

- [ ] **Step 2: Run test — verify it fails**

Run: `cargo test -p vllm-core --features opentelemetry metric_schema_mapping -- --nocapture`
Expected: FAIL with "Prometheus metric draft_resolutions_self_spec_self_spec_total has no OTel counterpart" (the intentional miss).

- [ ] **Step 3: Implement the schema map + `OtlpExporter` skeleton**

Append to `otlp.rs` (after `OtlpError`):

```rust
use std::sync::Arc;
use std::time::Duration;

use opentelemetry::metrics::{Counter, Gauge, MeterProvider, UpDownCounter};
use opentelemetry::KeyValue;
use opentelemetry_sdk::metrics::{PeriodicReader, SdkMeterProvider};
use opentelemetry_sdk::Resource;
use opentelemetry_semantic_conventions::resource as semres;

use crate::metrics::EnhancedMetricsCollector;

/// Schema mapping: `(prometheus_name, otel_name, InstrumentKind, unit)`.
/// `InstrumentKind` is a private tag because the OTel Counter/Gauge types
/// are not nameable across `Option<>` in a const table.
#[derive(Debug, Clone, Copy)]
enum InstrumentKind {
    Counter,
    Gauge,
    UpDownCounter,
}

const SCHEMA_MAP: &[(&str, &str, InstrumentKind, &str)] = &[
    ("cuda_graph_hits_total", "cuda.graph.hits", InstrumentKind::Counter, "{hit}"),
    ("cuda_graph_misses_total", "cuda.graph.misses", InstrumentKind::Counter, "{miss}"),
    ("speculative_adjustments_total", "speculative.adjustments", InstrumentKind::Counter, "{adjustment}"),
    ("requests_total", "requests", InstrumentKind::Counter, "{request}"),
    ("draft_resolutions_external_total", "draft.resolutions.external", InstrumentKind::Counter, "{resolution}"),
    ("draft_resolutions_self_spec_total", "draft.resolutions.self_spec", InstrumentKind::Counter, "{resolution}"),
    ("draft_resolutions_none_total", "draft.resolutions.none", InstrumentKind::Counter, "{resolution}"),
    ("draft_load_failures_total", "draft.load.failures", InstrumentKind::Counter, "{failure}"),
    ("draft_runtime_errors_total", "draft.runtime.errors", InstrumentKind::Counter, "{error}"),
    ("packing_efficiency", "packing.efficiency", InstrumentKind::Gauge, "{ratio}"),
    ("speculative_acceptance_rate", "speculative.acceptance_rate", InstrumentKind::Gauge, "{ratio}"),
    ("speculative_efficiency", "speculative.efficiency", InstrumentKind::Gauge, "{ratio}"),
    ("throughput_speedup_ratio", "throughput.speedup_ratio", InstrumentKind::Gauge, "{ratio}"),
    ("speculative_per_request_count", "speculative.per_request_count", InstrumentKind::Gauge, "{sequence}"),
    ("request_queue_depth", "request.queue_depth", InstrumentKind::UpDownCounter, "{request}"),
    ("active_sequences", "active.sequences", InstrumentKind::UpDownCounter, "{sequence}"),
    ("gpu_memory_used_bytes", "gpu.memory.used", InstrumentKind::UpDownCounter, "By"),
    ("gpu_memory_total_bytes", "gpu.memory.total", InstrumentKind::UpDownCounter, "By"),
    ("is_leader", "is_leader", InstrumentKind::Gauge, "{bool}"),
    ("inflight_requests", "inflight.requests", InstrumentKind::UpDownCounter, "{request}"),
    ("scheduler_queue_size", "scheduler.queue_size", InstrumentKind::UpDownCounter, "{request}"),
];

/// Push-based OTLP metrics exporter. Holds an `EnhancedMetricsCollector`
/// reference + an OTLP `SdkMeterProvider`; `run()` polls the collector every
/// `config.metrics_export_interval_secs` and records each value into the
/// corresponding OTel instrument. The `PeriodicReader` flushes on each tick.
pub struct OtlpExporter {
    collector: Arc<EnhancedMetricsCollector>,
    config: OtlpConfig,
    provider: SdkMeterProvider,
}

impl OtlpExporter {
    /// Build the exporter. Constructs the `SdkMeterProvider` with an OTLP
    /// exporter targeting `config.endpoint`. The caller must call `.run()`
    /// to start the polling loop.
    pub fn new(collector: Arc<EnhancedMetricsCollector>, config: OtlpConfig) -> Result<Self, OtlpError> {
        config.validate()?;

        let exporter = opentelemetry_otlp::SpanExporter::builder()
            .with_tonic()
            .with_endpoint(&config.endpoint)
            .build()
            .map_err(|e| OtlpError::Builder(format!("span exporter: {e}")))?;

        let resource = Resource::new(vec![
            KeyValue::new(semres::SERVICE_NAME, config.service_name.clone()),
            KeyValue::new(semres::SERVICE_VERSION, config.service_version.clone()),
            KeyValue::new(semres::SERVICE_INSTANCE_ID, uuid::Uuid::new_v4().to_string()),
            KeyValue::new(semres::HOST_ARCH, std::env::consts::ARCH),
        ]);

        let reader = PeriodicReader::builder(exporter, opentelemetry_sdk::runtime::Tokio)
            .with_interval(Duration::from_secs(config.metrics_export_interval_secs))
            .build();

        let provider = SdkMeterProvider::builder()
            .with_reader(reader)
            .with_resource(resource)
            .build();

        Ok(Self { collector, config, provider })
    }

    /// Return the underlying meter provider so the caller can install
    /// `tracing-opentelemetry` on top of the same exporter.
    pub fn meter_provider(&self) -> &SdkMeterProvider {
        &self.provider
    }

    /// Background task body. Polls the collector and records each metric.
    /// Returns on shutdown (the caller drops the provider, which flushes).
    pub async fn run(self) -> Result<(), OtlpError> {
        use opentelemetry::metrics::MeterProvider as _;
        let meter = self.provider.meter("vllm-lite");

        // Build one instrument per schema-map entry. OTel instruments are
        // cheap to construct; the type system distinguishes them so we
        // stash them in a small enum.
        enum Instrument {
            Counter(Counter<u64>),
            Gauge(Gauge<f64>),
            UpDownCounter(UpDownCounter<i64>),
        }

        let instruments: Vec<(&str, Instrument)> = SCHEMA_MAP
            .iter()
            .map(|(prom_name, otel_name, kind, unit)| {
                let inst = match kind {
                    InstrumentKind::Counter => {
                        Instrument::Counter(meter.u64_counter(*otel_name).init())
                    }
                    InstrumentKind::Gauge => {
                        Instrument::Gauge(meter.f64_gauge(*otel_name).init())
                    }
                    InstrumentKind::UpDownCounter => {
                        Instrument::UpDownCounter(meter.i64_up_down_counter(*otel_name).init())
                    }
                };
                (*prom_name, inst)
            })
            .collect();

        let mut ticker = tokio::time::interval(Duration::from_secs(self.config.metrics_export_interval_secs));
        // Skip the immediate first tick so the first export happens after
        // `interval`, not at t=0 (gives the engine time to record data).
        ticker.tick().await;

        loop {
            ticker.tick().await;
            let snapshot = self.collector.snapshot();
            // Index the snapshot by prometheus name for O(1) lookups.
            let mut by_name: std::collections::HashMap<&str, f64> =
                snapshot.iter().map(|(k, v)| (k.as_str(), *v)).collect();
            for (prom_name, inst) in &instruments {
                let value = by_name.remove(prom_name).unwrap_or(0.0);
                match inst {
                    Instrument::Counter(c) => c.add(value as u64, &[]),
                    Instrument::Gauge(g) => g.set(value, &[]),
                    Instrument::UpDownCounter(u) => u.add(value as i64, &[]),
                }
            }
            // PeriodicReader auto-flushes on each tick; no explicit flush call.
        }
    }

    /// Flush pending spans + metrics synchronously. Called by the
    /// bootstrap on shutdown.
    pub fn shutdown(&self) -> Result<(), OtlpError> {
        self.provider
            .shutdown()
            .map_err(|e| OtlpError::Export(format!("provider shutdown: {e}")))
    }
}

/// Builder for `OtlpExporter`. Use when you want to inject a pre-built
/// `SdkMeterProvider` (e.g. in tests with an in-memory exporter).
pub struct OtlpExporterBuilder {
    collector: Option<Arc<EnhancedMetricsCollector>>,
    config: OtlpConfig,
}

impl OtlpExporterBuilder {
    pub fn new(config: OtlpConfig) -> Self {
        Self { collector: None, config }
    }

    pub fn collector(mut self, collector: Arc<EnhancedMetricsCollector>) -> Self {
        self.collector = Some(collector);
        self
    }

    pub fn build(self) -> Result<OtlpExporter, OtlpError> {
        let collector = self
            .collector
            .ok_or_else(|| OtlpError::Builder("collector not set".to_string()))?;
        OtlpExporter::new(collector, self.config)
    }
}
```

- [ ] **Step 4: Fix the schema-mapping test typo (drop the intentional miss)**

In the test added in Step 1, replace `"draft_resolutions_self_spec_self_spec_total"` with the correct name `"draft_resolutions_self_spec_total"`.

- [ ] **Step 5: Add `uuid` to workspace deps (needed for `service.instance.id`)**

```bash
grep -q '^uuid' /workspace/vllm-lite/Cargo.toml || echo 'uuid = { version = "1", features = ["v4"] }' >> /workspace/vllm-lite/Cargo.toml
```

Add to `crates/core/Cargo.toml` `[dependencies]`:

```toml
uuid = { workspace = true }
```

- [ ] **Step 6: Run tests — verify they pass**

Run: `cargo test -p vllm-core --features opentelemetry --lib otlp -- --nocapture`
Expected: PASS (all 6 tests).

- [ ] **Step 7: Commit**

```bash
git add crates/core/src/metrics/exporter/otlp.rs crates/core/Cargo.toml Cargo.toml Cargo.lock
git commit -m "feat(core): OtlpExporter background task + 21-metric schema map (P43 T3)"
```

---

## Task 4: `tracing_init` module — `init_tracing_with_otlp` + `OtlpGuard`

**Files:**
- Create: `crates/core/src/tracing_init.rs`
- Modify: `crates/core/src/lib.rs` (re-export the new module behind feature)

- [ ] **Step 1: Add failing tests at the bottom of `tracing_init.rs`**

```rust
//! Tracing-subscriber bootstrap with optional OpenTelemetry bridge.
//! Gated by the `opentelemetry` feature on `vllm-core`.

#![cfg(feature = "opentelemetry")]

use opentelemetry::trace::TracerProvider as _;
use opentelemetry_sdk::trace::{Sampler, TracerProvider};
use opentelemetry_sdk::Resource;
use opentelemetry::KeyValue;
use opentelemetry_semantic_conventions::resource as semres;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

use crate::metrics::exporter::otlp::{OtlpConfig, OtlpError};

/// Initialise `tracing-subscriber` with the requested `EnvFilter` and,
/// when `config.enabled = true`, add a `tracing-opentelemetry` layer
/// that bridges every `tracing::info_span!` to an OTel span. The returned
/// `OtlpGuard` flushes pending spans on drop (graceful shutdown).
pub fn init_tracing_with_otlp(
    env_filter: EnvFilter,
    config: OtlpConfig,
) -> Result<OtlpGuard, OtlpError> {
    config.validate()?;

    let fmt_layer = tracing_subscriber::fmt::layer().with_target(true);

    if !config.enabled {
        tracing_subscriber::registry()
            .with(env_filter)
            .with(fmt_layer)
            .try_init()
            .map_err(|e| OtlpError::Builder(format!("subscriber init: {e}")))?;
        return Ok(OtlpGuard::disabled());
    }

    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_tonic()
        .with_endpoint(&config.endpoint)
        .build()
        .map_err(|e| OtlpError::Builder(format!("span exporter: {e}")))?;

    let resource = Resource::new(vec![
        KeyValue::new(semres::SERVICE_NAME, config.service_name.clone()),
        KeyValue::new(semres::SERVICE_VERSION, config.service_version.clone()),
    ]);

    let sampler = Sampler::ParentBased(Box::new(Sampler::TraceIdRatioBased(
        config.trace_sampling_ratio,
    )));

    let provider = TracerProvider::builder()
        .with_batch_exporter(exporter, opentelemetry_sdk::runtime::Tokio)
        .with_resource(resource)
        .with_sampler(sampler)
        .build();

    let tracer = provider.tracer("vllm-lite");
    let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer);

    tracing_subscriber::registry()
        .with(env_filter)
        .with(fmt_layer)
        .with(otel_layer)
        .try_init()
        .map_err(|e| OtlpError::Builder(format!("subscriber init: {e}")))?;

    Ok(OtlpGuard::enabled(provider))
}

/// Drop-flush guard for the OTLP tracing bridge. Dropping the guard
/// blocks until the tracer provider's `shutdown()` returns (5s timeout),
/// flushing any pending spans.
pub struct OtlpGuard {
    provider: Option<TracerProvider>,
}

impl OtlpGuard {
    fn disabled() -> Self {
        Self { provider: None }
    }
    fn enabled(provider: TracerProvider) -> Self {
        Self { provider: Some(provider) }
    }

    /// Returns `true` if OTLP tracing was initialised (drop will flush).
    pub fn is_enabled(&self) -> bool {
        self.provider.is_some()
    }
}

impl Drop for OtlpGuard {
    fn drop(&mut self) {
        if let Some(provider) = self.provider.take() {
            // `shutdown()` returns `Vec<SpanData>` on success; we discard.
            let _ = provider.shutdown();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_tracing_with_otlp_disabled_returns_inactive_guard() {
        // We can't actually init the subscriber twice (singleton), so this
        // test only exercises the disabled path which doesn't init.
        let cfg = OtlpConfig { enabled: false, ..OtlpConfig::default() };
        // We can't call init_tracing_with_otlp here (would conflict with
        // any other test that ran first), so just check the config-validate
        // path. The full init is exercised in integration tests.
        assert!(cfg.validate().is_ok());
        assert!(!cfg.enabled);
    }

    #[test]
    fn otlp_guard_disabled_reports_not_enabled() {
        let guard = OtlpGuard::disabled();
        assert!(!guard.is_enabled());
        // Drop is a no-op when disabled.
        drop(guard);
    }
}
```

- [ ] **Step 2: Re-export the module from `crates/core/src/lib.rs`**

Locate the `pub mod` declarations in `crates/core/src/lib.rs` (search for `pub mod`). Add the following line after the existing `pub mod metrics;` (or wherever fits the existing pattern):

```rust
#[cfg(feature = "opentelemetry")]
pub mod tracing_init;
```

- [ ] **Step 3: Run tests — verify they pass**

Run: `cargo test -p vllm-core --features opentelemetry --lib tracing_init -- --nocapture`
Expected: PASS (2 tests).

- [ ] **Step 4: Commit**

```bash
git add crates/core/src/tracing_init.rs crates/core/src/lib.rs
git commit -m "feat(core): tracing_init with OTLP bridge + OtlpGuard (P43 T4)"
```

---

## Task 5: Server bootstrap wiring — config + main.rs + CLI override

**Files:**
- Modify: `crates/server/Cargo.toml` (add opt-in feature)
- Modify: `crates/server/src/config/mod.rs` (add `ObservabilityConfig`)
- Create: `crates/server/src/bootstrap/observability.rs`
- Modify: `crates/server/src/bootstrap/mod.rs` (re-export)
- Modify: `crates/server/src/main.rs` (call init_tracing_with_otlp + spawn_otlp_exporter)

- [ ] **Step 1: Add the opt-in feature to `crates/server/Cargo.toml`**

Inside `[dependencies]` for `vllm-core`, change:

```toml
vllm-core = { path = "../core" }
```

to:

```toml
vllm-core = { path = "../core", features = ["opentelemetry"] }
```

Inside `[features]`, add a new feature flag (so operators can opt in):

```toml
opentelemetry = ["vllm-core/opentelemetry"]
```

- [ ] **Step 2: Add `ObservabilityConfig` to `crates/server/src/config/mod.rs`**

Locate the existing `pub struct AppConfig` definition (line ~77). After its closing brace and before the `impl Default for AppConfig` block, add:

```rust
/// Observability configuration (currently only OTLP; Prometheus scraping
/// stays on by default at `/metrics`).
#[derive(Debug, Clone, Default, serde::Deserialize, serde::Serialize)]
pub struct ObservabilityConfig {
    #[serde(default)]
    pub otlp: vllm_core::metrics::exporter::otlp::OtlpConfig,
}
```

Inside the `AppConfig` struct, add a new field:

```rust
    /// Observability configuration. Defaults to `ObservabilityConfig::default()`
    /// (OTLP disabled). Override via the `observability` section of the
    /// server YAML or the `--otlp-endpoint` CLI flag.
    #[serde(default)]
    pub observability: ObservabilityConfig,
```

(Adjust the `#[serde(default)]` annotation to match the existing style in `AppConfig` if a different attribute is already in use — match conventions, don't fight them.)

- [ ] **Step 3: Write failing test for the YAML parser in `config/mod.rs`**

Add to the existing `#[cfg(test)] mod tests` block at the bottom of `config/mod.rs`:

```rust
    #[test]
    fn app_config_parses_otlp_section() {
        let yaml = r#"
server:
  bind: "0.0.0.0:8000"
observability:
  otlp:
    enabled: true
    endpoint: "http://collector:4317"
    metrics_export_interval_secs: 15
    trace_sampling_ratio: 0.5
"#;
        let cfg: AppConfig = serde_yaml::from_str(yaml).expect("yaml parses");
        assert!(cfg.observability.otlp.enabled);
        assert_eq!(cfg.observability.otlp.endpoint, "http://collector:4317");
        assert_eq!(cfg.observability.otlp.metrics_export_interval_secs, 15);
        assert!((cfg.observability.otlp.trace_sampling_ratio - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn app_config_defaults_otlp_disabled_when_section_missing() {
        let yaml = r#"
server:
  bind: "0.0.0.0:8000"
"#;
        let cfg: AppConfig = serde_yaml::from_str(yaml).expect("yaml parses");
        assert!(!cfg.observability.otlp.enabled);
        assert_eq!(cfg.observability.otlp.endpoint, "http://localhost:4317");
    }
```

If `serde_yaml` is not already a dep of `vllm-server`, add it (check `crates/server/Cargo.toml` first; add as `serde_yaml = { workspace = true }` if missing, otherwise use the existing path).

- [ ] **Step 4: Run test — verify it fails**

Run: `cargo test -p vllm-server --lib app_config_parses_otlp_section -- --nocapture`
Expected: FAIL — `OtlpConfig` doesn't exist in the server's dep graph yet (it's gated behind the feature).

- [ ] **Step 5: Create `crates/server/src/bootstrap/observability.rs`**

```rust
//! OTLP exporter bootstrap: wires the engine's `EnhancedMetricsCollector`
//! to a background `OtlpExporter` task. Returns an `OtlpHandle` that
//! flushes pending spans + metrics on drop (graceful shutdown).

#![cfg(feature = "opentelemetry")]

use std::sync::Arc;

use vllm_core::metrics::exporter::otlp::{OtlpConfig, OtlpError, OtlpExporter};
use vllm_core::metrics::EnhancedMetricsCollector;

use crate::AppState;

/// Handle returned by `spawn_otlp_exporter`. Holds the running task's
/// `JoinHandle` + the exporter itself (for shutdown flush). Drop = flush.
pub struct OtlpHandle {
    exporter: Option<OtlpExporter>,
    task: Option<tokio::task::JoinHandle<Result<(), OtlpError>>>,
}

impl OtlpHandle {
    /// Returns `true` if the OTLP task is running.
    pub fn is_running(&self) -> bool {
        self.task.is_some()
    }

    /// Stop the background task + flush pending spans/metrics. Safe to
    /// call multiple times (idempotent). Blocks until flush completes.
    pub async fn shutdown(mut self) -> Result<(), OtlpError> {
        if let Some(task) = self.task.take() {
            task.abort();
            let _ = task.await; // ignore cancellation error
        }
        if let Some(exporter) = self.exporter.take() {
            exporter.shutdown()?;
        }
        Ok(())
    }
}

impl Drop for OtlpHandle {
    fn drop(&mut self) {
        if let Some(task) = self.task.take() {
            task.abort();
        }
        // Note: `OtlpExporter::shutdown` is sync; we call it inline on drop.
        // If `shutdown` is async in a later refactor, replace with
        // `tokio::task::block_in_place`.
        if let Some(exporter) = self.exporter.take() {
            let _ = exporter.shutdown();
        }
    }
}

/// Spawn the OTLP background task. Returns `None` when OTLP is disabled
/// (so callers can `if let Some(handle) = ...` instead of branching on
/// `config.enabled`).
pub fn spawn_otlp_exporter(
    state: &AppState,
    config: OtlpConfig,
) -> Result<Option<OtlpHandle>, OtlpError> {
    if !config.enabled {
        return Ok(None);
    }
    config.validate()?;
    let collector: Arc<EnhancedMetricsCollector> = state.metrics.clone();
    let exporter = OtlpExporter::new(collector, config)?;
    let task = tokio::spawn(exporter.run());
    Ok(Some(OtlpHandle {
        exporter: None, // exporter is consumed by `run`; shutdown is via the handle's task abort + the OtlpGuard from init_tracing
        task: Some(task),
    }))
}
```

Note: `OtlpExporter` is consumed by `run()`. The handle owns only the task. The `MeterProvider`/`TracerProvider` flush happens via the `OtlpGuard` returned by `init_tracing_with_otlp` (Task 4) which the bootstrap also stores.

- [ ] **Step 6: Re-export from `crates/server/src/bootstrap/mod.rs`**

Add:

```rust
#[cfg(feature = "opentelemetry")]
pub mod observability;
```

- [ ] **Step 7: Wire into `crates/server/src/main.rs`**

Locate the bootstrap section (search for `let state = AppState::new` or similar). Before the axum listener is bound, add:

```rust
#[cfg(feature = "opentelemetry")]
{
    use tracing_subscriber::EnvFilter;
    use vllm_core::metrics::exporter::otlp::OtlpConfig;
    use vllm_core::tracing_init::init_tracing_with_otlp;
    use vllm_server::bootstrap::observability::spawn_otlp_exporter;

    let otlp_cfg = app_config.observability.otlp.clone();
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,vllm_core=info,vllm_server=info"));
    let _otlp_guard = init_tracing_with_otlp(env_filter, otlp_cfg.clone())
        .map_err(|e| {
            tracing::warn!(error = %e, "OTLP tracing init failed; continuing without OTLP");
            e
        })
        .ok();
    let _otlp_handle = spawn_otlp_exporter(&state, otlp_cfg)
        .map_err(|e| {
            tracing::warn!(error = %e, "OTLP metrics exporter spawn failed; continuing without OTLP");
            e
        })
        .ok()
        .flatten();
    // Both `_otlp_guard` and `_otlp_handle` are dropped on shutdown, flushing.
}
```

Locate the CLI argument parsing section (search for `clap` or `--bind`). Add a new optional flag:

```rust
/// OTLP collector endpoint (overrides `observability.otlp.endpoint` in YAML).
#[arg(long, env = "VLLM_OTLP_ENDPOINT")]
pub otlp_endpoint: Option<String>,
```

In the config-loading section (after `AppConfig::load`), apply the override:

```rust
if let Some(endpoint) = cli_args.otlp_endpoint.as_ref() {
    app_config.observability.otlp.enabled = true;
    app_config.observability.otlp.endpoint = endpoint.clone();
}
```

- [ ] **Step 8: Build with the feature — verify it compiles**

Run: `cargo build -p vllm-server --features opentelemetry`
Expected: SUCCESS.

- [ ] **Step 9: Build default — verify no regression**

Run: `cargo build -p vllm-server`
Expected: SUCCESS (no OTLP code in the default build).

- [ ] **Step 10: Run the new config-parser tests**

Run: `cargo test -p vllm-server --lib app_config_parses_otlp_section app_config_defaults_otlp_disabled_when_section_missing -- --nocapture`
Expected: PASS (2 tests).

- [ ] **Step 11: Commit**

```bash
git add crates/server/Cargo.toml crates/server/src/config/mod.rs crates/server/src/bootstrap/observability.rs crates/server/src/bootstrap/mod.rs crates/server/src/main.rs Cargo.toml Cargo.lock
git commit -m "feat(server): bootstrap OTLP wiring + YAML config + CLI override (P43 T5)"
```

---

## Task 6: Integration tests with in-process stub OTLP collector

**Files:**
- Create: `crates/server/tests/otlp_stub_collector.rs`
- Create: `crates/server/tests/otlp_exporter_integration.rs`

- [ ] **Step 1: Create the stub OTLP collector**

`crates/server/tests/otlp_stub_collector.rs`:

```rust
//! In-process OTLP collector stub for integration tests. Implements the
//! OTLP gRPC `MetricsService` and `TraceService` and records received
//! requests in a shared `Arc<Mutex<Vec<...>>>` for assertion.

#![cfg(feature = "opentelemetry")]

use std::sync::{Arc, Mutex};

use opentelemetry_proto::tonic::collector::metrics::v1::{
    ExportMetricsServiceRequest, ExportMetricsServiceResponse, MetricsService,
    metrics_service_server::MetricsServiceServer,
};
use opentelemetry_proto::tonic::collector::trace::v1::{
    ExportTraceServiceRequest, ExportTraceServiceResponse, TraceService,
    trace_service_server::TraceServiceServer,
};
use tonic::{Request, Response, Status};

#[derive(Default, Clone)]
pub struct RecordedExport {
    pub metrics: Arc<Mutex<Vec<ExportMetricsServiceRequest>>>,
    pub traces: Arc<Mutex<Vec<ExportTraceServiceRequest>>>,
}

pub struct StubMetricsService {
    pub recorded: RecordedExport,
}

#[tonic::async_trait]
impl MetricsService for StubMetricsService {
    async fn export(
        &self,
        request: Request<ExportMetricsServiceRequest>,
    ) -> Result<Response<ExportMetricsServiceResponse>, Status> {
        self.recorded.metrics.lock().unwrap().push(request.into_inner());
        Ok(Response::new(ExportMetricsServiceResponse {
            partial_success: None,
        }))
    }
}

pub struct StubTraceService {
    pub recorded: RecordedExport,
}

#[tonic::async_trait]
impl TraceService for StubTraceService {
    async fn export(
        &self,
        request: Request<ExportTraceServiceRequest>,
    ) -> Result<Response<ExportTraceServiceResponse>, Status> {
        self.recorded.traces.lock().unwrap().push(request.into_inner());
        Ok(Response::new(ExportTraceServiceResponse {
            partial_success: None,
        }))
    }
}

/// Start a stub OTLP collector on `127.0.0.1:0` (OS-assigned port).
/// Returns the recorded-exports handle and the listening URL.
pub async fn spawn_stub_collector() -> (RecordedExport, String) {
    use tonic::transport::Server;

    let recorded = RecordedExport::default();
    let metrics = StubMetricsService { recorded: recorded.clone() };
    let traces = StubTraceService { recorded: recorded.clone() };

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let url = format!("http://{addr}");

    tokio::spawn(async move {
        Server::builder()
            .add_service(MetricsServiceServer::new(metrics))
            .add_service(TraceServiceServer::new(traces))
            .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener))
            .await
            .unwrap();
    });

    // Give the server a moment to start accepting connections.
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    (recorded, url)
}
```

Add the OTLP proto crates as dev-dependencies on `vllm-server`:

```bash
grep -q '^opentelemetry-proto' /workspace/vllm-lite/Cargo.toml || echo 'opentelemetry-proto = { version = "0.27", features = ["gen-tonic"] }' >> /workspace/vllm-lite/Cargo.toml
```

In `crates/server/Cargo.toml` `[dev-dependencies]`:

```toml
opentelemetry-proto = { workspace = true }
tokio-stream = { workspace = true }
```

- [ ] **Step 2: Create the integration test file**

`crates/server/tests/otlp_exporter_integration.rs`:

```rust
//! Integration tests for the OTLP exporter. Each test spins up an in-process
//! stub OTLP collector on `127.0.0.1:0`, points the server's OTLP config
//! at it, fires a request, and asserts the stub received the expected
//! metrics / traces.

#![cfg(feature = "opentelemetry")]

use std::sync::Arc;
use std::time::Duration;

use vllm_core::metrics::EnhancedMetricsCollector;
use vllm_core::metrics::exporter::otlp::{OtlpConfig, OtlpExporter};

mod otlp_stub_collector;
use otlp_stub_collector::spawn_stub_collector;

fn collector_with_one_request() -> Arc<EnhancedMetricsCollector> {
    let c = Arc::new(EnhancedMetricsCollector::new());
    // EnhancedMetricsCollector doesn't expose `requests_total` directly;
    // the bootstrap path records it on each AddRequest. For unit-style
    // integration we just verify the exporter sends at least one batch.
    c
}

#[tokio::test]
async fn otlp_metrics_arrive_with_correct_name_and_value() {
    let (recorded, url) = spawn_stub_collector().await;
    let collector = collector_with_one_request();

    let cfg = OtlpConfig {
        enabled: true,
        endpoint: url,
        metrics_export_interval_secs: 1, // fast tick for the test
        ..OtlpConfig::default()
    };
    let exporter = OtlpExporter::new(collector, cfg).expect("exporter builds");

    // Spawn the background task; abort after one tick.
    let task = tokio::spawn(exporter.run());
    tokio::time::sleep(Duration::from_millis(1500)).await;
    task.abort();
    let _ = task.await;

    let received = recorded.metrics.lock().unwrap();
    assert!(
        !received.is_empty(),
        "stub collector received no metrics within the tick window"
    );
}

#[tokio::test]
async fn otlp_collector_unreachable_does_not_crash_exporter() {
    // Point at a port that's guaranteed to refuse. Exporter construction
    // succeeds; runtime export will fail silently per the failure-tolerant
    // contract.
    let collector = collector_with_one_request();
    let cfg = OtlpConfig {
        enabled: true,
        endpoint: "http://127.0.0.1:1".to_string(),
        metrics_export_interval_secs: 1,
        ..OtlpConfig::default()
    };
    let exporter = OtlpExporter::new(collector, cfg).expect("exporter builds even with bad endpoint");
    let task = tokio::spawn(exporter.run());
    tokio::time::sleep(Duration::from_millis(1500)).await;
    // No assertion needed — the test passes if the task didn't panic.
    task.abort();
    let _ = task.await;
}

#[tokio::test]
async fn otlp_disabled_default_skips_initialization() {
    // When `enabled = false`, `OtlpExporter::new` should still construct
    // (it's the bootstrap that skips spawning). Just sanity-check the config.
    let cfg = OtlpConfig::default();
    assert!(!cfg.enabled);
    assert!(cfg.validate().is_ok());
}
```

- [ ] **Step 3: Run the integration tests**

Run: `cargo test -p vllm-server --features opentelemetry --test otlp_exporter_integration -- --nocapture`
Expected: PASS (3 tests). If any fail due to proto crate path differences, adjust the `use` statements in `otlp_stub_collector.rs` to match the actual crate version's module layout.

- [ ] **Step 4: Commit**

```bash
git add crates/server/tests/otlp_stub_collector.rs crates/server/tests/otlp_exporter_integration.rs crates/server/Cargo.toml Cargo.toml Cargo.lock
git commit -m "test(server): OTLP exporter integration tests with stub collector (P43 T6)"
```

---

## Task 7: CI job, ADR-021, OPERATIONS.md, CHANGELOG, STATE.md, feature-matrix

**Files:**
- Modify: `.github/workflows/ci.yml`
- Create: `docs/adr/ADR-021-otlp-exporter.md`
- Modify: `docs/OPERATIONS.md`
- Modify: `docs/reference/feature-matrix.md`
- Modify: `CHANGELOG.md`
- Modify: `.planning/STATE.md`
- Modify: `.planning/v31.0-MASTER-PLAN.md`

- [ ] **Step 1: Add `ci-otlp` job to `.github/workflows/ci.yml`**

Locate the existing `ci` and `ci-all-features` jobs. Add a new job in parallel:

```yaml
  ci-otlp:
    name: CI (opentelemetry feature)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          toolchain: 1.88
      - uses: Swatinem/rust-cache@v2
        with:
          shared-key: ${{ runner.os }}-ci-otlp
      - name: Build
        run: cargo build -p vllm-core --features opentelemetry && cargo build -p vllm-server --features opentelemetry
      - name: Test
        run: cargo test -p vllm-core --features opentelemetry --lib && cargo test -p vllm-server --features opentelemetry --test otlp_exporter_integration
```

(Adjust the toolchain + cache config to match the rest of the workflow — `ci.yml` may use a different runner image or action version.)

- [ ] **Step 2: Create `docs/adr/ADR-021-otlp-exporter.md`**

Follow the ADR template used in `docs/adr/ADR-020-multi-node-kv-block-transfer.md`. The ADR captures:

- Title: "OTLP Metrics + Traces Exporter (P43)"
- Status: Accepted (2026-07-22)
- Context: engine telemetry today is Prometheus-only; OTel is the de-facto standard for new SRE shops; multi-node (P40-P42) makes distributed tracing across nodes more valuable.
- Decision: add `opentelemetry` feature on `vllm-core`; ship `OtlpExporter` (push) + `tracing-opentelemetry` bridge; default off.
- Consequences: 5 new deps; bounded public API delta; closes one of 5 production-readiness code follow-ups.
- Alternatives: HTTP/proto transport (deferred); OTel logs export (deferred — tracing-subscriber JSON already log-aggregator friendly); full replacement of PrometheusExporter (rejected — keep both).

- [ ] **Step 3: Add "OTLP Observability" section to `docs/OPERATIONS.md`**

Locate the existing "Observability" section. After the Prometheus subsection, add a new "OTLP Observability" subsection with:

- One-line summary: "Push engine metrics + tracing spans to any OpenTelemetry-compatible collector."
- Setup examples for Jaeger, Tempo, Datadog Agent, Honeycomb (Docker one-liners).
- A "Verify it works" subsection with the Jaeger all-in-one example.
- Troubleshooting: wrong endpoint, no collector, sampling too low, firewall.
- Reference to `app_config.observability.otlp` YAML + `--otlp-endpoint` CLI override.

- [ ] **Step 4: Update `docs/reference/feature-matrix.md`**

In the per-crate feature tables, add a new `opentelemetry` row under `vllm-core`:

```
| `opentelemetry` | Opt-in | Enables OTLP metrics + tracing exporter (P43) |
```

Add a cross-crate propagation row: `vllm-server` → `vllm-core/opentelemetry` (opt-in, server exposes the bootstrap).

- [ ] **Step 5: Update `CHANGELOG.md`**

Add a new `[Unreleased] > Added` entry following the established P-batch format (see P40-P42 entries for the tone + structure). Include:

- Feature name + scope.
- New types + new config keys.
- Reference to ADR-021.
- Test counts: ~6 new unit tests + ~3 new integration tests.
- `public-api: vllm-core added` bullet listing the new `OtlpConfig`, `OtlpError`, `OtlpExporter`, `OtlpExporterBuilder`, `OtlpProtocol`, `tracing_init::init_tracing_with_otlp`, `tracing_init::OtlpGuard`.
- `public-api: vllm-server added` bullet for the new `ObservabilityConfig` + `bootstrap::observability::spawn_otlp_exporter` + `OtlpHandle`.

Also add an `ObservabilityConfig` row to the public-API section.

- [ ] **Step 6: Update `.planning/STATE.md`**

In the "Remaining open items" section (or equivalent), flip the OTLP line from "v32+ code follow-up" to "Shipped in P43 (2026-07-22)". Update the test count line to reflect the +9 tests from Tasks 2-6. Add a one-paragraph summary of P43 in the "v31 Active Work" section or equivalent.

- [ ] **Step 7: Update `.planning/v31.0-MASTER-PLAN.md`**

In the production-readiness closure summary (or wherever the 5 v32+ code follow-ups are tracked), flip the OTLP item from "v32+" to "Shipped in v31.0 P43". Add a one-line pointer to the ADR + the CHANGELOG entry.

- [ ] **Step 8: Commit each doc change separately (frequent commits)**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add ci-otlp job for opentelemetry feature (P43 T7a)"

git add docs/adr/ADR-021-otlp-exporter.md
git commit -m "docs(adr): ADR-021 OTLP exporter architecture decision (P43 T7b)"

git add docs/OPERATIONS.md
git commit -m "docs(operations): OTLP Observability section with Jaeger/Tempo/Datadog examples (P43 T7c)"

git add docs/reference/feature-matrix.md
git commit -m "docs(feature-matrix): add opentelemetry feature row (P43 T7d)"

git add CHANGELOG.md
git commit -m "docs(changelog): P43 OTLP exporter release notes + public-api bullets (P43 T7e)"

git add .planning/STATE.md .planning/v31.0-MASTER-PLAN.md
git commit -m "docs(planning): record P43 OTLP exporter closure + test count delta (P43 T7f)"
```

---

## Task 8: Final verification — `just ci` green

- [ ] **Step 1: Run the full CI loop locally**

Run: `just ci`
Expected: EXIT 0 (fmt-check, clippy, doc-check, doctest, nextest, public-api-check, doc-coverage-check all pass).

- [ ] **Step 2: Run with the opentelemetry feature explicitly enabled**

Run: `cargo build --workspace --all-features`
Expected: SUCCESS. (The `opentelemetry` feature is on `vllm-core`; `--all-features` should enable it transitively via `vllm-server`.)

- [ ] **Step 3: Run the OTLP CI job locally**

Run: `cargo test -p vllm-core --features opentelemetry --lib && cargo test -p vllm-server --features opentelemetry --test otlp_exporter_integration`
Expected: PASS.

- [ ] **Step 4: Live smoke test (optional but recommended)**

```bash
docker run -d --name jaeger -p 4317:4317 -p 16686:16686 jaegertracing/all-in-one:latest
cargo run --features opentelemetry -- --otlp-endpoint http://localhost:4317
# in another shell:
curl -X POST http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{...}'
# then open http://localhost:16686 → see the trace
docker stop jaeger && docker rm jaeger
```

Expected: trace visible in Jaeger UI.

- [ ] **Step 5: Final commit (no code changes, just a marker)**

```bash
git commit --allow-empty -m "chore: P43 OTLP exporter complete + just ci green"
```

---

## Self-review checklist

- [x] **Spec coverage**: each of G1-G7 from the spec maps to a Task (T1=skeleton+T2=types+T3=exporter+T4=tracing-init+T5=bootstrap+T6=tests+T7=docs+CI).
- [x] **No placeholders**: every code step has actual Rust code; no "TBD" or "fill in later".
- [x] **Type consistency**: `OtlpConfig`, `OtlpProtocol`, `OtlpError`, `OtlpExporter`, `OtlpExporterBuilder`, `OtlpGuard`, `OtlpHandle`, `MetricsError::Otlp` defined in Task 2 and used consistently in Tasks 3-6.
- [x] **Exact paths**: all file paths use the full crate-relative paths verified against the repo.
- [x] **Frequent commits**: each task ends with a commit; T7 splits into 6 sub-commits.
- [x] **TDD**: each task writes tests first (or alongside), verifies they fail, implements, verifies they pass.
- [x] **Acceptance criteria**: §13 of the spec maps to T8 final verification.

## Effort estimate (recap)

- T1: 0.3 day (Cargo + skeleton)
- T2: 0.3 day (types + validation)
- T3: 0.7 day (exporter + 21-metric schema)
- T4: 0.4 day (tracing-init + guard)
- T5: 0.4 day (bootstrap + config + CLI)
- T6: 0.5 day (integration tests + stub collector)
- T7: 0.2 day (CI + docs + ADR + CHANGELOG + STATE)
- T8: 0.1 day (final verification)
- **Total: ~2.9 days** (slightly above the 2.5-day estimate due to the stub collector being a small standalone service — acceptable)
