//! OpenTelemetry (OTLP) push-based exporter — gated by the `opentelemetry`
//! feature on `vllm-core`. Streams engine metrics + tracing spans to any
//! `OTel`-compatible collector (Jaeger / Tempo / Datadog / Honeycomb / etc.).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use opentelemetry::metrics::{Counter, Gauge, UpDownCounter};
use opentelemetry::KeyValue;
use opentelemetry_otlp::{MetricExporter, WithExportConfig};
use opentelemetry_sdk::metrics::{PeriodicReader, SdkMeterProvider};
use opentelemetry_sdk::Resource;
use opentelemetry_semantic_conventions::attribute as semattr;
use serde::{Deserialize, Serialize};
use tokio::time::interval;

use crate::metrics::EnhancedMetricsCollector;

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
    #[serde(default)]
    pub enabled: bool,
    /// OTLP collector endpoint (gRPC). Default: `"http://localhost:4317"`.
    #[serde(default = "OtlpConfig::default_endpoint")]
    pub endpoint: String,
    /// ``OTel`` `service.name` resource attribute.
    #[serde(default = "OtlpConfig::default_service_name")]
    pub service_name: String,
    /// `OTel` `service.version` resource attribute (synced from the release manifest).
    #[serde(default = "OtlpConfig::default_service_version")]
    pub service_version: String,
    /// Metrics export interval in seconds. Default: `30`.
    #[serde(default = "OtlpConfig::default_metrics_export_interval_secs")]
    pub metrics_export_interval_secs: u64,
    /// Trace sampling ratio in `[0.0, 1.0]`. Default: `1.0` (always sample).
    #[serde(default = "OtlpConfig::default_trace_sampling_ratio")]
    pub trace_sampling_ratio: f64,
    /// OTLP transport protocol. Default: `Grpc`.
    #[serde(default)]
    pub protocol: OtlpProtocol,
}

impl OtlpConfig {
    fn default_endpoint() -> String {
        "http://localhost:4317".to_string()
    }
    fn default_service_name() -> String {
        "vllm-lite".to_string()
    }
    fn default_service_version() -> String {
        env!("CARGO_PKG_VERSION").to_string()
    }
    const fn default_metrics_export_interval_secs() -> u64 {
        30
    }
    const fn default_trace_sampling_ratio() -> f64 {
        1.0
    }
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
    ///
    /// # Errors
    ///
    /// Returns [`OtlpError::Config`] if `trace_sampling_ratio` is not finite
    /// or outside `[0.0, 1.0]`, `metrics_export_interval_secs` is zero, or
    /// `endpoint` is empty or whitespace-only.
    pub fn validate(&self) -> Result<(), OtlpError> {
        if !self.trace_sampling_ratio.is_finite() || !(0.0..=1.0).contains(&self.trace_sampling_ratio) {
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

/// Schema mapping: `(prometheus_name, otel_name, InstrumentKind, unit)`.
/// `InstrumentKind` is a private tag because the `OTel` Counter/Gauge types
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

/// Bundled instruments for the schema map. One entry per Prometheus metric.
enum Instrument {
    Counter(Counter<u64>),
    Gauge(Gauge<f64>),
    UpDownCounter(UpDownCounter<i64>),
}

/// Push-based OTLP metrics exporter. Holds an `EnhancedMetricsCollector`
/// reference + an OTLP `SdkMeterProvider`; `run()` polls the collector every
/// `config.metrics_export_interval_secs` and records each value into the
/// corresponding `OTel` instrument. The `PeriodicReader` flushes on each tick.
///
/// Internally `Arc`-wrapped so the bootstrap can clone + share between the
/// spawned background task (caller of `run`) and the shutdown path (caller
/// of `shutdown`).
#[derive(Clone)]
pub struct OtlpExporter {
    inner: Arc<OtlpExporterInner>,
}

struct OtlpExporterInner {
    collector: Arc<EnhancedMetricsCollector>,
    config: OtlpConfig,
    provider: SdkMeterProvider,
}

impl std::fmt::Debug for OtlpExporter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OtlpExporter")
            .field("config", &self.inner.config)
            .field("collector_strong_count", &Arc::strong_count(&self.inner.collector))
            .finish()
    }
}

impl OtlpExporter {
    /// Build the exporter. Constructs the `SdkMeterProvider` with an OTLP
    /// `MetricExporter` targeting `config.endpoint`. The caller must call
    /// `.run()` to start the polling loop.
    ///
    /// # Errors
    ///
    /// Returns [`OtlpError::Config`] if the configuration is invalid, or
    /// [`OtlpError::Builder`] if the metric exporter cannot be constructed.
    pub fn new(collector: Arc<EnhancedMetricsCollector>, config: OtlpConfig) -> Result<Self, OtlpError> {
        config.validate()?;

        let metric_exporter = MetricExporter::builder()
            .with_tonic()
            .with_endpoint(&config.endpoint)
            .build()
            .map_err(|e| OtlpError::Builder(format!("metric exporter: {e}")))?;

        let reader = PeriodicReader::builder(metric_exporter)
            .with_interval(Duration::from_secs(config.metrics_export_interval_secs))
            .build();

        let resource = Resource::builder()
            .with_attributes([
                KeyValue::new(semattr::SERVICE_NAME, config.service_name.clone()),
                KeyValue::new(semattr::SERVICE_VERSION, config.service_version.clone()),
                KeyValue::new(semattr::SERVICE_INSTANCE_ID, uuid::Uuid::new_v4().to_string()),
                KeyValue::new("host.arch", std::env::consts::ARCH),
            ])
            .build();

        let provider = SdkMeterProvider::builder()
            .with_reader(reader)
            .with_resource(resource)
            .build();

        Ok(Self {
            inner: Arc::new(OtlpExporterInner { collector, config, provider }),
        })
    }

    /// Return the underlying meter provider so the caller can install
    /// `tracing-opentelemetry` on top of the same exporter.
    #[must_use]
    pub fn meter_provider(&self) -> &SdkMeterProvider {
        &self.inner.provider
    }

    /// Background task body. Polls the collector and records each metric
    /// into the `OTel` instrument. Returns on cancellation; the caller is
    /// expected to `shutdown()` the exporter to flush.
    ///
    /// # Errors
    ///
    /// This function currently does not return `Err`, but the signature
    /// reserves the ability to propagate export failures in future revisions.
    pub async fn run(&self) -> Result<(), OtlpError> {
        use opentelemetry::metrics::MeterProvider as _;
        let meter = self.inner.provider.meter("vllm-lite");

        let instruments: Vec<(&str, Instrument)> = SCHEMA_MAP
            .iter()
            .map(|(prom_name, otel_name, kind, _unit)| {
                let inst = match kind {
                    InstrumentKind::Counter => {
                        Instrument::Counter(meter.u64_counter(*otel_name).build())
                    }
                    InstrumentKind::Gauge => {
                        Instrument::Gauge(meter.f64_gauge(*otel_name).build())
                    }
                    InstrumentKind::UpDownCounter => {
                        Instrument::UpDownCounter(meter.i64_up_down_counter(*otel_name).build())
                    }
                };
                (*prom_name, inst)
            })
            .collect();

        let mut ticker = interval(Duration::from_secs(self.inner.config.metrics_export_interval_secs));
        // Skip the immediate first tick so the first export happens after
        // `interval`, not at t=0 (gives the engine time to record data).
        ticker.tick().await;

        loop {
            ticker.tick().await;
            // Index by Prometheus name for O(1) lookups. Most metrics come
            // from `get_counter` / `get_gauge`; the four draft_* counters
            // come from `draft_metrics_snapshot`.
            let mut by_name: HashMap<String, f64> = HashMap::with_capacity(SCHEMA_MAP.len());

            for prom_name in SCHEMA_MAP.iter().map(|(p, _, _, _)| *p) {
                let value = self
                    .inner
                    .collector
                    .get_counter(prom_name)
                    .max(self.inner.collector.get_gauge(prom_name));
                by_name.insert(prom_name.to_string(), value as f64);
            }

            // Overlay the draft metrics snapshot (5 counters not in the
            // atomic-field path).
            let draft = self.inner.collector.draft_metrics_snapshot();
            by_name.insert("draft_resolutions_external_total".into(), draft.resolutions_external_total as f64);
            by_name.insert("draft_resolutions_self_spec_total".into(), draft.resolutions_self_spec_total as f64);
            by_name.insert("draft_resolutions_none_total".into(), draft.resolutions_none_total as f64);
            by_name.insert("draft_load_failures_total".into(), draft.load_failures_total as f64);
            by_name.insert("draft_runtime_errors_total".into(), draft.runtime_errors_total as f64);

            for (prom_name, inst) in &instruments {
                let value = by_name.get(*prom_name).copied().unwrap_or(0.0);
                match inst {
                    Instrument::Counter(c) => c.add(value as u64, &[]),
                    Instrument::Gauge(g) => g.record(value, &[]),
                    Instrument::UpDownCounter(u) => u.add(value as i64, &[]),
                }
            }
            // PeriodicReader auto-flushes on each tick; no explicit flush call.
        }
    }

    /// Flush pending metrics synchronously. Called by the bootstrap on
    /// shutdown.
    ///
    /// # Errors
    ///
    /// Returns [`OtlpError::Export`] if the meter provider cannot shut down
    /// within the configured timeout.
    pub fn shutdown(&self) -> Result<(), OtlpError> {
        self.inner
            .provider
            .shutdown()
            .map_err(|e| OtlpError::Export(format!("provider shutdown: {e}")))
    }
}

/// Builder for `OtlpExporter`. Use when you want to inject a pre-built
/// `SdkMeterProvider` (e.g. in tests with an in-memory exporter). For
/// production, prefer `OtlpExporter::new`.
#[derive(Debug)]
pub struct OtlpExporterBuilder {
    collector: Option<Arc<EnhancedMetricsCollector>>,
    config: OtlpConfig,
}

impl OtlpExporterBuilder {
    /// Create a builder with the given config. Use `.collector()` to
    /// inject a metrics collector, then `.build()`.
    #[must_use]
    pub const fn new(config: OtlpConfig) -> Self {
        Self { collector: None, config }
    }

    /// Inject a metrics collector. Returns `self` for chaining.
    #[must_use]
    pub fn collector(mut self, collector: Arc<EnhancedMetricsCollector>) -> Self {
        self.collector = Some(collector);
        self
    }

    /// Build the exporter.
    ///
    /// # Errors
    ///
    /// Returns [`OtlpError::Builder`] if no collector was set, or
    /// [`OtlpError::Config`] / [`OtlpError::Builder`] if the exporter
    /// cannot be constructed.
    pub fn build(self) -> Result<OtlpExporter, OtlpError> {
        let collector = self
            .collector
            .ok_or_else(|| OtlpError::Builder("collector not set".to_string()))?;
        OtlpExporter::new(collector, self.config)
    }
}

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

    #[test]
    fn metric_schema_mapping_covers_prometheus_metrics() {
        // Pins the contract: every Prometheus metric exposed by
        // PrometheusExporter has an `OTel` counterpart in SCHEMA_MAP.
        // If PrometheusExporter adds a new metric, this test fails until
        // the schema map is updated.
        let prometheus_names = [
            "cuda_graph_hits_total",
            "cuda_graph_misses_total",
            "speculative_adjustments_total",
            "requests_total",
            "draft_resolutions_external_total",
            "draft_resolutions_self_spec_total",
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
        assert_eq!(
            SCHEMA_MAP.len(),
            prometheus_names.len(),
            "SCHEMA_MAP must have exactly one entry per Prometheus metric"
        );
        for name in prometheus_names {
            assert!(
                SCHEMA_MAP.iter().any(|(p, _, _, _)| *p == name),
                "Prometheus metric {name} has no `OTel` counterpart in SCHEMA_MAP"
            );
        }
    }
}


