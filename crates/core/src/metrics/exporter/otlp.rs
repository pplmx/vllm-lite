//! OpenTelemetry (OTLP) push-based exporter — gated by the `opentelemetry`
//! feature on `vllm-core`. Streams engine metrics + tracing spans to any
//! `OTel`-compatible collector (Jaeger / Tempo / Datadog / Honeycomb / etc.).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use opentelemetry::KeyValue;
use opentelemetry::metrics::{Counter, Gauge, UpDownCounter};
use opentelemetry_otlp::{MetricExporter, WithExportConfig};
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::metrics::{PeriodicReader, SdkMeterProvider};
use opentelemetry_semantic_conventions::attribute as semattr;
use serde::{Deserialize, Serialize};
use tokio::time::interval;

use crate::metrics::{EnhancedMetricsCollector, MetricsSnapshot};

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
        if !self.trace_sampling_ratio.is_finite()
            || !(0.0..=1.0).contains(&self.trace_sampling_ratio)
        {
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

/// Where an instrument's value is sourced from on each export tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MetricSource {
    /// The collector's atomic fields (via `get_counter` / `get_gauge`) plus
    /// the `draft_metrics_snapshot()` counters.
    Atomic,
    /// The lock-free runtime [`MetricsSnapshot`] (lifetime + live gauges:
    /// tokens, latency percentiles, throughput, KV/prefix usage, in-flight,
    /// scheduler wait). RIL ISS-103 — pre-fix the exporter read ONLY the
    /// atomic path, so every headline engine metric that lives on
    /// `/metrics` was absent from OTLP.
    Snapshot,
}

/// Schema mapping: `(prometheus_name, otel_name, InstrumentKind, unit, source)`.
/// `InstrumentKind` is a private tag because the `OTel` Counter/Gauge types
/// are not nameable across `Option<>` in a const table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InstrumentKind {
    Counter,
    Gauge,
    UpDownCounter,
}

/// Every instrument this exporter records, keyed by the Prometheus wire
/// name so a single name drives both `/metrics` and OTLP (parity is pinned
/// by `metric_schema_mapping_covers_all_exported_prometheus_metrics`).
///
/// Only instruments with a REAL production source are listed (RIL ISS-103):
/// the five fabricated always-0 instruments (`gpu_memory_used/total_bytes`
/// — no GPU-memory source; `is_leader` — no leader-election in production;
/// `scheduler_queue_size` / `inflight_requests` — stale names with no
/// writers, superseded by `request_queue_depth` / `requests_in_flight`)
/// were removed instead of exporting zeros forever.
const SCHEMA_MAP: &[(&str, &str, InstrumentKind, &str, MetricSource)] = &[
    (
        "cuda_graph_hits_total",
        "cuda.graph.hits",
        InstrumentKind::Counter,
        "{hit}",
        MetricSource::Atomic,
    ),
    (
        "cuda_graph_misses_total",
        "cuda.graph.misses",
        InstrumentKind::Counter,
        "{miss}",
        MetricSource::Atomic,
    ),
    (
        "speculative_adjustments_total",
        "speculative.adjustments",
        InstrumentKind::Counter,
        "{adjustment}",
        MetricSource::Atomic,
    ),
    (
        "requests_total",
        "requests",
        InstrumentKind::Counter,
        "{request}",
        MetricSource::Atomic,
    ),
    (
        "dropped_tokens_total",
        "engine.dropped_tokens",
        InstrumentKind::Counter,
        "{token}",
        MetricSource::Atomic,
    ),
    (
        // RIL ISS-133: engine step errors on the primary OTLP surface too
        // (was /debug/metrics-admin-only).
        "errors_total",
        "engine.errors",
        InstrumentKind::Counter,
        "{error}",
        MetricSource::Atomic,
    ),
    (
        "draft_resolutions_external_total",
        "draft.resolutions.external",
        InstrumentKind::Counter,
        "{resolution}",
        MetricSource::Atomic,
    ),
    (
        "draft_resolutions_self_spec_total",
        "draft.resolutions.self_spec",
        InstrumentKind::Counter,
        "{resolution}",
        MetricSource::Atomic,
    ),
    (
        "draft_resolutions_none_total",
        "draft.resolutions.none",
        InstrumentKind::Counter,
        "{resolution}",
        MetricSource::Atomic,
    ),
    (
        "draft_load_failures_total",
        "draft.load.failures",
        InstrumentKind::Counter,
        "{failure}",
        MetricSource::Atomic,
    ),
    (
        "draft_runtime_errors_total",
        "draft.runtime.errors",
        InstrumentKind::Counter,
        "{error}",
        MetricSource::Atomic,
    ),
    (
        "packing_efficiency",
        "packing.efficiency",
        InstrumentKind::Gauge,
        "{ratio}",
        MetricSource::Atomic,
    ),
    (
        "speculative_acceptance_rate",
        "speculative.acceptance_rate",
        InstrumentKind::Gauge,
        "{ratio}",
        MetricSource::Atomic,
    ),
    (
        "throughput_speedup_ratio",
        "throughput.speedup_ratio",
        InstrumentKind::Gauge,
        "{ratio}",
        MetricSource::Atomic,
    ),
    (
        "speculative_per_request_count",
        "speculative.per_request_count",
        InstrumentKind::Gauge,
        "{sequence}",
        MetricSource::Atomic,
    ),
    (
        "request_queue_depth",
        "request.queue_depth",
        InstrumentKind::UpDownCounter,
        "{request}",
        MetricSource::Atomic,
    ),
    (
        "active_sequences",
        "active.sequences",
        InstrumentKind::UpDownCounter,
        "{sequence}",
        MetricSource::Atomic,
    ),
    // ── Headline engine metrics (from the lock-free runtime snapshot) ─────
    (
        "tokens_total",
        "tokens.generated",
        InstrumentKind::Counter,
        "{token}",
        MetricSource::Snapshot,
    ),
    (
        "avg_latency_ms",
        "latency.avg",
        InstrumentKind::Gauge,
        "ms",
        MetricSource::Snapshot,
    ),
    (
        "latency_p50_ms",
        "latency.p50",
        InstrumentKind::Gauge,
        "ms",
        MetricSource::Snapshot,
    ),
    (
        "latency_p90_ms",
        "latency.p90",
        InstrumentKind::Gauge,
        "ms",
        MetricSource::Snapshot,
    ),
    (
        "latency_p99_ms",
        "latency.p99",
        InstrumentKind::Gauge,
        "ms",
        MetricSource::Snapshot,
    ),
    (
        "avg_batch_size",
        "scheduler.avg_batch_size",
        InstrumentKind::Gauge,
        "{sequence}",
        MetricSource::Snapshot,
    ),
    (
        "current_batch_size",
        "scheduler.current_batch_size",
        InstrumentKind::Gauge,
        "{sequence}",
        MetricSource::Snapshot,
    ),
    (
        "requests_in_flight",
        "engine.inflight_requests",
        InstrumentKind::UpDownCounter,
        "{request}",
        MetricSource::Snapshot,
    ),
    (
        "kv_cache_usage_percent",
        "kv_cache.usage_percent",
        InstrumentKind::Gauge,
        "%",
        MetricSource::Snapshot,
    ),
    (
        "prefix_cache_hit_rate",
        "prefix_cache.hit_rate",
        InstrumentKind::Gauge,
        "%",
        MetricSource::Snapshot,
    ),
    (
        "prefill_throughput_tps",
        "throughput.prefill_tps",
        InstrumentKind::Gauge,
        "tps",
        MetricSource::Snapshot,
    ),
    (
        "decode_throughput_tps",
        "throughput.decode_tps",
        InstrumentKind::Gauge,
        "tps",
        MetricSource::Snapshot,
    ),
    (
        "avg_scheduler_wait_time_ms",
        "scheduler.avg_wait_time_ms",
        InstrumentKind::Gauge,
        "ms",
        MetricSource::Snapshot,
    ),
];

/// Bundled instruments for the schema map. One entry per Prometheus metric.
enum Instrument {
    Counter(Counter<u64>),
    Gauge(Gauge<f64>),
    UpDownCounter(UpDownCounter<i64>),
}

/// Fixed-point scale for the ratio gauges (`packing_efficiency`,
/// `speculative_*`, `throughput_speedup_ratio`): the collector stores them
/// as `ratio * 100_000` (u64) and the Prometheus exporter prints
/// `value / 100_000.0`. `OTel` must apply the same scale before
/// `Gauge::record`, otherwise the exported value is ``100_000×`` too large.
const RATIO_FIXED_POINT_SCALE: f64 = 100_000.0;

/// Delta an `OTel` [`Counter`] should be `add`ed since the previous export
/// tick, given the collector's ABSOLUTE lifetime total (`value`) and the
/// previously-exported total (`prev`). `OTel` counters accumulate their
/// `add()`s server-side (cumulative temporality), so feeding the absolute
/// total every tick inflates the series by the sum of every observed total
/// (e.g. `requests_total` 100 then 200 would read 100 then 300). Sending
/// the delta makes the backend cumulative equal the true lifetime total.
/// `None` `prev` (the first tick) sends the full value — the backend starts
/// empty. Monotonic counters never decrease, so the delta saturates at 0.
const fn counter_delta(prev: Option<u64>, value: u64) -> u64 {
    match prev {
        Some(p) => value.saturating_sub(p),
        None => value,
    }
}

/// Signed delta an `OTel` [`UpDownCounter`] should be `add`ed since the
/// previous export tick. The [`UpDownCounter`] instruments back gauge-like
/// `CURRENT` snapshots (request queue depth, active sequences, memory
/// bytes), so the signed delta keeps the backend cumulative equal to the
/// current value — including when it falls (negative delta). The first
/// tick (no `prev`) sends the current value once, matching the backend's
/// empty start.
fn updown_delta(prev: Option<u64>, value: u64) -> i64 {
    prev.map_or_else(
        || i64::try_from(value).unwrap_or(i64::MAX),
        |p| {
            // invariant: the exact i128 subtraction never overflows for any
            // pair of u64s; the clamp only guards the i64 conversion.
            let delta = i128::from(value) - i128::from(p);
            i64::try_from(delta).unwrap_or(if delta < 0 { i64::MIN } else { i64::MAX })
        },
    )
}

/// Gauge value-scale for an `OTel` [`Gauge`]. Only the `{ratio}` unit
/// series are fixed-point (`ratio * 100_000`) and need the 100,000-scale;
/// all other gauges export the raw current value.
fn ratio_scale_for(unit: &str) -> f64 {
    if unit == "{ratio}" {
        RATIO_FIXED_POINT_SCALE
    } else {
        1.0
    }
}

/// Lifetime/u64-valued field of a [`MetricsSnapshot`], keyed by the
/// Prometheus wire name (for snapshot-sourced `Counter` /
/// `UpDownCounter` instruments — RIL ISS-103).
fn snapshot_u64(snap: &MetricsSnapshot, name: &str) -> u64 {
    match name {
        "tokens_total" => snap.tokens_total,
        "requests_in_flight" => snap.requests_in_flight,
        _ => 0,
    }
}

/// Float-valued field of a [`MetricsSnapshot`], keyed by the Prometheus
/// wire name (for snapshot-sourced `Gauge` instruments — RIL ISS-103).
/// Values are exported as-is (no fixed-point scale): `kv_cache_usage_percent`
/// and `prefix_cache_hit_rate` are already `0..=100`, latencies ms, throughput
/// tokens/sec, batch sizes sequences.
#[allow(clippy::cast_precision_loss)]
fn snapshot_f64(snap: &MetricsSnapshot, name: &str) -> f64 {
    match name {
        "avg_latency_ms" => snap.avg_latency_ms,
        "latency_p50_ms" => snap.p50_latency_ms,
        "latency_p90_ms" => snap.p90_latency_ms,
        "latency_p99_ms" => snap.p99_latency_ms,
        "avg_batch_size" => snap.avg_batch_size,
        "current_batch_size" => snap.current_batch_size as f64,
        "kv_cache_usage_percent" => snap.kv_cache_usage_percent,
        "prefix_cache_hit_rate" => snap.prefix_cache_hit_rate,
        "prefill_throughput_tps" => snap.prefill_throughput,
        "decode_throughput_tps" => snap.decode_throughput,
        "avg_scheduler_wait_time_ms" => snap.avg_scheduler_wait_time_ms,
        _ => 0.0,
    }
}

/// Push-based OTLP metrics exporter polling the collector and exporting via `OTel`.
///
/// Holds an `EnhancedMetricsCollector` reference + an OTLP `SdkMeterProvider`;
/// `run()` polls the collector every `config.metrics_export_interval_secs` and
/// records each value into the corresponding `OTel` instrument. The
/// `PeriodicReader` flushes on each tick.
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
            .field(
                "collector_strong_count",
                &Arc::strong_count(&self.inner.collector),
            )
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
    pub fn new(
        collector: Arc<EnhancedMetricsCollector>,
        config: OtlpConfig,
    ) -> Result<Self, OtlpError> {
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
                KeyValue::new(
                    semattr::SERVICE_INSTANCE_ID,
                    uuid::Uuid::new_v4().to_string(),
                ),
                KeyValue::new("host.arch", std::env::consts::ARCH),
            ])
            .build();

        let provider = SdkMeterProvider::builder()
            .with_reader(reader)
            .with_resource(resource)
            .build();

        Ok(Self {
            inner: Arc::new(OtlpExporterInner {
                collector,
                config,
                provider,
            }),
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

        let instruments: Vec<(&str, Instrument, MetricSource)> = SCHEMA_MAP
            .iter()
            .map(|(prom_name, otel_name, kind, _unit, source)| {
                let inst = match kind {
                    InstrumentKind::Counter => {
                        Instrument::Counter(meter.u64_counter(*otel_name).build())
                    }
                    InstrumentKind::Gauge => Instrument::Gauge(meter.f64_gauge(*otel_name).build()),
                    InstrumentKind::UpDownCounter => {
                        Instrument::UpDownCounter(meter.i64_up_down_counter(*otel_name).build())
                    }
                };
                (*prom_name, inst, *source)
            })
            .collect();

        let mut ticker = interval(Duration::from_secs(
            self.inner.config.metrics_export_interval_secs,
        ));
        // Skip the immediate first tick so the first export happens after
        // `interval`, not at t=0 (gives the engine time to record data).
        ticker.tick().await;

        // Per-schema gauge scaling: only the `{ratio}` series are stored
        // fixed-point and must be divided by `RATIO_FIXED_POINT_SCALE` —
        // snapshot-sourced gauges export raw f64 values and scale by 1.0.
        let gauge_scales: HashMap<&str, f64> = SCHEMA_MAP
            .iter()
            .filter(|(_, _, kind, _, source)| {
                *kind == InstrumentKind::Gauge && *source == MetricSource::Atomic
            })
            .map(|(p, _, _, unit, _)| (*p, ratio_scale_for(unit)))
            .collect();

        // Atomic-sourced Prometheus names. The draft counters ride on the
        // same `u64` path (their values also come from atomics).
        let atomic_names: Vec<&str> = SCHEMA_MAP
            .iter()
            .filter(|(_, _, _, _, source)| *source == MetricSource::Atomic)
            .map(|(p, _, _, _, _)| *p)
            .collect();

        // Last-exported absolute values per instrument, so Counter /
        // UpDownCounter can be fed DELTAS (their OTel semantics accumulate
        // `add()`s) instead of absolute totals (which double/triple-count).
        let mut prev_values: HashMap<String, u64> = HashMap::with_capacity(SCHEMA_MAP.len());

        loop {
            ticker.tick().await;
            // Index by Prometheus name for O(1) lookups. All atomic sources
            // (`get_counter`, `get_gauge`, `draft_metrics_snapshot`) return
            // `u64`, so we keep the native integer type throughout and only
            // cast to `f64` at the OTel API boundary for `Gauge`. This
            // avoids the `u64 → f64 → u64` round-trip that lost precision
            // for counters exceeding 2^53.
            let mut by_name: HashMap<String, u64> = HashMap::with_capacity(atomic_names.len());

            for prom_name in &atomic_names {
                let value = self
                    .inner
                    .collector
                    .get_counter(prom_name)
                    .max(self.inner.collector.get_gauge(prom_name));
                by_name.insert((*prom_name).to_string(), value);
            }

            // Overlay the draft metrics snapshot (5 counters not in the
            // atomic-field path).
            let draft = self.inner.collector.draft_metrics_snapshot();
            by_name.insert(
                "draft_resolutions_external_total".into(),
                draft.resolutions_external_total,
            );
            by_name.insert(
                "draft_resolutions_self_spec_total".into(),
                draft.resolutions_self_spec_total,
            );
            by_name.insert(
                "draft_resolutions_none_total".into(),
                draft.resolutions_none_total,
            );
            by_name.insert(
                "draft_load_failures_total".into(),
                draft.load_failures_total,
            );
            by_name.insert(
                "draft_runtime_errors_total".into(),
                draft.runtime_errors_total,
            );

            // RIL ISS-103: the lock-free runtime snapshot carries the
            // headline engine metrics every tick — read once, applied to
            // every snapshot-sourced instrument.
            let snapshot = self.inner.collector.runtime_snapshot();

            for (prom_name, inst, source) in &instruments {
                match (inst, source) {
                    // Counter takes u64 deltas: the backend (cumulative
                    // temporality) accumulates `add()`s, so send the delta
                    // since the last tick — the absolute total would be
                    // added AGAIN server-side, inflating the series.
                    (Instrument::Counter(c), MetricSource::Atomic) => {
                        let value = by_name.get(*prom_name).copied().unwrap_or(0);
                        let prev = prev_values.get(*prom_name).copied();
                        c.add(counter_delta(prev, value), &[]);
                        prev_values.insert((*prom_name).to_string(), value);
                    }
                    (Instrument::Counter(c), MetricSource::Snapshot) => {
                        let value = snapshot_u64(&snapshot, prom_name);
                        let prev = prev_values.get(*prom_name).copied();
                        c.add(counter_delta(prev, value), &[]);
                        prev_values.insert((*prom_name).to_string(), value);
                    }
                    // Gauge's OTel API takes f64 — an unavoidable cast;
                    // integers don't exceed 2^53 in practice so precision
                    // loss is negligible here. Gauges are snapshots (no
                    // delta): atomic `{ratio}` series are fixed-point and
                    // scaled by 100,000; snapshot-sourced gauges record the
                    // raw f64 value.
                    (Instrument::Gauge(g), MetricSource::Atomic) => {
                        let value = by_name.get(*prom_name).copied().unwrap_or(0);
                        let scale = gauge_scales.get(prom_name).copied().unwrap_or(1.0);
                        #[allow(clippy::cast_precision_loss)]
                        g.record(value as f64 / scale, &[]);
                    }
                    (Instrument::Gauge(g), MetricSource::Snapshot) => {
                        g.record(snapshot_f64(&snapshot, prom_name), &[]);
                    }
                    // UpDownCounter takes i64 signed deltas (can decrease),
                    // keeping the backend cumulative equal to the CURRENT
                    // value (queue depth, active sequences, in-flight).
                    (Instrument::UpDownCounter(u), MetricSource::Atomic) => {
                        let value = by_name.get(*prom_name).copied().unwrap_or(0);
                        let prev = prev_values.get(*prom_name).copied();
                        u.add(updown_delta(prev, value), &[]);
                        prev_values.insert((*prom_name).to_string(), value);
                    }
                    (Instrument::UpDownCounter(u), MetricSource::Snapshot) => {
                        let value = snapshot_u64(&snapshot, prom_name);
                        let prev = prev_values.get(*prom_name).copied();
                        u.add(updown_delta(prev, value), &[]);
                        prev_values.insert((*prom_name).to_string(), value);
                    }
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
        Self {
            collector: None,
            config,
        }
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
// `ratio_gauge_scale_applies_only_to_ratio_units` asserts exact equality
// against the fixed-point scale constants (not a tolerance-based float
// comparison), so `float_cmp` is allowed — same as the other test modules.
#[allow(clippy::float_cmp)]
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
        let cfg = OtlpConfig {
            trace_sampling_ratio: -0.1,
            ..OtlpConfig::default()
        };
        assert!(matches!(cfg.validate(), Err(OtlpError::Config(_))));
    }

    #[test]
    fn otlp_config_rejects_above_one_sampling_ratio() {
        let cfg = OtlpConfig {
            trace_sampling_ratio: 1.5,
            ..OtlpConfig::default()
        };
        assert!(matches!(cfg.validate(), Err(OtlpError::Config(_))));
    }

    #[test]
    fn otlp_config_rejects_zero_metrics_interval() {
        let cfg = OtlpConfig {
            metrics_export_interval_secs: 0,
            ..OtlpConfig::default()
        };
        assert!(matches!(cfg.validate(), Err(OtlpError::Config(_))));
    }

    #[test]
    fn otlp_config_validates_default_as_ok() {
        assert!(OtlpConfig::default().validate().is_ok());
    }

    /// RIL ISS-103: `SCHEMA_MAP` must match the `PrometheusExporter` surface
    /// EXACTLY — pre-fix it omitted every headline engine metric (tokens,
    /// latencies, throughput, KV/prefix usage, in-flight, scheduler wait —
    /// all live on `/metrics`) and instead carried 5 fabricated always-0
    /// instruments (`gpu_memory_used/total_bytes`, `is_leader`,
    /// `scheduler_queue_size`, `inflight_requests`) that had no writers in
    /// the whole codebase. The metric names are parsed from the exporter's
    /// live output (non-`#` lines = `name value`), so the parity check
    /// tracks reality rather than a hand-maintained list — any metric added
    /// to `/metrics` or any fabricated instrument reintroduced fails here.
    #[tokio::test]
    async fn metric_schema_mapping_covers_all_exported_prometheus_metrics() {
        let collector = crate::metrics::EnhancedMetricsCollector::new();
        let exporter =
            crate::metrics::PrometheusExporter::new(std::sync::Arc::new(collector), 9090);
        let out = exporter.export_to_string().await;

        let mut exported: Vec<&str> = out
            .lines()
            .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
            .map(|l| l.split_whitespace().next().unwrap_or_default())
            .filter(|n| !n.is_empty())
            .collect();
        exported.sort_unstable();
        exported.dedup();

        let mut schema: Vec<&str> = SCHEMA_MAP.iter().map(|(p, _, _, _, _)| *p).collect();
        schema.sort_unstable();
        schema.dedup();

        assert_eq!(
            exported, schema,
            "SCHEMA_MAP must match the PrometheusExporter surface exactly — no \
             missing headline metric (they live on /metrics), no fabricated \
             always-0 instrument, no stale name"
        );
    }

    /// RIL ISS-133: engine step errors must surface on the PRIMARY
    /// observability surface (`/metrics` Prometheus) with the real
    /// counter value — pre-fix `errors_total` was written in production
    /// (engine/run.rs) but only rendered on the admin-gated
    /// `/debug/metrics`, so Prometheus/OTLP scrapes showed zero
    /// step-error signal while the engine silently failed steps.
    #[tokio::test]
    async fn errors_total_exported_on_prometheus_with_live_value() {
        let collector = crate::metrics::EnhancedMetricsCollector::new();
        assert_eq!(collector.get_counter("errors_total"), 0);
        collector.record_engine_error();
        collector.record_engine_error();
        assert_eq!(collector.get_counter("errors_total"), 2);

        let exporter =
            crate::metrics::PrometheusExporter::new(std::sync::Arc::new(collector), 9090);
        let out = exporter.export_to_string().await;
        assert!(
            out.contains("# HELP errors_total "),
            "/metrics must document errors_total, got:\n{out}"
        );
        let value_line = out
            .lines()
            .find(|l| l.starts_with("errors_total "))
            .expect("errors_total must be exported to /metrics");
        assert!(
            value_line.ends_with("errors_total 2"),
            "errors_total must carry the live counter value, got: {value_line}"
        );
    }

    /// RIL ISS-103: the fabricated always-0 instruments must not come back.
    #[test]
    fn schema_map_excludes_fabricated_instruments() {
        for banned in [
            "gpu_memory_used_bytes",
            "gpu_memory_total_bytes",
            "is_leader",
            "scheduler_queue_size",
            "inflight_requests",
        ] {
            assert!(
                !SCHEMA_MAP.iter().any(|(p, _, _, _, _)| *p == banned),
                "fabricated always-0 instrument {banned} must not be in SCHEMA_MAP"
            );
        }
    }

    #[test]
    fn counter_delta_first_tick_is_full_value_then_deltas() {
        // The collector exposes ABSOLUTE lifetime totals from an atomic
        // counter. An OTel Counter (cumulative temporality) accumulates its
        // `add()`s server-side, so feeding the absolute value every tick
        // inflates the series by the sum of all observed totals. The
        // exporter must send the delta since the last tick.
        assert_eq!(counter_delta(None, 100), 100);
        assert_eq!(counter_delta(Some(100), 200), 100);
        assert_eq!(counter_delta(Some(200), 250), 50);
    }

    #[test]
    fn counter_delta_clamps_monotonic_overflow() {
        // A monotonic counter can never decrease; a decrease reads as 0
        // delta (never a negative add that would corrupt the backend).
        assert_eq!(counter_delta(Some(50), 30), 0);
    }

    #[test]
    fn updown_delta_preserves_decreases() {
        // UpDownCounter instruments back gauge-like CURRENT snapshots (queue
        // depth, active sequences). Signed deltas keep the backend cumulative
        // equal to the current value, including when it falls.
        assert_eq!(updown_delta(None, 5), 5);
        assert_eq!(updown_delta(Some(5), 8), 3);
        assert_eq!(updown_delta(Some(5), 3), -2);
    }

    #[test]
    fn ratio_gauge_scale_applies_only_to_ratio_units() {
        // `packing_efficiency` / `speculative_*` / `throughput_speedup_ratio`
        // are stored fixed-point as `ratio * 100_000` (u64); the Prometheus
        // exporter divides by 100_000.0. OTLP must apply the same scale,
        // otherwise `Gauge::record(85000.0)` is 100,000× too large.
        assert_eq!(ratio_scale_for("{ratio}"), RATIO_FIXED_POINT_SCALE);
        assert_eq!(ratio_scale_for("{sequence}"), 1.0);
        assert_eq!(ratio_scale_for("By"), 1.0);
    }
}
