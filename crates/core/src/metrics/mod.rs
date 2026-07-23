#![allow(clippy::module_name_repetitions)]
//! Metrics collection and export
/// Time-series metrics collector (enhanced per-request metrics).
pub mod collector;
/// Metrics export backends (Prometheus, in-memory).
pub mod exporter;
/// Metric value types, labels, and snapshots.
pub mod types;

pub use collector::{DraftResolutionKind, EnhancedMetricsCollector};
pub use exporter::{InMemoryMetricsExporter, MetricsExporter, PrometheusExporter};
pub use lock_free::{LockFreeMetrics, MetricsCollector, MetricsSnapshot};
pub use types::{MetricLabels, MetricType, MetricValue};

mod lock_free;
