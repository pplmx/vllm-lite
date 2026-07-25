#![allow(clippy::module_name_repetitions, clippy::too_long_first_doc_paragraph)]
//! Metrics collection and export.
/// Per-request metrics collector with enhanced timers and counters.
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
