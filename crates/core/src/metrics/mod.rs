//! Metrics collection and export
/// collector: collector module.
pub mod collector;
/// exporter: exporter module.
pub mod exporter;
/// types: types module.
pub mod types;

pub use collector::EnhancedMetricsCollector;
pub use exporter::{MetricsExporter, PrometheusExporter};
pub use lock_free::{LockFreeMetrics, MetricsCollector, MetricsSnapshot};
pub use types::{MetricLabels, MetricType, MetricValue};

mod lock_free;
