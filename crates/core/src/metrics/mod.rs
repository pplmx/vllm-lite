//! Metrics collection and export
pub mod collector;
pub mod exporter;
pub mod types;

pub use collector::EnhancedMetricsCollector;
pub use exporter::{MetricsExporter, PrometheusExporter};
pub use legacy::{LockFreeMetrics, MetricsCollector, MetricsSnapshot};
pub use types::{MetricLabels, MetricType, MetricValue};

mod legacy;
