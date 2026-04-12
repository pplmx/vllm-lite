//! Metrics collection and export
pub mod types;
pub mod collector;
pub mod exporter;

// New enhanced types
pub use types::{MetricLabels, MetricType, MetricValue};
pub use collector::EnhancedMetricsCollector;
pub use exporter::{MetricsExporter, PrometheusExporter};

// Legacy metrics for backward compatibility
mod legacy;
pub use legacy::{LockFreeMetrics, MetricsCollector, MetricsSnapshot};
