//! Metrics collection and export
pub mod collector;
pub mod exporter;
pub mod types;

// New enhanced types
pub use collector::EnhancedMetricsCollector;
pub use exporter::{MetricsExporter, PrometheusExporter};
pub use types::{MetricLabels, MetricType, MetricValue};

// Legacy metrics for backward compatibility
mod legacy;
pub use legacy::{LockFreeMetrics, MetricsCollector, MetricsSnapshot};
