//! Prometheus + OpenTelemetry + JSON exporters for engine metrics.
//!
//! `MetricsExporter` is the trait every concrete exporter implements;
//! `PrometheusExporter` writes the text format to `/metrics`, the OTLP
//! exporter streams to a configured collector, and the JSON exporter
//! serves `/debug/metrics`. Activated by feature flags.
//!
//! Module layout:
//!
//! - [`self`] (`mod.rs`) — `MetricsExporter` trait + `InMemoryMetricsExporter`
//!   + `MetricsError` + `dyn MetricsExporter::default_arc` + tests
//! - `prometheus` — `PrometheusExporter` struct + impl

// crates/core/src/metrics/exporter/mod.rs
// Prometheus text format requires explicit LF line terminators; explicit `\n` in
// `write!` calls is clearer than `writeln!` for this protocol.
#![allow(clippy::write_with_newline)]
mod prometheus;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Trait for metrics exporters
#[async_trait::async_trait]
pub trait MetricsExporter {
    async fn export(&self) -> Result<String, MetricsError>;
}

/// Default in-memory `MetricsExporter`.
///
/// `export()` returns the snapshot as a simple "name=value" text format. Useful
/// as the `Arc<dyn MetricsExporter>` default instance and for tests that only
/// care that `export()` succeeds.
#[derive(Debug, Default, Clone)]
pub struct InMemoryMetricsExporter {
    values: Arc<Mutex<HashMap<String, f64>>>,
}

impl InMemoryMetricsExporter {
    /// # Panics
    ///
    /// Panics if a required invariant is violated (e.g. a `None` value is force-unwrapped or an out-of-bounds index is used).
    /// Records a single metric value, overwriting any prior entry for `name`.
    pub fn record(&self, name: impl Into<String>, value: f64) {
        self.values
            .lock()
            // invariant: lock is only held for sync field access; poisoning only happens on panic during a critical section.
            .expect("metrics exporter mutex poisoned")
            .insert(name.into(), value);
    }

    /// # Panics
    ///
    /// Panics if a required invariant is violated (e.g. a `None` value is force-unwrapped or an out-of-bounds index is used).
    /// Returns the recorded values, in unspecified order.
    #[must_use]
    pub fn snapshot(&self) -> Vec<(String, f64)> {
        self.values
            .lock()
            // invariant: lock is only held for sync field access; poisoning only happens on panic during a critical section.
            .expect("metrics exporter mutex poisoned")
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect()
    }
}

#[async_trait::async_trait]
impl MetricsExporter for InMemoryMetricsExporter {
    async fn export(&self) -> Result<String, MetricsError> {
        let values = self.values.lock()?;
        let mut out = String::new();
        for (name, value) in values.iter() {
            out.push_str(name);
            out.push(' ');
            out.push_str(&value.to_string());
            out.push('\n');
        }
        drop(values);
        Ok(out)
    }
}

impl dyn MetricsExporter {
    /// Returns an `Arc<Self>` containing a fresh [`InMemoryMetricsExporter`].
    ///
    /// This is the closest equivalent to `Arc::<dyn MetricsExporter>::default()`;
    /// Rust's orphan rule prevents a direct `impl Default for Arc<dyn ...>`
    /// because `Arc` is foreign and there is no local type appearing before
    /// the uncovered trait-object parameter.
    #[must_use]
    pub fn default_arc() -> Arc<Self> {
        Arc::new(InMemoryMetricsExporter::default())
    }
}

/// Error type for the metrics export / serialization layer. Covers I/O failures on the export sink and Prometheus-format conversion errors.
#[derive(Debug, thiserror::Error)]
pub enum MetricsError {
    #[error("export failed: {0}")]
    ExportFailed(String),
    /// A `Mutex`/`RwLock` guard was poisoned by a panic while held.
    #[error("metrics exporter lock poisoned")]
    LockPoisoned,
}

/// Convert any `std::sync::PoisonError<T>` into [`MetricsError::LockPoisoned`].
impl<T> From<std::sync::PoisonError<T>> for MetricsError {
    fn from(_: std::sync::PoisonError<T>) -> Self {
        Self::LockPoisoned
    }
}

pub use prometheus::PrometheusExporter;

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_prometheus_exporter_format() {
        let collector = Arc::new(crate::metrics::EnhancedMetricsCollector::new());
        collector.record_cuda_graph_hit();
        let exporter = PrometheusExporter::new(collector, 9090);
        let output = exporter.export_to_string().await;
        assert!(output.contains("cuda_graph_hits_total"));
        assert!(output.contains('1'));
    }

    #[tokio::test]
    async fn test_prometheus_exporter_gauges() {
        let collector = Arc::new(crate::metrics::EnhancedMetricsCollector::new());
        collector.record_packing_efficiency(0.85);
        let exporter = PrometheusExporter::new(collector, 9090);
        let output = exporter.export_to_string().await;
        assert!(output.contains("packing_efficiency 0.850"));
    }

    #[tokio::test]
    async fn metrics_exporter_default_arc_returns_empty() {
        let exporter: Arc<dyn MetricsExporter> = <dyn MetricsExporter>::default_arc();
        // invariant: pre-conditions make this infallible at this call site.
        let output = exporter.export().await.expect("default export succeeds");
        assert!(output.is_empty());
    }
}
