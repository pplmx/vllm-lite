#![allow(clippy::module_name_repetitions)]
// crates/core/src/metrics/exporter.rs
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::metrics::EnhancedMetricsCollector;

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
            .expect("metrics exporter mutex poisoned")
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect()
    }
}

#[async_trait::async_trait]
impl MetricsExporter for InMemoryMetricsExporter {
    async fn export(&self) -> Result<String, MetricsError> {
        let values = self.values.lock().expect("metrics exporter mutex poisoned");
        let mut out = String::new();
        for (name, value) in values.iter() {
            out.push_str(name);
            out.push(' ');
            out.push_str(&value.to_string());
            out.push('\n');
        }
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

/// `MetricsError`: metrics error.
#[derive(Debug, thiserror::Error)]
pub enum MetricsError {
    #[error("export failed: {0}")]
    ExportFailed(String),
}

#[derive(Debug)]
/// Prometheus metrics exporter
pub struct PrometheusExporter {
    collector: Arc<EnhancedMetricsCollector>,
    port: u16,
}

impl PrometheusExporter {
    pub const fn new(collector: Arc<EnhancedMetricsCollector>, port: u16) -> Self {
        Self { collector, port }
    }

    /// Export metrics as Prometheus text format
    pub async fn export_to_string(&self) -> String {
        let mut output = String::new();

        // Counters
        output.push_str("# HELP cuda_graph_hits_total Number of CUDA graph cache hits\n");
        output.push_str("# TYPE cuda_graph_hits_total counter\n");
        output.push_str(&format!(
            "cuda_graph_hits_total {}\n",
            self.collector.get_counter("cuda_graph_hits_total")
        ));

        output.push_str("# HELP cuda_graph_misses_total Number of CUDA graph cache misses\n");
        output.push_str("# TYPE cuda_graph_misses_total counter\n");
        output.push_str(&format!(
            "cuda_graph_misses_total {}\n",
            self.collector.get_counter("cuda_graph_misses_total")
        ));

        output.push_str("# HELP packing_sequences_total Total sequences packed\n");
        output.push_str("# TYPE packing_sequences_total counter\n");
        output.push_str(&format!(
            "packing_sequences_total {}\n",
            self.collector.get_counter("packing_sequences_total")
        ));

        output.push_str(
            "# HELP speculative_adjustments_total Number of speculative draft adjustments\n",
        );
        output.push_str("# TYPE speculative_adjustments_total counter\n");
        output.push_str(&format!(
            "speculative_adjustments_total {}\n",
            self.collector.get_counter("speculative_adjustments_total")
        ));

        output.push_str("# HELP requests_total Total requests processed\n");
        output.push_str("# TYPE requests_total counter\n");
        output.push_str(&format!(
            "requests_total {}\n",
            self.collector.get_counter("requests_total")
        ));

        output.push_str("# HELP errors_total Total errors encountered\n");
        output.push_str("# TYPE errors_total counter\n");
        output.push_str(&format!(
            "errors_total {}\n",
            self.collector.get_counter("errors_total")
        ));

        // v18.0 multi-model speculative decoding metrics
        let draft_snap = self.collector.draft_metrics_snapshot();
        output.push_str(
            "# HELP draft_resolutions_external_total Total draft resolutions -> external backend\n",
        );
        output.push_str("# TYPE draft_resolutions_external_total counter\n");
        output.push_str(&format!(
            "draft_resolutions_external_total {}\n",
            draft_snap.resolutions_external_total
        ));

        output.push_str("# HELP draft_resolutions_self_spec_total Total draft resolutions -> self-spec fallback\n");
        output.push_str("# TYPE draft_resolutions_self_spec_total counter\n");
        output.push_str(&format!(
            "draft_resolutions_self_spec_total {}\n",
            draft_snap.resolutions_self_spec_total
        ));

        output.push_str("# HELP draft_resolutions_none_total Total draft resolutions -> no draft (pure target decode)\n");
        output.push_str("# TYPE draft_resolutions_none_total counter\n");
        output.push_str(&format!(
            "draft_resolutions_none_total {}\n",
            draft_snap.resolutions_none_total
        ));

        output.push_str(
            "# HELP draft_load_failures_total Total draft load failures (FALL-01 trigger)\n",
        );
        output.push_str("# TYPE draft_load_failures_total counter\n");
        output.push_str(&format!(
            "draft_load_failures_total {}\n",
            draft_snap.load_failures_total
        ));

        output.push_str(
            "# HELP draft_runtime_errors_total Total draft runtime errors (FALL-02 trigger)\n",
        );
        output.push_str("# TYPE draft_runtime_errors_total counter\n");
        output.push_str(&format!(
            "draft_runtime_errors_total {}\n",
            draft_snap.runtime_errors_total
        ));

        // Gauges
        output.push_str("# HELP packing_efficiency Batch efficiency (0-1)\n");
        output.push_str("# TYPE packing_efficiency gauge\n");
        let eff = self.collector.get_gauge("packing_efficiency") as f64 / 100000.0;
        output.push_str(&format!("packing_efficiency {eff:.3}\n"));

        output.push_str("# HELP packing_waste_ratio Waste ratio (0-1)\n");
        output.push_str("# TYPE packing_waste_ratio gauge\n");
        let waste = self.collector.get_gauge("packing_waste_ratio") as f64 / 100000.0;
        output.push_str(&format!("packing_waste_ratio {waste:.3}\n"));

        output.push_str("# HELP speculative_acceptance_rate Token acceptance rate (0-1)\n");
        output.push_str("# TYPE speculative_acceptance_rate gauge\n");
        let rate = self.collector.get_gauge("speculative_acceptance_rate") as f64 / 100000.0;
        output.push_str(&format!("speculative_acceptance_rate {rate:.3}\n"));

        output.push_str("# HELP speculative_draft_count Current draft tokens\n");
        output.push_str("# TYPE speculative_draft_count gauge\n");
        output.push_str(&format!(
            "speculative_draft_count {}\n",
            self.collector.get_gauge("speculative_draft_count")
        ));

        output.push_str("# HELP speculative_efficiency Draft token efficiency ratio (0-1)\n");
        output.push_str("# TYPE speculative_efficiency gauge\n");
        let eff = self.collector.get_gauge("speculative_efficiency") as f64 / 100000.0;
        output.push_str(&format!("speculative_efficiency {eff:.3}\n"));

        output.push_str(
            "# HELP throughput_speedup_ratio Speculative speedup vs baseline (1.0 = same)\n",
        );
        output.push_str("# TYPE throughput_speedup_ratio gauge\n");
        let speedup = self.collector.get_gauge("throughput_speedup_ratio") as f64 / 100000.0;
        output.push_str(&format!("throughput_speedup_ratio {speedup:.3}\n"));

        output.push_str(
            "# HELP speculative_per_request_count Number of tracked per-request acceptance rates\n",
        );
        output.push_str("# TYPE speculative_per_request_count gauge\n");
        output.push_str(&format!(
            "speculative_per_request_count {}\n",
            self.collector.get_gauge("speculative_per_request_count")
        ));

        output.push_str("# HELP request_queue_depth Pending requests\n");
        output.push_str("# TYPE request_queue_depth gauge\n");
        output.push_str(&format!(
            "request_queue_depth {}\n",
            self.collector.get_gauge("request_queue_depth")
        ));

        output.push_str("# HELP active_sequences Currently processing sequences\n");
        output.push_str("# TYPE active_sequences gauge\n");
        output.push_str(&format!(
            "active_sequences {}\n",
            self.collector.get_gauge("active_sequences")
        ));

        output.push_str("# HELP gpu_memory_used_bytes GPU memory usage in bytes\n");
        output.push_str("# TYPE gpu_memory_used_bytes gauge\n");
        output.push_str(&format!(
            "gpu_memory_used_bytes {}\n",
            self.collector.get_gauge("gpu_memory_used_bytes")
        ));

        output.push_str("# HELP gpu_memory_total_bytes Total GPU memory\n");
        output.push_str("# TYPE gpu_memory_total_bytes gauge\n");
        output.push_str(&format!(
            "gpu_memory_total_bytes {}\n",
            self.collector.get_gauge("gpu_memory_total_bytes")
        ));

        output.push_str("# HELP is_leader Whether this instance is the leader\n");
        output.push_str("# TYPE is_leader gauge\n");
        output.push_str(&format!(
            "is_leader {}\n",
            self.collector.get_gauge("is_leader")
        ));

        output.push_str("# HELP inflight_requests Currently in-flight requests\n");
        output.push_str("# TYPE inflight_requests gauge\n");
        output.push_str(&format!(
            "inflight_requests {}\n",
            self.collector.get_gauge("inflight_requests")
        ));

        output.push_str("# HELP scheduler_queue_size Pending scheduling queue\n");
        output.push_str("# TYPE scheduler_queue_size gauge\n");
        output.push_str(&format!(
            "scheduler_queue_size {}\n",
            self.collector.get_gauge("scheduler_queue_size")
        ));

        output
    }

    #[must_use]
    pub const fn port(&self) -> u16 {
        self.port
    }
}

#[async_trait::async_trait]
impl MetricsExporter for PrometheusExporter {
    async fn export(&self) -> Result<String, MetricsError> {
        Ok(self.export_to_string().await)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_prometheus_exporter_format() {
        let collector = Arc::new(EnhancedMetricsCollector::new());
        collector.record_cuda_graph_hit();
        let exporter = PrometheusExporter::new(collector, 9090);
        let output = exporter.export_to_string().await;
        assert!(output.contains("cuda_graph_hits_total"));
        assert!(output.contains('1'));
    }

    #[tokio::test]
    async fn test_prometheus_exporter_gauges() {
        let collector = Arc::new(EnhancedMetricsCollector::new());
        collector.record_packing_efficiency(0.85);
        let exporter = PrometheusExporter::new(collector, 9090);
        let output = exporter.export_to_string().await;
        assert!(output.contains("packing_efficiency 0.850"));
    }

    #[tokio::test]
    async fn metrics_exporter_default_arc_returns_empty() {
        let exporter: Arc<dyn MetricsExporter> = <dyn MetricsExporter>::default_arc();
        let output = exporter.export().await.expect("default export succeeds");
        assert!(output.is_empty());
    }
}
