// crates/core/src/metrics/exporter.rs
use std::sync::Arc;

use crate::metrics::EnhancedMetricsCollector;

/// Trait for metrics exporters
#[async_trait::async_trait]
pub trait MetricsExporter {
    async fn export(&self) -> Result<String, MetricsError>;
}

#[derive(Debug, thiserror::Error)]
pub enum MetricsError {
    #[error("export failed: {0}")]
    ExportFailed(String),
}

/// Prometheus metrics exporter
pub struct PrometheusExporter {
    collector: Arc<EnhancedMetricsCollector>,
    port: u16,
}

impl PrometheusExporter {
    pub fn new(collector: Arc<EnhancedMetricsCollector>, port: u16) -> Self {
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

        output.push_str("# HELP speculative_adjustments_total Number of speculative draft adjustments\n");
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

        // Gauges
        output.push_str("# HELP packing_efficiency Batch efficiency (0-1)\n");
        output.push_str("# TYPE packing_efficiency gauge\n");
        let eff = self.collector.get_gauge("packing_efficiency") as f64 / 100000.0;
        output.push_str(&format!("packing_efficiency {:.3}\n", eff));

        output.push_str("# HELP packing_waste_ratio Waste ratio (0-1)\n");
        output.push_str("# TYPE packing_waste_ratio gauge\n");
        let waste = self.collector.get_gauge("packing_waste_ratio") as f64 / 100000.0;
        output.push_str(&format!("packing_waste_ratio {:.3}\n", waste));

        output.push_str("# HELP speculative_acceptance_rate Token acceptance rate (0-1)\n");
        output.push_str("# TYPE speculative_acceptance_rate gauge\n");
        let rate = self.collector.get_gauge("speculative_acceptance_rate") as f64 / 100000.0;
        output.push_str(&format!("speculative_acceptance_rate {:.3}\n", rate));

        output.push_str("# HELP speculative_draft_count Current draft tokens\n");
        output.push_str("# TYPE speculative_draft_count gauge\n");
        output.push_str(&format!(
            "speculative_draft_count {}\n",
            self.collector.get_gauge("speculative_draft_count")
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

        output
    }

    pub fn port(&self) -> u16 {
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
        assert!(output.contains("1"));
    }

    #[tokio::test]
    async fn test_prometheus_exporter_gauges() {
        let collector = Arc::new(EnhancedMetricsCollector::new());
        collector.record_packing_efficiency(0.85);
        let exporter = PrometheusExporter::new(collector, 9090);
        let output = exporter.export_to_string().await;
        assert!(output.contains("packing_efficiency 0.850"));
    }
}
