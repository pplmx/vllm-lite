//! `PrometheusExporter` — write engine metrics in the Prometheus text
//! exposition format (one metric per line, with `# HELP` / `# TYPE`
//! headers, LF line terminators). Activated via the
//! `prometheus` feature flag on the server.

use std::fmt::Write;
use std::sync::Arc;

use crate::metrics::EnhancedMetricsCollector;

use super::MetricsError;
use super::MetricsExporter;

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
    #[allow(clippy::unused_async)]
    #[allow(clippy::too_many_lines)]
    // Prometheus text format: per-metric help/type/value triples in a fixed schema
    // invariant: gauge values are bounded counters/ratios (≤ 100_000), so u64 -> f64
    // precision loss is acceptable for Prometheus snapshot output.
    #[allow(clippy::cast_precision_loss)]
    pub async fn export_to_string(&self) -> String {
        let mut output = String::new();

        // Counters
        output.push_str("# HELP cuda_graph_hits_total Number of CUDA graph cache hits\n");
        output.push_str("# TYPE cuda_graph_hits_total counter\n");
        let _ = write!(
            output,
            "cuda_graph_hits_total {}\n",
            self.collector.get_counter("cuda_graph_hits_total")
        );

        output.push_str("# HELP cuda_graph_misses_total Number of CUDA graph cache misses\n");
        output.push_str("# TYPE cuda_graph_misses_total counter\n");
        let _ = write!(
            output,
            "cuda_graph_misses_total {}\n",
            self.collector.get_counter("cuda_graph_misses_total")
        );

        output.push_str("# HELP packing_sequences_total Total sequences packed\n");
        output.push_str("# TYPE packing_sequences_total counter\n");
        let _ = write!(
            output,
            "packing_sequences_total {}\n",
            self.collector.get_counter("packing_sequences_total")
        );

        output.push_str(
            "# HELP speculative_adjustments_total Number of speculative draft adjustments\n",
        );
        output.push_str("# TYPE speculative_adjustments_total counter\n");
        let _ = write!(
            output,
            "speculative_adjustments_total {}\n",
            self.collector.get_counter("speculative_adjustments_total")
        );

        output.push_str("# HELP requests_total Total requests processed\n");
        output.push_str("# TYPE requests_total counter\n");
        let _ = write!(
            output,
            "requests_total {}\n",
            self.collector.get_counter("requests_total")
        );

        output.push_str("# HELP errors_total Total errors encountered\n");
        output.push_str("# TYPE errors_total counter\n");
        let _ = write!(
            output,
            "errors_total {}\n",
            self.collector.get_counter("errors_total")
        );

        // v18.0 multi-model speculative decoding metrics
        let draft_snap = self.collector.draft_metrics_snapshot();
        output.push_str(
            "# HELP draft_resolutions_external_total Total draft resolutions -> external backend\n",
        );
        output.push_str("# TYPE draft_resolutions_external_total counter\n");
        let _ = write!(
            output,
            "draft_resolutions_external_total {}\n",
            draft_snap.resolutions_external_total
        );

        output.push_str("# HELP draft_resolutions_self_spec_total Total draft resolutions -> self-spec fallback\n");
        output.push_str("# TYPE draft_resolutions_self_spec_total counter\n");
        let _ = write!(
            output,
            "draft_resolutions_self_spec_total {}\n",
            draft_snap.resolutions_self_spec_total
        );

        output.push_str("# HELP draft_resolutions_none_total Total draft resolutions -> no draft (pure target decode)\n");
        output.push_str("# TYPE draft_resolutions_none_total counter\n");
        let _ = write!(
            output,
            "draft_resolutions_none_total {}\n",
            draft_snap.resolutions_none_total
        );

        output.push_str(
            "# HELP draft_load_failures_total Total draft load failures (FALL-01 trigger)\n",
        );
        output.push_str("# TYPE draft_load_failures_total counter\n");
        let _ = write!(
            output,
            "draft_load_failures_total {}\n",
            draft_snap.load_failures_total
        );

        output.push_str(
            "# HELP draft_runtime_errors_total Total draft runtime errors (FALL-02 trigger)\n",
        );
        output.push_str("# TYPE draft_runtime_errors_total counter\n");
        let _ = write!(
            output,
            "draft_runtime_errors_total {}\n",
            draft_snap.runtime_errors_total
        );

        // Gauges
        output.push_str("# HELP packing_efficiency Batch efficiency (0-1)\n");
        output.push_str("# TYPE packing_efficiency gauge\n");
        let eff = self.collector.get_gauge("packing_efficiency") as f64 / 100_000.0;
        let _ = write!(output, "packing_efficiency {eff:.3}\n");

        output.push_str("# HELP packing_waste_ratio Waste ratio (0-1)\n");
        output.push_str("# TYPE packing_waste_ratio gauge\n");
        let waste = self.collector.get_gauge("packing_waste_ratio") as f64 / 100_000.0;
        let _ = write!(output, "packing_waste_ratio {waste:.3}\n");

        output.push_str("# HELP speculative_acceptance_rate Token acceptance rate (0-1)\n");
        output.push_str("# TYPE speculative_acceptance_rate gauge\n");
        let rate = self.collector.get_gauge("speculative_acceptance_rate") as f64 / 100_000.0;
        let _ = write!(output, "speculative_acceptance_rate {rate:.3}\n");

        output.push_str("# HELP speculative_draft_count Current draft tokens\n");
        output.push_str("# TYPE speculative_draft_count gauge\n");
        let _ = write!(
            output,
            "speculative_draft_count {}\n",
            self.collector.get_gauge("speculative_draft_count")
        );

        output.push_str("# HELP speculative_efficiency Draft token efficiency ratio (0-1)\n");
        output.push_str("# TYPE speculative_efficiency gauge\n");
        let eff = self.collector.get_gauge("speculative_efficiency") as f64 / 100_000.0;
        let _ = write!(output, "speculative_efficiency {eff:.3}\n");

        output.push_str(
            "# HELP throughput_speedup_ratio Speculative speedup vs baseline (1.0 = same)\n",
        );
        output.push_str("# TYPE throughput_speedup_ratio gauge\n");
        let speedup = self.collector.get_gauge("throughput_speedup_ratio") as f64 / 100_000.0;
        let _ = write!(output, "throughput_speedup_ratio {speedup:.3}\n");

        output.push_str(
            "# HELP speculative_per_request_count Number of tracked per-request acceptance rates\n",
        );
        output.push_str("# TYPE speculative_per_request_count gauge\n");
        let _ = write!(
            output,
            "speculative_per_request_count {}\n",
            self.collector.get_gauge("speculative_per_request_count")
        );

        output.push_str("# HELP request_queue_depth Pending requests\n");
        output.push_str("# TYPE request_queue_depth gauge\n");
        let _ = write!(
            output,
            "request_queue_depth {}\n",
            self.collector.get_gauge("request_queue_depth")
        );

        output.push_str("# HELP active_sequences Currently processing sequences\n");
        output.push_str("# TYPE active_sequences gauge\n");
        let _ = write!(
            output,
            "active_sequences {}\n",
            self.collector.get_gauge("active_sequences")
        );

        output.push_str("# HELP gpu_memory_used_bytes GPU memory usage in bytes\n");
        output.push_str("# TYPE gpu_memory_used_bytes gauge\n");
        let _ = write!(
            output,
            "gpu_memory_used_bytes {}\n",
            self.collector.get_gauge("gpu_memory_used_bytes")
        );

        output.push_str("# HELP gpu_memory_total_bytes Total GPU memory\n");
        output.push_str("# TYPE gpu_memory_total_bytes gauge\n");
        let _ = write!(
            output,
            "gpu_memory_total_bytes {}\n",
            self.collector.get_gauge("gpu_memory_total_bytes")
        );

        output.push_str("# HELP is_leader Whether this instance is the leader\n");
        output.push_str("# TYPE is_leader gauge\n");
        let _ = write!(
            output,
            "is_leader {}\n",
            self.collector.get_gauge("is_leader")
        );

        output.push_str("# HELP inflight_requests Currently in-flight requests\n");
        output.push_str("# TYPE inflight_requests gauge\n");
        let _ = write!(
            output,
            "inflight_requests {}\n",
            self.collector.get_gauge("inflight_requests")
        );

        output.push_str("# HELP scheduler_queue_size Pending scheduling queue\n");
        output.push_str("# TYPE scheduler_queue_size gauge\n");
        let _ = write!(
            output,
            "scheduler_queue_size {}\n",
            self.collector.get_gauge("scheduler_queue_size")
        );

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
