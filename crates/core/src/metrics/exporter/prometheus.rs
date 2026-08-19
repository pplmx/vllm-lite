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

        output.push_str(
            "# HELP dropped_tokens_total Tokens dropped because the response \
             channel was full (slow consumer)\n",
        );
        output.push_str("# TYPE dropped_tokens_total counter\n");
        let _ = write!(
            output,
            "dropped_tokens_total {}\n",
            self.collector.get_counter("dropped_tokens_total")
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

        output.push_str("# HELP speculative_acceptance_rate Draft token acceptance rate, accepted/drafted (0-1)\n");
        output.push_str("# TYPE speculative_acceptance_rate gauge\n");
        let rate = self.collector.get_gauge("speculative_acceptance_rate") as f64 / 100_000.0;
        let _ = write!(output, "speculative_acceptance_rate {rate:.3}\n");

        output.push_str(
            "# HELP speculative_efficiency Draft token efficiency, accepted/drafted (0-1)\n",
        );
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

        // ── Core engine metrics (lock-free runtime snapshot) ────────────
        // These were previously only reachable via the unrouted
        // `api::get_prometheus` (GetMetrics round-trip). Reading the
        // collector's runtime snapshot directly keeps `/metrics`
        // self-contained and avoids a blocking engine round-trip on
        // every Prometheus scrape.
        let snap = self.collector.runtime_snapshot();

        output.push_str("# HELP tokens_total Total tokens generated\n");
        output.push_str("# TYPE tokens_total counter\n");
        let _ = write!(output, "tokens_total {}\n", snap.tokens_total);

        output.push_str("# HELP avg_latency_ms Average inference latency (ms)\n");
        output.push_str("# TYPE avg_latency_ms gauge\n");
        let _ = write!(output, "avg_latency_ms {:.3}\n", snap.avg_latency_ms);

        output.push_str("# HELP latency_p50_ms Inference latency p50 (ms)\n");
        output.push_str("# TYPE latency_p50_ms gauge\n");
        let _ = write!(output, "latency_p50_ms {:.3}\n", snap.p50_latency_ms);

        output.push_str("# HELP latency_p90_ms Inference latency p90 (ms)\n");
        output.push_str("# TYPE latency_p90_ms gauge\n");
        let _ = write!(output, "latency_p90_ms {:.3}\n", snap.p90_latency_ms);

        output.push_str("# HELP latency_p99_ms Inference latency p99 (ms)\n");
        output.push_str("# TYPE latency_p99_ms gauge\n");
        let _ = write!(output, "latency_p99_ms {:.3}\n", snap.p99_latency_ms);

        output.push_str("# HELP kv_cache_usage_percent KV cache usage (0-100)\n");
        output.push_str("# TYPE kv_cache_usage_percent gauge\n");
        let _ = write!(
            output,
            "kv_cache_usage_percent {:.3}\n",
            snap.kv_cache_usage_percent
        );

        output.push_str("# HELP prefix_cache_hit_rate Prefix cache hit rate (0-100)\n");
        output.push_str("# TYPE prefix_cache_hit_rate gauge\n");
        let _ = write!(
            output,
            "prefix_cache_hit_rate {:.3}\n",
            snap.prefix_cache_hit_rate
        );

        output.push_str("# HELP prefill_throughput_tps Prefill throughput (tokens/sec)\n");
        output.push_str("# TYPE prefill_throughput_tps gauge\n");
        let _ = write!(
            output,
            "prefill_throughput_tps {:.3}\n",
            snap.prefill_throughput
        );

        output.push_str("# HELP decode_throughput_tps Decode throughput (tokens/sec)\n");
        output.push_str("# TYPE decode_throughput_tps gauge\n");
        let _ = write!(
            output,
            "decode_throughput_tps {:.3}\n",
            snap.decode_throughput
        );

        output.push_str("# HELP avg_batch_size Average batch size\n");
        output.push_str("# TYPE avg_batch_size gauge\n");
        let _ = write!(output, "avg_batch_size {:.3}\n", snap.avg_batch_size);

        output.push_str("# HELP current_batch_size Current batch size\n");
        output.push_str("# TYPE current_batch_size gauge\n");
        let _ = write!(output, "current_batch_size {}\n", snap.current_batch_size);

        output.push_str("# HELP requests_in_flight Currently in-flight requests\n");
        output.push_str("# TYPE requests_in_flight gauge\n");
        let _ = write!(output, "requests_in_flight {}\n", snap.requests_in_flight);

        output.push_str("# HELP avg_scheduler_wait_time_ms Average scheduler wait (ms)\n");
        output.push_str("# TYPE avg_scheduler_wait_time_ms gauge\n");
        let _ = write!(
            output,
            "avg_scheduler_wait_time_ms {:.3}\n",
            snap.avg_scheduler_wait_time_ms
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::EnhancedMetricsCollector;

    #[tokio::test]
    async fn export_includes_core_engine_metrics() {
        let collector = EnhancedMetricsCollector::new();
        collector.record_tokens(42);
        collector.record_request();
        collector.record_latency(12.5);
        collector.record_kv_cache_usage(3, 8);

        let exporter = PrometheusExporter::new(std::sync::Arc::new(collector), 9090);
        let out = exporter.export_to_string().await;

        assert!(
            out.contains("tokens_total 42\n"),
            "missing tokens_total: {out}"
        );
        assert!(
            out.contains("avg_latency_ms 12.500\n"),
            "missing latency: {out}"
        );
        assert!(
            out.contains("kv_cache_usage_percent 37.500\n"),
            "missing kv usage: {out}"
        );
        assert!(
            out.contains("requests_in_flight 0\n"),
            "missing in-flight: {out}"
        );
        assert!(
            out.contains("current_batch_size 0\n"),
            "missing batch size: {out}"
        );
    }

    #[tokio::test]
    async fn export_reports_zero_when_no_activity() {
        let collector = EnhancedMetricsCollector::new();
        let exporter = PrometheusExporter::new(std::sync::Arc::new(collector), 9090);
        let out = exporter.export_to_string().await;
        assert!(
            out.contains("tokens_total 0\n"),
            "zero tokens expected: {out}"
        );
        assert!(out.contains("avg_latency_ms 0.000\n"));
        assert!(out.contains("kv_cache_usage_percent 0.000\n"));
    }
}
