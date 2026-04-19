//! Health check endpoint for production monitoring
//!
//! Provides HTTP endpoint for health checks and metrics export

use crate::metrics::enhanced::{EnhancedMetrics, HealthChecker, HealthStatus};
use std::sync::Arc;
use std::time::Instant;

/// Health check response
#[derive(Clone, Debug, serde::Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub timestamp: u64,
    pub uptime_seconds: u64,
    pub metrics: HealthMetrics,
}

/// Health metrics subset
#[derive(Clone, Debug, serde::Serialize)]
pub struct HealthMetrics {
    pub requests_total: u64,
    pub requests_failed: u64,
    pub failure_rate: f64,
    pub cuda_graph_hit_rate: f64,
    pub packing_efficiency: f64,
    pub speculative_acceptance_rate: f64,
}

/// Health endpoint handler
pub struct HealthEndpoint {
    metrics: Arc<EnhancedMetrics>,
    checker: HealthChecker,
    start_time: Instant,
}

impl HealthEndpoint {
    /// Create a new health endpoint
    pub fn new(metrics: Arc<EnhancedMetrics>) -> Self {
        let checker = HealthChecker::new(metrics.as_ref().clone());
        Self {
            metrics,
            checker,
            start_time: Instant::now(),
        }
    }

    /// Get health status
    pub fn check(&self) -> HealthResponse {
        let status = self.checker.check_health();
        let uptime = self.start_time.elapsed().as_secs();

        tracing::debug!(
            status = ?status,
            uptime_seconds = uptime,
            requests_total = self.metrics.requests_total(),
            failure_rate = self.metrics.failure_rate(),
            "Health check"
        );

        tracing::trace!(
            cuda_graph_hit_rate = self.metrics.cuda_graph_hit_rate(),
            packing_efficiency = self.metrics.packing_efficiency(),
            "Health metrics"
        );

        HealthResponse {
            status: format!("{:?}", status),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            uptime_seconds: uptime,
            metrics: HealthMetrics {
                requests_total: self
                    .metrics
                    .requests_total
                    .load(std::sync::atomic::Ordering::Relaxed),
                requests_failed: self
                    .metrics
                    .requests_failed
                    .load(std::sync::atomic::Ordering::Relaxed),
                failure_rate: self.metrics.failure_rate(),
                cuda_graph_hit_rate: self.metrics.cuda_graph_hit_rate(),
                packing_efficiency: self.metrics.packing_efficiency(),
                speculative_acceptance_rate: self.metrics.speculative_acceptance_rate(),
            },
        }
    }

    /// Check if healthy
    pub fn is_healthy(&self) -> bool {
        self.checker.check_health().is_healthy()
    }

    /// Get uptime
    pub fn uptime(&self) -> std::time::Duration {
        self.start_time.elapsed()
    }
}

/// Prometheus metrics format
pub struct PrometheusExporter {
    metrics: Arc<EnhancedMetrics>,
}

impl PrometheusExporter {
    /// Create a new Prometheus exporter
    pub fn new(metrics: Arc<EnhancedMetrics>) -> Self {
        Self { metrics }
    }

    /// Export metrics in Prometheus format
    pub fn export(&self) -> String {
        let mut output = String::new();

        // CUDA Graph metrics
        output.push_str("# HELP vllm_cuda_graph_hits Total CUDA Graph hits\n");
        output.push_str("# TYPE vllm_cuda_graph_hits counter\n");
        output.push_str(&format!(
            "vllm_cuda_graph_hits {}\n",
            self.metrics
                .cuda_graph_hits
                .load(std::sync::atomic::Ordering::Relaxed)
        ));

        output.push_str("# HELP vllm_cuda_graph_misses Total CUDA Graph misses\n");
        output.push_str("# TYPE vllm_cuda_graph_misses counter\n");
        output.push_str(&format!(
            "vllm_cuda_graph_misses {}\n",
            self.metrics
                .cuda_graph_misses
                .load(std::sync::atomic::Ordering::Relaxed)
        ));

        output.push_str("# HELP vllm_cuda_graph_hit_rate CUDA Graph hit rate\n");
        output.push_str("# TYPE vllm_cuda_graph_hit_rate gauge\n");
        output.push_str(&format!(
            "vllm_cuda_graph_hit_rate {:.4}\n",
            self.metrics.cuda_graph_hit_rate()
        ));

        // Packing metrics
        output.push_str("# HELP vllm_packing_efficiency Packing efficiency percentage\n");
        output.push_str("# TYPE vllm_packing_efficiency gauge\n");
        output.push_str(&format!(
            "vllm_packing_efficiency {:.4}\n",
            self.metrics.packing_efficiency()
        ));

        output.push_str("# HELP vllm_packing_waste_ratio Packing waste ratio\n");
        output.push_str("# TYPE vllm_packing_waste_ratio gauge\n");
        output.push_str(&format!(
            "vllm_packing_waste_ratio {:.4}\n",
            self.metrics.packing_waste_ratio()
        ));

        // Speculative metrics
        output.push_str("# HELP vllm_speculative_acceptance_rate Speculative acceptance rate\n");
        output.push_str("# TYPE vllm_speculative_acceptance_rate gauge\n");
        output.push_str(&format!(
            "vllm_speculative_acceptance_rate {:.4}\n",
            self.metrics.speculative_acceptance_rate()
        ));

        output.push_str("# HELP vllm_speculative_current_draft_count Current draft token count\n");
        output.push_str("# TYPE vllm_speculative_current_draft_count gauge\n");
        output.push_str(&format!(
            "vllm_speculative_current_draft_count {}\n",
            self.metrics
                .speculative_current_draft_count
                .load(std::sync::atomic::Ordering::Relaxed)
        ));

        // General metrics
        output.push_str("# HELP vllm_requests_total Total requests\n");
        output.push_str("# TYPE vllm_requests_total counter\n");
        output.push_str(&format!(
            "vllm_requests_total {}\n",
            self.metrics
                .requests_total
                .load(std::sync::atomic::Ordering::Relaxed)
        ));

        output.push_str("# HELP vllm_requests_failed Total failed requests\n");
        output.push_str("# TYPE vllm_requests_failed counter\n");
        output.push_str(&format!(
            "vllm_requests_failed {}\n",
            self.metrics
                .requests_failed
                .load(std::sync::atomic::Ordering::Relaxed)
        ));

        output.push_str("# HELP vllm_failure_rate Request failure rate\n");
        output.push_str("# TYPE vllm_failure_rate gauge\n");
        output.push_str(&format!(
            "vllm_failure_rate {:.4}\n",
            self.metrics.failure_rate()
        ));

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_endpoint() {
        let metrics = Arc::new(EnhancedMetrics::new());
        let endpoint = HealthEndpoint::new(metrics.clone());

        let response = endpoint.check();
        assert_eq!(response.status, "Healthy");
        assert!(response.uptime_seconds >= 0);
    }

    #[test]
    fn test_prometheus_export() {
        let metrics = Arc::new(EnhancedMetrics::new());
        let exporter = PrometheusExporter::new(metrics);

        let output = exporter.export();
        assert!(output.contains("vllm_cuda_graph_hits"));
        assert!(output.contains("vllm_packing_efficiency"));
        assert!(output.contains("vllm_speculative_acceptance_rate"));
        assert!(output.contains("vllm_requests_total"));
    }
}
