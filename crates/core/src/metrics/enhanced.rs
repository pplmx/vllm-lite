//! Enhanced metrics for production observability
//!
//! This module provides detailed metrics for all three optimizations:
//! - CUDA Graph: hit/miss rates, execution times
//! - Sequence Packing: waste ratios, efficiency
//! - Adaptive Speculative: acceptance rates, adjustments

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// Enhanced metrics for production monitoring
#[derive(Clone, Debug)]
pub struct EnhancedMetrics {
    // CUDA Graph metrics
    pub cuda_graph_hits: Arc<AtomicU64>,
    pub cuda_graph_misses: Arc<AtomicU64>,
    pub cuda_graph_execution_time_us: Arc<AtomicU64>,
    pub cuda_graph_capture_time_us: Arc<AtomicU64>,

    // Sequence Packing metrics
    pub packing_batches_created: Arc<AtomicU64>,
    pub packing_total_waste: Arc<AtomicU64>,
    pub packing_total_tokens: Arc<AtomicU64>,
    pub packing_efficiency_percent: Arc<AtomicU64>,

    // Adaptive Speculative metrics
    pub speculative_draft_tokens_generated: Arc<AtomicU64>,
    pub speculative_draft_tokens_accepted: Arc<AtomicU64>,
    pub speculative_verification_failures: Arc<AtomicU64>,
    pub speculative_adjustments: Arc<AtomicU64>,
    pub speculative_current_draft_count: Arc<AtomicU64>,

    // General health metrics
    pub requests_total: Arc<AtomicU64>,
    pub requests_failed: Arc<AtomicU64>,
    pub requests_cancelled: Arc<AtomicU64>,
    pub engine_restarts: Arc<AtomicU64>,
}

impl Default for EnhancedMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl EnhancedMetrics {
    /// Create a new enhanced metrics instance
    pub fn new() -> Self {
        Self {
            // CUDA Graph
            cuda_graph_hits: Arc::new(AtomicU64::new(0)),
            cuda_graph_misses: Arc::new(AtomicU64::new(0)),
            cuda_graph_execution_time_us: Arc::new(AtomicU64::new(0)),
            cuda_graph_capture_time_us: Arc::new(AtomicU64::new(0)),

            // Sequence Packing
            packing_batches_created: Arc::new(AtomicU64::new(0)),
            packing_total_waste: Arc::new(AtomicU64::new(0)),
            packing_total_tokens: Arc::new(AtomicU64::new(0)),
            packing_efficiency_percent: Arc::new(AtomicU64::new(0)),

            // Adaptive Speculative
            speculative_draft_tokens_generated: Arc::new(AtomicU64::new(0)),
            speculative_draft_tokens_accepted: Arc::new(AtomicU64::new(0)),
            speculative_verification_failures: Arc::new(AtomicU64::new(0)),
            speculative_adjustments: Arc::new(AtomicU64::new(0)),
            speculative_current_draft_count: Arc::new(AtomicU64::new(0)),

            // General health
            requests_total: Arc::new(AtomicU64::new(0)),
            requests_failed: Arc::new(AtomicU64::new(0)),
            requests_cancelled: Arc::new(AtomicU64::new(0)),
            engine_restarts: Arc::new(AtomicU64::new(0)),
        }
    }

    // CUDA Graph metrics
    pub fn record_cuda_graph_hit(&self) {
        self.cuda_graph_hits.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_cuda_graph_miss(&self) {
        self.cuda_graph_misses.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_cuda_graph_execution_time(&self, duration: Duration) {
        let us = duration.as_micros() as u64;
        self.cuda_graph_execution_time_us
            .fetch_add(us, Ordering::Relaxed);
    }

    pub fn record_cuda_graph_capture_time(&self, duration: Duration) {
        let us = duration.as_micros() as u64;
        self.cuda_graph_capture_time_us
            .fetch_add(us, Ordering::Relaxed);
    }

    /// Get CUDA Graph hit rate (0.0-1.0)
    pub fn cuda_graph_hit_rate(&self) -> f64 {
        let hits = self.cuda_graph_hits.load(Ordering::Relaxed);
        let misses = self.cuda_graph_misses.load(Ordering::Relaxed);
        let total = hits + misses;
        if total == 0 {
            0.0
        } else {
            hits as f64 / total as f64
        }
    }

    // Sequence Packing metrics
    pub fn record_packing_batch(&self, waste: usize, total_tokens: usize) {
        self.packing_batches_created.fetch_add(1, Ordering::Relaxed);
        self.packing_total_waste
            .fetch_add(waste as u64, Ordering::Relaxed);
        self.packing_total_tokens
            .fetch_add(total_tokens as u64, Ordering::Relaxed);

        // Update efficiency
        let total = self.packing_total_tokens.load(Ordering::Relaxed);
        let waste = self.packing_total_waste.load(Ordering::Relaxed);
        if total > 0 {
            let efficiency = ((total - waste) * 100) / total;
            self.packing_efficiency_percent
                .store(efficiency as u64, Ordering::Relaxed);
        }
    }

    /// Get packing efficiency percentage (0-100)
    pub fn packing_efficiency(&self) -> f64 {
        self.packing_efficiency_percent.load(Ordering::Relaxed) as f64
    }

    /// Get packing waste ratio (0.0-1.0)
    pub fn packing_waste_ratio(&self) -> f64 {
        let total = self.packing_total_tokens.load(Ordering::Relaxed);
        let waste = self.packing_total_waste.load(Ordering::Relaxed);
        if total == 0 {
            0.0
        } else {
            waste as f64 / total as f64
        }
    }

    // Adaptive Speculative metrics
    pub fn record_speculative_draft(&self, generated: usize, accepted: usize) {
        self.speculative_draft_tokens_generated
            .fetch_add(generated as u64, Ordering::Relaxed);
        self.speculative_draft_tokens_accepted
            .fetch_add(accepted as u64, Ordering::Relaxed);
    }

    pub fn record_speculative_verification_failure(&self) {
        self.speculative_verification_failures
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_speculative_adjustment(&self) {
        self.speculative_adjustments.fetch_add(1, Ordering::Relaxed);
    }

    pub fn set_speculative_current_draft_count(&self, count: usize) {
        self.speculative_current_draft_count
            .store(count as u64, Ordering::Relaxed);
    }

    /// Get speculative acceptance rate (0.0-1.0)
    pub fn speculative_acceptance_rate(&self) -> f64 {
        let generated = self
            .speculative_draft_tokens_generated
            .load(Ordering::Relaxed);
        let accepted = self
            .speculative_draft_tokens_accepted
            .load(Ordering::Relaxed);
        if generated == 0 {
            0.0
        } else {
            accepted as f64 / generated as f64
        }
    }

    // General health metrics
    pub fn record_request(&self) {
        self.requests_total.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_request_failed(&self) {
        self.requests_failed.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_request_cancelled(&self) {
        self.requests_cancelled.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_engine_restart(&self) {
        self.engine_restarts.fetch_add(1, Ordering::Relaxed);
    }

    /// Get requests per second
    pub fn request_rate(&self, elapsed_secs: u64) -> f64 {
        let total = self.requests_total.load(Ordering::Relaxed);
        if elapsed_secs == 0 {
            0.0
        } else {
            total as f64 / elapsed_secs as f64
        }
    }

    /// Get failure rate (0.0-1.0)
    pub fn failure_rate(&self) -> f64 {
        let total = self.requests_total.load(Ordering::Relaxed);
        let failed = self.requests_failed.load(Ordering::Relaxed);
        if total == 0 {
            0.0
        } else {
            failed as f64 / total as f64
        }
    }
}

/// Health check status
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
}

impl HealthStatus {
    /// Check if status is healthy
    pub fn is_healthy(&self) -> bool {
        matches!(self, HealthStatus::Healthy)
    }

    /// Check if status is unhealthy
    pub fn is_unhealthy(&self) -> bool {
        matches!(self, HealthStatus::Unhealthy)
    }
}

/// Health checker for production monitoring
pub struct HealthChecker {
    metrics: EnhancedMetrics,
    failure_threshold: f64,
    latency_threshold_ms: f64,
}

impl HealthChecker {
    /// Create a new health checker
    pub fn new(metrics: EnhancedMetrics) -> Self {
        Self {
            metrics,
            failure_threshold: 0.05,      // 5% failure rate
            latency_threshold_ms: 1000.0, // 1 second
        }
    }

    /// Get current health status
    pub fn check_health(&self) -> HealthStatus {
        let failure_rate = self.metrics.failure_rate();

        if failure_rate > self.failure_threshold {
            return HealthStatus::Unhealthy;
        }

        // Additional checks could be added here:
        // - Latency checks
        // - Memory usage
        // - Queue depth

        HealthStatus::Healthy
    }

    /// Set failure threshold
    pub fn set_failure_threshold(&mut self, threshold: f64) {
        self.failure_threshold = threshold;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_graph_metrics() {
        let metrics = EnhancedMetrics::new();

        metrics.record_cuda_graph_hit();
        metrics.record_cuda_graph_hit();
        metrics.record_cuda_graph_miss();

        assert_eq!(metrics.cuda_graph_hits.load(Ordering::Relaxed), 2);
        assert_eq!(metrics.cuda_graph_misses.load(Ordering::Relaxed), 1);
        assert!((metrics.cuda_graph_hit_rate() - 0.67).abs() < 0.01);
    }

    #[test]
    fn test_packing_metrics() {
        let metrics = EnhancedMetrics::new();

        // Batch with 100 tokens, 20 waste (80% efficiency)
        metrics.record_packing_batch(20, 100);

        assert_eq!(metrics.packing_efficiency(), 80.0);
        assert!((metrics.packing_waste_ratio() - 0.2).abs() < 0.01);
    }

    #[test]
    fn test_speculative_metrics() {
        let metrics = EnhancedMetrics::new();

        metrics.record_speculative_draft(10, 7);
        metrics.set_speculative_current_draft_count(5);

        assert!((metrics.speculative_acceptance_rate() - 0.7).abs() < 0.01);
        assert_eq!(
            metrics
                .speculative_current_draft_count
                .load(Ordering::Relaxed),
            5
        );
    }

    #[test]
    fn test_health_checker() {
        let metrics = EnhancedMetrics::new();
        let checker = HealthChecker::new(metrics.clone());

        // Initially healthy (no failures)
        assert!(checker.check_health().is_healthy());

        // Add some failures
        for _ in 0..10 {
            metrics.record_request();
        }
        metrics.record_request_failed();

        // Should still be healthy (< 5% failure rate)
        assert!(checker.check_health().is_healthy());

        // Add many failures
        for _ in 0..100 {
            metrics.record_request();
            metrics.record_request_failed();
        }

        // Should now be unhealthy
        assert!(checker.check_health().is_unhealthy());
    }
}
