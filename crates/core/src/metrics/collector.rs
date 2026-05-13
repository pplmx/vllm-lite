// crates/core/src/metrics/collector.rs
use dashmap::DashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use vllm_traits::SeqId;

/// Centralized metrics collector for all optimization components
#[derive(Debug)]
pub struct EnhancedMetricsCollector {
    // Counters
    cuda_graph_hits: AtomicU64,
    cuda_graph_misses: AtomicU64,
    packing_sequences: AtomicU64,
    speculative_adjustments: AtomicU64,
    requests_total: AtomicU64,
    errors_total: AtomicU64,
    // Gauges (stored as fixed-point u64)
    packing_waste_ratio: AtomicU64,
    packing_efficiency: AtomicU64,
    speculative_acceptance_rate: AtomicU64,
    speculative_draft_count: AtomicU64,
    speculative_efficiency: AtomicU64,
    throughput_speedup_ratio: AtomicU64,
    request_queue_depth: AtomicU64,
    active_sequences: AtomicU64,
    speculative_per_request_count: AtomicU64,
    // Histograms
    inference_latency_ns: DashMap<String, Vec<u64>>,
    // Per-request acceptance tracking
    per_request_acceptance: DashMap<SeqId, (AtomicU64, AtomicU64)>,
}

impl EnhancedMetricsCollector {
    pub fn new() -> Self {
        Self {
            cuda_graph_hits: AtomicU64::new(0),
            cuda_graph_misses: AtomicU64::new(0),
            packing_sequences: AtomicU64::new(0),
            speculative_adjustments: AtomicU64::new(0),
            requests_total: AtomicU64::new(0),
            errors_total: AtomicU64::new(0),
            packing_waste_ratio: AtomicU64::new(0),
            packing_efficiency: AtomicU64::new(0),
            speculative_acceptance_rate: AtomicU64::new(0),
            speculative_draft_count: AtomicU64::new(0),
            speculative_efficiency: AtomicU64::new(0),
            throughput_speedup_ratio: AtomicU64::new(0),
            request_queue_depth: AtomicU64::new(0),
            active_sequences: AtomicU64::new(0),
            speculative_per_request_count: AtomicU64::new(0),
            inference_latency_ns: DashMap::new(),
            per_request_acceptance: DashMap::new(),
        }
    }

    // CUDA Graph metrics
    pub fn record_cuda_graph_hit(&self) {
        self.cuda_graph_hits.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_cuda_graph_miss(&self) {
        self.cuda_graph_misses.fetch_add(1, Ordering::Relaxed);
    }

    // Packing metrics
    pub fn record_packing_sequence(&self) {
        self.packing_sequences.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_packing_efficiency(&self, ratio: f64) {
        let fixed = (ratio * 100000.0) as u64;
        self.packing_efficiency.store(fixed, Ordering::Relaxed);
    }

    pub fn record_packing_waste_ratio(&self, ratio: f64) {
        let fixed = (ratio * 100000.0) as u64;
        self.packing_waste_ratio.store(fixed, Ordering::Relaxed);
    }

    // Speculative metrics
    pub fn record_speculative_acceptance(&self, accepted: usize, total: usize) {
        if total > 0 {
            let rate = (accepted as f64 / total as f64 * 100000.0) as u64;
            self.speculative_acceptance_rate
                .store(rate, Ordering::Relaxed);
        }
    }

    pub fn record_speculative_draft_count(&self, count: u64) {
        self.speculative_draft_count.store(count, Ordering::Relaxed);
    }

    pub fn record_speculative_adjustment(&self) {
        self.speculative_adjustments.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_speculative_efficiency(&self, efficiency: f64) {
        let fixed = (efficiency * 100000.0) as u64;
        self.speculative_efficiency.store(fixed, Ordering::Relaxed);
    }

    pub fn record_throughput_speedup(&self, ratio: f64) {
        let fixed = (ratio * 100000.0) as u64;
        self.throughput_speedup_ratio
            .store(fixed, Ordering::Relaxed);
    }

    // Per-request acceptance tracking
    pub fn record_per_request_acceptance(&self, seq_id: SeqId, accepted: usize, total: usize) {
        let entry = self
            .per_request_acceptance
            .entry(seq_id)
            .or_insert_with(|| (AtomicU64::new(0), AtomicU64::new(0)));
        entry
            .value()
            .0
            .fetch_add(accepted as u64, Ordering::Relaxed);
        entry.value().1.fetch_add(total as u64, Ordering::Relaxed);
        self.speculative_per_request_count
            .store(self.per_request_acceptance.len() as u64, Ordering::Relaxed);
    }

    pub fn get_per_request_acceptance_rate(&self, seq_id: SeqId) -> f64 {
        self.per_request_acceptance
            .get(&seq_id)
            .map(|entry| {
                let accepted = entry.0.load(Ordering::Relaxed);
                let total = entry.1.load(Ordering::Relaxed);
                if total == 0 {
                    0.0
                } else {
                    accepted as f64 / total as f64
                }
            })
            .unwrap_or(0.0)
    }

    pub fn remove_per_request(&self, seq_id: SeqId) {
        self.per_request_acceptance.remove(&seq_id);
        self.speculative_per_request_count
            .store(self.per_request_acceptance.len() as u64, Ordering::Relaxed);
    }

    // System metrics
    pub fn record_request(&self) {
        self.requests_total.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_error(&self) {
        self.errors_total.fetch_add(1, Ordering::Relaxed);
    }

    pub fn set_queue_depth(&self, depth: u64) {
        self.request_queue_depth.store(depth, Ordering::Relaxed);
    }

    pub fn set_active_sequences(&self, count: u64) {
        self.active_sequences.store(count, Ordering::Relaxed);
    }

    pub fn record_inference_latency(&self, duration_ns: u64) {
        let mut buckets = self
            .inference_latency_ns
            .entry("inference".to_string())
            .or_default();
        buckets.push(duration_ns);
        if buckets.len() > 10000 {
            buckets.remove(0);
        }
    }

    // Getters for testing and export
    pub fn get_counter(&self, name: &str) -> u64 {
        match name {
            "cuda_graph_hits_total" => self.cuda_graph_hits.load(Ordering::Relaxed),
            "cuda_graph_misses_total" => self.cuda_graph_misses.load(Ordering::Relaxed),
            "packing_sequences_total" => self.packing_sequences.load(Ordering::Relaxed),
            "speculative_adjustments_total" => self.speculative_adjustments.load(Ordering::Relaxed),
            "requests_total" => self.requests_total.load(Ordering::Relaxed),
            "errors_total" => self.errors_total.load(Ordering::Relaxed),
            _ => 0,
        }
    }

    pub fn get_gauge(&self, name: &str) -> u64 {
        match name {
            "packing_efficiency" => self.packing_efficiency.load(Ordering::Relaxed),
            "packing_waste_ratio" => self.packing_waste_ratio.load(Ordering::Relaxed),
            "speculative_acceptance_rate" => {
                self.speculative_acceptance_rate.load(Ordering::Relaxed)
            }
            "speculative_draft_count" => self.speculative_draft_count.load(Ordering::Relaxed),
            "speculative_efficiency" => self.speculative_efficiency.load(Ordering::Relaxed),
            "throughput_speedup_ratio" => self.throughput_speedup_ratio.load(Ordering::Relaxed),
            "request_queue_depth" => self.request_queue_depth.load(Ordering::Relaxed),
            "active_sequences" => self.active_sequences.load(Ordering::Relaxed),
            "speculative_per_request_count" => {
                self.speculative_per_request_count.load(Ordering::Relaxed)
            }
            _ => 0,
        }
    }
}

impl Default for EnhancedMetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collector_records_cuda_graph_hit() {
        let collector = EnhancedMetricsCollector::new();
        collector.record_cuda_graph_hit();
        let hits = collector.get_counter("cuda_graph_hits_total");
        assert_eq!(hits, 1);
    }

    #[test]
    fn test_collector_records_packing_efficiency() {
        let collector = EnhancedMetricsCollector::new();
        collector.record_packing_efficiency(0.85);
        let efficiency = collector.get_gauge("packing_efficiency");
        assert_eq!(efficiency, 85000);
    }

    #[test]
    fn test_collector_records_speculative_acceptance() {
        let collector = EnhancedMetricsCollector::new();
        collector.record_speculative_acceptance(8, 10);
        let rate = collector.get_gauge("speculative_acceptance_rate");
        assert_eq!(rate, 80000);
    }

    #[test]
    fn test_collector_records_inference_latency() {
        let collector = EnhancedMetricsCollector::new();
        collector.record_inference_latency(1_000_000);
        collector.record_inference_latency(2_000_000);
        let buckets = collector.inference_latency_ns.get("inference").unwrap();
        assert_eq!(buckets.len(), 2);
    }

    // ---- Plan 17.4-H: Metrics Tests ----

    #[test]
    fn test_speculative_efficiency_basic() {
        let collector = EnhancedMetricsCollector::new();
        collector.record_speculative_efficiency(0.6667);
        let gauge = collector.get_gauge("speculative_efficiency");
        assert!(gauge > 66000 && gauge < 67000);
    }

    #[test]
    fn test_speculative_efficiency_zero() {
        let collector = EnhancedMetricsCollector::new();
        let gauge = collector.get_gauge("speculative_efficiency");
        assert_eq!(gauge, 0);
    }

    #[test]
    fn test_throughput_speedup_set_get() {
        let collector = EnhancedMetricsCollector::new();
        collector.record_throughput_speedup(1.5);
        let gauge = collector.get_gauge("throughput_speedup_ratio");
        assert_eq!(gauge, 150000);
    }

    #[test]
    fn test_throughput_speedup_default() {
        let collector = EnhancedMetricsCollector::new();
        let gauge = collector.get_gauge("throughput_speedup_ratio");
        assert_eq!(gauge, 0);
    }

    #[test]
    fn test_collector_records_speculative_efficiency() {
        let collector = EnhancedMetricsCollector::new();
        collector.record_speculative_efficiency(0.75);
        let gauge = collector.get_gauge("speculative_efficiency");
        assert_eq!(gauge, 75000);
    }
}
