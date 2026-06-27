//! collector: collector.

// crates/core/src/metrics/collector.rs
use super::lock_free::{LockFreeMetrics, MetricsSnapshot};
use dashmap::DashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use vllm_traits::SeqId;

/// Unified metrics collector for scheduler, engine, and HTTP export.
pub struct EnhancedMetricsCollector {
    runtime: LockFreeMetrics,
    cuda_graph_hits: AtomicU64,
    cuda_graph_misses: AtomicU64,
    packing_sequences: AtomicU64,
    speculative_adjustments: AtomicU64,
    errors_total: AtomicU64,
    packing_waste_ratio: AtomicU64,
    packing_efficiency: AtomicU64,
    speculative_acceptance_rate: AtomicU64,
    speculative_draft_count: AtomicU64,
    speculative_efficiency: AtomicU64,
    throughput_speedup_ratio: AtomicU64,
    request_queue_depth: AtomicU64,
    active_sequences: AtomicU64,
    speculative_per_request_count: AtomicU64,
    // v18.0 multi-model speculative decoding
    draft_resolutions_external_total: AtomicU64,
    draft_resolutions_self_spec_total: AtomicU64,
    draft_resolutions_none_total: AtomicU64,
    draft_load_failures_total: AtomicU64,
    draft_runtime_errors_total: AtomicU64,
    inference_latency_ns: DashMap<String, Vec<u64>>,
    per_request_acceptance: DashMap<SeqId, (AtomicU64, AtomicU64)>,
}

impl EnhancedMetricsCollector {
/// new: new.
    pub fn new() -> Self {
        Self {
            runtime: LockFreeMetrics::new(),
            cuda_graph_hits: AtomicU64::new(0),
            cuda_graph_misses: AtomicU64::new(0),
            packing_sequences: AtomicU64::new(0),
            speculative_adjustments: AtomicU64::new(0),
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
            draft_resolutions_external_total: AtomicU64::new(0),
            draft_resolutions_self_spec_total: AtomicU64::new(0),
            draft_resolutions_none_total: AtomicU64::new(0),
            draft_load_failures_total: AtomicU64::new(0),
            draft_runtime_errors_total: AtomicU64::new(0),
            inference_latency_ns: DashMap::new(),
            per_request_acceptance: DashMap::new(),
        }
    }

    /// Snapshot for engine/API export (latency, throughput, KV usage).
    pub fn snapshot(&self) -> MetricsSnapshot {
        self.runtime.snapshot()
    }

/// record_tokens: record tokens.
    pub fn record_tokens(&self, count: u64) {
        self.runtime.record_tokens(count);
    }

/// record_batch_size: record batch size.
    pub fn record_batch_size(&self, size: usize) {
        self.runtime.record_batch_size(size);
    }

/// record_latency: record latency.
    pub fn record_latency(&self, ms: f64) {
        self.runtime.record_latency(ms);
    }

/// record_kv_cache_usage: record kv cache usage.
    pub fn record_kv_cache_usage(&self, used: u64, total: u64) {
        self.runtime.record_kv_cache_usage(used, total);
    }

/// record_prefix_cache_hit: record prefix cache hit.
    pub fn record_prefix_cache_hit(&self) {
        self.runtime.record_prefix_cache_hit();
    }

/// record_prefix_cache_request: record prefix cache request.
    pub fn record_prefix_cache_request(&self) {
        self.runtime.record_prefix_cache_request();
    }

    // CUDA Graph metrics
/// record_cuda_graph_hit: record cuda graph hit.
    pub fn record_cuda_graph_hit(&self) {
        self.cuda_graph_hits.fetch_add(1, Ordering::Relaxed);
    }

/// record_cuda_graph_miss: record cuda graph miss.
    pub fn record_cuda_graph_miss(&self) {
        self.cuda_graph_misses.fetch_add(1, Ordering::Relaxed);
    }

    // Packing metrics
/// record_packing_sequence: record packing sequence.
    pub fn record_packing_sequence(&self) {
        self.packing_sequences.fetch_add(1, Ordering::Relaxed);
    }

/// record_packing_efficiency: record packing efficiency.
    pub fn record_packing_efficiency(&self, ratio: f64) {
        let fixed = (ratio * 100000.0) as u64;
        self.packing_efficiency.store(fixed, Ordering::Relaxed);
    }

/// record_packing_waste_ratio: record packing waste ratio.
    pub fn record_packing_waste_ratio(&self, ratio: f64) {
        let fixed = (ratio * 100000.0) as u64;
        self.packing_waste_ratio.store(fixed, Ordering::Relaxed);
    }

    // Speculative metrics
/// record_speculative_acceptance: record speculative acceptance.
    pub fn record_speculative_acceptance(&self, accepted: usize, total: usize) {
        if total > 0 {
            let rate = (accepted as f64 / total as f64 * 100000.0) as u64;
            self.speculative_acceptance_rate
                .store(rate, Ordering::Relaxed);
        }
    }

/// record_speculative_draft_count: record speculative draft count.
    pub fn record_speculative_draft_count(&self, count: u64) {
        self.speculative_draft_count.store(count, Ordering::Relaxed);
    }

/// record_speculative_adjustment: record speculative adjustment.
    pub fn record_speculative_adjustment(&self) {
        self.speculative_adjustments.fetch_add(1, Ordering::Relaxed);
    }

/// record_speculative_efficiency: record speculative efficiency.
    pub fn record_speculative_efficiency(&self, efficiency: f64) {
        let fixed = (efficiency * 100000.0) as u64;
        self.speculative_efficiency.store(fixed, Ordering::Relaxed);
    }

/// record_throughput_speedup: record throughput speedup.
    pub fn record_throughput_speedup(&self, ratio: f64) {
        let fixed = (ratio * 100000.0) as u64;
        self.throughput_speedup_ratio
            .store(fixed, Ordering::Relaxed);
    }

    // Per-request acceptance tracking
/// record_per_request_acceptance: record per request acceptance.
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

/// get_per_request_acceptance_rate: get per request acceptance rate.
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

/// remove_per_request: remove per request.
    pub fn remove_per_request(&self, seq_id: SeqId) {
        self.per_request_acceptance.remove(&seq_id);
        self.speculative_per_request_count
            .store(self.per_request_acceptance.len() as u64, Ordering::Relaxed);
    }

    // System metrics
/// record_request: record request.
    pub fn record_request(&self) {
        self.runtime.record_request();
    }

/// record_error: record error.
    pub fn record_error(&self) {
        self.errors_total.fetch_add(1, Ordering::Relaxed);
    }

/// set_queue_depth: set queue depth.
    pub fn set_queue_depth(&self, depth: u64) {
        self.request_queue_depth.store(depth, Ordering::Relaxed);
    }

/// set_active_sequences: set active sequences.
    pub fn set_active_sequences(&self, count: u64) {
        self.active_sequences.store(count, Ordering::Relaxed);
    }

/// record_inference_latency: record inference latency.
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

    // ───────────────────── v18.0 multi-model spec metrics ─────────────────────

    /// Increment the draft-resolution counter for the given result kind.
    /// `kind` is one of "external", "self_spec", "none".
    pub fn inc_draft_resolution(&self, kind: &str) {
        match kind {
            "external" => {
                self.draft_resolutions_external_total
                    .fetch_add(1, Ordering::Relaxed);
            }
            "self_spec" => {
                self.draft_resolutions_self_spec_total
                    .fetch_add(1, Ordering::Relaxed);
            }
            "none" => {
                self.draft_resolutions_none_total
                    .fetch_add(1, Ordering::Relaxed);
            }
            _ => {}
        }
    }

    /// Increment the draft-load-failure counter (FALL-01 trigger).
    pub fn inc_draft_load_failure(&self) {
        self.draft_load_failures_total
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Increment the draft-runtime-error counter (FALL-02 trigger).
    pub fn inc_draft_runtime_error(&self) {
        self.draft_runtime_errors_total
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Snapshot the v18.0 counters as a struct (for tests / exporters).
    pub fn draft_metrics_snapshot(&self) -> DraftMetricsSnapshot {
        DraftMetricsSnapshot {
            resolutions_external_total: self
                .draft_resolutions_external_total
                .load(Ordering::Relaxed),
            resolutions_self_spec_total: self
                .draft_resolutions_self_spec_total
                .load(Ordering::Relaxed),
            resolutions_none_total: self.draft_resolutions_none_total.load(Ordering::Relaxed),
            load_failures_total: self.draft_load_failures_total.load(Ordering::Relaxed),
            runtime_errors_total: self.draft_runtime_errors_total.load(Ordering::Relaxed),
        }
    }

    // Getters for testing and export
/// get_counter: get counter.
    pub fn get_counter(&self, name: &str) -> u64 {
        match name {
            "cuda_graph_hits_total" => self.cuda_graph_hits.load(Ordering::Relaxed),
            "cuda_graph_misses_total" => self.cuda_graph_misses.load(Ordering::Relaxed),
            "packing_sequences_total" => self.packing_sequences.load(Ordering::Relaxed),
            "speculative_adjustments_total" => self.speculative_adjustments.load(Ordering::Relaxed),
            "requests_total" => self.runtime.requests_total(),
            "errors_total" => self.errors_total.load(Ordering::Relaxed),
            _ => 0,
        }
    }

/// get_gauge: get gauge.
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

    #[test]
    fn test_collector_records_draft_resolution_metrics() {
        let collector = EnhancedMetricsCollector::new();
        collector.inc_draft_resolution("external");
        collector.inc_draft_resolution("external");
        collector.inc_draft_resolution("self_spec");
        collector.inc_draft_resolution("none");
        let snap = collector.draft_metrics_snapshot();
        assert_eq!(snap.resolutions_external_total, 2);
        assert_eq!(snap.resolutions_self_spec_total, 1);
        assert_eq!(snap.resolutions_none_total, 1);
    }

    #[test]
    fn test_collector_records_draft_failures() {
        let collector = EnhancedMetricsCollector::new();
        collector.inc_draft_load_failure();
        collector.inc_draft_load_failure();
        collector.inc_draft_runtime_error();
        let snap = collector.draft_metrics_snapshot();
        assert_eq!(snap.load_failures_total, 2);
        assert_eq!(snap.runtime_errors_total, 1);
    }
}

/// Snapshot of v18.0 multi-model speculative decoding metrics.
#[derive(Debug, Clone, Default)]
pub struct DraftMetricsSnapshot {
    pub resolutions_external_total: u64,
    pub resolutions_self_spec_total: u64,
    pub resolutions_none_total: u64,
    pub load_failures_total: u64,
    pub runtime_errors_total: u64,
}
