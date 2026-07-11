// crates/core/src/metrics/collector/sampler/runtime.rs
//
// Lock-free delegation + CUDA Graph metrics + system counters.
// All methods here either forward to `self.runtime` (the lock-free hot path)
// or touch small `AtomicU64` fields / a single `DashMap` for inference latency.

use super::super::super::lock_free::MetricsSnapshot;
use super::EnhancedMetricsCollector;
use std::sync::atomic::Ordering;

impl EnhancedMetricsCollector {
    /// Snapshot for engine/API export (latency, throughput, KV usage).
    pub fn snapshot(&self) -> MetricsSnapshot {
        self.runtime.snapshot()
    }

    /// Accumulate generated token count into throughput counters.
    pub fn record_tokens(&self, count: u64) {
        self.runtime.record_tokens(count);
    }

    /// Record the batch size of the most recent forward pass.
    pub fn record_batch_size(&self, size: usize) {
        self.runtime.record_batch_size(size);
    }

    /// Record end-to-end step latency in milliseconds.
    pub fn record_latency(&self, ms: f64) {
        self.runtime.record_latency(ms);
    }

    /// Update KV-cache occupancy counters (`used` blocks vs `total` capacity).
    pub fn record_kv_cache_usage(&self, used: u64, total: u64) {
        self.runtime.record_kv_cache_usage(used, total);
    }

    /// Increment prefix-cache hit counter after a radix-tree match.
    pub fn record_prefix_cache_hit(&self) {
        self.runtime.record_prefix_cache_hit();
    }

    /// Increment prefix-cache lookup counter (hit or miss).
    pub fn record_prefix_cache_request(&self) {
        self.runtime.record_prefix_cache_request();
    }

    /// Total prefix-cache hits since process start.
    pub fn prefix_cache_hits(&self) -> u64 {
        self.runtime.prefix_cache_hits()
    }

    /// Total prefix-cache lookups since process start.
    pub fn prefix_cache_requests(&self) -> u64 {
        self.runtime.prefix_cache_requests()
    }

    /// Increment CUDA Graph fast-path hit counter.
    pub fn record_cuda_graph_hit(&self) {
        self.cuda_graph_hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment CUDA Graph miss counter (fell back to eager forward).
    pub fn record_cuda_graph_miss(&self) {
        self.cuda_graph_misses.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment completed-request counter.
    pub fn record_request(&self) {
        self.runtime.record_request();
    }

    /// Publish current waiting-queue depth for observability exporters.
    pub fn set_queue_depth(&self, depth: u64) {
        self.request_queue_depth.store(depth, Ordering::Relaxed);
    }

    /// Publish count of sequences currently in the running set.
    pub fn set_active_sequences(&self, count: u64) {
        self.active_sequences.store(count, Ordering::Relaxed);
    }

    /// Append a single inference latency sample (nanoseconds) to the rolling histogram.
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
}
