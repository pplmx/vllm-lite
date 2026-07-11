//! Per-step scheduler statistics: running count, waiting count, preempted count, last-update timestamp.
//!
//! Emitted every scheduler step into the metrics channel; used by the
//! Prometheus exporter to render queue-depth and preemption-rate gauges.
#![allow(clippy::module_name_repetitions)]
use std::time::Instant;

/// Telemetry snapshot of the scheduler: queue length, preemption count, prefix-cache hit rate, average batch fill ratio. Updated on every step and exposed via metrics.
#[derive(Debug, Clone)]
pub struct SchedulerStats {
    /// Cumulative batches emitted by the scheduler since process start.
    pub total_batches: usize,
    /// Cumulative prefill-phase requests processed.
    pub total_prefill_requests: usize,
    /// Cumulative decode-phase requests processed.
    pub total_decode_requests: usize,
    /// Cumulative preemption events.
    pub total_preemptions: usize,
    /// Cumulative KV-cache block evictions.
    pub total_evictions: usize,
    /// Running average batch size across all emitted batches.
    pub avg_batch_size: f64,
    /// Batch size of the most recently emitted batch.
    pub last_batch_size: usize,
    /// Cumulative batch-size sum (basis for the running average).
    pub batch_size_sum: u64,
    /// Instant of the most recent stats update.
    pub last_update: Instant,
}

impl Default for SchedulerStats {
    fn default() -> Self {
        Self::new()
    }
}

impl SchedulerStats {
    /// Construct a fresh stats record with all counters at zero and
    /// `last_update` set to the current instant.
    #[must_use]
    pub fn new() -> Self {
        Self {
            total_batches: 0,
            total_prefill_requests: 0,
            total_decode_requests: 0,
            total_preemptions: 0,
            total_evictions: 0,
            avg_batch_size: 0.0,
            last_batch_size: 0,
            batch_size_sum: 0,
            last_update: Instant::now(),
        }
    }

    /// Increment the lifetime prefill counter.
    pub const fn record_prefill(&mut self) {
        self.total_prefill_requests += 1;
    }

    /// Increment the lifetime decode counter.
    pub const fn record_decode(&mut self) {
        self.total_decode_requests += 1;
    }

    /// Increment the lifetime preemption counter.
    pub const fn record_preemption(&mut self) {
        self.total_preemptions += 1;
    }

    /// Increment the lifetime eviction counter.
    pub const fn record_eviction(&mut self) {
        self.total_evictions += 1;
    }
}
