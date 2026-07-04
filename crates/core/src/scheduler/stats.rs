#![allow(clippy::module_name_repetitions)]
use std::time::Instant;

/// Telemetry snapshot of the scheduler: queue length, preemption count, prefix-cache hit rate, average batch fill ratio. Updated on every step and exposed via metrics.
#[derive(Debug, Clone)]
pub struct SchedulerStats {
    pub total_batches: usize,
    pub total_prefill_requests: usize,
    pub total_decode_requests: usize,
    pub total_preemptions: usize,
    pub total_evictions: usize,
    pub avg_batch_size: f64,
    pub last_batch_size: usize,
    pub batch_size_sum: u64,
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

    /// Record a completed batch of `batch_size` sequences. Updates
    /// `total_batches`, `last_batch_size`, and `avg_batch_size` (running
    /// average). Also stamps `last_update` to "now".
    pub fn record_batch(&mut self, batch_size: usize) {
        self.total_batches += 1;
        self.last_batch_size = batch_size;
        self.batch_size_sum += u64::try_from(batch_size).unwrap_or(u64::MAX);
        // invariant: counters are bounded; u64/usize -> f64 precision loss is
        // acceptable for the running-average metric.
        #[allow(clippy::cast_precision_loss)]
        let avg = self.batch_size_sum as f64 / self.total_batches as f64;
        self.avg_batch_size = avg;
        self.last_update = Instant::now();
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
