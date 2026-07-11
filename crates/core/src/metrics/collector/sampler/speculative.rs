// crates/core/src/metrics/collector/sampler/speculative.rs
//
// Speculative decoding metrics: aggregate counters (acceptance rate,
// efficiency, throughput speedup, draft count, adjustment count) plus
// per-request accepted/total tracking.

use super::EnhancedMetricsCollector;
use std::sync::atomic::{AtomicU64, Ordering};
use vllm_traits::SeqId;

impl EnhancedMetricsCollector {
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        dead_code
    )]
    pub(crate) fn record_speculative_acceptance(&self, accepted: usize, total: usize) {
        if total > 0 {
            let rate = (accepted as f64 / total as f64 * 100_000.0) as u64;
            self.speculative_acceptance_rate
                .store(rate, Ordering::Relaxed);
        }
    }

    pub fn record_speculative_adjustment(&self) {
        self.speculative_adjustments.fetch_add(1, Ordering::Relaxed);
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn record_speculative_efficiency(&self, efficiency: f64) {
        let fixed = (efficiency * 100_000.0) as u64;
        self.speculative_efficiency.store(fixed, Ordering::Relaxed);
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, dead_code)]
    pub(crate) fn record_throughput_speedup(&self, ratio: f64) {
        let fixed = (ratio * 100_000.0) as u64;
        self.throughput_speedup_ratio
            .store(fixed, Ordering::Relaxed);
    }

    // Per-request acceptance tracking
    ///
    /// Records accepted/total counts for a per-request acceptance tracker.
    /// Implementation note: we cannot hold a `DashMap::Entry` across a
    /// `DashMap::len()` call (the shard held by the entry blocks the
    /// shard iteration that `len()` performs internally, causing a deadlock).
    /// We use a release-then-reacquire pattern: upsert the entry under its
    /// own shard lock, then drop the entry, then call `len()`.
    pub fn record_per_request_acceptance(&self, seq_id: SeqId, accepted: usize, total: usize) {
        // Upsert under a single shard lock; entry guard is dropped before len().
        {
            let entry = self
                .per_request_acceptance
                .entry(seq_id)
                .or_insert_with(|| (AtomicU64::new(0), AtomicU64::new(0)));
            entry
                .value()
                .0
                .fetch_add(accepted as u64, Ordering::Relaxed);
            entry.value().1.fetch_add(total as u64, Ordering::Relaxed);
        }
        // Entry guard is now dropped — safe to call len().
        let len = self.per_request_acceptance.len();
        self.speculative_per_request_count
            .store(len as u64, Ordering::Relaxed);
    }

    pub fn remove_per_request(&self, seq_id: SeqId) {
        self.per_request_acceptance.remove(&seq_id);
        self.speculative_per_request_count
            .store(self.per_request_acceptance.len() as u64, Ordering::Relaxed);
    }
}
