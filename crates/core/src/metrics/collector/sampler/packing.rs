// crates/core/src/metrics/collector/sampler/packing.rs
//
// Packing metrics: efficiency ratio + waste ratio. Both stored as
// fixed-point integers (× 100_000) for lock-free updates.

use super::EnhancedMetricsCollector;
use std::sync::atomic::Ordering;

impl EnhancedMetricsCollector {
    pub fn record_packing_sequence(&self) {
        self.packing_sequences.fetch_add(1, Ordering::Relaxed);
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn record_packing_efficiency(&self, ratio: f64) {
        let fixed = (ratio * 100_000.0) as u64;
        self.packing_efficiency.store(fixed, Ordering::Relaxed);
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn record_packing_waste_ratio(&self, ratio: f64) {
        let fixed = (ratio * 100_000.0) as u64;
        self.packing_waste_ratio.store(fixed, Ordering::Relaxed);
    }
}
