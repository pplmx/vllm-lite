// crates/core/src/metrics/collector/sampler/packing.rs
//
// Packing metrics: efficiency ratio + waste ratio. Both stored as
// fixed-point integers (× 100_000) for lock-free updates.

use super::EnhancedMetricsCollector;
use std::sync::atomic::Ordering;

impl EnhancedMetricsCollector {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn record_packing_efficiency(&self, ratio: f64) {
        let fixed = (ratio * 100_000.0) as u64;
        self.packing_efficiency.store(fixed, Ordering::Relaxed);
    }
}
