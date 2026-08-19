// crates/core/src/metrics/collector/sampler/packing.rs
//
// Packing metrics: efficiency ratio + waste ratio. Both stored as
// fixed-point integers (× 100_000) for lock-free updates.

use super::EnhancedMetricsCollector;
use std::sync::atomic::Ordering;

impl EnhancedMetricsCollector {
    /// Records the batch packing efficiency (0-1, × `100_000` fixed-point).
    ///
    /// RIL ISS-089: no production caller exists — while the sequence-packing
    /// feature (ISS-064) is inert, `SequencePacker::pack_sequences` (the
    /// natural data source, it computes `padding_waste`) is never invoked, so
    /// this gauge is pinned at 0 on /metrics and OTLP. Wire it when ISS-064
    /// ships; keeping the gauge exposed preserves the metrics contract.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn record_packing_efficiency(&self, ratio: f64) {
        let fixed = (ratio * 100_000.0) as u64;
        self.packing_efficiency.store(fixed, Ordering::Relaxed);
    }
}
