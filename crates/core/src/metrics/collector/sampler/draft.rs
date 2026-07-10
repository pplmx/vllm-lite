// crates/core/src/metrics/collector/sampler/draft.rs
//
// v18.0 multi-model speculative decoding: draft-resolution counters
// (External / SelfSpec / None), draft-load-failure counter (FALL-01),
// and draft-runtime-error counter (FALL-02). All stored as plain
// `AtomicU64`s; snapshot helper bundles them into a typed struct.

use super::super::metrics::{DraftMetricsSnapshot, DraftResolutionKind};
use super::EnhancedMetricsCollector;
use std::sync::atomic::Ordering;

impl EnhancedMetricsCollector {
    /// Increment the draft-resolution counter for the given result kind.
    pub fn inc_draft_resolution(&self, kind: DraftResolutionKind) {
        match kind {
            DraftResolutionKind::External => {
                self.draft_resolutions_external_total
                    .fetch_add(1, Ordering::Relaxed);
            }
            DraftResolutionKind::SelfSpec => {
                self.draft_resolutions_self_spec_total
                    .fetch_add(1, Ordering::Relaxed);
            }
            DraftResolutionKind::None => {
                self.draft_resolutions_none_total
                    .fetch_add(1, Ordering::Relaxed);
            }
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
}
