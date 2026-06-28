// crates/core/src/metrics/collector/mod.rs
//
// Facade for the metrics collector subsystem. Sub-modules:
// - `metrics` — metric type definitions (`DraftResolutionKind`, `DraftMetricsSnapshot`).
// - `sampler` — runtime sampling/recording via `EnhancedMetricsCollector`.

mod metrics;
mod sampler;

pub use metrics::{DraftMetricsSnapshot, DraftResolutionKind};
pub use sampler::EnhancedMetricsCollector;
