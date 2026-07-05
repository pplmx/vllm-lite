//! Metrics collector namespace: re-exports [`EnhancedMetricsCollector`] + the metric value types in `metrics` and the sampling pipeline in `sampler`.
#![allow(clippy::module_name_repetitions)]
// crates/core/src/metrics/collector/mod.rs
//
// Facade for the metrics collector subsystem. Sub-modules:
// - `metrics` — metric type definitions (`DraftResolutionKind`, `DraftMetricsSnapshot`).
// - `sampler` — runtime sampling/recording via `EnhancedMetricsCollector`.

mod metrics;
mod sampler;

pub use metrics::{DraftMetricsSnapshot, DraftResolutionKind};
pub use sampler::EnhancedMetricsCollector;
