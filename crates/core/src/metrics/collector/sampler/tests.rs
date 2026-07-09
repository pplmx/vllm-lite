//! Unit tests for `EnhancedMetricsCollector` record / snapshot /
//! `DraftResolutionKind` parsing.
//!
//! Extracted from `sampler.rs` to keep the implementation file under
//! the project's 800-line soft cap. Exercises:
//!
//! - Counter / gauge recorders (`cuda_graph_hit`, `packing_efficiency`,
//!   `speculative_acceptance`, `inference_latency`,
//!   `speculative_efficiency`, `throughput_speedup`)
//! - Draft-resolution metric counters
//!   (`DraftResolutionKind::{External, SelfSpec, None}`)
//! - `DraftResolutionKind::parse` round-trip + aliases + invalid input
//! - `DraftResolutionKind::Display` formatting

use super::*;

#[test]
fn test_collector_records_cuda_graph_hit() {
    let collector = EnhancedMetricsCollector::new();
    collector.record_cuda_graph_hit();
    let hits = collector.get_counter("cuda_graph_hits_total");
    assert_eq!(hits, 1);
}

#[test]
fn test_collector_records_packing_efficiency() {
    let collector = EnhancedMetricsCollector::new();
    collector.record_packing_efficiency(0.85);
    let efficiency = collector.get_gauge("packing_efficiency");
    assert_eq!(efficiency, 85000);
}

#[test]
fn test_collector_records_speculative_acceptance() {
    let collector = EnhancedMetricsCollector::new();
    collector.record_speculative_acceptance(8, 10);
    let rate = collector.get_gauge("speculative_acceptance_rate");
    assert_eq!(rate, 80000);
}

#[test]
fn test_collector_records_inference_latency() {
    let collector = EnhancedMetricsCollector::new();
    collector.record_inference_latency(1_000_000);
    collector.record_inference_latency(2_000_000);
    assert_eq!(
        collector
            .inference_latency_ns
            .get("inference")
            .unwrap()
            .len(),
        2
    );
}

// ---- Plan 17.4-H: Metrics Tests ----

#[test]
fn test_speculative_efficiency_basic() {
    let collector = EnhancedMetricsCollector::new();
    collector.record_speculative_efficiency(0.6667);
    let gauge = collector.get_gauge("speculative_efficiency");
    assert!(gauge > 66000 && gauge < 67000);
}

#[test]
fn test_speculative_efficiency_zero() {
    let collector = EnhancedMetricsCollector::new();
    let gauge = collector.get_gauge("speculative_efficiency");
    assert_eq!(gauge, 0);
}

#[test]
fn test_throughput_speedup_set_get() {
    let collector = EnhancedMetricsCollector::new();
    collector.record_throughput_speedup(1.5);
    let gauge = collector.get_gauge("throughput_speedup_ratio");
    assert_eq!(gauge, 150_000);
}

#[test]
fn test_throughput_speedup_default() {
    let collector = EnhancedMetricsCollector::new();
    let gauge = collector.get_gauge("throughput_speedup_ratio");
    assert_eq!(gauge, 0);
}

#[test]
fn test_collector_records_speculative_efficiency() {
    let collector = EnhancedMetricsCollector::new();
    collector.record_speculative_efficiency(0.75);
    let gauge = collector.get_gauge("speculative_efficiency");
    assert_eq!(gauge, 75000);
}

#[test]
fn test_collector_records_draft_resolution_metrics() {
    let collector = EnhancedMetricsCollector::new();
    collector.inc_draft_resolution(DraftResolutionKind::External);
    collector.inc_draft_resolution(DraftResolutionKind::External);
    collector.inc_draft_resolution(DraftResolutionKind::SelfSpec);
    collector.inc_draft_resolution(DraftResolutionKind::None);
    let snap = collector.draft_metrics_snapshot();
    assert_eq!(snap.resolutions_external_total, 2);
    assert_eq!(snap.resolutions_self_spec_total, 1);
    assert_eq!(snap.resolutions_none_total, 1);
}

#[test]
fn draft_resolution_kind_parse_roundtrip() {
    for kind in [
        DraftResolutionKind::External,
        DraftResolutionKind::SelfSpec,
        DraftResolutionKind::None,
    ] {
        assert_eq!(DraftResolutionKind::parse(kind.as_str()), Some(kind));
    }
}

#[test]
fn draft_resolution_kind_parse_aliases() {
    assert_eq!(
        DraftResolutionKind::parse("self-spec"),
        Some(DraftResolutionKind::SelfSpec)
    );
    assert_eq!(
        DraftResolutionKind::parse("selfspec"),
        Some(DraftResolutionKind::SelfSpec)
    );
    assert_eq!(
        DraftResolutionKind::parse("SELF_SPEC"),
        Some(DraftResolutionKind::SelfSpec)
    );
    assert_eq!(
        DraftResolutionKind::parse("External"),
        Some(DraftResolutionKind::External)
    );
    assert_eq!(
        DraftResolutionKind::parse("NONE"),
        Some(DraftResolutionKind::None)
    );
}

#[test]
fn draft_resolution_kind_parse_invalid() {
    assert_eq!(DraftResolutionKind::parse("invalid"), None);
    assert_eq!(DraftResolutionKind::parse(""), None);
    assert_eq!(DraftResolutionKind::parse("shared"), None);
    assert_eq!(DraftResolutionKind::parse("per_request"), None);
}

#[test]
fn draft_resolution_kind_display() {
    assert_eq!(DraftResolutionKind::External.to_string(), "external");
    assert_eq!(DraftResolutionKind::SelfSpec.to_string(), "self_spec");
    assert_eq!(DraftResolutionKind::None.to_string(), "none");
}

#[test]
fn test_collector_records_draft_failures() {
    let collector = EnhancedMetricsCollector::new();
    collector.inc_draft_load_failure();
    collector.inc_draft_load_failure();
    collector.inc_draft_runtime_error();
    let snap = collector.draft_metrics_snapshot();
    assert_eq!(snap.load_failures_total, 2);
    assert_eq!(snap.runtime_errors_total, 1);
}
